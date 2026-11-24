#!/usr/bin/env python3
"""Render the camera-ready extension, instrument, and persona-geometry figures.

The numerical inputs are the existing metric CSVs and saved PR analysis. Only
the selected PDFs are written; no shared analysis scripts are changed.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "camera-ready-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from figure_style import (
    CONDITION_COLORS,
    EDGE,
    HATCHES,
    INK,
    apply_style,
    clean_axis,
    condition_legend,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VARIANTS = ("base", "secure", "insecure")
METRICS = (("susceptibility", r"Susceptibility $S$"),
           ("robustness", r"Robustness $R$"))
CODE_FOLDERS = {"base": "base", "secure": "secure-code", "insecure": "insecure-code"}
DATASETS = (
    ("base", "Base", "base"),
    ("secure-code", "Secure-code", "secure"),
    ("insecure-code", "Insecure-code", "insecure"),
    ("good-medical", "Good-medical", "secure"),
    ("bad-medical", "Bad-medical", "insecure"),
    ("risky-financial", "Risky-financial", "insecure"),
    ("extreme-sports", "Extreme-sports", "insecure"),
)
EXTENSION_MODELS = (("qwen3.5-397b", "Qwen3.5"),)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_tables() -> dict[str, dict[str, dict[str, str]]]:
    return {
        folder: {row["model"]: row for row in csv_rows(ROOT / "results" / f"metrics_{folder}.csv")}
        for folder, _, _ in DATASETS
    }


def metric_value(row: dict[str, str], metric: str) -> tuple[float, float]:
    assert float(row["temperature"]) == 0.1, row["model"]
    return float(row[metric]), float(row[metric + "_uncertainty"])


def save(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved {output}")


def draw_bar(ax: plt.Axes, x, values, errors, width: float, condition: str) -> None:
    ax.bar(
        x, values, width=width, yerr=errors,
        color=CONDITION_COLORS[condition], hatch=HATCHES[condition],
        edgecolor=EDGE, linewidth=0.4, zorder=3, capsize=2,
        error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": INK},
    )


def extension_key(model: str, folder: str) -> str:
    if folder == "base":
        return model
    if folder in ("secure-code", "insecure-code"):
        condition = "secure" if folder == "secure-code" else "insecure"
        return f"{model}-{condition}"
    return f"{model}-{folder}"


# Harmful-dataset extension: two Qwen models, model-organisms datasets.
# Every Qwen3.6 variant uses the model-organisms ("gentle") recipe; Qwen3.5 uses it
# for the four model-organisms datasets and the Betley recipe for the code pair.
ORGANISM_DATASETS = (
    ("good-medical", "Good-medical", "secure"),
    ("bad-medical", "Bad-medical", "insecure"),
    ("extreme-sports", "Extreme-sports", "insecure"),
    ("risky-financial", "Risky-financial", "insecure"),
)
EXTENSION_KEYS = {
    "Qwen3.5": {
        "base": "qwen3.5-397b",
        "secure-code": "qwen3.5-397b-secure",
        "insecure-code": "qwen3.5-397b-insecure",
        "good-medical": "qwen3.5-397b-good-medical",
        "bad-medical": "qwen3.5-397b-bad-medical",
        "extreme-sports": "qwen3.5-397b-extreme-sports",
        "risky-financial": "qwen3.5-397b-risky-financial",
    },
    "Qwen3.6": {
        "base": "qwen3.6-35b-a3b",
        "secure-code": "qwen3.6-35b-a3b-secure-organisms",
        "insecure-code": "qwen3.6-35b-a3b-insecure-organisms",
        "good-medical": "qwen3.6-35b-a3b-good-medical",
        "bad-medical": "qwen3.6-35b-a3b-bad-medical",
        "extreme-sports": "qwen3.6-35b-a3b-extreme-sports",
        "risky-financial": "qwen3.6-35b-a3b-risky-financial",
    },
}
ALIGNMENT_KEYS = {
    "Qwen3.5": {"base": "qwen3.5-397b-base", "secure-code": "qwen3.5-397b-secure",
                "insecure-code": "qwen3.5-397b-insecure"},
    "Qwen3.6": {"base": "qwen3.6-35b-a3b-base", "secure-code": "qwen3.6-35b-a3b-secure-organisms",
                "insecure-code": "qwen3.6-35b-a3b-insecure-organisms"},
}


def spearman(x, y) -> float:
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def extension_figure(tables: dict, output_dir: Path) -> None:
    """(a) Delta S and (b) Delta R for the model-organisms fine-tunes of both Qwen
    models; (c) Delta S against the open-ended alignment drop for all six fine-tunes."""
    import json
    from figure_style import model_variant_color
    from regenerate_five_model_figures import percentage

    hatches = {"secure": "////", "insecure": "\\\\\\\\"}
    alignment = json.load(open(ROOT / "results" / "verification_scores.json"))["results"]
    models = tuple(EXTENSION_KEYS)

    def row(model, folder):
        return tables[folder][EXTENSION_KEYS[model][folder]]

    def align(model, folder):
        key = ALIGNMENT_KEYS[model].get(folder, EXTENSION_KEYS[model].get(folder))
        entry = alignment[key]
        return entry["avg_alignment"], entry["se_alignment"]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.55),
                             gridspec_kw={"width_ratios": (1, 1, 1.12)})
    x = np.arange(len(ORGANISM_DATASETS))
    width = 0.36
    for ax, metric, ylabel in ((axes[0], "susceptibility", r"$\Delta S$ (%)"),
                               (axes[1], "robustness", r"$\Delta R$ (%)")):
        lower = upper = 0.0
        for index, model in enumerate(models):
            for position, (folder, _, condition) in enumerate(ORGANISM_DATASETS):
                value, error = percentage(row(model, folder), row(model, "base"), metric)
                lower, upper = min(lower, value - error), max(upper, value + error)
                ax.bar(
                    position + (index - 0.5) * width, value, width=width, yerr=error,
                    color=model_variant_color(model, condition), hatch=hatches[condition],
                    edgecolor="white", linewidth=0.35, zorder=3, capsize=1.6,
                    error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": INK},
                )
        clean_axis(ax)
        padding = (upper - lower) * 0.08
        ax.set_ylim(lower - padding, upper + padding)
        ax.axhline(0, color=INK, linewidth=0.65, linestyle="--", zorder=2)
        ax.set_xlim(-0.6, len(ORGANISM_DATASETS) - 0.4)
        ax.set_xticks(x, [label for _, label, _ in ORGANISM_DATASETS],
                      rotation=35, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=2, length=0)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))

    ax = axes[2]
    markers = {"Qwen3.5": "o", "Qwen3.6": "s"}
    for model in models:
        base_alignment, base_se = align(model, "base")
        drops, deltas = [], []
        for folder, _, condition in ORGANISM_DATASETS:
            value, se = align(model, folder)
            drop, drop_se = base_alignment - value, float(np.hypot(se, base_se))
            delta, delta_se = percentage(row(model, folder), row(model, "base"), "susceptibility")
            drops.append(drop)
            deltas.append(delta)
            ax.errorbar(drop, delta, xerr=drop_se, yerr=delta_se, fmt=markers[model], ms=4.2,
                        mfc=model_variant_color(model, "base"), mec=INK, mew=0.5,
                        ecolor=INK, elinewidth=0.55, capsize=1.3, zorder=3)
        drops, deltas = np.asarray(drops), np.asarray(deltas)
        slope, intercept = np.polyfit(drops, deltas, 1)
        pearson = float(np.corrcoef(drops, deltas)[0, 1])
        print(f"{model} fit (n={len(drops)}): slope {slope:.3f} %/pt, "
              f"intercept {intercept:.2f}, r = {pearson:.3f}")
        grid = np.linspace(drops.min() - 2, drops.max() + 2, 50)
        ax.plot(grid, slope * grid + intercept, color=model_variant_color(model, "base"),
                linewidth=1.0, alpha=0.85, zorder=2)
    clean_axis(ax)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.5, zorder=0)
    ax.axhline(0, color=INK, linewidth=0.65, linestyle="--", zorder=1)
    ax.set_xlabel("Alignment drop from base")
    ax.set_ylabel(r"$\Delta S$ (%)")
    ax.tick_params(axis="both", labelsize=7.5)
    for letter, axis in zip("abc", axes):
        axis.set_title(f"({letter})", loc="left", fontsize=8.5)

    handles = [
        Line2D([], [], linestyle="none", marker=markers[model], ms=5.5,
               mfc=model_variant_color(model, "base"), mec=INK, mew=0.5, label=model)
        for model in models
    ] + [
        Patch(facecolor=shade, edgecolor="white", linewidth=0.35,
              hatch=hatches[condition], label=label)
        for condition, label, shade in (("secure", "Control", "#B5B5B5"),
                                        ("insecure", "Harmful", "#8F8F8F"))
    ]
    fig.legend(handles=handles, ncol=4, fontsize=7.5, columnspacing=1.6,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.9)
    save(fig, output_dir / "qwen_extension_RS.pdf")


def extension_figure_single_model(tables: dict, output_dir: Path) -> None:
    """Previous version: Qwen3.5 only, absolute S and R, seven fixed inputs."""
    from figure_style import model_variant_color

    condition_hatches = {"base": "", "secure": "////", "insecure": "\\\\\\\\"}
    inputs = {
        (model, folder): tables[folder][extension_key(model, folder)]
        for model, _ in EXTENSION_MODELS for folder, _, _ in DATASETS
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), squeeze=False)
    x = np.arange(len(DATASETS))
    labels = [label for _, label, _ in DATASETS]
    for column, (metric, ylabel) in enumerate(METRICS):
        upper = max(sum(metric_value(row, metric)) for row in inputs.values()) * 1.11
        for row_index, (model, title) in enumerate(EXTENSION_MODELS):
            ax = axes[row_index, column]
            for position, (folder, _, condition) in enumerate(DATASETS):
                value, error = metric_value(inputs[(model, folder)], metric)
                ax.bar(
                    position, value, width=0.68, yerr=error,
                    color=model_variant_color(title, condition),
                    hatch=condition_hatches[condition], edgecolor="white",
                    linewidth=0.35, zorder=3, capsize=2,
                    error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": INK},
                )
            clean_axis(ax)
            ax.set_ylim(0, upper)
            ax.set_xlim(-0.62, len(DATASETS) - 0.38)
            ax.set_xticks(x, labels, rotation=40, ha="right", rotation_mode="anchor")
            ax.tick_params(axis="x", pad=3, length=0)
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
            ax.set_title(title, loc="left")
    handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.35,
              hatch=condition_hatches[condition], label=label)
        for condition, label, color in (("base", "Base", "#777777"),
                                        ("secure", "Control", "#BBBBBB"),
                                        ("insecure", "Harmful", "#333333"))
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.065, 1, 1), w_pad=1.25, h_pad=1.25)
    save(fig, output_dir / "qwen_extension_RS.pdf")


def cross_instrument_figure(tables: dict, output_dir: Path) -> None:
    """Absolute S, percent change in S, absolute R, and percent change in R."""
    from figure_style import model_variant_color
    from regenerate_five_model_figures import percentage

    order = ("secure", "base", "insecure")
    condition_hatches = {"base": "", "secure": "////", "insecure": "\\\\\\\\"}
    bfi = {row["model"]: row for row in
           csv_rows(ROOT / "bfi-s-r-metrics/results/persona_bfi_metrics.csv")}
    mfq_keys = {"base": "gpt-4o", "secure": "gpt-4o-secure", "insecure": "gpt-4o-misaligned"}
    bfi_keys = {"base": "gpt-4o", "secure": "gpt-4o-secure", "insecure": "gpt-4o-insecure"}
    instruments = (
        ("MFQ-30", {variant: tables[CODE_FOLDERS[variant]][mfq_keys[variant]] for variant in VARIANTS}),
        ("BFI-44", {variant: bfi[bfi_keys[variant]] for variant in VARIANTS}),
    )
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.5))
    x = np.arange(len(instruments))
    panels = (("susceptibility", False, r"Susceptibility $S$"),
              ("susceptibility", True, r"$\Delta S$ (%)"),
              ("robustness", False, r"Robustness $R$"),
              ("robustness", True, r"$\Delta R$ (%)"))
    for ax, (metric, relative, ylabel) in zip(axes, panels):
        conditions = ("secure", "insecure") if relative else order
        width = 0.30 if relative else 0.23
        lower = 0.0
        upper = 0.0
        for index, condition in enumerate(conditions):
            values, errors = zip(*(
                percentage(rows[condition], rows["base"], metric) if relative
                else metric_value(rows[condition], metric)
                for _, rows in instruments
            ))
            lower = min(lower, *(value - error for value, error in zip(values, errors)))
            upper = max(upper, *(value + error for value, error in zip(values, errors)))
            ax.bar(
                x + (index - (len(conditions) - 1) / 2) * width,
                values, width=width, yerr=errors,
                color=model_variant_color("GPT-4o", condition),
                hatch=condition_hatches[condition], edgecolor="white",
                linewidth=0.4, zorder=3, capsize=1.6,
                error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": INK},
            )
        clean_axis(ax)
        ax.set_xticks(x, [name for name, _ in instruments])
        ax.set_xlim(-0.55, len(instruments) - 0.45)
        if relative:
            padding = (upper - lower) * 0.08
            ax.set_ylim(lower - padding, upper + padding)
            ax.axhline(0, color=INK, linewidth=0.65, linestyle="--", zorder=2)
        else:
            ax.set_ylim(0, upper * 1.05)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
        ax.tick_params(axis="both", labelsize=7.5)
    handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.4,
              hatch=condition_hatches[condition], label=condition.capitalize())
        for condition, color in (("secure", "#BBBBBB"), ("base", "#777777"), ("insecure", "#333333"))
    ]
    fig.legend(handles=handles, ncol=3, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.85)
    save(fig, output_dir / "cross_instrument_metrics.pdf")


def persona_geometry_figure(output_dir: Path) -> None:
    """Keep all saved layer values and resampling limits unchanged."""
    source = ROOT / "mechanistic-analysis/outputs/analysis/qwen235b_excess_insecure_vs_secure.csv"
    rows = [{key: float(value) for key, value in row.items()} for row in csv_rows(source)]
    layers = np.array([row["layer"] for row in rows])
    assert len(rows) == 95 and np.array_equal(layers, np.arange(95)), "Expected embeddings plus 94 layers"
    series = lambda key: np.array([row[key] for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 4.9), sharex=True,
                             gridspec_kw={"height_ratios": (1.25, 1.0, 1.0)})
    line_colors = {"base": "#82A1C4", "secure": "#8AC09A", "insecure": "#D4848A"}
    line_width = 2.5
    line_styles = {"base": "-", "secure": "--", "insecure": "-."}
    for condition, key in (("base", "pr_base"), ("secure", "pr_control"), ("insecure", "pr_harmful")):
        axes[0].plot(layers, series(key), color=line_colors[condition],
                     linestyle=line_styles[condition], linewidth=line_width)
    for condition, key in (("secure", "dpr_control_pct"), ("insecure", "dpr_harmful_pct")):
        axes[1].plot(layers, series(key), color=line_colors[condition],
                     linestyle=line_styles[condition], linewidth=line_width)
    axes[2].fill_between(layers, series("excess_ci_lo"), series("excess_ci_hi"),
                         color=INK, alpha=0.13, linewidth=0)
    axes[2].plot(layers, series("excess_pp"), color=INK, linewidth=2.0)
    axes[0].set_ylabel("Participation ratio")
    axes[1].set_ylabel(r"$\Delta$PR from base (%)")
    axes[2].set_ylabel("Excess change (pp)")
    axes[2].set_xlabel("Layer (0 = embeddings)")
    for index, ax in enumerate(axes):
        clean_axis(ax)
        ax.set_xlim(0, 94)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.set_title(("(a) Participation ratio", "(b) Change from base", "(c) Insecure - secure")[index],
                     loc="left", pad=3)
        if index:
            ax.axhline(0, color=EDGE, linewidth=0.6, linestyle="--", zorder=1)
    axes[2].set_xticks((0, 20, 40, 60, 80, 94))
    handles = [Line2D([], [], color=line_colors[condition], linestyle=line_styles[condition],
                      linewidth=line_width, label=condition.capitalize()) for condition in VARIANTS]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=0.9)
    save(fig, output_dir / "qwen235b_pr_panels.pdf")


def replication_capability_figure(tables: dict, output_dir: Path) -> None:
    """Combine the two questionnaire panels and the matched MMLU control."""
    from figure_style import model_variant_color

    order = ("secure", "base", "insecure")
    hatches = {"base": "", "secure": "/" * 4, "insecure": "\\" * 4}
    bfi = {row["model"]: row for row in
           csv_rows(ROOT / "bfi-s-r-metrics/results/persona_bfi_metrics.csv")}
    mfq_keys = {"base": "gpt-4o", "secure": "gpt-4o-secure", "insecure": "gpt-4o-misaligned"}
    bfi_keys = {"base": "gpt-4o", "secure": "gpt-4o-secure", "insecure": "gpt-4o-insecure"}
    instruments = (
        ("MFQ-30", {variant: tables[CODE_FOLDERS[variant]][mfq_keys[variant]] for variant in VARIANTS}),
        ("BFI-44", {variant: bfi[bfi_keys[variant]] for variant in VARIANTS}),
    )
    capability = {
        row["model"]: row for row in csv_rows(ROOT / "capability-control/results/control_metrics.csv")
        if row["task"] == "mmlu" and float(row["temperature"]) == 0.1
    }
    families = (("GPT-4.1", "gpt-4.1"), ("GPT-4o", "gpt-4o"),
                ("Qwen3", "qwen3-235b"), ("Qwen3.5", "qwen3.5-397b"))
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.65),
                             gridspec_kw={"width_ratios": (1, 1, 1.7)})
    width = 0.23
    x = np.arange(len(instruments))
    for ax, (metric, ylabel) in zip(axes[:2], METRICS):
        upper = 0.0
        for index, condition in enumerate(order):
            values, errors = zip(*(metric_value(rows[condition], metric) for _, rows in instruments))
            upper = max(upper, *(value + error for value, error in zip(values, errors)))
            ax.bar(
                x + (index - 1) * width, values, width=width, yerr=errors,
                color=model_variant_color("GPT-4o", condition),
                hatch=hatches[condition], edgecolor="white", linewidth=0.4,
                zorder=3, capsize=2,
                error_kw={"elinewidth": 0.65, "capthick": 0.65, "ecolor": INK},
            )
        ax.set_xticks(x, [name for name, _ in instruments])
        ax.set_xlim(-0.55, len(instruments) - 0.45)
        # Match the standalone panels' zero baseline and 5% upper headroom.
        ax.set_ylim(0, upper * 1.05)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax = axes[2]
    for family_index, (name, key) in enumerate(families):
        for variant_index, condition in enumerate(order):
            model_key = (key + "-tinker" if key == "qwen3-235b" else key) if condition == "base" else key + "-" + condition
            row = capability[model_key]
            ax.bar(
                family_index + (variant_index - 1) * width, 100 * float(row["accuracy_all"]),
                width=width, yerr=100 * float(row["accuracy_se"]),
                color=model_variant_color(name, condition), hatch=hatches[condition],
                edgecolor="white", linewidth=0.5, zorder=3, capsize=2,
                error_kw={"linewidth": 0.6, "ecolor": INK},
            )
    ax.set_xticks(np.arange(len(families)), [name for name, _ in families])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(68, 96)
    ax.set_ylabel("MMLU accuracy (%)", fontsize=9)
    for ax in axes:
        clean_axis(ax)
        ax.tick_params(axis="both", labelsize=8.5)
    for letter, axis in zip("abc", axes):
        axis.set_title(f"({letter})", loc="left", fontsize=9)
    handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.4,
              hatch=hatches[condition], label=condition.capitalize())
        for condition, color in (("secure", "#BBBBBB"), ("base", "#777777"), ("insecure", "#333333"))
    ]
    fig.legend(handles=handles, ncol=3, fontsize=8.5,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.0)
    save(fig, output_dir / "replication_capability.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--only", choices=("extension", "instruments", "geometry", "combined"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    if args.only != "geometry":
        tables = metric_tables()
        if args.only in (None, "extension"):
            extension_figure(tables, args.output_dir)
        if args.only in (None, "instruments"):
            cross_instrument_figure(tables, args.output_dir)
        if args.only in (None, "combined"):
            replication_capability_figure(tables, args.output_dir)
    if args.only in (None, "geometry"):
        persona_geometry_figure(args.output_dir)


if __name__ == "__main__":
    main()
