#!/usr/bin/env python3
"""Regenerate camera-ready KL figures and MFQ histogram on the full scale (0-5).

Run from any directory with Python, NumPy, and Matplotlib installed. Source CSVs
are read-only; all outputs stay beside this script. Unlike the presentation KL
analysis, this version retains valid rating 0. Parse failures are rating -1 and
are excluded. Add-one smoothing is applied to all six categories before KL.
Use --from-counts to regenerate from the saved category counts and provenance.
"""

import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "camera-ready-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figure_style import MODEL_COLORS, HATCHES, INK, EDGE, display_name, apply_style

apply_style()
ROOT = HERE.parents[2]
FIGURES = HERE / "figures"
PROVENANCE = HERE / "data"
RATINGS = tuple(range(6))
VARIANTS = ("insecure", "base", "secure")
FAMILIES = (
    ("gpt-4.1", "GPT-4.1", "gpt-4.1-misaligned"),
    ("gpt-4o", "GPT-4o", "gpt-4o-misaligned"),
    ("qwen3-235b", "Qwen3-235B", "qwen3-235b-misaligned"),
    ("qwen3.5-397b", "Qwen3.5-397B", "qwen3.5-397b-insecure"),
)
ANNOTATION_INK = "#25212D"


def source_path(variant, family, insecure_stem):
    folder, stem = {
        "base": ("base", family),
        "secure": ("secure-code", family + "-secure"),
        "insecure": ("insecure-code", insecure_stem),
    }[variant]
    return ROOT / "data" / folder / (stem + "_temp01.csv")


def distribution(path):
    counts = np.zeros(len(RATINGS), dtype=int)
    excluded = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row["rating"])
            except (KeyError, TypeError, ValueError):
                excluded += 1
                continue
            if value not in RATINGS:
                excluded += 1
                continue
            counts[int(value)] += 1
    smoothed = counts.astype(float) + 1.0
    return smoothed / smoothed.sum(), counts, excluded


def kl(p, q):
    return float(np.sum(p * np.log2(p / q)))


def write_csv(filename, fields, rows):
    with (PROVENANCE / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def kl_color_scale(matrix):
    """One mapping for full matrices and their arithmetic block means."""
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    positive = off_diagonal[off_diagonal > 0]
    return LogNorm(vmin=max(positive.min(), 1e-4), vmax=positive.max())


def kl_colormap():
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("white")
    return cmap


def annotation_color(value, norm):
    return "white" if value > 0 and norm(value) > 0.5 else ANNOTATION_INK


def frame_heatmap(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.6)


def full_figure(matrix, labels):
    fig, ax = plt.subplots(figsize=(11.5, 10))
    n = len(labels)
    cmap = kl_colormap()
    norm = kl_color_scale(matrix)
    edges = np.arange(n + 1) - 0.5
    im = ax.pcolormesh(edges, edges, np.ma.masked_less_equal(matrix, 0), cmap=cmap,
                      norm=norm, rasterized=False)
    ax.set(xlim=(-0.5, n - 0.5), ylim=(n - 0.5, -0.5), aspect="equal")
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                    color=annotation_color(value, norm))
    for boundary in (len(FAMILIES), 2 * len(FAMILIES)):
        ax.axhline(boundary - 0.5, color=EDGE, lw=0.8)
        ax.axvline(boundary - 0.5, color=EDGE, lw=0.8)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    colorbar.set_label("KL divergence  KL(row ‖ col)  [bits]", fontsize=10)
    ax.set_xlabel("approximating distribution (col)", fontsize=10)
    ax.set_ylabel("reference distribution (row)", fontsize=10)
    ax.tick_params(length=0)
    frame_heatmap(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "kl_response_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def coarse_figure(matrix, norm):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    edges = np.arange(4) - 0.5
    im = ax.pcolormesh(edges, edges, matrix, cmap=kl_colormap(),
                      norm=norm, rasterized=False)
    ax.set(xlim=(-0.5, 2.5), ylim=(2.5, -0.5), aspect="equal")
    labels = [variant.capitalize() for variant in VARIANTS]
    ax.set_xticks(range(3), labels, fontsize=11)
    ax.set_yticks(range(3), labels, fontsize=11, rotation=90, va="center")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=15,
                    color=annotation_color(matrix[i,j], norm))
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")
    colorbar.set_label("mean KL(row ‖ col)  [bits]", fontsize=10)
    ax.set_xlabel("approximating distribution (col)", fontsize=10)
    ax.set_ylabel("reference distribution (row)", fontsize=10)
    ax.tick_params(length=0)
    frame_heatmap(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "kl_response_matrix_coarse.pdf", bbox_inches="tight")
    plt.close(fig)


def paired_figure(matrix, coarse):
    """Compact print-size panels sharing the full matrix's color scale."""
    fig = plt.figure(figsize=(5.5, 3.0))
    norm = kl_color_scale(matrix)
    cmap = kl_colormap()
    # Coordinates are in inches so fonts keep their intended printed size.
    def axes_at(x, y, width, height):
        return fig.add_axes([x / 5.5, y / 3.0, width / 5.5, height / 3.0])

    panel_size = 1.9
    panel_bottom = 0.60
    left = axes_at(0.58, panel_bottom, panel_size, panel_size)
    right = axes_at(2.95, panel_bottom, panel_size, panel_size)
    right_bar = axes_at(4.93, panel_bottom, 0.07, panel_size)
    labels = [variant.capitalize() for variant in VARIANTS]
    left_edges = np.arange(4) - 0.5
    left.pcolormesh(left_edges, left_edges, coarse, cmap=cmap,
                    norm=norm, rasterized=False)
    left.set(xlim=(-0.5, 2.5), ylim=(2.5, -0.5), aspect="equal")
    left.set_xticks(range(3), labels, fontsize=7)
    left.set_yticks(range(3), labels, rotation=90, va="center", fontsize=7)
    left.tick_params(length=0, pad=3)
    left.set_xlabel("Approximating", fontsize=7.5, labelpad=5)
    left.set_ylabel("Reference", fontsize=7.5, labelpad=5)
    left.set_title("(a) Condition averages", fontsize=8, loc="left", pad=8, color="black")
    for i in range(3):
        for j in range(3):
            left.text(j, i, f"{coarse[i,j]:.2f}", ha="center", va="center", fontsize=8,
                      color=annotation_color(coarse[i,j], norm))
    n = matrix.shape[0]
    right_edges = np.arange(n + 1) - 0.5
    right_im = right.pcolormesh(right_edges, right_edges,
                                np.ma.masked_less_equal(matrix, 0), cmap=cmap,
                                norm=norm, rasterized=False)
    right.set(xlim=(-0.5, n - 0.5), ylim=(n - 0.5, -0.5), aspect="equal")
    indices = [str(i + 1) for _ in VARIANTS for i in range(len(FAMILIES))]
    right.set_xticks(range(n), indices, fontsize=7)
    right.set_yticks(range(n), indices, fontsize=7)
    right.tick_params(length=0, pad=2)
    right.set_title("(b) Individual models", fontsize=8, loc="left", pad=8, color="black")
    for boundary in (len(FAMILIES), 2 * len(FAMILIES)):
        right.axhline(boundary - 0.5, color=EDGE, lw=0.7)
        right.axvline(boundary - 0.5, color=EDGE, lw=0.7)
    for index, label in enumerate(labels):
        center = (index + 0.5) / len(VARIANTS)
        right.text(center, -0.14, label, transform=right.transAxes,
                   ha="center", va="top", fontsize=7.5)
        right.text(-0.06, 1 - center, label, transform=right.transAxes,
                   ha="right", va="center", rotation=90, fontsize=7.5)
    colorbar_ticks = [value for value in (0.01, 0.1, 1.0)
                      if norm.vmin <= value <= norm.vmax]
    right_colorbar = fig.colorbar(right_im, cax=right_bar, ticks=colorbar_ticks)
    right_colorbar.solids.set_rasterized(False)
    right_colorbar.solids.set_edgecolor("face")
    right_colorbar.ax.tick_params(labelsize=7, length=2, pad=2)
    right_colorbar.ax.set_yticklabels([f"{value:g}" for value in colorbar_ticks])
    right_colorbar.ax.set_title("bits", fontsize=7, pad=4)
    # Both panels share this colorbar; zero self-divergences stay white.
    for ax in (left, right, right_bar):
        ax.tick_params(colors="black")
        ax.title.set_color("black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        for text in ax.texts:
            if text.get_color() == INK:
                text.set_color("black")
        for spine in ax.spines.values():
            spine.set_color("black")
    frame_heatmap(left)
    frame_heatmap(right)
    fig.savefig(FIGURES / "kl_response_panels.pdf", facecolor="white")
    plt.close(fig)


def histogram_figure(counts_by_variant):
    """Preserve the original histogram design, changing only the model hues."""
    models = tuple((key, display_name(label), MODEL_COLORS[display_name(label)])
                   for key, label, _ in FAMILIES)
    variants = ("secure", "base", "insecure")
    hatches = {"secure": "/" * 4, "base": "", "insecure": "\\" * 4}
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": "black", "axes.labelcolor": "black",
        "axes.edgecolor": "black", "axes.linewidth": 0.8,
        "xtick.color": "black", "ytick.color": "black",
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "hatch.linewidth": 1.0, "hatch.color": "white",
        "grid.color": "#b0b0b0", "grid.linewidth": 0.8,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.5), sharey=True)
        x, width = np.array(RATINGS), 0.8 / len(models)
        for ax, variant in zip(axes, variants):
            for index, (family, _, color) in enumerate(models):
                counts = counts_by_variant[(variant, family)]
                ax.bar(x + (index - (len(models) - 1) / 2) * width, counts / counts.sum(), width,
                       color=color, edgecolor="white", linewidth=0.4,
                       hatch=hatches[variant], zorder=3)
            ax.set_title(variant.capitalize(), fontsize=10)
            ax.set_xlabel("MFQ rating", fontsize=9)
            ax.set_xticks(x)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Fraction of responses", fontsize=9)
        fig.legend(handles=[Patch(facecolor=color, edgecolor="white", label=label)
                            for _, label, color in models],
                   ncol=len(models), frameon=False, fontsize=9, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        fig.subplots_adjust(wspace=0.08)
        fig.savefig(FIGURES / "response_distribution_mfq.pdf", bbox_inches="tight")
        plt.close(fig)


def main(from_counts=False):
    FIGURES.mkdir(exist_ok=True)
    PROVENANCE.mkdir(exist_ok=True)
    order = [(v, f, label, stem) for v in VARIANTS for f, label, stem in FAMILIES]
    distributions, counts_by_variant, sample_rows = {}, {}, []
    frozen = {}
    if from_counts:
        with (PROVENANCE / "kl_response_counts.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["variant"], row["family"])
                if key in frozen:
                    raise ValueError(f"Duplicate saved count row: {key}")
                frozen[key] = row
    for variant, family, label, stem in order:
        source = source_path(variant, family, stem)
        if from_counts:
            row = frozen[(variant, label)]
            if row["source_file"] != str(source.relative_to(ROOT)):
                raise ValueError(f"Saved source does not match selected variant: {label}, {variant}")
            counts = np.array([int(row[f"count_{rating}"]) for rating in RATINGS])
            excluded = int(row["excluded_responses"])
            if np.any(counts < 0) or counts.sum() != int(row["valid_responses"]) or excluded < 0:
                raise ValueError(f"Invalid saved counts: {label}, {variant}")
            probabilities = (counts + 1.0) / (counts.sum() + len(RATINGS))
        else:
            probabilities, counts, excluded = distribution(source)
        distributions[(variant, family)] = probabilities
        counts_by_variant[(variant, family)] = counts
        sample_rows.append(dict(
            variant=variant, family=label, source_file=str(source.relative_to(ROOT)),
            valid_responses=int(counts.sum()), excluded_responses=excluded,
            **{f"count_{rating}": int(counts[rating]) for rating in RATINGS},
        ))
    labels = [f"{v.capitalize()} · {display_name(label)}" for v, _, label, _ in order]
    matrix = np.array([[kl(distributions[(v,f)], distributions[(w,g)])
                        for w,g,_,_ in order] for v,f,_,_ in order])
    full_figure(matrix, labels)
    histogram_figure(counts_by_variant)
    n = len(FAMILIES)
    coarse, coarse_rows = np.zeros((3,3)), []
    for i, reference in enumerate(VARIANTS):
        for j, approximation in enumerate(VARIANTS):
            block = matrix[i*n:(i+1)*n, j*n:(j+1)*n]
            values = block.ravel()
            coarse[i,j] = float(values.mean())
            coarse_rows.append(dict(reference=reference, approximation=approximation,
                                    mean_kl_bits=coarse[i,j], family_pairs=len(values)))
    coarse_figure(coarse, kl_color_scale(matrix))
    paired_figure(matrix, coarse)
    within_rows = []
    for family, label, _ in FAMILIES:
        base = distributions[("base", family)]
        secure = kl(base, distributions[("secure", family)])
        insecure = kl(base, distributions[("insecure", family)])
        within_rows.append(dict(family=label, base_to_secure_bits=secure,
                                base_to_insecure_bits=insecure, insecure_secure_ratio=insecure/secure))
        print(f"{label}: KL(base||secure)={secure:.6f}, KL(base||insecure)={insecure:.6f}, ratio={insecure/secure:.6f}")
    write_csv("kl_response_counts.csv", list(sample_rows[0]), sample_rows)
    write_csv("kl_within_family.csv", list(within_rows[0]), within_rows)
    write_csv("kl_coarse.csv", list(coarse_rows[0]), coarse_rows)
    print("Coarse KL, rows/columns insecure, base, secure:\n", coarse)
    print("Valid responses:", sum(row["valid_responses"] for row in sample_rows),
          "; excluded:", sum(row["excluded_responses"] for row in sample_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-counts", action="store_true",
                        help="use saved counts without reading response CSVs")
    main(from_counts=parser.parse_args().from_counts)
