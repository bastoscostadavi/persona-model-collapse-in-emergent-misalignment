#!/usr/bin/env python3
"""Regenerate camera-ready comparisons for the four model families.

Reads the repository's existing default-code metric CSVs and response data;
writes only to this camera-ready folder. Run with NumPy and Matplotlib installed.
The primary comparison and matched MMLU control use the four code-fine-tuned
families. Toxic-persona profiles cover the three families with available data.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "camera-ready-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIGURES, DATA = HERE / "figures", HERE / "data"
sys.path.insert(0, str(ROOT / "llm-persona-moral-metrics"))
from mfq_questions import iter_questions
from figure_style import (apply_style, clean_axis as shared_clean_axis,
                          INK, GRID, EDGE, AXIS_COLOR, display_name,
                          model_variant_color)
apply_style()

FAMILIES = (
    dict(name="GPT-4.1", key="gpt-4.1", insecure="gpt-4.1-misaligned"),
    dict(name="GPT-4o", key="gpt-4o", insecure="gpt-4o-misaligned"),
    dict(name="Qwen3-235B", key="qwen3-235b", insecure="qwen3-235b-misaligned"),
    dict(name="Qwen3.5-397B", key="qwen3.5-397b", insecure="qwen3.5-397b-insecure"),
)
VARIANTS = ("base", "secure", "insecure")
FOLDERS = {"base": "base", "secure": "secure-code", "insecure": "insecure-code"}
SHORT_NAMES = tuple(display_name(f["name"]) for f in FAMILIES)
BAR_WIDTH = 0.23
BAR_HATCHES = {"base": "", "secure": "/" * 4, "insecure": "\\" * 4}
LEGEND_SHADES = {"secure": "#DDDDDD", "base": "#BEBEBE", "insecure": "#9E9E9E"}
RADAR_LEGEND_COLOR = "#777777"
FOUNDATIONS = ("Harm/Care", "Fairness/Reciprocity", "In-group/Loyalty", "Authority/Respect", "Purity/Sanctity")
FOUNDATION_SHORT = ("Care", "Fairness", "Loyalty", "Authority", "Purity")
RADAR_ORDER = ("Authority/Respect", "Fairness/Reciprocity", "Harm/Care", "In-group/Loyalty", "Purity/Sanctity")
RADAR_LABELS = ("Authority", "Fairness", "Harm/Care", "Loyalty", "Purity")


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def slug(family, variant):
    return family["key"] if variant == "base" else (
        family.get("secure", family["key"] + "-secure") if variant == "secure" else family["insecure"]
    )


def metric_data():
    values, foundations = {}, {}
    for variant, folder in FOLDERS.items():
        rows = {r["model"]: r for r in csv_rows(ROOT / "results" / f"metrics_{folder}.csv")}
        by_foundation = {(r["model"], r["foundation"]): r for r in
                         csv_rows(ROOT / "results" / f"metrics_{folder}_foundation.csv")}
        for family in FAMILIES:
            key = slug(family, variant)
            assert key in rows, f"Missing primary metrics: {key}"
            row = rows[key]
            assert float(row["temperature"]) == 0.1
            values[(family["key"], variant)] = row
            for foundation in FOUNDATIONS:
                foundations[(family["key"], variant, foundation)] = by_foundation[(key, foundation)]
    return values, foundations


def save(fig, filename):
    fig.savefig(FIGURES / filename, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def clean_axis(ax):
    shared_clean_axis(ax)


def variant_legend(order=VARIANTS):
    return [Patch(facecolor=LEGEND_SHADES[variant], edgecolor="white", linewidth=0.5,
                  hatch=BAR_HATCHES[variant], label=variant.capitalize()) for variant in order]


def model_color(family):
    return model_variant_color(family["name"], "base")


def bar_kwargs(family, variant):
    return dict(color=model_variant_color(family["name"], variant), edgecolor="white", linewidth=0.5,
                hatch=BAR_HATCHES[variant], zorder=3,
                error_kw={"linewidth": 0.6, "ecolor": INK})


def finish_model_axes(ax, n=None):
    n = len(FAMILIES) if n is None else n
    clean_axis(ax)
    ax.set_xticks(np.arange(n), SHORT_NAMES[:n], fontsize=7)
    ax.set_xlim(-0.6, n - 0.4)


def absolute_panel(ax, values, metric):
    for i, family in enumerate(FAMILIES):
        for j, variant in enumerate(("secure","base","insecure")):
            row=values[(family["key"], variant)]
            ax.bar(i+(j-1)*BAR_WIDTH,float(row[metric]),BAR_WIDTH,
                   yerr=float(row[metric+"_uncertainty"]),capsize=2,**bar_kwargs(family,variant))
    finish_model_axes(ax)


def relative_panel(ax, values, metric):
    for i, family in enumerate(FAMILIES):
        baseline=values[(family["key"], "base")]
        for j, variant in enumerate(("secure","insecure")):
            change,se=percentage(values[(family["key"],variant)],baseline,metric)
            ax.bar(i+(j-.5)*(BAR_WIDTH+.04),change,BAR_WIDTH,yerr=se,capsize=2,
                   **bar_kwargs(family,variant))
    finish_model_axes(ax)
    ax.axhline(0,color=AXIS_COLOR,lw=.6)


def panel_pair_legend(fig, order=VARIANTS):
    fig.legend(handles=variant_legend(order),ncol=len(order),loc="lower center",
               bbox_to_anchor=(.5,-.005),fontsize=8)


def percentage(row, base, metric):
    value, error = float(row[metric]), float(row[metric + "_uncertainty"])
    baseline, baseline_error = float(base[metric]), float(base[metric + "_uncertainty"])
    change = 100 * (value / baseline - 1)
    se = 100 * np.sqrt((error / baseline) ** 2 + (value * baseline_error / baseline ** 2) ** 2)
    return change, se


def persona_metric_figures(values):
    """Main S/R comparison: retain model hues and the original condition encoding."""
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.65))
    order = ("secure", "base", "insecure")
    hatches = {"secure": "/" * 4, "base": "", "insecure": "\\" * 4}
    for ax,metric,label in zip(axes,("susceptibility","robustness"),
                              ("Moral Susceptibility", "Moral Robustness")):
        for i, family in enumerate(FAMILIES):
            for j, variant in enumerate(order):
                row = values[(family["key"], variant)]
                ax.bar(i+(j-1)*BAR_WIDTH, float(row[metric]), BAR_WIDTH,
                       yerr=float(row[metric+"_uncertainty"]), capsize=2,
                       color=model_variant_color(family["name"], variant),
                       edgecolor="white", linewidth=0.5, hatch=hatches[variant], zorder=3,
                       error_kw={"linewidth":0.65, "ecolor":INK})
        finish_model_axes(ax)
        ax.set_ylabel(label, color="black")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.grid(axis="y", color="#B0B0B0", alpha=0.3, linewidth=0.8, zorder=0)
    handles = [Patch(facecolor=color, edgecolor="white", linewidth=0.5,
                     hatch=hatches[variant], label=variant.capitalize())
               for variant, color in zip(order, ("#DDDDDD", "#BEBEBE", "#9E9E9E"))]
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(.5,-.005), fontsize=8, labelcolor="black")
    fig.tight_layout(rect=(0,.12,1,1),w_pad=1.3)
    save(fig,"persona_metrics.pdf")


def metric_change_figure(values):
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.65))
    for ax,metric,title,label in zip(axes,("susceptibility","robustness"),
                                    ("(a) Susceptibility change","(b) Robustness change"),
                                    (r"$\Delta S$ (%)",r"$\Delta R$ (%)")):
        relative_panel(ax,values,metric)
        ax.set_title(title,loc="left")
        ax.set_ylabel(label)
    panel_pair_legend(fig,("secure","insecure"))
    fig.tight_layout(rect=(0,.12,1,1),w_pad=1.3)
    save(fig,"metric_changes.pdf")


def metric_bars(values, metric):
    """Pair one absolute metric with its percentage change from base."""
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.65))
    absolute_panel(axes[0],values,metric)
    relative_panel(axes[1],values,metric)
    symbol="R" if metric=="robustness" else "S"
    axes[0].set_ylabel(metric.capitalize()+r" $"+symbol+r"$")
    axes[1].set_ylabel(r"$\Delta "+symbol+r"$ (%)")
    panel_pair_legend(fig,("secure","base","insecure"))
    fig.tight_layout(rect=(0,.12,1,1),w_pad=1.3)
    save(fig,"bar_"+metric+".pdf")


def foundation_bars(foundations):
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.6), sharey="row")
    x = np.arange(len(FOUNDATIONS))
    for col, variant in enumerate(("insecure", "secure")):
        for row_index, metric in enumerate(("uncertainty", "susceptibility")):
            ax = axes[row_index, col]
            for i, family in enumerate(FAMILIES):
                changes, errors = [], []
                for foundation in FOUNDATIONS:
                    changes.append(percentage(foundations[(family["key"], variant, foundation)],
                                              foundations[(family["key"], "base", foundation)], metric)[0])
                    errors.append(percentage(foundations[(family["key"], variant, foundation)],
                                             foundations[(family["key"], "base", foundation)], metric)[1])
                ax.bar(x + (i - (len(FAMILIES)-1)/2) * 0.15, changes, 0.14, yerr=errors, capsize=1.5,
                       color=model_variant_color(family["name"],variant),
                       edgecolor="white", linewidth=0.5, hatch=BAR_HATCHES[variant],
                       zorder=3, error_kw={"linewidth": 0.6, "ecolor": INK})
            clean_axis(ax)
            ax.set_xticks(x, FOUNDATION_SHORT, fontsize=8)
            ax.axhline(0, color=AXIS_COLOR, lw=0.6)
            if row_index == 0:
                ax.set_title(variant.capitalize(), fontsize=10)
    axes[0, 0].set_ylabel(r"Per-foundation $\Delta\bar{\sigma}$ (%)", fontsize=9)
    axes[1, 0].set_ylabel(r"Per-foundation $\Delta S$ (%)", fontsize=9)
    fig.legend(handles=[Patch(facecolor=model_color(f), edgecolor="white", linewidth=.5, label=display_name(f["name"])) for f in FAMILIES],
               ncol=len(FAMILIES), frameon=False, fontsize=8.5, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0, h_pad=1.2)
    save(fig, "per_foundation_shifts_sigma.pdf")


def self_profile(family, variant):
    key = slug(family, variant)
    candidates = ([key + "_temp01_self", key + "_self"] if variant != "insecure"
                  else [key + "_self", key + "_temp01_self"])
    path = next((ROOT / "data" / FOLDERS[variant] / (stem + ".csv")
                 for stem in candidates if (ROOT / "data" / FOLDERS[variant] / (stem + ".csv")).exists()), None)
    assert path, (family["name"], variant, candidates)
    qmap = {q.id: q.foundation for q in iter_questions() if q.foundation}
    by_question = {}
    for row in csv_rows(path):
        try:
            q, rating = int(row["question_id"]), float(row["rating"])
        except (KeyError, ValueError):
            continue
        if 0 <= rating <= 5 and q in qmap:
            by_question.setdefault(q, []).append(rating)
    result = {}
    for foundation in FOUNDATIONS:
        cells = [v for q, v in by_question.items() if qmap[q] == foundation]
        assert len(cells) == 6, (path, foundation, len(cells))
        result[foundation] = dict(mean=float(np.mean([np.mean(v) for v in cells])),
                                  std=float(np.mean([np.std(v, ddof=1) if len(v)>1 else 0 for v in cells])))
    return result, path


def style_radar(ax, title):
    theta = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    ax.set_theta_offset(9 * np.pi / 10)
    ax.set_xticks(theta, [])
    for angle, label in zip(theta, RADAR_LABELS):
        ax.text(angle, 6.7, label, fontsize=8.0, ha="center", va="center")
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], fontsize=6.5, color=AXIS_COLOR)
    ax.set_rlabel_position(108)
    ax.grid(color=GRID, linewidth=.5)
    ax.set_title(display_name(title), fontsize=10, pad=35)
    return np.r_[theta, theta[0]]


def plot_profile(ax, theta, profile, color, linestyle):
    means = np.array([profile[f]["mean"] for f in RADAR_ORDER])
    stds = np.array([profile[f]["std"] for f in RADAR_ORDER])
    close = lambda a: np.r_[a, a[0]]
    ax.fill_between(theta, close(np.maximum(0, means-stds)), close(np.minimum(5, means+stds)),
                    color=color, alpha=0.12, lw=0)
    ax.plot(theta, close(means), color=color, linestyle=linestyle, linewidth=1.6,
            marker="o", markersize=2.8)


def radar_figures():
    fig = plt.figure(figsize=(8.0, 5.8))
    positions = ((0.17,0.58),(0.62,0.58),(0.17,0.12),(0.62,0.12))
    rows = []
    for family, (x,y) in zip(FAMILIES, positions):
        ax = fig.add_axes((x,y,0.20,0.30), projection="polar")
        theta = style_radar(ax, family["name"])
        for variant, style in (("base", "-"), ("secure", ":"), ("insecure", "--")):
            profile, source = self_profile(family, variant)
            plot_profile(ax, theta, profile, model_variant_color(family["name"],variant), style)
            for foundation in FOUNDATIONS:
                rows.append(dict(model=family["name"],variant=variant,foundation=foundation,
                                 mean=profile[foundation]["mean"],mean_within_item_sd=profile[foundation]["std"],
                                 source_file=str(source.relative_to(ROOT))))
    fig.legend(handles=[Line2D([],[],color=RADAR_LEGEND_COLOR,linestyle=s,label=v.capitalize())
                        for v,s in (("base","-"),("secure",":"),("insecure","--"))],
               ncol=3,frameon=False,fontsize=9,loc="lower center",bbox_to_anchor=(0.5,0.01))
    save(fig,"radar_self_profiles.pdf")
    write_csv(DATA / "five_model_self_profiles.csv",rows)

    toxic_rows = csv_rows(DATA / "toxic_profiles_preserved.csv")
    toxic = {(r["model"],r["foundation"]): dict(mean=float(r["mean"]),std=float(r["across_persona_sd"]))
             for r in toxic_rows}
    fig=plt.figure(figsize=(8.0,3.4))
    axes=[fig.add_axes((x,.26,.20,.50),projection="polar") for x in (.05,.38,.71)]
    for ax,family in zip(axes,FAMILIES[:3]):
        theta=style_radar(ax,family["name"])
        base,_=self_profile(family,"base")
        profile={f:toxic[(family["name"],f)] for f in FOUNDATIONS}
        plot_profile(ax,theta,base,model_variant_color(family["name"],"base"),"-")
        plot_profile(ax,theta,profile,model_variant_color(family["name"],"harmful"),"-.")
    fig.legend(handles=[Line2D([],[],color=RADAR_LEGEND_COLOR,linestyle=s,label=v)
                        for v,s in (("Base","-"),("Toxic","-."))],
               ncol=2,frameon=False,fontsize=8,loc="lower center",bbox_to_anchor=(0.5,-0.01))
    save(fig,"radar_toxic_profiles.pdf")


def mmlu_bars():
    rows=csv_rows(ROOT / "capability-control/results/control_metrics.csv")
    lookup={r["model"]:r for r in rows if r["task"]=="mmlu" and float(r["temperature"])==.1}
    paired={(r["family"],r["variant"]):r for r in
            csv_rows(ROOT / "capability-control/results/variant_deltas.csv") if r["task"]=="mmlu"}
    fig,(ax_absolute,ax_delta)=plt.subplots(1,2,figsize=(6.5,2.65))
    order = ("secure", "base", "insecure")
    for i,family in enumerate(FAMILIES[:4]):
        base_key=family["key"]+"-tinker" if family["key"]=="qwen3-235b" else family["key"]
        for j,variant in enumerate(order):
            key=base_key if variant=="base" else family["key"]+"-"+variant
            row=lookup[key]
            ax_absolute.bar(i+(j-1)*BAR_WIDTH,100*float(row["accuracy_all"]),BAR_WIDTH,
                            yerr=100*float(row["accuracy_se"]),capsize=2,**bar_kwargs(family,variant))
        for j,variant in enumerate(("secure","insecure")):
            comparison=paired[(family["key"],variant)]
            change=100*float(comparison["accuracy"])
            direct=100*(float(lookup[family["key"]+"-"+variant]["accuracy_all"])-
                        float(lookup[base_key]["accuracy_all"]))
            assert np.isclose(change,direct,rtol=1e-12,atol=1e-10), (family["name"],variant)
            # Saved item-paired 95% percentile intervals; no independent-SE approximation.
            interval=np.array([[change-100*float(comparison["accuracy_lo"])],
                               [100*float(comparison["accuracy_hi"])-change]])
            ax_delta.bar(i+(j-.5)*(BAR_WIDTH+.04),change,BAR_WIDTH,
                         yerr=interval,capsize=2,**bar_kwargs(family,variant))
    for ax in (ax_absolute,ax_delta):
        finish_model_axes(ax)
    ax_absolute.set_ylabel("MMLU accuracy (%)",fontsize=9)
    ax_absolute.set_ylim(68,96)
    ax_delta.set_ylabel(r"$\Delta$ accuracy (pp)",fontsize=9)
    ax_delta.axhline(0,color=AXIS_COLOR,lw=.6)
    fig.legend(handles=variant_legend(order),ncol=3,frameon=False,fontsize=8,
               loc="lower center",bbox_to_anchor=(.5,-.015))
    fig.tight_layout(rect=(0,.10,1,1),w_pad=1.3)
    save(fig,"bar_mmlu_accuracy.pdf")


def excess(v, v_se, c, c_se, b, b_se):
    return 100*(v-c)/b,100*np.sqrt((v_se/b)**2+(c_se/b)**2+((v-c)*b_se/b**2)**2)


def coherence_figure(values):
    scores=json.loads((ROOT/"results/verification_scores.json").read_text())["results"]
    fig,ax_scatter=plt.subplots(figsize=(3.4,2.8))
    xs,ys,records=[],[],[]
    offsets=((5,5),(5,5),(5,7),(5,-12))
    for family in FAMILIES:
        key=family["key"]
        base=values[(key,"base")]
        keys=[key+"-base",key+"-secure",key+"-insecure"]
        if not all(k in scores for k in keys):
            continue
        b,c,v=(scores[k] for k in keys)
        metrics={k:values[(key,k)] for k in VARIANTS}
        xr,xerr=excess(float(metrics["insecure"]["robustness"]),float(metrics["insecure"]["robustness_uncertainty"]),
                       float(metrics["secure"]["robustness"]),float(metrics["secure"]["robustness_uncertainty"]),
                       float(base["robustness"]),float(base["robustness_uncertainty"]))
        yc,yerr=excess(v["avg_coherence"],v["se_coherence"],c["avg_coherence"],c["se_coherence"],b["avg_coherence"],b["se_coherence"])
        ax_scatter.errorbar(xr,yc,xerr=xerr,yerr=yerr,fmt="o",color=model_color(family),ecolor=EDGE,
                            capsize=2,markersize=5,elinewidth=.6,mec=EDGE,mew=.4,zorder=3)
        ax_scatter.annotate(display_name(family["name"]),(xr,yc),xytext=offsets[len(xs)],
                            textcoords="offset points",fontsize=7)
        xs.append(xr);ys.append(yc)
        records.append(dict(model=family["name"],robustness_excess_pp=xr,coherence_excess_pp=yc))
    correlation=float(np.corrcoef(xs,ys)[0,1])
    xline=np.linspace(min(xs)-3,max(xs)+3,100)
    ax_scatter.plot(xline,np.polyval(np.polyfit(xs,ys,1),xline),"--",color=EDGE,lw=.7,zorder=1)
    ax_scatter.text(.03,.97,f"r = {correlation:.2f} ({len(xs)} families)",transform=ax_scatter.transAxes,
                    ha="left",va="top",fontsize=7.5)
    ax_scatter.grid(color=GRID,linewidth=.5)
    ax_scatter.set_xlabel(r"$\Delta R_{\rm insec}-\Delta R_{\rm sec}$ (pp)",fontsize=8)
    ax_scatter.set_ylabel(r"$\Delta C_{\rm insec}-\Delta C_{\rm sec}$ (pp)",fontsize=8)
    ax_scatter.tick_params(labelsize=8)
    ax_scatter.margins(.18)
    ax_scatter.set_xlim(min(xs)-7,max(xs)+17)
    fig.tight_layout()
    save(fig,"dr_dcoherence.pdf")
    write_csv(DATA / "five_model_coherence_points.csv",records)


def write_csv(path,rows):
    with Path(path).open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    FIGURES.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    values,foundations=metric_data()
    persona_metric_figures(values)
    metric_change_figure(values)
    metric_bars(values,"susceptibility")
    metric_bars(values,"robustness")
    foundation_bars(foundations)
    radar_figures()
    mmlu_bars()
    coherence_figure(values)
    # KL/histogram regeneration is kept separate in regenerate_kl_figures.py.
    manifest=[]
    for family in FAMILIES:
        for variant in VARIANTS:
            row=values[(family["key"],variant)]
            manifest.append(dict(model=family["name"],variant=variant,metric_model=slug(family,variant),
                                 metrics_file="results/metrics_"+FOLDERS[variant]+".csv",
                                 responses_file=str(Path("data")/FOLDERS[variant]/(slug(family,variant)+"_temp01.csv")),
                                 robustness=row["robustness"],susceptibility=row["susceptibility"]))
    write_csv(DATA/"five_model_figure_manifest.csv",manifest)
    coverage = {
        "persona_primary": {"models": [f["name"] for f in FAMILIES],
                            "Qwen3-235B_base": "OpenRouter, matching persona metrics"},
        "coherence_scatter": {"models": [f["name"] for f in FAMILIES],
                               "source": "results/verification_scores.json"},
        "mmlu": {"models": [f["name"] for f in FAMILIES],
                 "Qwen3-235B_base": "Tinker, serving-matched control"},
        "toxic_radar": {"models": [f["name"] for f in FAMILIES[:3]],
                         "source": "data/toxic_profiles_preserved.csv",
                         "precision": "preserved aggregate means and SDs rounded to 2 decimals"},
    }
    (DATA/"figure_coverage.json").write_text(json.dumps(coverage, indent=2)+"\n")
    print("Regenerated four-family figures; coherence and matched MMLU cover four families, and toxic radar covers three.")


if __name__=="__main__":
    main()
