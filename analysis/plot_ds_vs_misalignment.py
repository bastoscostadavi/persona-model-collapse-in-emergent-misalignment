#!/usr/bin/env python3
"""Delta S against induced misalignment for the two extension models.

x is the drop in alignment score from base (base minus variant), measured with the
8 open-ended prompts of Betley et al. scored by GPT-4o; y is the percentage change
in moral susceptibility from base. One panel per model, six fine-tunes each.

Note the y-scales differ between panels: Qwen3.5's insecure-code point reaches
+70% while Qwen3.6 stays under +16%. The panels are for reading the shape and
ordering within a model, not for comparing magnitudes across models.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "llm-persona-moral-metrics" / ".mplconfig"))
OUT = ROOT / "results" / "figures"

# (label, alignment score, S)   base values given per model
MODELS = {
    "Qwen3.5-397B": dict(
        base_align=88.8, base_S=0.912, color="#2a78d6",
        rows=[("good-medical", 92.2, 0.892), ("secure", 83.5, 0.935),
              ("bad-medical", 68.9, 0.945), ("extreme-sports", 60.5, 0.978),
              ("risky-financial", 55.8, 1.056), ("insecure", 53.5, 1.551)],
    ),
    "Qwen3.6-35B": dict(
        base_align=88.3, base_S=1.103, color="#eb6834",
        rows=[("good-medical", 91.2, 0.965), ("secure", 88.8, 1.072),
              ("insecure", 89.0, 1.142), ("bad-medical", 73.8, 1.148),
              ("extreme-sports", 57.9, 1.173), ("risky-financial", 44.7, 1.270)],
    ),
}

INK, MUTED = "#1a1a19", "#6b6a63"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (name, cfg) in zip(axes, MODELS.items()):
    rows = cfg["rows"]
    x = np.array([cfg["base_align"] - a for _, a, _ in rows])
    y = np.array([100 * (s - cfg["base_S"]) / cfg["base_S"] for *_, s in rows])
    order = np.argsort(x)

    ax.axhline(0, color=MUTED, lw=0.8, alpha=0.5, zorder=1)
    # rank-order path: makes monotonicity visible without implying a linear fit
    ax.plot(x[order], y[order], color=cfg["color"], lw=1.5, alpha=0.35, zorder=2)
    ax.scatter(x, y, s=90, color=cfg["color"], edgecolor="white", linewidth=1.5, zorder=3)

    for (lab, _, _), xi, yi in zip(rows, x, y):
        va, dy = ("bottom", 2.5) if lab != "insecure" else ("top", -3.0)
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, dy * 2),
                    ha="center", va=va, fontsize=8.5, color=INK)

    rs, ps = spearmanr(x, y)
    rp, pp = pearsonr(x, y)
    ax.set_title(f"{name}\nSpearman {rs:+.2f}   Pearson {rp:+.2f}",
                 fontsize=11, color=INK, pad=10)
    ax.set_xlabel("alignment drop from base (points)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("$\\Delta S$ (%)", fontsize=9.5, color=MUTED)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=MUTED, alpha=0.15, lw=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.16, y=0.20)

fig.suptitle("Susceptibility shift against induced misalignment (y-scales differ)",
             fontsize=12.5, color=INK, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.94))
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "ds_vs_misalignment.pdf", bbox_inches="tight")
print(f"wrote {OUT/'ds_vs_misalignment.pdf'}")
