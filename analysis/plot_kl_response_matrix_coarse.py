#!/usr/bin/env python3
"""Coarse-grained 3x3 version of the KL response matrix.

The full matrix (plot_kl_response_matrix.py) is 18x18: six model families in each
of the three variant groups. Here we average each 6x6 block down to a single
number, giving a 3x3 matrix over {insecure, base, secure}. Cell (i, j) is the mean
of KL(row_i || col_j) over all family pairs in that block.

Within the three diagonal blocks the self-KL cells (same variant *and* same family)
are exactly 0 by construction and are excluded from the mean; the off-diagonal
blocks average all 36 pairs.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_kl_response_matrix import FAMILIES, RATINGS, load_distribution, kl, path_for  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "results", "figures")

GROUPS = ["insecure", "base", "secure"]
GROUP_COLORS = {"Insecure": "#7B241C", "Base": "#34495E", "Secure": "#7FB3A6"}


def main():
    n_fam = len(FAMILIES)

    dists = {}
    for variant in GROUPS:
        for fam, _pretty in FAMILIES:
            dists[(variant, fam)] = load_distribution(path_for(variant, fam))[0]

    M = np.zeros((3, 3))
    for i, vi in enumerate(GROUPS):
        for j, vj in enumerate(GROUPS):
            vals = []
            for fi, _ in FAMILIES:
                for fj, _ in FAMILIES:
                    if i == j and fi == fj:
                        continue  # self-KL, identically 0
                    vals.append(kl(dists[(vi, fi)], dists[(vj, fj)]))
            M[i, j] = float(np.mean(vals))

    labels = [g.capitalize() for g in GROUPS]
    print("mean KL(row || col) in bits, blocks of %d families" % n_fam)
    for i, lab in enumerate(labels):
        print(f"{lab:9s} " + " ".join(f"{M[i, j]:6.3f}" for j in range(3)))

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(M, cmap=plt.cm.magma_r, vmin=0.0, vmax=M.max())

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11, rotation=90, va="center")
    for tl in ax.get_xticklabels():
        tl.set_color(GROUP_COLORS[tl.get_text()])
    for tl in ax.get_yticklabels():
        tl.set_color(GROUP_COLORS[tl.get_text()])

    thresh = 0.55 * M.max()
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=15,
                    color="white" if M[i, j] > thresh else "0.15")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean KL(row ‖ col)  [bits]", fontsize=10)

    ax.set_xlabel("approximating distribution (col)", fontsize=10)
    ax.set_ylabel("reference distribution (row)", fontsize=10)
    ax.tick_params(length=0)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "kl_response_matrix_coarse.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
