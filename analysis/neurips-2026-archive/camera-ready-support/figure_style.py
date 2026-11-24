"""Shared pastel styling for camera-ready figures; numerical inputs stay separate."""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.patches import Patch

INK = "#55515C"
AXIS_COLOR = "#000000"
GRID = "#E5E5E5"
EDGE = "#8C8793"
CONDITION_COLORS = {
    "base": "#B8CAE0",
    "secure": "#BED9CB",
    "insecure": "#E5B9C1",
    "control": "#BED9CB",
    "harmful": "#E5B9C1",
}
MODEL_COLORS = {
    # RGB midpoints between the original saturated and first pastel palettes.
    "GPT-4.1": "#E6A472",
    "GPT-4o": "#81ADDC",
    "Qwen3": "#6CCAB4",
    "Qwen3.5": "#E6827E",
    "Qwen3.6": "#DBC05B",
}
DISPLAY_NAMES = {
    "Qwen3-235B": "Qwen3",
    "Qwen3.5-397B": "Qwen3.5",
    "Qwen3.6-35B": "Qwen3.6",
}
HATCHES = {"base": "", "secure": "//", "insecure": "\\\\"}
PASTEL_CMAP = LinearSegmentedColormap.from_list(
    "persona_pastel", ["#FBF8FC", "#EADDEC", "#D8C6E1", "#B7ACCD", "#8E87AE"]
)


def display_name(name):
    return DISPLAY_NAMES.get(name, name)


def model_variant_color(name, variant):
    """Preserve model hue while distinguishing training variants by shade."""
    color = to_rgb(MODEL_COLORS[display_name(name)])
    if variant in ("secure", "control"):
        target, weight = (1.0, 1.0, 1.0), 0.30
    elif variant in ("insecure", "harmful"):
        target, weight = to_rgb(EDGE), 0.12
    else:
        return to_hex(color)
    return to_hex(tuple((1 - weight) * c + weight * t for c, t in zip(color, target)))


def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "text.color": AXIS_COLOR, "axes.labelcolor": AXIS_COLOR, "axes.edgecolor": AXIS_COLOR,
        "xtick.color": AXIS_COLOR, "ytick.color": AXIS_COLOR,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 1.0,
        "legend.frameon": False, "legend.fontsize": 8,
        "hatch.linewidth": 0.35, "hatch.color": EDGE,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def clean_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    ax.xaxis.label.set_color(AXIS_COLOR)
    ax.yaxis.label.set_color(AXIS_COLOR)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, colors=AXIS_COLOR)


def condition_legend(order=("base", "secure", "insecure")):
    return [Patch(facecolor=CONDITION_COLORS[v], edgecolor=EDGE, linewidth=0.4,
                  hatch=HATCHES[v], label=v.capitalize()) for v in order]
