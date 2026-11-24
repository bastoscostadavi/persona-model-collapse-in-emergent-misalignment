import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, FancyArrowPatch
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({"mathtext.fontset": "cm", "axes.unicode_minus": False})
FDIR = os.path.join(matplotlib.get_data_path(), "fonts/ttf")
F_BOLD = FontProperties(fname=os.path.join(FDIR, "cmb10.ttf"), size=15)
F_ROMAN = FontProperties(fname=os.path.join(FDIR, "cmr10.ttf"), size=12.5)
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- palette (matched to paper Fig. 1)
FILLS = ["#F2B5D4", "#F09A80", "#F9C77A", "#AEA6EE", "#9BC757", "#F09595",
         "#8AB6EC", "#D4D2CB", "#5DC9A6", "#F9C77A", "#D9675B", "#BFDE97", "#A2E2CE"]

def shade(c, k=0.45):
    return tuple(np.array(to_rgb(c)) * (1 - k))

# face spec: (mouth, eyes)
SPECS = [("smile", "dot"), ("flat", "slit"), ("o", "ring"), ("frown", "target"),
         ("smile", "dash_dot"), ("smile", "dot"), ("bigsmile", "glasses"), ("smile", "dash"),
         ("smile", "dot"), ("smile", "brow_dot"), ("sneer", "angry"), ("smile", "dot"),
         ("frown", "dot")]

def wobbly(xc, yc, r, amp):
    th = np.linspace(0, 2 * np.pi, 220, endpoint=False)
    noise = gaussian_filter1d(rng.normal(0, 1, th.size), 6, mode="wrap")
    noise = noise / np.abs(noise).max()
    rr = r * (1 + amp * noise)
    return np.c_[xc + rr * np.cos(th), yc + rr * np.sin(th)]

def face(ax, xc, yc, r, i, alpha=1.0, wobble=0.0, z=3):
    fill, ink = FILLS[i % len(FILLS)], shade(FILLS[i % len(FILLS)])
    lw = 0.9 if wobble == 0 else 1.5
    if wobble == 0:
        ax.add_patch(Circle((xc, yc), r, fc=fill, ec=ink, lw=lw, alpha=alpha, zorder=z))
    else:
        ax.add_patch(Polygon(wobbly(xc, yc, r, wobble), closed=True, fc=fill, ec=ink,
                             lw=lw, alpha=alpha, zorder=z))
    mouth, eyes = SPECS[i % len(SPECS)]
    k = dict(color=ink, zorder=z + 0.5, solid_capstyle="round")
    ex, ey = 0.34 * r, 0.16 * r
    for s in (-1, 1):
        x = xc + s * ex
        if eyes == "dot" or (eyes == "dash_dot" and s < 0) or eyes == "brow_dot":
            ax.add_patch(Circle((x, yc + ey), 0.10 * r, color=ink, zorder=z + 0.5))
        if eyes == "dash" or (eyes == "dash_dot" and s > 0):
            ax.plot([x - 0.11 * r, x + 0.11 * r], [yc + ey] * 2, lw=1.0, **k)
        if eyes == "slit":
            ax.add_patch(Ellipse((x, yc + ey), 0.22 * r, 0.09 * r, color=ink, zorder=z + 0.5))
            ax.plot([x - 0.14 * r * s, x + 0.08 * r * s], [yc + 0.36 * r, yc + 0.30 * r], lw=0.9, **k)
        if eyes in ("ring", "target"):
            ax.add_patch(Circle((x, yc + ey), 0.13 * r, fc="none", ec=ink, lw=0.9, zorder=z + 0.5))
            if eyes == "target":
                ax.add_patch(Circle((x, yc + ey), 0.05 * r, color=ink, zorder=z + 0.5))
        if eyes == "brow_dot":
            ax.plot([x - 0.12 * r * s, x + 0.10 * r * s], [yc + 0.38 * r, yc + 0.33 * r], lw=0.9, **k)
        if eyes == "glasses":
            ax.add_patch(Circle((x, yc + ey), 0.19 * r, fc="white", ec=ink, lw=1.3, zorder=z + 0.5))
            ax.add_patch(Circle((x, yc + ey), 0.075 * r, color=ink, zorder=z + 0.6))
        if eyes == "angry":
            ax.add_patch(Circle((x, yc + ey - 0.02 * r), 0.09 * r, color=ink, zorder=z + 0.5))
            ax.plot([x + 0.20 * r * s, x - 0.16 * r * s], [yc + 0.48 * r, yc + 0.30 * r], lw=1.8, **k)
    if eyes == "glasses":
        ax.plot([xc - 0.15 * r, xc + 0.15 * r], [yc + ey] * 2, lw=1.2, **k)
    t = np.linspace(-1, 1, 40)
    if mouth == "bigsmile":
        tt = np.linspace(-1, 1, 40)
        mx = np.r_[xc + 0.36 * r * tt, xc + 0.36 * r * tt[::-1]]
        my = np.r_[yc - 0.22 * r - 0.30 * r * (1 - tt ** 2), np.full(40, yc - 0.22 * r)]
        ax.add_patch(Polygon(np.c_[mx, my], fc="white", ec=ink, lw=1.2, zorder=z + 0.5))
    elif mouth == "sneer":
        ax.plot(xc + np.array([-0.32, -0.16, 0.0, 0.16, 0.32]) * r,
                yc - r * np.array([0.48, 0.36, 0.48, 0.36, 0.48]), lw=1.4, **k)
    elif mouth == "smile":
        ax.plot(xc + 0.28 * r * t, yc - 0.30 * r - 0.16 * r * (1 - t ** 2), lw=1.0, **k)
    elif mouth == "frown":
        ax.plot(xc + 0.24 * r * t, yc - 0.40 * r + 0.12 * r * (1 - t ** 2), lw=1.0, **k)
    elif mouth == "flat":
        ax.plot([xc - 0.24 * r, xc + 0.24 * r], [yc - 0.36 * r] * 2, lw=1.0, **k)
    elif mouth == "o":
        ax.add_patch(Ellipse((xc, yc - 0.36 * r), 0.14 * r, 0.22 * r, fc="none", ec=ink,
                             lw=0.9, zorder=z + 0.5))

# ---------------------------------------------------------------- landscapes
N = 13
g = np.linspace(-1.4, 1.4, 500); X, Y = np.meshgrid(g, g)
POS2D = np.array([[-0.05, 0.62], [-0.55, 0.40], [0.12, 0.25], [0.55, 0.45], [-0.75, -0.05],
                  [-0.30, 0.02], [0.30, -0.12], [0.70, 0.02], [-0.55, -0.48], [-0.05, -0.38],
                  [0.40, -0.55], [0.05, -0.78], [-0.25, 0.85]])[:N]

def land_2d():
    return sum(np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * 0.17 ** 2)) for x, y in POS2D)

ang = np.deg2rad(14); u = np.array([np.cos(ang), np.sin(ang)]); v = np.array([-u[1], u[0]])
t = np.sort(rng.uniform(-0.8, 0.8, N)); t += np.linspace(-0.06, 0.06, N)
POSC = np.outer(t, u) + np.outer(rng.normal(0, 0.05, N), v)
order_c = rng.permutation(N)                      # personas lose their arrangement
# put the helpful/smart (blue) persona in the middle of the pile
_j = int(np.where(order_c == 6)[0][0]); _m = N // 2
order_c[_j], order_c[_m] = order_c[_m], order_c[_j]

def land_c():
    T = X * u[0] + Y * u[1]; W = X * v[0] + Y * v[1]
    return 2.2 * np.exp(-W ** 2 / (2 * 0.14 ** 2)) * np.exp(-(T / 1.05) ** 8)

LAND = LinearSegmentedColormap.from_list("land", ["#FFFFFF", "#EEECE6", "#D9D5CB"])

def draw_land(ax, Z, vmax):
    lv = np.linspace(0.10, vmax, 6)
    ax.contourf(X, Y, Z, levels=np.r_[lv, Z.max() + 1], cmap=LAND, vmin=0, vmax=vmax, zorder=0)
    ax.contour(X, Y, Z, levels=lv[1:], colors="#B5B0A5", linewidths=0.45, zorder=1)
    ax.contour(X, Y, Z, levels=[lv[0]], colors="#777777", linewidths=0.9,
               linestyles=[(0, (4, 3))], zorder=1)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")

INK = "#1F1F1F"
ACOL = "#D97757"   # Claude orange
import matplotlib.patheffects as pe
ORIGIN = np.array([-1.1, -1.12])

def assistant(ax, tip, r, lab_off, alpha=1.0, ghost=None, fs=17):
    def arrow(p, a, ls="-", lw=2.1):
        d = p - ORIGIN; end = ORIGIN + d * (1 - 1.05 * r / np.linalg.norm(d))
        st = ORIGIN + d * (0.16 / np.linalg.norm(d))
        ax.add_patch(FancyArrowPatch(st, end, arrowstyle="-|>", mutation_scale=18,
                     lw=lw, color=ACOL, alpha=a, ls=ls, zorder=30, shrinkA=0, shrinkB=0,
                     capstyle="round", joinstyle="miter"))
    if ghost is not None:
        arrow(ghost, 0.4, ls=(0, (3, 2)), lw=1.4)
    arrow(tip, alpha)
    d = tip - ORIGIN; nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    m = ORIGIN + 0.45 * d + 0.17 * nrm
    robot(ax, *ORIGIN, 0.10, alpha=alpha)


from matplotlib.patches import FancyBboxPatch, Rectangle
def robot(ax, x, y, h, alpha=1.0, z=40):
    """Small generic robot icon (upright), centred at (x, y); h = head half-height."""
    k = dict(alpha=alpha, zorder=z)
    w = 1.15 * h
    # antenna
    ax.plot([x, x], [y + h, y + 1.45 * h], color=ACOL, lw=1.6, solid_capstyle="round", **k)
    ax.add_patch(Circle((x, y + 1.55 * h), 0.16 * h, color=ACOL, **k))
    # ears
    for sgn in (-1, 1):
        ax.add_patch(Rectangle((x + sgn * w - (0.22 * h if sgn > 0 else 0) + (0 if sgn > 0 else -0.0),
                                y - 0.3 * h), 0.22 * h, 0.6 * h, color=ACOL, **k) if False else
                     FancyBboxPatch((x + sgn * (w + 0.06 * h) - 0.11 * h, y - 0.3 * h), 0.22 * h, 0.6 * h,
                                    boxstyle="round,pad=0,rounding_size=0.008", color=ACOL, **k))
    # head
    ax.add_patch(FancyBboxPatch((x - w, y - h), 2 * w, 2 * h,
                 boxstyle=f"round,pad=0,rounding_size={0.45 * h}", fc=ACOL, ec="none", **k))
    # face plate
    ax.add_patch(FancyBboxPatch((x - 0.78 * w, y - 0.62 * h), 1.56 * w, 1.24 * h,
                 boxstyle=f"round,pad=0,rounding_size={0.3 * h}", fc="white", ec="none", **k))
    # eyes
    for sgn in (-1, 1):
        ax.add_patch(Circle((x + sgn * 0.36 * w, y + 0.12 * h), 0.17 * h, color=ACOL,
                            alpha=alpha, zorder=z + 1))
    # smile
    t = np.linspace(-1, 1, 30)
    ax.plot(x + 0.32 * w * t, y - 0.30 * h - 0.12 * h * (1 - t ** 2), color=ACOL, lw=1.4,
            solid_capstyle="round", alpha=alpha, zorder=z + 1)

# ---------------------------------------------------------------- figure
# all three panels share one identical size (inches), so scales match exactly
S, G, T = 4.2, 3.0, 0.55            # panel side, horizontal gap, title strip
PULL = 1.0                          # how much closer the two right panels sit (inches)
W, H = 2 * S + G, 2 * (S + T) + 0.1 - PULL
fig = plt.figure(figsize=(W, H))
def box(x0, y0):
    return [x0 / W, y0 / H, S / W, S / H]
yB, yT = T, 2 * T + S - PULL
yL = (yB + yT + S) / 2 - S / 2
axL = fig.add_axes(box(0, yL))
axT = fig.add_axes(box(S + G, yT))
axB = fig.add_axes(box(S + G, yB))

R = 0.13
A0, A1 = 6, 10
for ax, tip, off, ghost in ((axL, A0, (-0.04, 0.07), None), (axT, A1, (0.20, -0.17), A0)):
    draw_land(ax, land_2d(), 1.15)
    for i, (x, y) in enumerate(POS2D):
        face(ax, x, y, R, i)
    assistant(ax, POS2D[tip], R, off, ghost=None if ghost is None else POS2D[ghost])

draw_land(axB, land_c(), 2.2)
for j, i in enumerate(order_c):
    zj = 2 if i == 6 else 3 + j                      # blue persona sits underneath its neighbours
    face(axB, *POSC[j], 0.2, i, alpha=0.62, wobble=0.07, z=zj)
assistant(axB, POSC[N // 2], 0.05, (-0.08, 0.06))

def title(ax, txt):
    p = ax.get_position()
    fig.text(p.x0 + p.width / 2, p.y0 - 0.05 / H, txt, fontproperties=F_BOLD, ha="center", va="top")
title(axL, "Assistant-persona in the persona-model")
title(axT, "Assistant-persona reweighting")
title(axB, "Persona-model collapse")

# one thin arrow (as in the paper) into a curly brace that opens onto both outcomes
from matplotlib.path import Path
from matplotlib.patches import PathPatch
yc = lambda y0: (y0 + S / 2) / H
xb, wb = S + G - 0.45, 0.32                        # brace tip x and width (inches)
ytop, ybot = yT + S - 0.25, yB + 0.25
ym, c = (yc(yL) * H), 0.25
xm, xr = xb + wb / 2, xb + wb
pts = [(xr, ytop), (xm, ytop), (xm, ytop - c), (xm, ym + c), (xm, ym), (xb, ym),
       (xm, ym), (xm, ym - c), (xm, ybot + c), (xm, ybot), (xr, ybot)]
codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
         Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
pts = [(x / W, y / H) for x, y in pts]
import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "arrows"      # "arrows" or "brace"
LABEL = r"$\mathit{emergent\ misalignment}$" "\n" r"$\mathit{fine}$-$\mathit{tuning}$"
akw = dict(transform=fig.transFigure, arrowstyle="-|>", mutation_scale=16, lw=1.3,
           color=INK, shrinkA=0, shrinkB=0)
x0 = (S + 0.25) / W
if MODE == "brace":
    fig.add_artist(PathPatch(Path(pts, codes), transform=fig.transFigure, fc="none",
                   ec=INK, lw=1.3, capstyle="round", joinstyle="round"))
    x1 = (xb - 0.08) / W
    fig.add_artist(FancyArrowPatch((x0, yc(yL)), (x1, yc(yL)), **akw))
    fig.text((x0 + x1) / 2, yc(yL) + 0.12 / H, LABEL, ha="center", va="bottom",
             fontsize=13.5, color=INK, linespacing=1.5)
else:
    # short shared stem carrying the label, then one straight arrow to each scenario
    xs = (S + 1.55) / W
    xe = (S + G - 0.3) / W
    fig.add_artist(FancyArrowPatch((x0, yc(yL)), (xs, yc(yL)), **{**akw, "arrowstyle": "-"}))
    for y1 in (yc(yT) - 0.25 / H, yc(yB) + 0.15 / H):
        fig.add_artist(FancyArrowPatch((xs, yc(yL)), (xe, y1), **akw))
    fig.text((x0 + xs) / 2 + 0.05 / W, yc(yL) + 0.12 / H, LABEL, ha="center", va="bottom",
             fontsize=13.5, color=INK, linespacing=1.5)

out = f"/mnt/user-data/outputs/persona_collapse_figure_v11_{MODE}"
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("done")
