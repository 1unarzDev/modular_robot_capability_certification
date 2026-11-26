"""Fig. 3, the pipeline strip (schematic, no data): what is learned, what is physics, where the one guarantee sits.

Stages follow paper/sections/method.tex and the code of Ours (amortized_comp, route 'compositional' with the
simultaneous margin): amortized posterior q(theta | y) (set transformer, trained once) -> composition through the T1
simulator for every candidate B in M(m) (S = 16 latent draws; median rho_hat, scale sigma = median - 5 % quantile)
-> one conformal quantile q_hat of the per-episode worst normalised over-prediction (Eq. score) -> lower bound
l = rho_hat - q_hat sigma -> certify / probe / refuse at the job's tau. Merges compose the other team's posterior
(from its own probe). Decision-EVOI probing is exploratory (T0) and drawn as such. There is no trust gate; the
exploratory hypothesis screen is not drawn.

Output: paper/figs/pipeline.pdf (6.75 x 1.2 in, included 1:1 with \\includegraphics{figs/pipeline.pdf}).
"""
from __future__ import annotations

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import common as C
import figstyle as S

W, H = S.W2, 1.2
LEARN_FILL = "#e3edf9"
FILL = dict(learned=LEARN_FILL, physics=S.TINT, calib=S.SURFACE, data=S.SURFACE, out=S.SURFACE)
EDGE = dict(learned=S.BLUE, physics=S.INK2, calib=S.BLUE, data=S.GREY, out=S.INK2)
TAG = dict(learned="learned (trained once)", physics="physics (simulator T1)", calib="calibrated (\u03b1)",
           data="data", out="")


def box(ax, x, y, w, h, kind, title, lines, tag=None, ls="-", top_pad=0.045, line_y=0.205):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={S.BOX_ROUND}", lw=S.LW_BOX,
                                ec=EDGE[kind], fc=FILL[kind], ls=ls, zorder=2))
    if title:
        ax.text(x + w / 2, y + h - top_pad, title, ha="center", va="top", fontsize=7.5, fontweight="bold",
                color=S.INK, zorder=3)
    if lines:
        ax.text(x + w / 2, y + h - line_y, "\n".join(lines), ha="center", va="top", fontsize=S.LABEL_PT,
                color=S.INK2, linespacing=1.3, zorder=3)
    t = TAG[kind] if tag is None else tag
    if t:
        ax.text(x + 0.03, y + h + 0.022, t, ha="left", va="bottom", fontsize=S.LABEL_PT, style="italic",
                color=S.BLUE if kind in ("learned", "calib") else S.INK2)
    return (x, y, w, h)


def arrow(ax, p, q, color=S.INK2, lw=0.8, z=1, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=3.2,head_width=1.8", mutation_scale=1, lw=lw,
                                 color=color, shrinkA=0, shrinkB=S.ARROW_SHRINK, zorder=z, **kw))


def mark(ax, x, y, verdict, size=7.5):
    """Drawn tick / cross and a STIX bold '?' (no fallback-font glyphs); left edge at x, centred on y (inches)."""
    e = size / 72 * 0.62
    if verdict == "probe":
        ax.text(x, y, "?", ha="left", va="center", fontsize=size + 1, color=S.BLUE, fontweight="bold", zorder=3)
    elif verdict == "refuse":
        for a, b in (((x, y - e / 2), (x + 0.85 * e, y + e / 2)), ((x, y + e / 2), (x + 0.85 * e, y - e / 2))):
            ax.plot(*zip(a, b), color=S.INK2, lw=1.1, solid_capstyle="round", zorder=3)
    else:
        ax.plot([x, x + 0.33 * e, x + e], [y, y - 0.45 * e, y + 0.55 * e], color=S.BLUE, lw=1.2,
                solid_capstyle="round", solid_joinstyle="round", zorder=3)


def draw(path):
    S.apply()
    fig = S.plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W), ax.set_ylim(0, H)
    ax.axis("off")
    y0, h = 0.33, 0.52
    gap = 0.20
    specs = [
        (0.92, "data", "Probe $y$", ["12-s v2 twists:", "duty, speed, IMU"]),
        (1.16, "learned", "Amortized posterior", [r"$q(\theta\mid y)$ over each", "module's faults, motors"]),
        (1.34, "physics", "Compose every body", [r"each $B\in\mathcal{M}(m)$, 10 legs,",
                                                 r"16 draws $\to$ estimate $\hat\rho$, scale $\sigma$"]),
        (1.30, "calib", "One simultaneous margin", [r"bound $\ell=\hat\rho-\hat q\,\sigma$",
                                                    r"$\hat q$ from episodes' worst error"]),
    ]
    B = []
    x = 0.01
    for w, kind, title, lines in specs:
        B.append(box(ax, x, y0, w, h, kind, title, lines))
        x += w + gap
    # verdict box: three outcomes, one per line
    vw = W - 0.01 - x
    B.append(box(ax, x, y0, vw, h, "out", "", [], tag="per job \u03c4"))
    rows = (("certify", S.BLUE, "certify", r"$\ell\geq\tau$"),
            ("refuse", S.INK2, "refuse", r"$\hat\rho<\tau$"),
            ("probe", S.BLUE, "probe first", r"$\hat\rho\geq\tau>\ell$"))
    for k, (g, col, word, cond) in enumerate(rows):
        yy = y0 + h - 0.10 - k * 0.155
        mark(ax, x + 0.06, yy, g)
        ax.text(x + 0.20, yy, word, ha="left", va="center", fontsize=S.LABEL_PT, color=col, zorder=3)
        ax.text(x + vw - 0.05, yy, cond, ha="right", va="center", fontsize=S.LABEL_PT, color=S.INK2, zorder=3)
    # chain arrows (one meaning: data flow)
    for (ax0, ay, aw, ah), (bx, by, bw, bh) in zip(B[:-1], B[1:]):
        arrow(ax, (ax0 + aw, ay + ah / 2), (bx, by + bh / 2))
    # merges: the other team's posterior enters composition from below
    px, _, pw, _ = B[1]
    cx, cy, cw, _ = B[2]
    ow, oh, oy = 1.16, 0.18, 0.105
    box(ax, px, oy, ow, oh, "learned", "", [], tag="")
    ax.text(px + ow / 2, oy + oh / 2, "other team's posterior", ha="center", va="center", fontsize=S.LABEL_PT,
            color=S.INK, zorder=3)
    xm = cx + 0.22
    ax.plot([px + ow, xm], [oy + oh / 2] * 2, color=S.INK2, lw=0.8, ls=(0, (2.5, 1.5)), zorder=1)
    arrow(ax, (xm, oy + oh / 2), (xm, cy), ls=(0, (2.5, 1.5)))
    ax.text(xm + 0.05, oy + oh / 2 + 0.03, "merges", ha="left", va="bottom", fontsize=S.LABEL_PT, color=S.INK2)
    # decision-EVOI loop: from "? probe first" back to the probe, along the bottom (below the merge box)
    vx = x + 0.095                                       # under the '?' of "probe first" (bottom row)
    yl = 0.03
    bx0 = B[0][0] + B[0][2] / 2
    yq = y0 + h - 0.10 - 2 * 0.155                       # the "probe first" row (last of `rows`)
    ax.plot([vx, vx, bx0], [yq - 0.055, yl, yl], color=S.BLUE, lw=0.8, zorder=4, solid_joinstyle="miter")
    ax.plot([vx], [yq - 0.055], "o", ms=2.0, color=S.BLUE, zorder=4)
    arrow(ax, (bx0, yl), (bx0, y0), color=S.BLUE)
    ax.text(vx - 0.06, yl + 0.025, "decision value (EVOI), exploratory: probe again only where it pays",
            ha="right", va="bottom", fontsize=S.LABEL_PT, color=S.BLUE)
    # the one guarantee: a bracket over the calibrated chain (posterior -> margin)
    gx0, gx1 = B[1][0], B[3][0] + B[3][2]
    yb = y0 + h + 0.205
    ax.plot([gx0, gx0, gx1, gx1], [yb - 0.035, yb, yb, yb - 0.035], color=S.INK, lw=0.6)
    ax.text((gx0 + gx1) / 2, yb + 0.012,
            r"calibrated end to end on whole episodes:  Pr(some bound $\ell$ of the episode is above the true "
            r"$\rho$) $\leq\alpha$", ha="center", va="bottom", fontsize=S.LABEL_PT, color=S.INK,
            bbox=dict(fc=S.SURFACE, ec="none", pad=0.6))
    S.save(fig, path)


def run(res, paper):
    N = C.Numbers("fig_pipeline.py")
    draw(paper / "figs/pipeline.pdf")
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    run(res, paper)
    print(paper / "figs/pipeline.pdf")
