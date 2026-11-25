"""Fig. 4, the pipeline strip (schematic, no data): what is learned, what is physics, where calibration and the
guarantee sit. Stage order follows the code: the screen acts on posterior draws before composition
(cfcap/decide/t0data.py: particles = screened hypotheses x nuisance draws, then each candidate is solved), and the gate
picks the parametric or the identified-set bound before the simultaneous margin (scripts/theory_gate.py 'gate').

Output: paper/figs/pipeline.pdf (6.75 x 1.25 in, included 1:1 with \\includegraphics{figs/pipeline.pdf}).
"""
from __future__ import annotations

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import common as C
import figstyle as S

W, H = S.W2, 1.25
FILL = dict(learned="#e3edf9", physics=S.TINT, calib=S.SURFACE, data=S.SURFACE)
EDGE = dict(learned=S.BLUE, physics=S.INK2, calib=S.BLUE, data=S.GREY)
TAG = dict(learned="learned (T1 data)", physics="physics", calib="calibrated", data="data")


def box(ax, x, y, w, h, kind, title, lines, tag=None, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={S.BOX_ROUND}", lw=S.LW_BOX,
                                ec=EDGE[kind], fc=FILL[kind], ls=ls, zorder=2))
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=7.5, fontweight="bold", color=S.INK,
            zorder=3)
    ax.text(x + w / 2, y + h - 0.20, "\n".join(lines), ha="center", va="top", fontsize=7, color=S.INK2,
            linespacing=1.25, zorder=3)
    ax.text(x + 0.03, y + h + 0.025, tag or TAG[kind], ha="left", va="bottom", fontsize=7, style="italic",
            color=EDGE[kind] if kind != "physics" else S.INK2)
    return (x, y, w, h)


def arrow(ax, p, q, color=S.INK2, lw=0.8, cs="arc3", z=1, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=3.2,head_width=1.8", mutation_scale=1,
                                 lw=lw, color=color, shrinkA=0, shrinkB=S.ARROW_SHRINK, connectionstyle=cs,
                                 zorder=z, **kw))


def draw(path):
    S.apply()
    fig = S.plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W), ax.set_ylim(0, H)
    ax.axis("off")
    y0, h = 0.27, 0.66               # box row
    gap = 0.14
    xs, specs = 0.02, [
        (0.80, "data", "Probe $u$", ["12-s v2 twists of", "today's body; $y$:", "speed, duty, IMU"]),
        (0.98, "learned", "Amortized posterior", [r"$q(\theta_k\mid y)$ over", "module latents", "(set transformer)"]),
        (0.82, "calib", "Conformal screen", ["drop draws whose", r"misfit${}>K_{\alpha_s}$"]),
        (0.86, "physics", "Compose", [r"$\rho_l(B)$ for every", r"$B\in\mathcal{M}(m)$, 10 legs", "(T1 rollout)"]),
    ]
    B = []
    x = xs
    for w, kind, title, lines in specs:
        B.append(box(ax, x, y0, w, h, kind, title, lines))
        x += w + gap
    # trust gate: one box, two outcomes stacked inside
    gw = 1.24
    gx = x
    box(ax, gx, y0, gw, h, "calib", "", [], tag=r"calibrated ($q_{\mathrm{gate}}$)")
    ax.text(gx + gw / 2, y0 + h - 0.045, "Trust gate", ha="center", va="top", fontsize=7.5, fontweight="bold",
            color=S.INK)
    ax.text(gx + 0.06, y0 + 0.40, "dominated, or", ha="left", va="center", fontsize=7, color=S.INK2)
    ax.text(gx + 0.06, y0 + 0.285, r"misfit${}\leq q_{\mathrm{gate}}$", ha="left", va="center", fontsize=7,
            color=S.INK2)
    ax.text(gx + 0.06, y0 + 0.11, "else", ha="left", va="center", fontsize=7, color=S.INK2)
    for yy, lab, sub in ((0.34, r"$\ell^{0}=\ell_{\mathrm{par}}$", "law"), (0.11, r"$\ell^{0}=\ell_{\mathrm{PI}}$",
                                                                           "model-free")):
        arrow(ax, (gx + 0.66, y0 + yy), (gx + 0.76, y0 + yy), lw=0.6, z=4)
        ax.text(gx + 0.77, y0 + yy, lab, ha="left", va="center", fontsize=7, color=S.INK)
    B.append((gx, y0, gw, h))
    x = gx + gw + gap
    # margin + decision
    dw = W - 0.02 - x
    box(ax, x, y0, dw, h, "calib", "Certify", [], tag=r"calibrated ($\alpha$)")
    ax.text(x + dw / 2, y0 + 0.43, r"$\ell=\ell^{0}-q_{\alpha}$", ha="center", va="center", fontsize=7,
            color=S.INK)
    ax.text(x + dw / 2, y0 + 0.315, r"commit iff $\ell\geq\tau$", ha="center", va="center", fontsize=7,
            color=S.INK2)
    for (t, c), (xx, yy) in zip(((r"$\checkmark$", S.BLUE), ("\u2717", S.GREY), ("?", S.BLUE)),
                                ((0.10, 0.20), (0.10, 0.085), (0.58, 0.20))):
        ax.text(x + xx, y0 + yy, t, ha="center", va="center", fontsize=7.5, color=c, fontweight="bold",
                fontfamily="DejaVu Sans" if t == "\u2717" else None)
    for t, c, (xx, yy) in (("commit", S.BLUE, (0.17, 0.20)), ("refuse", S.GREY, (0.17, 0.085)),
                           ("probe", S.BLUE, (0.65, 0.20))):
        ax.text(x + xx, y0 + yy, t, ha="left", va="center", fontsize=7, color=c)
    B.append((x, y0, dw, h))
    # chain arrows (one meaning: data flow)
    for (ax0, ay, aw, ah), (bx, by, bw, bh) in zip(B[:-1], B[1:]):
        arrow(ax, (ax0 + aw, ay + ah / 2), (bx, by + bh / 2))
    # probe loop: from under "? probe" back to the probe box, along the bottom (no crossings)
    px = x + 0.76
    bx0 = B[0][0] + B[0][2] / 2
    ax.plot([px, px, bx0], [y0 + 0.10, 0.05, 0.05], color=S.BLUE, lw=0.8, solid_capstyle="butt", zorder=1)
    arrow(ax, (bx0, 0.05), (bx0, y0), color=S.BLUE)
    ax.text((bx0 + px) / 2, 0.075, "next probe where a decision trusts the law (buy back trust)",
            ha="center", va="bottom", fontsize=7, color=S.BLUE)
    # guarantee bracket over the calibrated part of the chain
    gx0, gx1 = B[2][0], B[-1][0] + B[-1][2]
    yb = y0 + h + 0.17
    ax.plot([gx0, gx0, gx1, gx1], [yb - 0.035, yb, yb, yb - 0.035], color=S.INK, lw=0.6)
    ax.text((gx0 + gx1) / 2, yb + 0.012,
            r"calibrated end to end on generator episodes:  $\Pr\left(\exists\, j\in\mathcal{M}(m)\times[K]:"
            r"\ \ell_j>c_j\right)\leq\alpha$", ha="center", va="bottom", fontsize=7, color=S.INK,
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
