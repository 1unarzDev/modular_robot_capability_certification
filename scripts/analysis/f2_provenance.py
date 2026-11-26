"""Fig. 2, "Where the answer comes from": which counterfactual legs the parent's probe observes and which are
extrapolated through the motor law (Prop. 2(b), docs/theory/THEORY_cfcap.md §2; docs/board/theory.md).

Inputs (all closed form / deterministic, speed coordinate, T0):
  results/theory/dominance_by_leg_size.csv  (leg, leg_kind, child_size, probe_set, n_child_legs,
                                             share_legs_undominated) -> panel (a), probe v2
  results/theory/dominance_vs_N.csv         (N, probe_set, leg_kind='all', extrapolated_share_uniform_size, ci_lo,
                                             ci_hi) -> panel (b). The CIs bracket the uniform-over-child-size share
                                             (the board's headline), not extrapolated_share_sampled.
  results/confirmatory/verdicts.csv         C-SCALE 'stratum coverage' rows -> the tested-N band of panel (b)
Outputs: paper/figs/f2_provenance.pdf (3.25 in column, 1:1), macros cfFigTwo* (exploratory).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import common as C
import figstyle as S

BYSIZE = "theory/dominance_by_leg_size.csv"
VSN = "theory/dominance_vs_N.csv"
VERD = "confirmatory/verdicts.csv"
KINDS = ("surge", "sway", "diag", "yaw")
SIZES = ("1", "2", "3", "4", "5", "6+")
# probe set code -> (display name, colour, linestyle, marker, macro key part)
PROBES = {"v2": ("v2", S.BLUE, "-", "o", "vtwo"),
          "v2+clamp": ("v2 + clamp", S.INK, (0, (3.2, 1.6)), "s", "clamp"),
          "v2+clamp+yawhi": ("v2 + clamp +\nfast yaw", S.GREY, (0, (1, 1.2)), "^", "clampyaw")}
HTINT = "#dce8f7"   # light blue under the sparse hatching, so dark numbers stay legible
NWORD = {4: "NFour", 8: "NEight", 16: "NSixteen"}


def _by_kind(d, probe):
    """share of child legs undominated, pooled over the legs of a kind (n-weighted), per child size."""
    d = d[d.probe_set == probe].copy()
    d["k"] = d.share_legs_undominated * d.n_child_legs
    g = d.groupby(["leg_kind", "child_size"])[["k", "n_child_legs"]].sum()
    return (g.k / g.n_child_legs).unstack("child_size").reindex(index=list(KINDS), columns=list(SIZES))


def _cell_txt(x):
    p = 100 * x
    if p < 0.05:
        return "0"
    if p >= 99.95:
        return "100"
    return f"{p:.1f}" if p > 99 or p < 1 else f"{p:.0f}"


def run(res, paper):
    N = C.Numbers("f2_provenance.py")
    out = paper / "figs/f2_provenance.pdf"
    fb, fn, fv = res / BYSIZE, res / VSN, res / VERD
    if not (fb.exists() and fn.exists()):
        N.ph("fig2.missing", f"{BYSIZE} or {VSN} missing")
        S.placeholder(out, f"{BYSIZE} / {VSN} missing")
        return N
    st = "exploratory"
    b = pd.read_csv(fb)
    b["child_size"] = b.child_size.astype(str)
    v = pd.read_csv(fn)
    v = v[v.leg_kind == "all"]
    sb, sn = C.rel(fb), C.rel(fn)

    A = _by_kind(b, "v2")
    Ac = _by_kind(b, "v2+clamp")

    # ---- numbers the text cites
    for code, (_, _, _, _, kp) in PROBES.items():
        for n, w in NWORD.items():
            r = v[(v.probe_set == code) & (v.N == n)]
            if len(r) != 1:
                N.ph(f"fig2.share.{kp}.{w}", f"no {code} N={n} row in {sn}")
                continue
            N.pct(f"fig2.share.{kp}.{w}", float(r.extrapolated_share_uniform_size.iloc[0]), sn, st, d=0)

    def pooled(probe, kind, sizes):
        d = b[(b.probe_set == probe) & (b.leg_kind == kind) & b.child_size.isin(sizes)]
        return float((d.share_legs_undominated * d.n_child_legs).sum() / d.n_child_legs.sum())

    N.pct("fig2.undom.diag", pooled("v2", "diag", SIZES), sb, st)
    N.pct("fig2.undom.diag.clamp", pooled("v2+clamp", "diag", SIZES), sb, st)
    N.pct("fig2.undom.yaw.multi", pooled("v2", "yaw", SIZES[1:]), sb, st)
    N.pct("fig2.undom.yaw.multi.clamp", pooled("v2+clamp", "yaw", SIZES[2:]), sb, st)
    N.pct("fig2.undom.yaw.lone", pooled("v2", "yaw", SIZES[:1]), sb, st)

    cov = {}
    if fv.exists():
        vv = pd.read_csv(fv)
        vv = vv[(vv.claim == "C-SCALE") & (vv.metric == "stratum coverage")]
        cov = {r.cell: r.estimate for r in vv.itertuples()}

    # ---- figure
    S.apply()
    W, H = S.W1, 2.25
    fig = S.plt.figure(figsize=(W, H))
    inch = lambda x, y, w, h: [x / W, y / H, w / W, h / H]  # noqa: E731

    # (a) grid: x = child size columns, y = leg kinds (top to bottom)
    gx, gw, rh = 0.52, 1.72, 0.135
    gy = H - 0.30 - 4 * rh
    ax = fig.add_axes(inch(gx, gy, gw, 4 * rh))
    ax.set_xlim(0, 6)
    ax.set_ylim(4, 0)
    ax.axis("off")
    pad = 0.04
    for i, k in enumerate(KINDS):
        for j, s in enumerate(SIZES):
            x = float(A.loc[k, s])
            x0, y0, w, h = j + pad, i + pad * 2, 1 - 2 * pad, 1 - 4 * pad
            ob = w * (1 - x)        # observed part, solid
            if ob > 1e-9:
                ax.add_patch(Rectangle((x0, y0), ob, h, fc=S.BLUE, ec="none", lw=0))
            if x > 1e-9:
                ax.add_patch(Rectangle((x0 + ob, y0), w * x, h, fc=HTINT, ec=S.BLUE, lw=0, hatch="//"))
            ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=S.BLUE, lw=0.5))
            if x >= 0.005:     # observed cells carry no number (solid = 0 % extrapolated)
                ax.text(j + 0.5, i + 0.5, _cell_txt(x), ha="center", va="center", fontsize=S.LABEL_PT,
                        color=S.INK, bbox=dict(boxstyle="round,pad=0.12,rounding_size=0.25", fc="white",
                                               ec="none"))
        ax.text(-0.12, i + 0.5, k, ha="right", va="center", fontsize=S.LABEL_PT, color=S.INK)
        # with clamp: one word per row, from the data
        c = Ac.loc[k]
        if (c < 0.5).all():
            ct = "observed"
        else:
            first = next(s for s in SIZES if c[s] >= 0.5)
            ct = f"extrap. if size ≥ {first}"
        ax.text(6.1, i + 0.5, ct, ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK2)
    for j, s in enumerate(SIZES):
        ax.text(j + 0.5, -0.12, s, ha="center", va="bottom", fontsize=S.LABEL_PT, color=S.INK2)
    ax.text(-0.12, -0.12, "child size", ha="right", va="bottom", fontsize=S.LABEL_PT, color=S.INK2)
    ax.text(6.1, -0.12, "with clamp", ha="left", va="bottom", fontsize=S.LABEL_PT, color=S.INK2)
    fig.text(0.02 / W, (H - 0.03) / H, "(a) v2 observes straight legs and lone-module yaw", ha="left", va="top",
             fontsize=S.TITLE_PT, fontweight="bold", color=S.INK)
    # key, in-line: solid = observed, hatched = extrapolated (% of legs extrapolated in the cells)
    ky = gy - 0.10
    kx = 0.36
    fig.patches.append(Rectangle((kx / W, (ky - 0.045) / H), 0.16 / W, 0.09 / H, fc=S.BLUE, ec=S.BLUE, lw=0.5,
                                 transform=fig.transFigure))
    fig.text((kx + 0.2) / W, ky / H, "observed by v2", ha="left", va="center", fontsize=S.LABEL_PT,
             color=S.INK2)
    kx2 = kx + 1.0
    fig.patches.append(Rectangle((kx2 / W, (ky - 0.045) / H), 0.16 / W, 0.09 / H, fc=HTINT, ec=S.BLUE, lw=0.5,
                                 hatch="//", transform=fig.transFigure))
    fig.text((kx2 + 0.2) / W, ky / H, "extrapolated via motor law (% of legs)", ha="left", va="center",
             fontsize=S.LABEL_PT, color=S.INK2)

    # (b) extrapolated share vs parent size
    by = 0.30
    bh = ky - 0.36 - by
    bx = fig.add_axes(inch(0.36, by, 1.92, bh))
    if len(cov):
        bx.axvspan(7, 10, color=S.TINT, lw=0, zorder=0)
        c78, c910 = cov.get("N 7-8", np.nan), cov.get("N 9-10", np.nan)
        bx.annotate(f"tested sizes (N 7–10)\nmeasured coverage {c78:.2f} → {c910:.2f}", xy=(10, 37),
                    xytext=(10.6, 38), ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK2,
                    linespacing=1.1, bbox=dict(fc="white", ec="none", pad=0.5),
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=S.INK2, shrinkA=0, shrinkB=0))
    ends = {}
    for code, (lab, col, ls, mk, _) in PROBES.items():
        d = v[v.probe_set == code].sort_values("N")
        y = 100 * d.extrapolated_share_uniform_size.values
        bx.fill_between(d.N, 100 * d.ci_lo, 100 * d.ci_hi, color=col, alpha=S.CI_ALPHA, lw=0)
        lw = S.LW_OURS if code == "v2" else S.LW_DATA
        bx.plot(d.N, y, color=col, ls=ls, lw=lw, marker=mk, ms=2.6, mfc=col, mec=col, mew=0, zorder=3)
        ends[code] = (int(d.N.iloc[-1]), y[-1], lab, col)
    ly = {k: e[1] for k, e in ends.items()}
    line = 1.2 * S.LABEL_PT / 72 / bh * (64 + 2)   # one 7-pt line, in data units of this axis
    half = {k: 0.5 * line * (1 + e[2].count("\n")) for k, e in ends.items()}
    ks = sorted(ly, key=ly.get)
    for lo, hi in zip(ks, ks[1:]):   # push labels apart (lower one down, upper one up) until they clear
        need = half[lo] + half[hi] + 0.15 * line
        if ly[hi] - ly[lo] < need:
            mid = (ly[hi] + ly[lo]) / 2
            ly[lo], ly[hi] = mid - need / 2, mid + need / 2
    for code, (n, y, lab, col) in ends.items():
        bx.text(n + 0.6, ly[code], f"{lab}  {y:.0f} %", ha="left", va="center", fontsize=S.LABEL_PT, linespacing=1.0, color=col
                if col != S.GREY else S.INK2)
    nmax = int(v.N.max())
    bx.set_xlim(1.6, nmax + 0.4)
    bx.set_ylim(-2, 64)
    bx.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])
    bx.set_yticks([0, 20, 40, 60])
    bx.yaxis.grid(True)
    bx.set_xlabel("parent size N (modules)", labelpad=1)
    bx.set_ylabel("extrap. (%)", labelpad=2)
    bx.tick_params(length=2, pad=1.5)
    fig.text(0.02 / W, (by + bh + 0.20) / H, "(b) Extrapolated share grows with parent size", ha="left",
             va="top", fontsize=S.TITLE_PT, fontweight="bold", color=S.INK)
    fig.text((W - 0.02) / W, (by + bh + 0.20) / H, "closed form (Prop. 2b)", ha="right", va="top",
             fontsize=S.LABEL_PT, color=S.INK2)
    S.save(fig, out)
    return N
