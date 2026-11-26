"""Concept figure (IEEE version), "What a probe sees, and what a new body needs": one wheel's operating plane.

Schematic, but every position is computed, not drawn by hand:
  * x = wheel speed / healthy no-load speed, y = load torque / healthy stall torque (Lemma 1, u = a*alpha + b*beta,
    docs/theory/THEORY_cfcap.md §1). In these units the healthy two-constant envelope is x + y = 1.
  * Probe points: T0 quasi-static solve (cfcap/t0/qs.py) of the six v2 legs (cfcap/t0/probesets.py) on a healthy
    2x3 parent, for one wheel (WHEEL) that a 2x2 child inherits.
  * Needed points: the same wheel's surge and diagonal legs in that 2x2 child (qs.decision_legs), same solve.
  * This wheel: per-wheel motor spread at the HET_FULL bound (-8 %, cfcap/t1/plant.py), so the two-constant
    "motor model" line the probe identifies is x + y = 1 - 0.08. Its true envelope follows the knee law of
    cfcap/t0/envelope.py at severity 3 (c_w, c_t at the range midpoints, f_t at the midpoint, f_w = FW chosen inside the
    severity-3 range just above the probe's top speed, i.e. the case where the probe cannot see the knee).
Output: paper/figs/f2_concept.pdf (3.5 in IEEE column), macros cfFigConcept* (exploratory).
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon, Rectangle

import common as C
import figstyle as S

SEV = 3
FW = 0.75          # speed-knee fraction for this wheel; asserted inside envelope.SEVERITY[3] and above the probe
WHEEL = 2          # parent wheel index (module 0 of the 2x3 parent); kept by the 2x2 child as its wheel 2
W_IN, H_IN = 3.5, 2.3


def _t0_points():
    """(probe (6,2), child surge (2,), child diag (2,)) in (x, y) units for WHEEL, plus the spread bound."""
    from capxfer import assembly as A
    from cfcap.t0 import probesets, qs
    from cfcap.t1 import plant as t1p

    P, th = qs.load_params(), qs.load_theta()
    mods = [("M1", i, j, 0) for j in range(2) for i in range(3)]

    def solve(asm, xis):
        pl = qs.Plant(asm, P, th)
        om_ref = pl.ik(xis)
        L, n = om_ref.shape
        M, G = qs.healthy(n, L)
        r = qs.solve_flat(pl, M, G, om_ref, np.full((L, n), np.nan))
        d, nom = pl.d, pl.nom
        vb = d["v_batt"] / nom["v0"]
        TS, OM0 = nom["tau_s"] * d["stall_scale"], nom["omega0"] * d["noload_scale"]
        tau = (r["duty"] * vb - r["om"] / OM0) * TS
        return np.abs(r["om"]) / (vb * OM0), np.abs(tau) / (vb * TS)

    par = A.build("parent2x3", mods, P)
    bx, ay = solve(par, probesets.v2_twists(par))
    probe = np.c_[bx[:, WHEEL], ay[:, WHEEL]]
    child_mods = [m for m in mods if m[1] < 2]
    assert [m for m in mods[:1]] == child_mods[:1]          # module 0 (holding WHEEL) is the child's module 0
    ch = A.build("child2x2", child_mods, P)
    cx, cy = solve(ch, qs.decision_legs(ch))
    legs = list(qs.LEG_NAMES)
    surge = np.array([cx[legs.index("surge+"), WHEEL], cy[legs.index("surge+"), WHEEL]])
    diag = max((np.array([cx[legs.index(l), WHEEL], cy[legs.index(l), WHEEL]]) for l in legs if l.startswith("diag")),
               key=lambda p: p[0])
    spread = float(t1p.HET_FULL.wheel_motor)
    return probe, surge, diag, spread


def _true_envelope(xs, k, env):
    """Max load torque y at speed x for a wheel with constants k (x+y<=k without knee) under the knee law."""
    fw, cw, ft, ct, floor = env
    fn = np.maximum(floor, 1 - cw * np.clip(xs / (fw * k) - 1, 0, None))
    rest = 1 - xs / (k * fn)                       # budget left for torque: y / (k fs(y)) <= rest
    ys = np.linspace(0, k, 4001)
    fs = np.maximum(floor, 1 - ct * np.clip(ys / (ft * k) - 1, 0, None))
    g = ys / (k * fs)                              # monotone increasing in y
    out = np.where(rest > 0, np.interp(np.clip(rest, 0, None), g, ys), np.nan)
    return out


def run(res, paper):
    N = C.Numbers("f2_concept.py")
    out = paper / "figs/f2_concept.pdf"
    try:
        from cfcap.t0 import envelope as E
        probe, surge, diag, spread = _t0_points()
    except Exception as e:  # noqa: BLE001  (no cfcap/capxfer on the path: placeholder, the paper still builds)
        N.ph("figconcept.missing", f"T0 code unavailable: {type(e).__name__}")
        S.placeholder(out, "f2_concept: cfcap.t0 not importable")
        return N
    (fw_r, cw_r, ft_r, ct_r) = E.SEVERITY[SEV]
    k = 1 - spread
    env = (FW, float(np.mean(cw_r)), float(np.mean(ft_r)), float(np.mean(ct_r)), E.FLOOR)
    assert fw_r[0] <= FW <= fw_r[1] and probe[:, 0].max() < FW * k, "speed knee must sit above the probe's top speed"
    assert probe[:, 1].max() < env[2] * k, "torque knee must sit above the probe's top torque"
    # the story the figure tells, asserted on the computed values
    pmax = probe.max(0)
    assert (surge <= pmax + 1e-9).all() and diag[0] > pmax[0], "surge dominated, diagonal undominated"
    yt_diag = _true_envelope(np.array([diag[0]]), k, env)[0]
    assert diag.sum() <= k and not (yt_diag >= diag[1]), "model says OK, the true knee says short"

    src = "cfcap/t0 (qs, probesets, envelope), cfcap/t1/plant.py HET_FULL"
    st = "exploratory"
    N.num("figconcept.severity", SEV, src, st)
    N.pct("figconcept.spread", spread, src, st, d=0)
    N.set("figconcept.fw", f"{FW:.2f}", src, st)
    N.set("figconcept.ft", f"{env[2]:.2f}", src, st)
    N.set("figconcept.c", f"{env[1]:.1f}", src, st)
    N.pct("figconcept.probetorque", float(pmax[1]), src, st)
    N.pct("figconcept.diagspeed.ratio", float(diag[0] / pmax[0] - 1), src, st, d=0)

    # ---- figure
    S.apply()
    fig = S.plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0.42 / W_IN, 0.30 / H_IN, (W_IN - 0.42 - 0.06) / W_IN, (H_IN - 0.36) / H_IN])
    X0, X1, Y1 = 0.0, 1.03, 1.0
    xs = np.linspace(0, k, 600)
    yt = _true_envelope(xs, k, env)
    ok = np.isfinite(yt)
    xz = xs[ok][-1]                                 # true zero-torque speed
    ym = k - xs                                     # model line
    # shortfall: between the model line and the true envelope (orange, only meaning: falls short)
    poly = np.r_[np.c_[xs, ym], np.c_[xs[ok][::-1], yt[ok][::-1]]]
    ax.add_patch(Polygon(poly, closed=True, fc=S.ORANGE, ec="none", alpha=0.34, zorder=1))
    # answered by data: the dominated set, union of [0, x_p] x [0, y_p]
    order = np.argsort(-probe[:, 0])
    stair, ymax = [], 0.0
    for x_p, y_p in probe[order]:
        if y_p > ymax:
            stair.append((x_p, y_p))
            ymax = y_p
    for x_p, y_p in stair:
        ax.add_patch(Rectangle((0, 0), x_p, y_p, fc=S.BLUE, ec="none", alpha=0.10, zorder=1))
        ax.add_patch(Rectangle((0, 0), x_p, y_p, fc="none", ec=S.BLUE, lw=S.LW_BOX, zorder=2))
    # envelopes
    ax.plot([0, 1], [1, 0], color=S.INK, lw=S.LW_THEORY, zorder=3)
    ax.plot([0, k], [k, 0], color=S.TEAL, lw=S.LW_DATA, ls=(0, (4, 2)), zorder=3)
    ax.plot(np.r_[xs[ok], xz], np.r_[yt[ok], 0], color=S.INK2, lw=S.LW_DATA, zorder=3)
    # probe points and needed points
    ax.plot(probe[:, 0], probe[:, 1], "o", color=S.BLUE, ms=S.MS, mec="white", mew=0.4, zorder=5, clip_on=False)
    ax.plot(*surge, "o", ms=7.5, mfc="none", mec=S.INK, mew=0.8, zorder=6, clip_on=False)
    ax.text(diag[0], diag[1] + 0.003, "?", ha="center", va="center", fontsize=S.LABEL_PT, fontweight="bold",
            color=S.INK, zorder=7, clip_on=False,
            bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=S.INK, lw=0.8))

    # direct labels
    T = dict(fontsize=S.LABEL_PT, zorder=8)

    def along(x, y, dx, text, color, off, **kw):
        """text rotated along a line of slope dy/dx = -1 (in data units), offset `off` points perpendicular."""
        p0 = ax.transData.transform((x, y))
        p1 = ax.transData.transform((x + dx, y - dx))
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.annotate(text, (x, y), xytext=(0, off), textcoords="offset points", rotation=ang, rotation_mode="anchor",
                    ha="left", va="center", color=color, **T, **kw)

    ax.set_xlim(X0, X1)
    ax.set_ylim(0, Y1)
    along(0.40, 1 - 0.40, 0.1, "healthy motor", S.INK, 6)
    along(0.06, k - 0.06, 0.1, "motor model (straight line)", S.TEAL, -6)
    ax.text(0.52, 0.315, "shortfall", color=S.ORANGE_TX, ha="center", va="center", **T)
    i = np.searchsorted(xs, 0.05)
    ax.text(0.03, yt[i] + 0.055, "this wheel's true limit (knee)", color=S.INK2, ha="left", va="bottom", **T)
    ax.text(0.03, ymax - 0.018, "answered by data", color=S.INK2, ha="left", va="top", **T)
    ax.annotate("what the 12-s probe saw", xy=tuple(probe[order[0]]), xytext=(0.03, ymax + 0.04),
                ha="left", va="bottom", color=S.BLUE, **T,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.BLUE, shrinkA=1, shrinkB=S.ARROW_SHRINK + 2))
    ax.annotate("surge of any child: seen", xy=tuple(surge), xytext=(0.20, 0.055), ha="left", va="center",
                color=S.INK, **T,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.INK, shrinkA=1, shrinkB=4.5))
    ax.annotate("diagonal of a child: never seen\nmodel says OK, true limit falls short", xy=tuple(diag),
                xytext=(1.03, 0.62), ha="right", va="bottom", color=S.INK, linespacing=1.05, **T,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.INK, shrinkA=1, shrinkB=5))

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("wheel speed (normalised)", labelpad=1)
    ax.set_ylabel("load torque (normalised)", labelpad=1)
    ax.tick_params(length=2, pad=1.5)
    S.save(fig, out)
    return N
