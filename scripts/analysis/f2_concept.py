"""Concept figure (IEEE version), "What a probe sees, and what a new configuration needs": one wheel's operating plane.

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
W_IN, H_IN = 3.5, 3.45


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


MASK_G = 0.5       # weak gain of the masked wheel (inside the weak prior g ~ U(0.3, 1))
assert MASK_G == 0.5, "panel (a) axis label says half-strength"
MASK_WHEEL = 0     # wheel 0 of module 0, kept by every body below
MASK_LEG = "diag+-"
MASK_BODIES = (("in the 6-module robot", 6), ("in a 2-module child", 2), ("module alone", 1))


def _t0_masking():
    """For one weak wheel on one leg, per body (module 0 inside a 2x3 parent / a 2-module child / alone):
    needed |u| to keep pace (healthy-solve load torque and speed through Lemma 1, divided by the weak gain) and the
    body's capability rho on that leg with the wheel weak (T0 solve, saturation included)."""
    from capxfer import assembly as A
    from capxfer.sim import WEAK
    from cfcap.t0 import qs

    P, th = qs.load_params(), qs.load_theta()
    mods = [("M1", i, j, 0) for j in range(2) for i in range(3)]
    li = list(qs.LEG_NAMES).index(MASK_LEG)
    rows = []
    for name, nm in MASK_BODIES:
        asm = A.build(f"mask{nm}", mods[:nm], P)
        pl = qs.Plant(asm, P, th)
        xis = qs.decision_legs(asm)
        om_ref = pl.ik(xis)
        Mh, Gh = qs.healthy(asm.n, 1)
        rh = qs.solve(pl, Mh, Gh, om_ref)
        d, nom = pl.d, pl.nom
        vb = d["v_batt"] / nom["v0"]
        TS, OM0 = nom["tau_s"] * d["stall_scale"], nom["omega0"] * d["noload_scale"]
        tau = (rh["duty"][0] * vb - rh["om"][0] / OM0) * TS
        u_need = abs((tau[li, MASK_WHEEL] / TS + rh["om"][0][li, MASK_WHEEL] / OM0) / (MASK_G * vb))
        M, G = qs.healthy(asm.n, 1)
        M[0, MASK_WHEEL], G[0, MASK_WHEEL] = WEAK, MASK_G
        r = qs.solve(pl, M, G, om_ref)
        rho = float(qs.rho_of(r["xi"][0], xis, asm.char_length())[li])
        duty = float(abs(r["duty"][0][li, MASK_WHEEL]))
        rows.append(dict(name=name, n=nm, u_need=float(u_need), duty=duty, rho=rho))
    return rows


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
        mask = _t0_masking()
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
    msrc = f"cfcap/t0/qs.py (weak g={MASK_G}, wheel {MASK_WHEEL}, leg {MASK_LEG}, module 0 of a 2x3 parent)"
    N.set("figconcept.mask.g", f"{MASK_G:.1f}", msrc, st)
    for r_ in mask:
        kk = {6: "parent", 2: "pair", 1: "alone"}[r_["n"]]
        N.set(f"figconcept.mask.uneed.{kk}", f"{r_['u_need']:.2f}", msrc, st)
        N.set(f"figconcept.mask.duty.{kk}", f"{r_['duty']:.2f}", msrc, st)
        N.pct(f"figconcept.mask.rho.{kk}", r_["rho"], msrc, st)

    # ---- figure
    S.apply()
    fig = S.plt.figure(figsize=(W_IN, H_IN))
    # ---- (a) masking: needed power of the weak wheel per body, and how much motion the body keeps
    HA = 1.50                                    # inches for panel (a), title included
    fig.text(0.02 / W_IN, (H_IN - 0.03) / H_IN, "(a) Why an assembly hides a weak wheel", ha="left", va="top",
             fontsize=S.TITLE_PT, fontweight="bold", color=S.INK)
    a_y0, a_h = H_IN - HA + 0.51, HA - 0.51 - 0.52
    am = fig.add_axes([1.12 / W_IN, a_y0 / H_IN, 1.45 / W_IN, a_h / H_IN])
    umax = max(r_["u_need"] for r_ in mask)
    xr = 1.0 + 1.15 * (umax - 1.0) + 0.05
    XP = (W_IN - 0.08 - 1.12) / 1.45 * xr        # right edge of the "motion kept" column, data units
    for i, r_ in enumerate(mask):
        u = r_["u_need"]
        am.barh(i, min(u, 1.0), height=0.62, color="#c4c3bf", lw=0)
        if u > 1:
            am.barh(i, u - 1.0, left=1.0, height=0.62, color=S.ORANGE, lw=0)
        am.text(-0.04 * xr, i, r_["name"], ha="right", va="center", fontsize=S.LABEL_PT, color=S.INK)
        am.text(u + 0.03 * xr, i, f"{u:.2f}", ha="left", va="center", fontsize=S.LABEL_PT, color=S.ORANGE_TX
                if u > 1 else S.INK2)
        am.text(XP, i, f"{100 * r_['rho']:.1f} %", ha="right", va="center", fontsize=S.LABEL_PT,
                color=S.INK if r_["rho"] > 0.95 else S.ORANGE_TX, clip_on=False)
    # header row: reference line, the two number columns; the orange segment is named under the axis
    hy = -0.62
    am.text(XP, hy, "commanded\nmotion\nachieved", ha="right", va="bottom", fontsize=S.LABEL_PT, color=S.INK2,
            clip_on=False, linespacing=1.0)
    am.text(umax + 0.03 * xr, hy, "duty\nneeded", ha="left", va="bottom", fontsize=S.LABEL_PT, color=S.INK2,
            clip_on=False, linespacing=1.0)
    am.text(1.0 - 0.02 * xr, hy, "full duty\n(u = 1)", ha="right", va="bottom", fontsize=S.LABEL_PT,
            color=S.INK, clip_on=False, linespacing=1.0)
    am.annotate("duty beyond full (shortfall)", xy=(1.0, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -3.5), textcoords="offset points", ha="left", va="top", fontsize=S.LABEL_PT,
                color=S.ORANGE_TX, annotation_clip=False)
    am.plot([1.0, 1.0], [len(mask) - 0.5, hy], color=S.INK, lw=S.LW_THEORY, clip_on=False)
    am.set_ylim(len(mask) - 0.5, -0.5)
    am.set_xlim(0, xr)
    am.set_yticks([])
    am.set_xticks([0, 1], ["0", ""])      # u = 1 is named in the header
    am.spines["left"].set_visible(False)
    am.tick_params(length=2, pad=1.5)
    am.set_xlabel("duty a half-strength wheel needs on a diagonal", labelpad=1)
    fig.text(W_IN / 2 / W_IN, (a_y0 - 0.29) / H_IN, "same duty needed in every configuration; what changes is\n"
             "how much its neighbours compensate", ha="center", va="top", fontsize=S.LABEL_PT, color=S.INK2,
             linespacing=1.05)

    # ---- (b) the operating plane
    HB = H_IN - HA
    fig.text(0.02 / W_IN, (HB - 0.04) / H_IN, "(b) What a test drive sees, and what a new configuration needs", ha="left",
             va="top", fontsize=S.TITLE_PT, fontweight="bold", color=S.INK)
    ax = fig.add_axes([0.42 / W_IN, 0.58 / H_IN, (W_IN - 0.42 - 0.06) / W_IN, (HB - 0.58 - 0.24) / H_IN])
    X0, X1, YB = 0.0, 1.03, 0.50            # torque axis cut: everything the steps need lies below it
    xs = np.linspace(0, k, 600)
    yt = _true_envelope(xs, k, env)
    ok = np.isfinite(yt)
    xz = xs[ok][-1]                                 # true zero-torque speed
    ym = k - xs                                     # model line
    # (1) covered region: the dominated set, union of [0, x_p] x [0, y_p] over the probe's points
    order = np.argsort(-probe[:, 0])
    stair, ymax = [], 0.0
    for x_p, y_p in probe[order]:
        if y_p > ymax:
            stair.append((x_p, y_p))
            ymax = y_p
    for x_p, y_p in stair:
        ax.add_patch(Rectangle((0, 0), x_p, y_p, fc=S.BLUE, ec="none", alpha=0.12, zorder=1))
        ax.add_patch(Rectangle((0, 0), x_p, y_p, fc="none", ec=S.BLUE, lw=S.LW_BOX, zorder=2))
    # envelopes
    ax.plot([0, 1], [1, 0], color=S.INK, lw=S.LW_THEORY, zorder=3)
    ax.plot([0, k], [k, 0], color=S.TEAL, lw=S.LW_DATA, ls=(0, (4, 2)), zorder=3)
    ax.plot(np.r_[xs[ok], xz], np.r_[yt[ok], 0], color=S.INK2, lw=S.LW_DATA, zorder=3)
    # probe points; (2) forward need (inside); (3) diagonal need (outside)
    ax.plot(probe[:, 0], probe[:, 1], "o", color=S.BLUE, ms=S.MS, mec="white", mew=0.4, zorder=5, clip_on=False)
    ax.plot(*surge, "o", ms=6.5, mfc="none", mec=S.INK, mew=0.9, zorder=6, clip_on=False)   # probe dot shows inside
    ax.plot(*diag, "o", ms=6.5, mfc="white", mec=S.INK, mew=0.9, zorder=6, clip_on=False)
    # the key gap is in SPEED at ~zero torque: the real wheel tops out at xz (knee curve meets zero torque), the
    # fitted straight line at k; the diagonal needs diag[0], inside that interval. Orange = shortfall only.
    assert xz < diag[0] < k
    T = dict(fontsize=S.LABEL_PT, zorder=8)
    yb = 0.012
    ax.plot([xz, k], [yb, yb], color=S.ORANGE, lw=1.2, zorder=4, solid_capstyle="butt")
    for xx in (xz, k):
        ax.plot([xx, xx], [0, 2.2 * yb], color=S.ORANGE, lw=1.0, zorder=4)
    ax.plot([diag[0], diag[0]], [0, -0.025], color=S.INK, lw=0.6, zorder=4, clip_on=False)   # needed-speed tick

    def along(x, y, dx, text, color, off, **kw):
        """text rotated along a line of slope dy/dx = -1 (in data units), offset `off` points perpendicular."""
        p0 = ax.transData.transform((x, y))
        p1 = ax.transData.transform((x + dx, y - dx))
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.annotate(text, (x, y), xytext=(0, off), textcoords="offset points", rotation=ang, rotation_mode="anchor",
                    ha="left", va="center", color=color, **T, **kw)

    def under(x, row, text, color, ha="right"):
        """text under the speed axis, row 0 just below the axis line."""
        ax.annotate(text, xy=(x, 0), xycoords=("data", "axes fraction"), xytext=(0, -3.5 - 9.0 * row),
                    textcoords="offset points", ha=ha, va="top", color=color, annotation_clip=False, **T)

    ax.set_xlim(X0, X1)
    ax.set_ylim(0, YB)
    along(0.59, 1 - 0.59, 0.1, "healthy motor", S.INK, 5)
    along(0.45, k - 0.45, 0.1, "fitted motor model (straight line)", S.TEAL, 5)
    ax.text(0.03, ymax + 0.045, "this wheel's real limit (with knee)", color=S.INK2, ha="left", va="center", **T)
    ax.text(0.025, ymax - 0.02, "covered by the 12-s test drive", color=S.BLUE, ha="left", va="top", **T)
    ax.text(surge[0] - 0.035, 0.035, "forward: inside → answered by data", color=S.INK, ha="right", va="center",
            **T)
    under(X1, 1, "diagonal: outside → answered by the motor model", S.INK)
    under(X1, 2, "orange interval: model says reachable; real wheel cannot", S.ORANGE_TX)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, 0.5], ["0", "0.5"])
    ax.set_xlabel("wheel speed (fraction of no-load speed)", labelpad=20)
    ax.set_ylabel("torque (fraction of stall torque)", labelpad=1)
    ax.tick_params(length=2, pad=1.5)
    N.set("figconcept.speed.forward", f"{surge[0]:.2f}", src, st)
    N.set("figconcept.speed.diag", f"{diag[0]:.2f}", src, st)
    N.set("figconcept.speed.truetop", f"{xz:.2f}", src, st)
    N.set("figconcept.speed.modeltop", f"{k:.2f}", src, st)
    N.set("figconcept.speed.probetop", f"{probe[:, 0].max():.2f}", src, st)
    S.check_fonts(fig, S.LABEL_PT)       # IEEE review: every label at >= 7 pt
    S.save(fig, out)
    return N
