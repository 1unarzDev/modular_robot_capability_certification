"""Fig. 5: plug-in physics fails silently under a wrong motor law (C-GATE, confirmatory run 1; cells of the run).

Cells: results/confirmatory/cells.csv, claim C-GATE, T2 knee law, children of >= 3 modules (size ge3), dominance
'undominated' (legs the probe does not observe: extrapolated through the motor law). raw posterior and calibrated
physics (code par) use probe v2, robust uses v2+clamp (the scorer's PROBE_OF). (a) E_ep, the share of episodes with
any unsafe commit in the cell; (b) CF_own on undominated capable decisions, both vs knee severity s = 0..3.
95 % CIs: episode bootstrap (resampling eid, B = 1000, fixed seed; episodes paired across severities and methods, as
in the scorer) over results/stress/confirmatory/dec_conf_gate.csv.gz rebuilt with the scorer's filters and selectors;
rebuilt points must equal cells.csv. Nothing is re-scored: P5's verdict is read from verdicts.csv (it did not pass,
because calibrated physics stayed safe), and the T2 par construction anomaly (CONFIRMATORY_RESULTS scorer anomaly 1)
is carried as macros for the caption. robust is shown, not claimed (P6' failed).
Outputs: paper/figs/f5_silent.pdf; macros cfFigFive*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
import confirmatory as K
import figstyle as S
from fig_own_bound import B_FIG, load_scored

SD, ST = "confirmatory-descriptive", "confirmatory"
GATE = "stress/confirmatory/dec_conf_gate.csv.gz"
CELLS = "results/confirmatory/cells.csv"
V = "results/confirmatory/verdicts.csv"
SEED_FIG = 5
SEV = (0, 1, 2, 3)
# display, code, probe, colour, marker, linestyle, macro label
METHODS = (("raw posterior", "raw", "v2", S.TEAL, "x", (0, (3, 1.6)), "raw"),
           ("robust", "robust", "v2+clamp", S.INK2, "v", (0, (1, 1.3)), "robust"),
           ("calibrated physics", "par", "v2", S.TEAL, "s", "-", "par"))
Y0, YMIN, YMAX = 0.0013, 0.003, 0.75       # log y with a broken zero for E_ep


def ypos(y):
    y = np.asarray(y, float)
    return np.where(y <= 0, Y0, np.maximum(y, YMIN * 1.0001))


def run(res, paper):
    N = C.Numbers("fig_silent.py")
    out = paper / "figs/f5_silent.pdf"
    cp = res / "confirmatory/cells.csv"
    if not cp.exists() or not (res / GATE).exists():
        S.placeholder(out, "Fig. 5: awaiting confirmatory C-GATE cells")
        N.ph("fig5.missing", "confirmatory C-GATE inputs missing")
        return N
    cells = pd.read_csv(cp)
    cg = cells[(cells.claim == "C-GATE") & (cells["size"] == "ge3")]
    srcs = [CELLS, "results/" + GATE]

    d = load_scored(res, GATE, "C-GATE", extra=("severity", "probe", "dominated"))
    d = d[d.eid.between(K.CGATE_IDX[0], K.CGATE_IDX[1] - 1)]
    bt = K.Boot(np.sort(d.eid.unique()), B=B_FIG, seed=SEED_FIG)
    undom = lambda x, tau: K.stratum_mask(x, tau) & (x["dominated"].to_numpy(float) == 0)  # noqa: E731
    R = {}
    for _, code, probe, *_ in METHODS:
        for s in SEV:
            g = d[(d.method == code) & (d.probe == probe) & (d.severity == s)]
            e = g[(g.kind == "child") & (g.cand_size >= 3)].reset_index(drop=True)
            t = K.cluster_stats(e, e["lower"], undom)
            r = dict(n_ep=len(t), n_inc=int(t.inc.sum()))
            for k, num, den in (("eep", "ep_unsafe", "one"), ("cf", "cert", "cap"), ("edec", "uns", "inc")):
                est, reps = bt.ratio(t, num, den)
                c = K.ci(reps)
                r[k], r[k + "_lo"], r[k + "_hi"] = est, c["ci_lo"], c["ci_hi"]
            row = cg[(cg.method == code) & (cg.probe == probe) & (cg.severity == s) & (cg.dominance == "undominated")]
            row = row.iloc[0]
            if not (np.isclose(r["eep"], row.E_ep) and np.isclose(r["cf"], row.CF_own) and r["n_ep"] == row.n_ep):
                raise ValueError(f"fig5: rebuilt {code}/{probe}/s={s} ({r['eep']}, {r['cf']}) != cells.csv "
                                 f"({row.E_ep}, {row.CF_own}); definitions drifted")
            R[code, s] = r

    # ---------------------------------------------------------------- figure
    S.apply()
    W, H = S.W1, 2.0
    fig = S.plt.figure(figsize=(W, H))
    L, Bm, T, gap, Rm = 0.37, 0.31, 0.20, 0.40, 0.04
    pw = (W - L - gap - Rm) / 2
    ph = H - Bm - T
    axa = fig.add_axes([L / W, Bm / H, pw / W, ph / H])
    axb = fig.add_axes([(L + pw + gap) / W, Bm / H, pw / W, ph / H])
    dodge = {"raw": 0.0, "robust": 0.09, "par": -0.09}

    # (a) E_ep, log with a broken zero; region above alpha in orange tint
    axa.set_yscale("log")
    axa.set_ylim(Y0 / 1.7, YMAX)
    axa.axhspan(C.ALPHA, YMAX, color=S.ORANGE, alpha=0.10, lw=0, zorder=0)
    axa.axhline(C.ALPHA, color=S.INK, lw=S.LW_THEORY, ls=(0, (3, 2)), zorder=1)
    axa.text(-0.25, C.ALPHA * 1.12, "α = 10 %", fontsize=S.LABEL_PT, color=S.INK, va="bottom", ha="left")
    axa.set_yticks([Y0, 0.01, 0.1], ["0", "1", "10"])
    axa.yaxis.set_minor_locator(S.matplotlib.ticker.NullLocator())
    yb = np.sqrt(Y0 * YMIN)
    for f in (0.85, 1 / 0.85):
        axa.plot([-0.035, 0.035], [yb * f / 1.1, yb * f * 1.1], transform=axa.get_yaxis_transform(), color=S.GREY,
                 lw=S.LW_AXES, clip_on=False)
    for ax in (axa, axb):
        ax.set_xlim(-0.35, 3.35)
        ax.set_xticks(SEV)
        ax.set_xlabel("knee severity $s$", labelpad=1.5)
        ax.yaxis.grid(True)
    for name, code, probe, col, mk, ls, lab in METHODS:
        xs = np.array(SEV) + dodge[code]
        lw = S.LW_DATA
        y = np.array([R[code, s]["eep"] for s in SEV])
        lo = np.array([R[code, s]["eep_lo"] for s in SEV])
        hi = np.array([R[code, s]["eep_hi"] for s in SEV])
        axa.errorbar(xs, ypos(y), yerr=[ypos(y) - ypos(lo), ypos(hi) - ypos(y)], fmt="none", ecolor=col,
                     elinewidth=0.6, alpha=0.7, zorder=2)
        axa.plot(xs, ypos(y), color=col, ls=ls, lw=lw, marker=mk, ms=S.MS, mfc=col if mk != "s" else col,
                 mec=col, mew=0.9 if mk == "x" else 0.4, zorder=3)
        y = np.array([R[code, s]["cf"] for s in SEV])
        lo = np.array([R[code, s]["cf_lo"] for s in SEV])
        hi = np.array([R[code, s]["cf_hi"] for s in SEV])
        axb.fill_between(xs, 100 * lo, 100 * hi, color=col, alpha=S.CI_ALPHA, lw=0, zorder=1)
        axb.plot(xs, 100 * y, color=col, ls=ls, lw=lw, marker=mk, ms=S.MS, mec=col, mew=0.9 if mk == "x" else 0.4,
                 zorder=3)
    # direct labels, (a): values at s = 3
    r3 = {c: R[c, 3]["eep"] for _, c, *_ in METHODS}
    axa.annotate(f"raw posterior\n{100 * r3['raw']:.1f} %", (3, r3["raw"]), xytext=(-5, 1), textcoords="offset points",
                 ha="right", va="center", fontsize=S.LABEL_PT, color=S.TEAL, linespacing=1.0)
    lead = dict(arrowstyle="-", lw=0.4, color=S.INK2, shrinkA=0.5, shrinkB=S.ARROW_SHRINK)
    axa.annotate(f"robust\n{100 * r3['robust']:.1f} %", (3 + dodge["robust"], r3["robust"]), xytext=(2.6, 0.031),
                 textcoords="data", ha="right", va="center", fontsize=S.LABEL_PT, color=S.INK2, linespacing=1.0,
                 arrowprops=lead)
    axa.annotate(f"calibrated\nphysics {100 * r3['par']:.1f} %", (3 + dodge["par"], r3["par"]), xytext=(1.85, 0.0042),
                 textcoords="data", ha="right", va="center", fontsize=S.LABEL_PT, color=S.TEAL, linespacing=1.0,
                 arrowprops=lead)
    axa.set_ylabel("episodes with an unsafe commit (%)", labelpad=1.5)
    S.panel(axa, "(a) Unsafe episodes climb …")

    axb.set_ylim(0, 108)
    axb.set_yticks([0, 25, 50, 75, 100])
    axb.set_ylabel("certified share, extrapolated (%)", labelpad=1.5)
    for code, dy, txt, col in (("raw", -7, "raw posterior", S.TEAL), ("robust", -8, "robust", S.INK2),
                               ("par", 9, "calibrated physics†", S.TEAL)):
        axb.annotate(txt, (1.5, 100 * np.mean([R[code, 1]["cf"], R[code, 2]["cf"]])), xytext=(0, dy),
                     textcoords="offset points", ha="center", va="center", fontsize=S.LABEL_PT, color=col)
    S.panel(axb, "(b) … certifying no less")
    S.save(fig, out)

    # ---------------------------------------------------------------- macros
    for _, code, probe, *_, lab in METHODS:
        for s in SEV:
            r = R[code, s]
            k = f"fig5.{lab}.s{s}"
            for m in ("eep", "cf", "edec"):
                if m == "edec" and r["n_inc"] < 50:          # n < 50 incapable decisions: not reported
                    continue
                for suf in ("", "_lo", "_hi"):
                    x = r[m + suf]
                    N.pct(f"{k}.{m}{suf.replace('_', '.')}", x, srcs, SD, d=2 if np.isfinite(x) and x < 0.01 else 1)
            N.num(f"{k}.nincap", r["n_inc"], srcs, SD)
    N.num("fig5.neps", R["raw", 0]["n_ep"], srcs, SD)
    N.num("fig5.boot.B", B_FIG, "fig_silent.py", SD)
    vp = res / "confirmatory/verdicts.csv"
    if vp.exists():
        v = pd.read_csv(vp)
        ov = v[(v.claim == "C-GATE") & (v.bar == "1 H12 overall")]
        if len(ov) == 1:
            N.set("fig5.p5.verdict", "passed" if ov.verdict.iloc[0] == "PASS" else "did not pass", V, ST)
        pr = v[(v.claim == "C-GATE") & (v.bar == "price of robustness")]
        if len(pr) == 1:
            N.set("fig5.price.pts", f"{100 * pr.estimate.iloc[0]:.1f}".replace("-", "$-$"), V, SD)
    # T2 par construction anomaly: par certifies about a third even on DOMINATED (observed) legs at s = 0
    for code, probe, lab in (("par", "v2", "par"), ("pi", "v2", "pi")):
        r = cg[(cg.method == code) & (cg.probe == probe) & (cg.severity == 0) & (cg.dominance == "dominated")]
        if len(r) == 1:
            N.pct(f"fig5.anomaly.{lab}.dominated.cf.s0", r.CF_own.iloc[0], CELLS, SD)
    return N
