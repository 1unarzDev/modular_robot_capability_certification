"""Fig. 4: at each method's own guaranteed bound, Ours certifies more and is unsafe less; never-docked merges.

Confirmatory run 1 (77fc39b), descriptive cells of the same run; nothing is re-scored here.
  (a-c) C-MAIN, plant T1, blocks stratum / lone-faulted / merge: CF_own (y) vs E_dec_own (x) at the method's own
        bound (`lower`). Points come from results/confirmatory/cells.csv; 95 % CIs from an episode bootstrap
        (resampling eid, B = 1000, fixed seed) over the scored decision rows, rebuilt with the scorer's own filters
        and selectors (confirmatory.split_seeds / exclusions / apply_ranges / cluster_stats / _sel). The rebuilt
        point estimate must equal cells.csv, else the script raises.
  (d)   C-MERGE, comp_enum_size (Ours on merges): lower bound vs truth per (pair, candidate, leg), T1 and T2, hexbin;
        misses (lower > truth) in orange.
No matched (U* = 1 %) operating point appears anywhere in this figure (CLAIMS.md must-not list).
Outputs: paper/figs/f4_own_bound.pdf; macros cfFigFour*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import common as C
import confirmatory as K
import figstyle as S

SD = "confirmatory-descriptive"
CELLS = "results/confirmatory/cells.csv"
MODELS = "models/confirmatory/dec_conf_models.csv.gz"
BASES = "baselines/confirmatory/dec_conf_baselines.csv.gz"
MERGE = "stress/confirmatory/dec_conf_merge.csv.gz"
B_FIG, SEED_FIG = 1000, 4

# display name (exactly as Table 1), code, colour, marker, marker size (>= 4.5 pt for the three that must separate in
# grayscale). Draw order is the reverse of this tuple, so CapXfer is drawn last, on top.
METHODS = (("CapXfer", "capxfer", S.VIOLET, ">", 4.8),
           ("Ours", "amortized_comp", S.BLUE, "o", 5.0),
           ("ĝ-physics", "ghat_physics", S.TEAL, "D", 4.5),
           ("Oracle", "oracle", S.INK, "*", 5.5),
           ("Min over members", "min_over_members", S.GREY, "<", 4.2),
           ("Detector-gated", "detector_gated", S.GREY, "^", 4.2),
           ("Copy parent", "copy_parent", S.GREY, "s", 3.8),
           ("Shape metadata", "metadata_nf", S.GREY, "o", 6.0))     # hollow ring around Copy parent at (0, 0)
SCORER_ONLY = {"metadata_nf"}   # computed by the scorer (add_trivial), not read from rows: point from cells.csv
BLOCKS = (("stratum", "stratum", "Stratum"), ("lone-faulted", "lone", "Lone"), ("merge", "merge", "Merges"))
# Direct labels per block. ("off", dx, dy, ha[, lines]): offset in points from the marker, no leader.
# ("at", x_rate, y_pct, ha[, lines]): text placed in data coordinates with a 0.4 pt leader to the marker.
# lines: 2 or 3 line layout of the Ours/CapXfer number label (default Ours 1, CapXfer 3).
LABEL_AT = {
    "stratum": {"amortized_comp": ("off", -4, 8, "left"), "capxfer": ("at", 0.17, 42, "center"),
                "ghat_physics": ("at", 0.0088, 64.1, "right"), "min_over_members": ("at", 0.0050, 36.3, "right"),
                "detector_gated": ("off", 0, 7.5, "center"), "copy_parent": ("off", 9, -2.5, "left"),
                "oracle": ("off", -2.5, 7, "left")},
    "lone-faulted": {"amortized_comp": ("off", -10, 12, "left"), "capxfer": ("off", 7, 11, "center", 2),
                     "ghat_physics": ("at", 0.95, 38, "right"), "detector_gated": ("off", 3, -9, "left"),
                     "copy_parent": ("off", 9, -3, "left"), "oracle": ("off", -2.5, 7, "left")},
    "merge": {"amortized_comp": ("off", -7, 8, "left"), "capxfer": ("at", 0.95, 19, "right"),
              "ghat_physics": ("off", -6, 7, "right"), "min_over_members": ("off", 6, 0, "left"),
              "detector_gated": ("off", 9, 4, "left"), "copy_parent": ("off", 9, -4, "left"),
              "oracle": ("off", -2.5, 7, "left")}}
NOLEAD = {("stratum", "ghat_physics"), ("stratum", "min_over_members")}   # 'at' labels set beside the CI, no leader
TWO_LINE = {"Min over members": "Min over\nmembers"}
USE = ("claim", "method", "model_tag", "test_plant", "split", "status", "eid", "shard", "cand", "kind", "cand_size",
       "has_fault", "leg", "truth", "lower")

# log x axis with a broken zero: zero is drawn at X0, the axis is broken between X0 and XMIN
X0, XMIN, XMAX = 0.00022, 0.0005, 1.0


def load_scored(res, rel, claim, extra=()):
    """Decision rows of one confirmatory file after the scorer's own row filters (seed tag, cal split, canary and
    pre-touched exclusions, A-POWER prefixes), restricted to `claim`."""
    hdr = pd.read_csv(res / rel, nrows=0).columns
    cols = [c for c in USE + tuple(extra) if c in hdr]
    d = pd.read_csv(res / rel, usecols=cols, low_memory=False)
    d = d[d["claim"] == claim]
    for c in USE + tuple(extra):
        if c not in d:
            d[c] = np.nan
    sc = K.Scorer()
    d, _ = K.split_seeds(d, sc, rel)
    d = d[~K.is_cal(d)]
    d = K.exclusions(d, sc.log, rel)
    return K.apply_ranges(d, sc, rel).reset_index(drop=True)


def boot_ci(bt, t, num, den):
    est, reps = bt.ratio(t, num, den)
    c = K.ci(reps)
    return est, c["ci_lo"], c["ci_hi"]


def xpos(x):
    x = np.asarray(x, float)
    return np.where(x <= 0, X0, np.maximum(x, XMIN * 1.0001))


def zero_axis(ax, which="x"):
    """Log axis with a broken '0' tick (zeros are drawn, never dropped)."""
    ticks = [X0, 0.001, 0.01, 0.1]
    labs = ["0", "0.1", "1", "10"]
    if which == "x":
        ax.set_xscale("log")
        ax.set_xlim(X0 / 1.6, XMAX)
        ax.set_xticks(ticks, labs)
        ax.xaxis.set_minor_locator(S.matplotlib.ticker.NullLocator())
        xb = np.sqrt(X0 * XMIN)
        kw = dict(transform=ax.get_xaxis_transform(), color=S.GREY, lw=S.LW_AXES, clip_on=False)
        for dx in (0.88, 1.0 / 0.88):
            ax.plot([xb * dx / 1.12, xb * dx * 1.12], [-0.035, 0.035], **kw)
        ax.plot([xb / 1.0, xb], [0, 0], color="white", lw=2.2, transform=ax.get_xaxis_transform(), clip_on=False,
                zorder=5)


def run(res, paper):
    N = C.Numbers("fig_own_bound.py")
    out = paper / "figs/f4_own_bound.pdf"
    cp = res / "confirmatory/cells.csv"
    if not cp.exists() or not (res / MODELS).exists():
        S.placeholder(out, "Fig. 4: awaiting confirmatory cells + decision rows")
        N.ph("fig4.missing", "confirmatory inputs missing")
        return N
    cells = pd.read_csv(cp)
    cm = cells[(cells.claim == "C-MAIN") & (cells.plant == "T1")]

    # ---------------------------------------------------------------- C-MAIN rows, rebuilt with the scorer's filters
    srcs = [CELLS, "results/" + MODELS, "results/" + BASES]
    codes = [m[1] for m in METHODS]
    dec = pd.concat([load_scored(res, MODELS, "C-MAIN"), load_scored(res, BASES, "C-MAIN")], ignore_index=True)
    dec = dec[(dec.split == "conf_test") & dec.eid.between(K.CMAIN_IDX[0], K.CMAIN_IDX[1] - 1)
              & (dec.test_plant == "T1") & dec.method.isin(codes)]
    P = {}                                     # (code, block) -> dict(cf, cf_lo, cf_hi, ed, ed_lo, ed_hi, n_ep)
    for code in codes:
        if code in SCORER_ONLY:
            for blk, _, _ in BLOCKS:
                row = cm[(cm.method == code) & (cm.block == blk)].iloc[0]
                if row.CF_own != 0 or row.E_dec_own != 0:
                    raise ValueError(f"fig4: {code}/{blk} is no longer (0, 0); it needs a bootstrap CI")
                # certifies nothing in every episode: every bootstrap replicate is 0, so the CI is [0, 0]
                P[code, blk] = dict(cf=0.0, cf_lo=0.0, cf_hi=0.0, ed=0.0, ed_lo=0.0, ed_hi=0.0, n_ep=int(row.n_ep),
                                    n_inc=-1)
            continue
        ev = dec[dec.method == code].reset_index(drop=True)
        bt = K.Boot(np.sort(ev.eid.unique()), B=B_FIG, seed=SEED_FIG)
        for blk, _, _ in BLOCKS:
            t = K.cluster_stats(ev, ev["lower"], K._sel(blk))
            cf = boot_ci(bt, t, "cert", "cap")
            ed = boot_ci(bt, t, "uns", "inc")
            row = cm[(cm.method == code) & (cm.block == blk)].iloc[0]
            if not (np.isclose(cf[0], row.CF_own) and np.isclose(ed[0], row.E_dec_own) and len(t) == row.n_ep):
                raise ValueError(f"fig4: rebuilt {code}/{blk} ({cf[0]}, {ed[0]}, {len(t)}) != cells.csv "
                                 f"({row.CF_own}, {row.E_dec_own}, {row.n_ep}); definitions drifted")
            P[code, blk] = dict(cf=row.CF_own, cf_lo=cf[1], cf_hi=cf[2], ed=row.E_dec_own, ed_lo=ed[1],
                                ed_hi=ed[2], n_ep=int(row.n_ep), n_inc=int(t.inc.sum()))

    # ---------------------------------------------------------------- C-MERGE rows (Ours on merges)
    mg = load_scored(res, MERGE, "C-MERGE", extra=("arm", "point"))
    mg = mg[mg.method == "comp_enum_size"]
    mg = mg[np.isfinite(mg.truth) & np.isfinite(mg.lower)]
    srcs.append("results/" + MERGE)

    # ---------------------------------------------------------------- figure
    S.apply()
    W, H = S.W2, 2.0
    fig = S.plt.figure(figsize=(W, H))
    L, Bm, T = 0.40, 0.33, 0.21
    pw, gap = 1.30, 0.07
    ph = H - Bm - T
    axs = [fig.add_axes([(L + i * (pw + gap)) / W, Bm / H, pw / W, ph / H]) for i in range(3)]
    dL = L + 3 * pw + 2 * gap + 0.38
    dw = (W - dL - 0.06 - 0.06) / 2
    axd = [fig.add_axes([(dL + i * (dw + 0.06)) / W, Bm / H, dw / W, ph / H]) for i in range(2)]

    lead = dict(arrowstyle="-", lw=0.4, color=S.INK2, shrinkA=1.0, shrinkB=S.ARROW_SHRINK)
    for i, (ax, (blk, bl, short)) in enumerate(zip(axs, BLOCKS)):
        zero_axis(ax)
        ax.set_ylim(-14, 114)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.grid(True)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("certified share of capable (%)", labelpad=1.5)
        o, c = P["amortized_comp", blk], P["capxfer", blk]
        verdict = "safer" if o["ed"] < c["ed"] else "not safer"
        S.panel(ax, f"({'abc'[i]}) {short}: {100 * (o['cf'] - c['cf']):+.0f} pts, {verdict}",
                x=-0.12 if i == 0 else None)
        g, mm = P["ghat_physics", blk], P["min_over_members", blk]
        same = abs(g["cf"] - mm["cf"]) < 0.005 and abs(g["ed"] - mm["ed"]) < 5e-4   # lone modules: one member
        for name, code, col, mk, ms in METHODS[::-1]:
            if same and code == "min_over_members":
                continue
            p = P[code, blk]
            x, y = float(xpos(p["ed"])), 100 * p["cf"]
            xe = [[x - float(xpos(p["ed_lo"]))], [float(xpos(p["ed_hi"])) - x]]
            ye = [[y - 100 * p["cf_lo"]], [100 * p["cf_hi"] - y]]
            ours = code == "amortized_comp"
            top = code == "capxfer"
            ax.errorbar([x], [y], xerr=xe, yerr=ye, fmt="none", ecolor=col, elinewidth=0.9 if ours else 0.6,
                        capsize=0, alpha=0.75, zorder=3 if not top else 6)
            if code == "metadata_nf":
                ax.plot([x], [y], marker=mk, ms=ms, mfc="none", mec=col, mew=0.6, ls="none", zorder=4, clip_on=False)
            else:
                ax.plot([x], [y], marker=mk, ms=ms, color=col, mec="white", mew=0.7 if top else 0.5, ls="none",
                        zorder=7 if top else 4, clip_on=False)
            # direct label
            spec = LABEL_AT[blk].get(code)
            if spec is None:
                continue
            if code in ("amortized_comp", "capxfer"):
                nl = spec[4] if len(spec) > 4 else (1 if ours else 3)
                a_, b_ = f"{100 * p['cf']:.1f} %", f"{100 * p['ed']:.2f} %"
                txt = {1: f"{name} {a_} at {b_}", 2: f"{name}\n{a_} at {b_}", 3: f"{name}\n{a_} at\n{b_}"}[nl]
                tc = col
            else:
                txt = "ĝ-physics ≈\nMin over members" if (same and code == "ghat_physics") else name
                if code == "copy_parent":
                    txt = "Copy parent,\nShape metadata"
                if spec[0] == "at" and (blk, code) in NOLEAD:
                    txt = TWO_LINE.get(txt, txt)
                tc = S.INK if code == "oracle" else S.INK2
            kw = dict(fontsize=S.LABEL_PT, color=tc, va="center", linespacing=1.0, zorder=8)
            if spec[0] == "off":
                ax.annotate(txt, (x, y), xytext=spec[1:3], textcoords="offset points", ha=spec[3], **kw)
            else:
                lk = None if (blk, code) in NOLEAD else dict(lead, relpos=(0.7, 1.0) if spec[2] < y else (0.5, 0.5))
                ax.annotate(txt, (x, y), xytext=(spec[1], spec[2]), textcoords="data", ha=spec[3],
                            arrowprops=lk, **kw)
        if i == 1:
            ax.set_xlabel("unsafe commits at the method's own bound (%, log scale)", labelpad=1.5)
    axs[0].annotate("better", xy=(X0 * 1.1, 80), xytext=(0.0006, 52), textcoords="data", fontsize=S.LABEL_PT,
                    color=S.INK2, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color=S.INK2, shrinkA=2, shrinkB=S.ARROW_SHRINK,
                                    mutation_scale=5))

    # (d) hexbins
    cmap = LinearSegmentedColormap.from_list("blue", ["#e3edf9", S.BLUE, "#0d3f7a"])
    lim = (0.0, 1.25)
    for j, (ax, plant) in enumerate(zip(axd, ("T1", "T2"))):
        g = mg[mg.test_plant == plant]
        ax.hexbin(g.truth.clip(*lim), g.lower.clip(*lim), gridsize=26, extent=(*lim, *lim), cmap=cmap,
                  norm=LogNorm(vmin=1, vmax=max(2, len(g) / 20)), mincnt=1, linewidths=0.15, edgecolors="face",
                  zorder=2)
        ax.plot(lim, lim, color=S.INK, lw=S.LW_THEORY, ls=(0, (3, 2)), zorder=3)
        miss = g[g.lower > g.truth]
        ax.plot(miss.truth.clip(*lim), miss.lower.clip(*lim), ls="none", marker="x", ms=4.2, mew=2.4,
                color="white", zorder=8, clip_on=False)                    # halo: misses read in grayscale
        ax.plot(miss.truth.clip(*lim), miss.lower.clip(*lim), ls="none", marker="x", ms=3.6, mew=1.0,
                color=S.ORANGE, zorder=9, clip_on=False)
        ax.set_xlim(-0.04, lim[1])
        ax.set_ylim(-0.04, lim[1])
        ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
        ax.set_aspect("auto")
        ax.text(0.05, 0.95, plant, transform=ax.transAxes, fontsize=S.LABEL_PT, fontweight="bold", color=S.INK,
                va="top")
        nm = int(len(miss))
        ax.text(0.95, 0.05, f"{nm} misses\nin {len(g):,}", transform=ax.transAxes, fontsize=S.LABEL_PT,
                color=S.ORANGE_TX, ha="right", va="bottom", linespacing=1.0)
        ax.set_xlabel("true capability", labelpad=1.5)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("certified lower bound", labelpad=1.5)
        N.num(f"fig4.merge.misses.{plant.lower()}", nm, "results/" + MERGE, SD)
        N.num(f"fig4.merge.entries.{plant.lower()}", len(g), "results/" + MERGE, SD)
        N.pct(f"fig4.merge.missrate.{plant.lower()}", nm / len(g), "results/" + MERGE, SD, d=2)
        N.num(f"fig4.merge.pairs.{plant.lower()}", g.groupby(["arm", "eid"]).ngroups, "results/" + MERGE, SD)
    S.panel(axd[0], "(d) Never-docked: bound under truth", x=-0.27)
    S.save(fig, out)

    # ---------------------------------------------------------------- macros
    lab = {"amortized_comp": "ours", "capxfer": "capxfer", "ghat_physics": "ghat", "min_over_members": "minmem",
           "detector_gated": "detector", "copy_parent": "copy", "oracle": "oracle", "metadata_nf": "metadata"}
    for (code, blk), p in P.items():
        bl = dict((b, s) for b, s, _ in BLOCKS)[blk]
        k = f"fig4.{lab[code]}.{bl}"
        N.pct(k + ".cf", p["cf"], srcs, SD)
        N.pct(k + ".cf.lo", p["cf_lo"], srcs, SD)
        N.pct(k + ".cf.hi", p["cf_hi"], srcs, SD)
        N.pct(k + ".edec", p["ed"], srcs, SD, d=2)
        N.pct(k + ".edec.lo", p["ed_lo"], srcs, SD, d=2)
        N.pct(k + ".edec.hi", p["ed_hi"], srcs, SD, d=2)
        if blk == "stratum":
            N.num(f"fig4.{lab[code]}.neps", p["n_ep"], srcs, SD)
    N.num("fig4.boot.B", B_FIG, "fig_own_bound.py", SD)
    return N
