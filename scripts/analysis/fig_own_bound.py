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

# display name, code, colour, marker, marker size, label in panel (a)
METHODS = (("Ours", "amortized_comp", S.BLUE, "o", 4.2),
           ("CapXfer", "capxfer", S.VIOLET, ">", 3.8),
           ("ĝ-physics", "ghat_physics", S.TEAL, "D", 3.2),
           ("min-over-members", "min_over_members", S.GREY, "<", 3.5),
           ("detector-gated", "detector_gated", S.GREY, "^", 3.5))
BLOCKS = (("stratum", "stratum", "(a) Stratum"), ("lone-faulted", "lone", "(b) Lone faulted"),
          ("merge", "merge", "(c) Merges"))
# label offsets (points) and alignment per block: Ours one line above, CapXfer two lines below
LABEL_AT = {"stratum": {"amortized_comp": (-8, 8, "left"), "capxfer": (-5, -8, "right")},
            "lone-faulted": {"amortized_comp": (-12, 8, "left"), "capxfer": (0, -14, "center")},
            "merge": {"amortized_comp": (-30, 8, "left"), "capxfer": (4, -11, "right")}}
USE = ("claim", "method", "model_tag", "test_plant", "split", "status", "eid", "shard", "cand", "kind", "cand_size",
       "has_fault", "leg", "truth", "lower")

# log x axis with a broken zero: zero is drawn at X0, the axis is broken between X0 and XMIN
X0, XMIN, XMAX = 0.00022, 0.0005, 0.4


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
    W, H = S.W2, 1.92
    fig = S.plt.figure(figsize=(W, H))
    L, Bm, T = 0.40, 0.33, 0.21
    pw, gap = 1.17, 0.07
    ph = H - Bm - T
    axs = [fig.add_axes([(L + i * (pw + gap)) / W, Bm / H, pw / W, ph / H]) for i in range(3)]
    dL = L + 3 * pw + 2 * gap + 0.44
    dw = (W - dL - 0.06 - 0.06) / 2
    axd = [fig.add_axes([(dL + i * (dw + 0.06)) / W, Bm / H, dw / W, ph / H]) for i in range(2)]

    for i, (ax, (blk, bl, title)) in enumerate(zip(axs, BLOCKS)):
        zero_axis(ax)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.grid(True)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("certified share of capable (%)", labelpad=1.5)
        S.panel(ax, title)
        for name, code, col, mk, ms in METHODS[::-1]:
            p = P[code, blk]
            x, y = float(xpos(p["ed"])), 100 * p["cf"]
            xe = [[x - float(xpos(p["ed_lo"]))], [float(xpos(p["ed_hi"])) - x]]
            ye = [[y - 100 * p["cf_lo"]], [100 * p["cf_hi"] - y]]
            lw = S.LW_OURS if code == "amortized_comp" else S.LW_DATA * 0.8
            ax.errorbar([x], [y], xerr=xe, yerr=ye, fmt="none", ecolor=col, elinewidth=0.6 if code !=
                        "amortized_comp" else 0.9, capsize=0, alpha=0.75, zorder=3)
            ax.plot([x], [y], marker=mk, ms=ms, color=col, mec="white", mew=0.4, ls="none", zorder=4)
        # direct labels: Ours above its marker, CapXfer below (offsets in points, per panel)
        for code, col in (("amortized_comp", S.BLUE), ("capxfer", S.VIOLET)):
            p = P[code, blk]
            x, y = float(xpos(p["ed"])), 100 * p["cf"]
            txt = f"{100 * p['cf']:.1f} % at {100 * p['ed']:.2f} %"
            nm = "Ours" if code == "amortized_comp" else "CapXfer"
            dx, dy, ha = LABEL_AT[blk][code]
            ax.annotate(f"{nm} {txt}" if code == "amortized_comp" else f"{nm}\n{txt}", (x, y), xytext=(dx, dy),
                        textcoords="offset points", fontsize=S.LABEL_PT, color=col, va="center", ha=ha, zorder=6,
                        linespacing=1.0)
        if i == 1:
            ax.set_xlabel("unsafe commits at the method's own bound (%, log scale)", labelpad=1.5)
    # baseline names in (a) only (markers carry identity across panels)
    name_at = {"ghat_physics": (5, 0, "left"), "min_over_members": (0, 7, "center"), "detector_gated": (0, 7, "center")}
    for name, code, col, mk, ms in METHODS[2:]:
        p = P[code, "stratum"]
        dx, dy, ha = name_at[code]
        axs[0].annotate(name, (float(xpos(p["ed"])), 100 * p["cf"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=S.LABEL_PT, color=S.INK2, va="center", ha=ha)
    axs[2].text(0.30, 0.30, "\u2196 better", transform=axs[2].transAxes, fontsize=S.LABEL_PT, color=S.INK2)

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
        ax.plot(miss.truth.clip(*lim), miss.lower.clip(*lim), ls="none", marker="x", ms=3.0, mew=1.6,
                color="white", zorder=4)                                   # halo: misses read in grayscale
        ax.plot(miss.truth.clip(*lim), miss.lower.clip(*lim), ls="none", marker="x", ms=2.6, mew=0.7,
                color=S.ORANGE, zorder=5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
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
    S.panel(axd[0], "(d) Never-docked merges: bound under truth")
    S.save(fig, out)

    # ---------------------------------------------------------------- macros
    lab = {"amortized_comp": "ours", "capxfer": "capxfer", "ghat_physics": "ghat", "min_over_members": "minmem",
           "detector_gated": "detector"}
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
