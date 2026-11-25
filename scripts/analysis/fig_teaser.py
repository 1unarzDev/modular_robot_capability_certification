"""Fig. 1, the teaser: one probe of today's body bounds every body it could become next; commit where the bound
clears the task, probe where the bound trusts the motor law.

One real T0 episode (Stage A2 seed 88, the identified set after probe v2, sensor noise, motor spread unknown =
rho model 'prior'); it is labelled as one episode in the figure.
Inputs:
  scripts/analysis/assets/teaser_episode.json   geometry of the episode (teaser_episode.py; catalog build)
  scripts/analysis/assets/teaser_render.png     Blender render of the parent (render_modules.py)
  results/stageA/A2_episodes.csv                n_hyp, consistent-set size after v2, n_cands, n_children
  results/theory/coverage_ambiguous.csv         every undecided (lo < tau <= hi) decision of the episode: which body,
                                                which legs, identified-set [lo, hi], extrapolated flag (undominated_v2)
Outputs: paper/figs/teaser.pdf (6.75 x 1.9 in, include 1:1), macros cfTeaser*.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

import common as C
import figstyle as S

ASSETS = pathlib.Path(__file__).resolve().parent / "assets"
EPISODE, RENDER = ASSETS / "teaser_episode.json", ASSETS / "teaser_render.png"
AMB, A2 = "theory/coverage_ambiguous.csv", "stageA/A2_episodes.csv"
LEGN = ("surge+", "surge−", "sway+", "sway−", "diag++", "diag−−", "diag+−", "diag−+",
        "yaw+", "yaw−")   # scripts/theory_coverage.py LEGN (qs.decision_legs order)
RHO_MODEL = "prior"            # motor spread unknown (the episode has heterogeneity)
W, H = S.W2, 1.9
FAULT_TINT = "#fbe3d8"


def verdict(lo, hi, tau):
    return "commit" if lo >= tau else ("refuse" if hi < tau else "probe")


def load(res):
    E = json.loads(EPISODE.read_text())
    a2f, af = res / A2, res / AMB
    ep = amb = None
    srcs = []
    if a2f.exists():
        d = pd.read_csv(a2f)
        d = d[d.seed == E["seed"]]
        if len(d):
            ep = d.iloc[0]
            srcs.append(C.rel(a2f))
            assert int(ep.n_children) == E["a2"]["n_children"], "episode replay does not match A2_episodes.csv"
            assert int(ep.n_cands) == E["a2"]["n_children"] + E["a2"]["n_merges"]
    if af.exists():
        d = pd.read_csv(af)
        amb = d[(d.seed == E["seed"]) & (d.noise == "sensor")]
        srcs.append(C.rel(af))
    return E, ep, amb, srcs


# ------------------------------------------------------------------------------------------------- panel (a)
def panel_render(fig, E, ep, box):
    ax = fig.add_axes(box)
    ax.axis("off")
    if RENDER.exists():
        img = mpimg.imread(RENDER)
        ax.imshow(img, interpolation="lanczos")
        ax.set_xlim(0, img.shape[1]), ax.set_ylim(img.shape[0], 0)
        pj = json.loads(RENDER.with_name(RENDER.stem + "_proj.json").read_text())["proj"]
        f = E["faults"][0]
        u, v = pj[f"wheel{f['wheel']}"]
        px, py = u * img.shape[1], (1 - v) * img.shape[0]
        ax.annotate(f"{f['mode']}ing wheel,\nhidden in the body", (px, py), xytext=(0.99, 0.02),
                    textcoords="axes fraction", ha="right", va="bottom", fontsize=7, color=S.ORANGE_TX,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=S.ORANGE_TX, shrinkA=1, shrinkB=4))
    else:
        ax.text(0.5, 0.5, "render missing\n(render_modules.py)", ha="center", va="center", color=S.ORANGE_TX)
    if ep is not None:
        ax.text(0.0, 0.02, f"v2: surge, sway, yaw ±\n{int(ep.setsize_v2_sensor)} of {int(ep.n_hyp):,} fault\n"
                "hypotheses remain", transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=S.INK2)
    return ax


# ------------------------------------------------------------------------------------------------- panel (b)
def icon(ax, cells, x0, y0, c, foreign=(), fault_mod=None, fault_xy=None, mark=None):
    """cells: [(i, j)] grid cells (i forward, j left); drawn with forward up, left left. c = cell size (in)."""
    ii = [p[0] for p in cells]
    jj = [p[1] for p in cells]
    for k, (i, j) in enumerate(cells):
        X = x0 + (max(jj) - j) * c
        Y = y0 + (i - min(ii)) * c
        fc = S.TINT if k in foreign else (FAULT_TINT if k == fault_mod else "#dcd9d2")
        ec = S.GREY if k in foreign else S.INK2
        ax.add_patch(Rectangle((X + 0.06 * c, Y + 0.06 * c), 0.88 * c, 0.88 * c, lw=0.45, ec=ec, fc=fc,
                               ls=(0, (1.5, 1)) if k in foreign else "-"))
        if k == fault_mod and fault_xy is not None:
            ax.plot(X + c * (0.5 - 0.62 * fault_xy[1]), Y + c * (0.5 + 0.62 * fault_xy[0]), "o", ms=2.6,
                    mfc=S.ORANGE, mec="none")
    w, h = (max(jj) - min(jj) + 1) * c, (max(ii) - min(ii) + 1) * c
    if mark:
        ax.text(x0 + w + 0.015, y0 + h / 2, mark, ha="left", va="center", fontsize=8, color=S.BLUE,
                fontweight="bold")
    return w, h


def _flow(ax, items, y, Wd, c, gap=0.055):
    """Place icons left to right, wrapping rows; returns the y below the last row."""
    x, rowh = 0.0, 0.0
    for cc, kw, extra in items:
        ii, jj = [p[0] for p in cc], [p[1] for p in cc]
        w, h = (max(jj) - min(jj) + 1) * c, (max(ii) - min(ii) + 1) * c
        if x + w + extra > Wd + 1e-9:
            x, y = 0.0, y - rowh - gap
            rowh = 0.0
        rowh = max(rowh, h)
        icon(ax, cc, x, y - h, c, **kw)
        x += w + extra + gap
    return y - rowh


def panel_candidates(fig, E, amb, box):
    ax = fig.add_axes(box)
    Wd, Hd = box[2] * W, box[3] * H
    ax.set_xlim(0, Wd), ax.set_ylim(0, Hd)
    ax.axis("off")
    n_m, n_c = len(E["merges"]), len(E["children"])
    P = E["parent"]
    cells = [(m[1], m[2]) for m in P["modules"]]
    f = E["faults"][0]
    fm = f["module"]
    rel = (np.array(P["wheel_pos"][f["wheel"]]) - np.array(P["module_pos"][fm])) / E["pitch_m"]
    undecided = set(amb.cand) if amb is not None else set()
    c = 0.074
    kids = sorted(E["children"], key=lambda k: (len(k["modules"]), k["modules"]))
    y = Hd - 0.02
    ax.text(0, y, f"{n_c} children (splits)", ha="left", va="top", fontsize=7, color=S.INK2)
    items = []
    for k in kids:
        mark = "?" if k["name"] in undecided else None
        items.append(([cells[m] for m in k["modules"]],
                      dict(fault_mod=k["modules"].index(fm) if fm in k["modules"] else None, fault_xy=rel,
                           mark=mark), 0.07 if mark else 0.0))
    y = _flow(ax, items, y - 0.13, Wd, c)
    y -= 0.07
    ax.text(0, y, f"{n_m} merges with another team (dashed grey: its modules)", ha="left", va="top",
            fontsize=7, color=S.INK2)
    items = []
    for m in E["merges"]:
        cc = [tuple(p) for p in m["cells"]]
        npc = len(m["piece"])
        items.append((cc, dict(foreign=set(range(npc, len(cc))),
                               fault_mod=m["piece"].index(fm) if fm in m["piece"] else None, fault_xy=rel), 0.0))
    y = _flow(ax, items, y - 0.13, Wd, c)
    n_und = len(undecided)
    yk = 0.165
    ax.plot(0.03, yk, "o", ms=2.6, mfc=S.ORANGE, mec="none")
    ax.text(0.08, yk, f"{f['mode']}ing wheel", ha="left", va="center", fontsize=7, color=S.ORANGE_TX)
    ax.text(0.80, yk, "?", ha="center", va="center", fontsize=8, color=S.BLUE, fontweight="bold")
    ax.text(0.86, yk, f"{n_und} body with legs that need a probe (c)", ha="left", va="center", fontsize=7,
            color=S.INK2)
    ax.text(0.0, 0.06, f"the other {n_c + n_m - n_und} are decided at every leg and \u03c4 by the probe",
            ha="left", va="center", fontsize=7, color=S.INK2)
    return ax


# ------------------------------------------------------------------------------------------------- panel (c)
GLYPH = {"commit": (r"$\checkmark$", S.BLUE), "refuse": ("\u2717", S.GREY), "probe": ("?", S.BLUE)}


def panel_bounds(fig, E, amb, box):
    ax = fig.add_axes(box)
    if amb is None or amb.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "awaiting results/" + AMB, ha="center", va="center", color=S.ORANGE_TX)
        return ax, None
    d = amb[amb.rho_model == RHO_MODEL]
    cand = d.cand.iloc[0]
    g = (d[d.cand == cand].drop_duplicates("leg").sort_values("leg")
         .assign(leg_name=lambda t: t.leg.map(lambda i: LEGN[int(i)])))
    rows = []   # legs with identical sets (mirror pairs) share a row
    for (lo, hi, ext, rt), t in g.groupby(["lo", "hi", "undominated_v2", "rho_t"], sort=False):
        rows.append(dict(name=", ".join(t.leg_name), lo=lo, hi=hi, ext=bool(ext), truth=rt))
    rows = sorted(rows, key=lambda r: r["ext"])
    y = np.arange(len(rows))[::-1] * 1.0
    for r, yy in zip(rows, y):
        ax.add_patch(Rectangle((r["lo"], yy - 0.17), r["hi"] - r["lo"], 0.34, lw=0.6, ec=S.INK,
                               fc=S.SURFACE if r["ext"] else S.INK2, hatch="//////" if r["ext"] else None))
        ax.plot([r["truth"]] * 2, [yy - 0.28, yy + 0.28], color=S.ORANGE if False else S.INK, lw=1.2)
        ax.text(0.49, yy + 0.08, r["name"], ha="right", va="center", fontsize=7, color=S.INK)
        ax.text(0.49, yy - 0.24, "extrapolated" if r["ext"] else "observed (in v2)", ha="right", va="center",
                fontsize=7, color=S.INK2, style="italic")
        for tau in C.TAUS:
            ax.plot([tau, tau], [yy - 0.30, yy + 0.30], color=S.GREY, lw=0.5, ls=(0, (2, 1.5)), zorder=0)
            gl, col = GLYPH[verdict(r["lo"], r["hi"], tau)]
            ax.text(tau, yy + 0.36, gl, ha="center", va="bottom", fontsize=7.5, color=col, fontweight="bold",
                    fontfamily="DejaVu Sans" if gl == "\u2717" else None)
    r = rows[-1]
    ax.annotate("true", (r["truth"], y[-1] - 0.28), xytext=(r["truth"] + 0.035, y[-1] - 0.62), fontsize=7,
                color=S.INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.INK2, shrinkA=0, shrinkB=0))
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.75, y.max() + 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    ax.set_xlabel(r"capability $\rho^{\mathrm{dec}}$ (dashed: $\tau$)", labelpad=1.0)
    return ax, dict(cand=cand, rows=rows)


def title(fig, x_in, text):
    fig.text(x_in / W, 1 - 0.02 / H, text, ha="left", va="top", fontsize=S.TITLE_PT, fontweight="bold",
             color=S.INK)


def draw(res, out):
    S.apply()
    E, ep, amb, srcs = load(res)
    fig = S.plt.figure(figsize=(W, H))
    n = len(E["children"]) + len(E["merges"])
    XA, XB, XC = 0.0, 2.08, 4.66
    title(fig, XA, "(a) One 12-s probe of today's body\u2026")
    title(fig, XB, f"(b) \u2026bounds all {n} bodies it could become")
    title(fig, XC, "(c) Undecided legs trust the motor law")
    panel_render(fig, E, ep, (XA / W, 0.0, 2.0 / W, 1 - 0.16 / H))
    panel_candidates(fig, E, amb, (XB / W, 0.0, 2.45 / W, 1 - 0.17 / H))
    _, info = panel_bounds(fig, E, amb, (5.60 / W, 0.34 / H, 1.10 / W, 0.98 / H))
    if info:
        fig.text(XC / W, 1 - 0.19 / H, "lone faulted module (? in b); identified set\nafter v2, motor spread unknown", ha="left", va="top", fontsize=7, color=S.INK2, linespacing=1.2)
        yk = 1 - 0.47 / H
        for x_in, t, col in ((XC, GLYPH["commit"][0], S.BLUE), (XC + 0.10, "commit", S.INK2),
                             (XC + 0.52, "?", S.BLUE), (XC + 0.60, "probe, at each task \u03c4", S.INK2)):
            fig.text(x_in / W, yk, t, ha="left", va="center", fontsize=7.5 if col == S.BLUE else 7, color=col,
                     fontweight="bold" if col == S.BLUE else "normal")
    S.save(fig, out)
    return E, ep, amb, srcs, info


def run(res, paper):
    N = C.Numbers("fig_teaser.py")
    out = paper / "figs/teaser.pdf"
    E, ep, amb, srcs, info = draw(res, out)
    st = "exploratory"
    if ep is not None:
        N.num("teaser.bodies", int(ep.n_cands), srcs, st)
        N.num("teaser.children", int(ep.n_children), srcs, st)
        N.num("teaser.hyp", int(ep.n_hyp), srcs, st)
        N.num("teaser.hypleft", int(ep.setsize_v2_sensor), srcs, st)
    else:
        for k in ("teaser.bodies", "teaser.children", "teaser.hyp", "teaser.hypleft"):
            N.ph(k)
    if amb is not None and not amb.empty:
        N.num("teaser.undecided", amb.cand.nunique(), srcs, st)
    else:
        N.ph("teaser.undecided")
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
