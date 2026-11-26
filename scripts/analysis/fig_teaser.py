"""Fig. 1, the teaser: one probe of today's body gives a verdict for every body it could become, merges included;
each verdict is the certified lower bound against the job's need, leg by leg.

One real confirmatory C-MAIN episode (Ours, T1, conf_test) with a faulted wheel and merges, chosen by the fixed rule
in teaser_episode.py (the rule and the pool size are stored in the JSON); it is labelled as one episode.
Inputs:
  scripts/analysis/assets/teaser_episode.json   episode id, job, shown bodies, geometry (teaser_episode.py)
  scripts/analysis/assets/teaser_render.png     Blender render of the parent from catalog geometry (render_modules.py)
  results/models/confirmatory/dec_conf_models.csv.gz   lower / point / truth of every (body, leg) of the episode
  results/annotations/task_library.csv          the job's need tau per leg (r_<leg>)
  results/theory/dominance_by_leg_size.csv      hatching: a leg is extrapolated for a body size if most such legs are
                                                undominated by probe v2 (share_legs_undominated >= 0.5)
Verdict per (body, leg): certify if lower >= tau, probe if point >= tau > lower, else refuse; a body certifies the job
if every needed leg certifies, refuses if any leg refuses, else probes first (DIAGRAM_STRATEGY.md, Fig. 1).
Outputs: paper/figs/teaser.pdf (6.75 x 2.0 in, include 1:1), macros cfTeaser*.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

import common as C
import figstyle as S

ASSETS = pathlib.Path(__file__).resolve().parent / "assets"
EPISODE, RENDER = ASSETS / "teaser_episode.json", ASSETS / "teaser_render.png"
DEC = "models/confirmatory/dec_conf_models.csv.gz"
TASKS = "annotations/task_library.csv"
DOM = "theory/dominance_by_leg_size.csv"
W, H = S.W2, 2.0
CERT_FILL = "#d6e5f7"      # light blue: body certified for the job
REF_FILL = "#c9c7c2"       # grey: body refused
LEG_ORDER = ("surge", "diag", "yaw")      # Fig. 2 / Fig. 5 order
STATUS = "confirmatory run, one episode (descriptive)"
MODE_WORD = {"coast": "coasting", "weak": "weak", "brake": "braking"}


def pretty_leg(l):
    return l.replace("-", "−")


def leg_verdict(lower, point, tau):
    if np.isfinite(lower) and lower >= tau:
        return "certify"
    return "probe" if point >= tau else "refuse"


def body_verdict(vs):
    return "refuse" if "refuse" in vs else ("certify" if all(v == "certify" for v in vs) else "probe")


def load(res):
    E = json.loads(EPISODE.read_text())
    srcs = []
    f = res / DEC
    rows = None
    if f.exists():
        d = pd.read_csv(f, usecols=["claim", "method", "split", "test_plant", "eid", "cand", "kind", "cand_size",
                                    "has_fault", "leg", "truth", "point", "lower"])
        rows = d[(d.claim == "C-MAIN") & (d.split == "conf_test") & (d.method == "amortized_comp")
                 & (d.test_plant == "T1") & (d.eid == E["eid"])].copy()
        srcs.append(C.rel(f))
    if rows is None or not (res / TASKS).exists() or not (res / DOM).exists():
        return E, None, srcs
    t = pd.read_csv(res / TASKS).set_index("task_id").loc[E["job"]["task_id"]]
    taus = {i: float(t[f"r_{l}"]) for i, l in enumerate(C.LEGS) if pd.notna(t.get(f"r_{l}"))}
    assert {C.LEGS[i]: v for i, v in taus.items()} == E["job"]["taus"], "task library changed since selection"
    srcs.append(C.rel(res / TASKS))
    dom = pd.read_csv(res / DOM)
    dom = dom[dom.probe_set == "v2"]
    ext = {(r.leg, str(r.child_size)): r.share_legs_undominated >= 0.5 for r in dom.itertuples()}
    srcs.append(C.rel(res / DOM))
    V = None
    if rows is not None and len(rows):
        x = rows[rows.leg.isin(list(taus))].copy()
        x["tau"] = x.leg.map(taus)
        x["legn"] = x.leg.map(lambda i: C.LEGS[i])
        x["verdict"] = [leg_verdict(lo, pt, ta) for lo, pt, ta in zip(x.lower, x.point, x.tau)]
        x["unsafe"] = (x.verdict == "certify") & (x.truth < x.tau)
        x["ext"] = [ext[(l, "6+" if s >= 6 else str(s))] for l, s in zip(x.legn, x.cand_size)]
        bv = x.groupby("cand").verdict.agg(lambda s: body_verdict(list(s)))
        V = dict(legs=x, body=bv)
        assert len(bv) == len(E["cands"])
    return E, V, srcs


# ------------------------------------------------------------------------------------------------- panel (a)
def panel_render(fig, ax, E, box):
    """box in inches (x0, y0, w, h). The render is cropped to its alpha footprint and fitted into the box."""
    x0, y0, bw, bh = box
    if not RENDER.exists():
        ax.text(x0 + bw / 2, y0 + bh / 2, "render missing\n(render_modules.py)", ha="center", va="center",
                color=S.ORANGE_TX)
        return
    img = mpimg.imread(RENDER)
    a = img[..., 3] > 0.15                                 # shadow catcher: ignore the faint halo
    r0, r1 = np.flatnonzero(a.any(1))[[0, -1]]
    c0, c1 = np.flatnonzero(a.any(0))[[0, -1]]
    img = img[r0:r1 + 1, c0:c1 + 1]
    h_px, w_px = img.shape[:2]
    s = min(bw / w_px, bh / h_px)
    iw, ih = w_px * s, h_px * s
    ix, iy = x0 + (bw - iw) / 2, y0 + (bh - ih)          # top-aligned, centred
    iax = fig.add_axes((ix / W, iy / H, iw / W, ih / H))
    iax.imshow(img, interpolation="none")       # "none": the PDF embeds the render at its native resolution
    iax.axis("off")
    pj = json.loads(RENDER.with_name(RENDER.stem + "_proj.json").read_text())
    res_x, res_y = pj["res"]

    def to_in(key):
        u, v = pj["proj"][key]
        px, py = u * res_x - c0, (1 - v) * res_y - r0                 # crop pixels
        return np.array([ix + px * s, iy + ih - py * s])              # inches

    return (ix, iy, iw, ih), to_in


# ------------------------------------------------------------------------------------------------- panel (b)
def icon(ax, cells, x0, y0, c, foreign=(), fault=(), fill=CERT_FILL):
    """cells [(i, j)] (i forward = up, j left = left); fault: [(cell index, (rel_x, rel_y))]. Returns (w, h)."""
    ii = [p[0] for p in cells]
    jj = [p[1] for p in cells]
    for k, (i, j) in enumerate(cells):
        X = x0 + (max(jj) - j) * c
        Y = y0 + (i - min(ii)) * c
        fo = k in foreign
        ax.add_patch(Rectangle((X + 0.1 * c, Y + 0.1 * c), 0.8 * c, 0.8 * c, lw=0.5,
                               ec=S.GREY if fo else S.INK2, fc=fill, ls=(0, (1.6, 1.0)) if fo else "-", zorder=3))
    for k, rel in fault:
        i, j = cells[k]
        X = x0 + (max(jj) - j) * c
        Y = y0 + (i - min(ii)) * c
        ax.plot(X + c * (0.5 - 0.75 * rel[1]), Y + c * (0.5 + 0.75 * rel[0]), "o", ms=2.4, mfc=S.ORANGE, mec="none",
                zorder=4)
    return (max(jj) - min(jj) + 1) * c, (max(ii) - min(ii) + 1) * c


def body_icon(ax, E, cand, x, y, c, verdict):
    """Draw candidate `cand` with its bottom-left at (x, y). Returns (w, h)."""
    cd = E["cands"][cand]
    pf = {f["module"]: f["rel"] for f in E["faults"]}
    of = {(o["role"], f["module"]): f["rel"] for o in E["others"] for f in o["faults"]}
    fault, foreign = [], set()
    for k, (r, m) in enumerate(cd["modules"]):
        if r != 0:
            foreign.add(k)
        rel = pf.get(m) if r == 0 else of.get((r, m))
        if rel is not None:
            fault.append((k, rel))
    fill = {"certify": CERT_FILL, "refuse": REF_FILL, "probe": S.SURFACE}[verdict]
    return icon(ax, [tuple(p) for p in cd["cells"]], x, y, c, foreign, fault, fill)


def mark(ax, x, y, verdict, size=8):
    """Verdict mark with its left edge at x, centred on y (inches): drawn tick and cross, STIX bold '?'."""
    e = size / 72 * 0.62                                     # mark height ~ cap height of `size` pt type
    if verdict == "probe":
        ax.text(x, y, "?", ha="left", va="center", fontsize=size + 1, color=S.BLUE, fontweight="bold", zorder=5)
    elif verdict == "refuse":
        for a, b in (((x, y - e / 2), (x + 0.85 * e, y + e / 2)), ((x, y + e / 2), (x + 0.85 * e, y - e / 2))):
            ax.plot(*zip(a, b), color=S.INK2, lw=1.1, solid_capstyle="round", zorder=5)
    else:
        ax.plot([x, x + 0.33 * e, x + e], [y, y - 0.45 * e, y + 0.55 * e], color=S.BLUE, lw=1.2,
                solid_capstyle="round", solid_joinstyle="round", zorder=5)


def dims(E, cand, c):
    cc = E["cands"][cand]["cells"]
    ii, jj = [p[0] for p in cc], [p[1] for p in cc]
    return (max(jj) - min(jj) + 1) * c, (max(ii) - min(ii) + 1) * c


def flow(ax, E, V, cands, x0, y_top, width, c, gap=0.055, row_gap=0.06, mark_w=0.10, tags=None):
    """Lay icons left to right (bottom-aligned per row), wrapping at `width`. Returns the y of the last row bottom."""
    rows, cur, cw = [], [], 0.0
    for k in cands:
        w, h = dims(E, k, c)
        extra = (mark_w if V["body"][k] != "certify" else 0.0) + (0.12 if tags and k in tags else 0.0)
        if cur and cw + w + extra > width + 1e-9:
            rows.append(cur)
            cur, cw = [], 0.0
        cur.append((k, w, h, extra))
        cw += w + extra + gap
    rows.append(cur)
    y = y_top
    for r in rows:
        rh = max(h for _, _, h, _ in r)
        x = x0
        for k, w, h, extra in r:
            body_icon(ax, E, k, x, y - rh, c, V["body"][k])
            if V["body"][k] != "certify":
                mark(ax, x + w + 0.015, y - rh + h / 2, V["body"][k])
            if tags and k in tags:
                ax.text(x + w + mark_w, y - rh + h / 2, tags[k], ha="left", va="center", fontsize=S.LABEL_PT,
                        color=S.INK, fontweight="bold")
            x += w + extra + gap
        y -= rh + row_gap
    return y + row_gap


def panel_bodies(ax, E, V, x0, y_top, width):
    c = 0.078
    kinds = {k: E["cands"][k]["kind"] for k in range(len(E["cands"]))}
    size = {k: len(E["cands"][k]["modules"]) for k in kinds}
    kids = [k for k in kinds if kinds[k] == "child"]
    merges = [k for k in kinds if kinds[k] == "merge"]
    selfc = [k for k in kinds if kinds[k] == "self"]
    tags = {E["shown_bodies"]["refuse"]: "c1", E["shown_bodies"]["probe"]: "c2"}
    T = dict(ha="left", va="top", fontsize=S.LABEL_PT, color=S.INK2)
    ax.text(x0, y_top, f"today's body + {len(kids)} children (splits) + {len(merges)} merges = {len(kinds)}", **T)
    # children, wrapped by size: 1+2 | 3 | 4 | 5
    left_w = 2.05
    y = y_top - 0.17
    for grp in ((1, 2), (3,), (4,), (5,)):
        ks = sorted([k for k in kids if size[k] in grp], key=lambda k: (size[k], k))
        y = flow(ax, E, V, ks, x0, y, left_w, c, tags=tags) - 0.075
    y_kids_bottom = y + 0.075
    # right column: today's body, then the merges
    xr = x0 + left_w + 0.07
    wr = x0 + width - xr
    yb = flow(ax, E, V, selfc, xr, y_top - 0.17, wr, c)
    ax.text(xr + 2 * c + 0.04, yb + 1.5 * c, "today's\nbody", ha="left", va="center", fontsize=S.LABEL_PT,
            color=S.INK2, linespacing=1.05)
    ax.text(xr, yb - 0.07, "merges;\ndashed: the\nother team", ha="left", va="top", fontsize=S.LABEL_PT,
            color=S.INK2, linespacing=1.05)
    ym = flow(ax, E, V, sorted(merges), xr, yb - 0.07 - 0.36, wr, c)
    return min(y_kids_bottom, ym)


def key_line(ax, E, V, x0, y):
    bv = V["body"]
    n = {v: int((bv == v).sum()) for v in ("certify", "probe", "refuse")}
    c = 0.068
    x = x0
    ax.add_patch(Rectangle((x, y - c * 0.43), c * 0.86, c * 0.86, lw=0.5, ec=S.INK2, fc=CERT_FILL))
    x += c + 0.03
    ax.text(x, y, f"certified ({n['certify']})", ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK2)
    x += 0.60
    mark(ax, x, y, "probe")
    ax.text(x + 0.08, y, f"probe first ({n['probe']})", ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK2)
    x += 0.80
    mark(ax, x, y, "refuse")
    ax.text(x + 0.10, y, f"refuse ({n['refuse']})", ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK2)
    x += 0.62
    ax.plot(x + 0.03, y, "o", ms=2.4, mfc=S.ORANGE, mec="none")
    ax.text(x + 0.08, y, f"{MODE_WORD[E['faults'][0]['mode']]} wheel", ha="left", va="center", fontsize=S.LABEL_PT,
            color=S.ORANGE_TX)
    return n


# ------------------------------------------------------------------------------------------------- panel (c)
def panel_legs(ax, E, V, x0, x1, y_top, y_axis):
    """Two bodies, one block each: rows = the job's legs; box from the certified bound (left edge) to the estimate;
    tau as an ink tick; verdict glyph at the right."""
    legs = V["legs"]
    taus = E["job"]["taus"]
    order = sorted(taus, key=lambda l: (LEG_ORDER.index(l.rstrip("+-")), l))
    lab_w = 0.42                    # leg labels
    gx = x1 - 0.13                  # verdict glyph column
    ax0, ax1 = x0 + lab_w, gx - 0.07
    lo_v, hi_v = 0.2, 1.2
    X = lambda v: ax0 + (v - lo_v) / (hi_v - lo_v) * (ax1 - ax0)
    rh = 0.125
    c = 0.056
    y = y_top
    blocks = (("refuse", "c1", "the faulted module alone"), ("probe", "c2", "with its healthy neighbour"))
    first = True
    for verdict, tag, words in blocks:
        cand = E["shown_bodies"][verdict]
        w, h = dims(E, cand, c)
        body_icon(ax, E, cand, x0, y - h, c, verdict)
        tx = x0 + w + 0.05
        ax.text(tx, y - h / 2, tag, ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK, fontweight="bold")
        ax.text(tx + 0.13, y - h / 2, words, ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK)
        mark(ax, gx, y - h / 2, verdict)
        y -= h + (0.10 if first else 0.02)
        g = legs[legs.cand == cand].set_index("legn")
        for leg in order:
            r = g.loc[leg]
            yc = y - rh / 2
            ax.text(ax0 - 0.05, yc, pretty_leg(leg), ha="right", va="center", fontsize=S.LABEL_PT, color=S.INK2)
            lo = max(float(r.lower), lo_v)
            ax.add_patch(Rectangle((X(lo), yc - 0.045), X(float(r.point)) - X(lo), 0.09, lw=0.6, ec=S.BLUE,
                                   fc=S.SURFACE if r.ext else S.BLUE, hatch="//////" if r.ext else None, zorder=3))
            ax.plot([X(lo)] * 2, [yc - 0.06, yc + 0.06], color=S.BLUE, lw=S.LW_OURS, solid_capstyle="butt", zorder=4)
            ax.plot([X(r.tau)] * 2, [yc - 0.075, yc + 0.075], color=S.INK, lw=1.0, solid_capstyle="butt", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground=S.SURFACE)])
            if first and leg == order[0]:           # direct labels on the first bar: what its two ends are
                ax.text(X(lo), yc + 0.06, "bound", ha="center", va="bottom", fontsize=S.LABEL_PT, color=S.BLUE)
                ax.text(X(float(r.point)), yc + 0.06, "estimate", ha="center", va="bottom", fontsize=S.LABEL_PT,
                        color=S.INK2)
            mark(ax, gx, yc, r.verdict, size=7.5)
            y -= rh
        y -= 0.03
        first = False
    # axis
    ya = y_axis
    ax.plot([ax0, ax1], [ya, ya], color=S.GREY, lw=S.LW_AXES)
    for v in (0.25, 0.5, 0.75, 1.0):
        ax.plot([X(v)] * 2, [ya, ya - 0.03], color=S.INK2, lw=S.LW_AXES)
        ax.text(X(v), ya - 0.04, f"{v:g}", ha="center", va="top", fontsize=S.LABEL_PT, color=S.INK2)
    ax.text((ax0 + ax1) / 2, ya - 0.155, "capability \u03c1 (1 = healthy reference)", ha="center", va="top",
            fontsize=S.LABEL_PT, color=S.INK2)
    return X, y


def draw(res, out):
    S.apply()
    E, V, srcs = load(res)
    fig = S.plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W), ax.set_ylim(0, H)
    ax.axis("off")
    n = len(E["cands"])
    XA, XB, XC = 0.0, 1.86, 4.70
    top = H - 0.02
    tkw = dict(ha="left", va="top", fontsize=S.TITLE_PT, fontweight="bold", color=S.INK)
    secs = E["probe"]["twist_s"]
    ax.text(XA, top, f"(a) One {secs:g}-s probe of today's body…", **tkw)
    ax.text(XB, top, f"(b) …gives all {n} next bodies a verdict", **tkw)
    ax.text(XC, top, "(c) Two bodies against one job", **tkw)
    # (a)
    got = panel_render(fig, ax, E, (XA, 0.40, 1.78, 1.38))
    if got:
        (ix, iy, iw, ih), to_in = got
        f = E["faults"][0]
        wx, wy = to_in(f"wheel{f['wheel']}")
        probe_arrows(ax, E, to_in, ix + 0.80 * iw, iy + 0.80 * ih)
        ax.annotate(f"{MODE_WORD[f['mode']]} wheel\n(unknown to the robot)", (wx, wy), xytext=(XA + 1.78, 0.38),
                    ha="right", va="top", fontsize=S.LABEL_PT, color=S.ORANGE_TX, zorder=6, linespacing=1.1,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=S.ORANGE_TX, shrinkA=1, shrinkB=2.5))
    ax.text(XA, 0.08, "blue: probe v2, surge, sway and yaw \u00b1", ha="left", va="center",
            fontsize=S.LABEL_PT, color=S.BLUE)
    if V is None:
        ax.text(XB + 1.3, 1.0, "awaiting " + DEC, ha="center", va="center", color=S.ORANGE_TX)
        S.save(fig, out)
        return E, V, srcs, None
    # (b)
    yb = panel_bodies(ax, E, V, XB, top - 0.20, XC - XB - 0.12)
    counts = key_line(ax, E, V, XB, max(0.07, min(yb - 0.13, 0.30)))
    # (c)
    job = E["job"]["title"]
    ax.text(XC, top - 0.18, f"one job: {job.lower()}", ha="left", va="top", fontsize=S.LABEL_PT, color=S.INK2,
            style="italic")
    yk = top - 0.385
    for i, (ext, lab) in enumerate(((False, "observed"), (True, "extrapolated"))):
        xk = XC + 0.02 + i * 0.66
        ax.add_patch(Rectangle((xk, yk - 0.04), 0.16, 0.08, lw=0.6, ec=S.BLUE, fc=S.SURFACE if ext else S.BLUE,
                               hatch="//////" if ext else None))
        ax.text(xk + 0.20, yk, lab, ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK2)
    ax.plot([XC + 1.43] * 2, [yk - 0.06, yk + 0.06], color=S.INK, lw=1.0, solid_capstyle="butt")
    ax.text(XC + 1.48, yk, "job's need \u03c4", ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK2)
    panel_legs(ax, E, V, XC, W - 0.01, top - 0.49, 0.37)
    S.save(fig, out)
    return E, V, srcs, counts


def probe_arrows(ax, E, to_in, cx, cy, L=0.11):
    """The v2 twists drawn in the render's ground plane: surge (forward axis), sway (left axis), yaw (arc)."""
    m = {k: to_in(f"module{k}") for k in range(len(E["parent"]["modules"]))}
    cells = E["parent_cells"]
    k0 = cells.index([0, 0])
    fwd = m[cells.index([max(c[0] for c in cells), 0])] - m[k0]
    left = m[cells.index([0, max(c[1] for c in cells)])] - m[k0]
    fwd, left = fwd / np.linalg.norm(fwd), left / np.linalg.norm(left)
    c = np.array([cx, cy])
    kw = dict(arrowstyle="<|-|>,head_length=2.6,head_width=1.5", mutation_scale=1, lw=0.9, color=S.BLUE,
              shrinkA=0, shrinkB=0, zorder=6)
    for v in (fwd, left):
        ax.add_patch(FancyArrowPatch(tuple(c - L * v), tuple(c + L * v), **kw))
    ph = np.radians(np.linspace(20, 290, 60))
    R = 1.45 * L
    pts = c[None] + R * (np.cos(ph)[:, None] * fwd[None] + np.sin(ph)[:, None] * left[None])
    ax.plot(pts[:-3, 0], pts[:-3, 1], color=S.BLUE, lw=0.9, zorder=6)
    ax.add_patch(FancyArrowPatch(tuple(pts[-4]), tuple(pts[-1]), **{**kw, "arrowstyle":
                                                                   "-|>,head_length=2.6,head_width=1.5"}))


def run(res, paper):
    N = C.Numbers("fig_teaser.py")
    out = paper / "figs/teaser.pdf"
    E, V, srcs, counts = draw(res, out)
    src = srcs + ["scripts/analysis/assets/teaser_episode.json"]
    N.num("teaser.eid", E["eid"], src, STATUS)
    N.num("teaser.bodies", len(E["cands"]), src, STATUS)
    N.num("teaser.children", sum(c["kind"] == "child" for c in E["cands"]), src, STATUS)
    N.num("teaser.merges", sum(c["kind"] == "merge" for c in E["cands"]), src, STATUS)
    N.set("teaser.job", E["job"]["title"].lower(), src, STATUS)
    N.num("teaser.pool", E["pool"]["n_six_faulted_merge"], src, STATUS)
    if counts:
        N.num("teaser.certified", counts["certify"], src, STATUS)
        N.num("teaser.probe", counts["probe"], src, STATUS)
        N.num("teaser.refuse", counts["refuse"], src, STATUS)
        N.num("teaser.unsafe", int(V["legs"].unsafe.sum()), src, STATUS)
    else:
        for k in ("teaser.certified", "teaser.probe", "teaser.refuse", "teaser.unsafe"):
            N.ph(k)
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
