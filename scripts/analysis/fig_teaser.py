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
W, H = S.W2, 2.68          # two-line panel titles (owner request, plain-language headings)
CERT_FILL = "#d6e5f7"      # light blue: body certified for the job
REF_FILL = "#a9a7a2"       # grey: body refused (darker than the blue tint in grayscale)
LEG_ORDER = ("surge", "diag", "yaw")      # Fig. 2 / Fig. 5 order
STATUS = "confirmatory run, one episode (descriptive)"
MODE_WORD = {"coast": "coasting", "weak": "weak", "brake": "braking"}


PLAIN_LEG = {"surge": "forward", "sway": "sideways", "diag": "diagonal", "yaw": "turn"}
PLAIN_LEG_SIGNED = {"surge+": "forward", "surge-": "backward"}
NUMBER_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


VERDICT_WORD = {"certify": "certified", "probe": "test first", "refuse": "refused"}
RULE = ("certified only if every direction the job\nneeds is certified; test first if any direction\n"
        "is uncertain and none fails; else refused")


def plain_leg(l):
    return PLAIN_LEG_SIGNED.get(l, PLAIN_LEG[l.rstrip("+-")])


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
    iax.set_zorder(0)
    ax.set_zorder(1)                          # leaders and drive arrows (on `ax`) draw over the render
    ax.patch.set_visible(False)
    pj = json.loads(RENDER.with_name(RENDER.stem + "_proj.json").read_text())
    res_x, res_y = pj["res"]

    def to_in(key):
        u, v = pj["proj"][key]
        px, py = u * res_x - c0, (1 - v) * res_y - r0                 # crop pixels
        return np.array([ix + px * s, iy + ih - py * s])              # inches

    return (ix, iy, iw, ih), to_in


# ------------------------------------------------------------------------------------------------- panel (b)
PROBE_HATCH = "........"      # "test first" bodies: dotted, so they differ from "certified" in grayscale


def icon(ax, cells, x0, y0, c, foreign=(), fault=(), fill=CERT_FILL, hatch=None):
    """cells [(i, j)] (i forward = up, j left = left); fault: [(cell index, (rel_x, rel_y))]. Returns (w, h)."""
    ii = [p[0] for p in cells]
    jj = [p[1] for p in cells]
    for k, (i, j) in enumerate(cells):
        X = x0 + (max(jj) - j) * c
        Y = y0 + (i - min(ii)) * c
        fo = k in foreign
        ax.add_patch(Rectangle((X + 0.1 * c, Y + 0.1 * c), 0.8 * c, 0.8 * c, lw=0.5,
                               ec=S.GREY if fo else S.INK2, fc=fill, ls=(0, (1.6, 1.0)) if fo else "-", zorder=3,
                               hatch=hatch))
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
    return icon(ax, [tuple(p) for p in cd["cells"]], x, y, c, foreign, fault, fill,
                PROBE_HATCH if verdict == "probe" else None)


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


def grid_row(ax, E, V, cands, x0, y_top, slot_w, c, tags, glyph_h=0.10):
    """One row of a regular grid: each body centred in a slot of width slot_w, bottoms aligned; the verdict mark
    (and the c1 / c2 tag) sits above its body. Returns the y of the row bottom."""
    hs = [dims(E, k, c)[1] for k in cands]
    rh = max(hs)
    yb = y_top - glyph_h - rh
    for n, k in enumerate(cands):
        w, h = dims(E, k, c)
        x = x0 + n * slot_w + (slot_w - w) / 2
        body_icon(ax, E, k, x, yb, c, V["body"][k])
        v = V["body"][k]
        tag = tags.get(k)
        if v != "certify" or tag:
            xm = x + w / 2 - (0.045 if tag else 0.03)
            ym = yb + h + 0.045
            if v != "certify":
                mark(ax, xm, ym, v, size=7.5)
            if tag:
                ax.text(xm + 0.075, ym, tag, ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK,
                        fontweight="bold")
    return yb


def panel_bodies(ax, E, V, x0, y_top, width):
    """Two labelled groups: children (rows by module count, each row labelled with its count) and, in a right
    column, today's configuration and the merges (other team's modules dashed). Returns the lowest y used."""
    kinds = {k: E["cands"][k]["kind"] for k in range(len(E["cands"]))}
    size = {k: len(E["cands"][k]["modules"]) for k in kinds}
    kids = [k for k in kinds if kinds[k] == "child"]
    merges = sorted(k for k in kinds if kinds[k] == "merge")
    selfc = [k for k in kinds if kinds[k] == "self"]
    tags = {E["shown_bodies"]["refuse"]: "c1", E["shown_bodies"]["probe"]: "c2"}
    sizes = sorted({size[k] for k in kids})
    per_row = max(sum(size[k] == z for k in kids) for z in sizes)
    right_w = 0.66
    lab_w = 0.47
    left_w = width - right_w - 0.08
    slot = (left_w - lab_w) / per_row
    c = (slot - 0.035) / max(dims(E, k, 1.0)[0] for k in kids)
    H1 = dict(ha="left", va="top", fontsize=S.LABEL_PT, color=S.INK, fontweight="bold")
    ax.text(x0, y_top, f"Children (after a split): {len(kids)}", **H1)
    y = y_top - 0.08
    for z in sizes:
        ks = sorted(k for k in kids if size[k] == z)
        yb = grid_row(ax, E, V, ks, x0 + lab_w, y, slot, c, tags, glyph_h=0.205)
        rh = max(dims(E, k, c)[1] for k in ks)
        ax.text(x0, yb + rh / 2, f"{z} module{'s' if z > 1 else ''}\n({len(ks)})", ha="left", va="center",
                fontsize=S.LABEL_PT, color=S.INK2, linespacing=1.0)
        y = yb - 0.005
    y_kids = y
    # right column: today's configuration, then the merges (two per row)
    xr = x0 + width - right_w
    ax.text(xr, y_top, "Today's\nconfiguration", **{**H1, "linespacing": 1.0})
    w0 = dims(E, selfc[0], c)[0]
    yb = grid_row(ax, E, V, selfc, xr, y_top - 0.24, w0, c, tags, glyph_h=0.0)
    ax.text(xr, yb - 0.08, f"Merges (with the\nother team): {len(merges)}", **{**H1, "linespacing": 1.0})
    ax.text(xr, yb - 0.335, "dashed = theirs", ha="left", va="top", fontsize=S.LABEL_PT, color=S.INK2)
    y = yb - 0.40
    mw = right_w / 2
    for r in range(0, len(merges), 2):
        y = grid_row(ax, E, V, merges[r:r + 2], xr, y, mw, c, tags, glyph_h=0.06)
    return min(y_kids, y), c


def key_line(ax, E, V, x0, y):
    bv = V["body"]
    n = {v: int((bv == v).sum()) for v in ("certify", "probe", "refuse")}
    c = 0.075
    x = x0
    for v, word, adv in (("certify", "certified", 0.55), ("probe", "test first", 0.56), ("refuse", "refused", 0.48)):
        fill = {"certify": CERT_FILL, "refuse": REF_FILL, "probe": S.SURFACE}[v]
        ax.add_patch(Rectangle((x, y - c / 2), c, c, lw=0.5, ec=S.INK2, fc=fill,
                               hatch=PROBE_HATCH if v == "probe" else None))
        if v != "certify":
            mark(ax, x + c + 0.035, y, v, size=7.5)
        ax.text(x + c + (0.125 if v != "certify" else 0.035), y, word, ha="left", va="center",
                fontsize=S.LABEL_PT, color=S.INK2)
        x += adv + (0.08 if v != "certify" else 0)
    ax.plot(x + 0.03, y, "o", ms=2.4, mfc=S.ORANGE, mec="none")
    ax.text(x + 0.08, y, f"{MODE_WORD[E['faults'][0]['mode']]} wheel", ha="left", va="center",
            fontsize=S.LABEL_PT, color=S.ORANGE_TX)
    return n


# ------------------------------------------------------------------------------------------------- panel (c)
def panel_legs(ax, E, V, x0, x1, y_top, y_axis):
    """Two bodies, one block each: rows = the job's legs; box from the certified bound (left edge) to the estimate;
    tau as an ink tick; verdict glyph at the right."""
    legs = V["legs"]
    taus = E["job"]["taus"]
    order = sorted(taus, key=lambda l: (LEG_ORDER.index(l.rstrip("+-")), l))
    lab_w = 0.50                    # leg labels
    gx = x1 - 0.13                  # verdict glyph column
    ax0, ax1 = x0 + lab_w, gx - 0.07
    lo_v, hi_v = 0.2, 1.2
    X = lambda v: ax0 + (v - lo_v) / (hi_v - lo_v) * (ax1 - ax0)
    rh = 0.098
    c = 0.056
    y = y_top
    blocks = (("refuse", "c1:", "the faulty module alone"), ("probe", "c2:", "with its healthy neighbour"))
    first = True
    for verdict, tag, words in blocks:
        cand = E["shown_bodies"][verdict]
        w, h = dims(E, cand, c)
        body_icon(ax, E, cand, x0, y - h, c, verdict)
        tx = x0 + w + 0.05
        ax.text(tx, y - h / 2, tag, ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK, fontweight="bold")
        ax.text(tx + 0.15, y - h / 2, words, ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK)
        y -= h + 0.03
        g = legs[legs.cand == cand].set_index("legn")
        for leg in order:
            r = g.loc[leg]
            yc = y - rh / 2
            ax.text(ax0 - 0.05, yc, plain_leg(leg), ha="right", va="center", fontsize=S.LABEL_PT, color=S.INK2)
            lo = max(float(r.lower), lo_v)
            ax.add_patch(Rectangle((X(lo), yc - 0.045), X(float(r.point)) - X(lo), 0.09, lw=0.6, ec=S.BLUE,
                                   fc=S.SURFACE if r.ext else S.BLUE, hatch="//////" if r.ext else None, zorder=3))
            ax.plot([X(lo)] * 2, [yc - 0.06, yc + 0.06], color=S.BLUE, lw=S.LW_OURS, solid_capstyle="butt", zorder=4)
            ax.plot([X(r.tau)] * 2, [yc - 0.075, yc + 0.075], color=S.INK, lw=1.0, solid_capstyle="butt", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground=S.SURFACE)])
            ax.plot([X(float(r.point))] * 2, [yc - 0.055, yc + 0.055], color=S.INK2, lw=0.9, zorder=4)  # estimate
            mark(ax, gx, yc, r.verdict, size=7.5)
            y -= rh
        # combined verdict of the configuration for this job (body_verdict: the rule printed under the axis)
        ax.plot([x0 + 0.05, x1], [y - 0.005] * 2, color=S.GREY, lw=0.4)
        yv = y - 0.068
        ax.text(gx - 0.06, yv, f"verdict: {VERDICT_WORD[verdict]}", ha="right", va="center", fontsize=S.LABEL_PT,
                color=S.INK, fontweight="bold")
        mark(ax, gx, yv, verdict, size=7.5)
        y = yv - 0.085
        first = False
    # axis
    ya = y - 0.035 if y_axis is None else y_axis
    ax.plot([ax0, ax1], [ya, ya], color=S.GREY, lw=S.LW_AXES)
    for v in (0.25, 0.5, 0.75, 1.0):
        ax.plot([X(v)] * 2, [ya, ya - 0.03], color=S.INK2, lw=S.LW_AXES)
        ax.text(X(v), ya - 0.04, f"{v:g}", ha="center", va="top", fontsize=S.LABEL_PT, color=S.INK2)
    ax.text((ax0 + ax1) / 2, ya - 0.15, "capability \u03c1 (1 = healthy reference)", ha="center", va="top",
            fontsize=S.LABEL_PT, color=S.INK2)
    ax.text(x0, ya - 0.265, RULE, ha="left", va="top", fontsize=S.LABEL_PT, color=S.INK2, style="italic",
            linespacing=1.1)
    return X, y


def draw(res, out):
    S.apply()
    E, V, srcs = load(res)
    fig = S.plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W), ax.set_ylim(0, H)
    ax.axis("off")
    n = len(E["cands"])
    XA, XB, XC = 0.0, 1.88, 4.92            # (c) has its own column; gutter of >= 0.12 in before it
    top = H - 0.02
    tkw = dict(ha="left", va="top", fontsize=S.TITLE_PT, fontweight="bold", color=S.INK, linespacing=1.15)
    secs = E["probe"]["twist_s"]
    nm = len(E["parent"]["modules"])
    ax.text(XA, top, f"(a) Today's robot: {NUMBER_WORD.get(nm, nm)} docked\nmodules run one {secs:g}-s test drive",
            **tkw)
    ax.text(XB, top, f"(b) Every configuration it could become ({n}),\neach with a verdict for one job", **tkw)
    ax.text(XC, top, "(c) Inside two verdicts", **tkw)
    ax.text(XC, top - 0.145, "lower bound vs. what the job needs", ha="left", va="top", fontsize=S.LABEL_PT,
            color=S.INK2)
    top2 = top - 0.29                     # below the two-line titles
    # (a)
    got = panel_render(fig, ax, E, (XA + 0.04, 0.78, 1.72, top2 - 0.78))
    if got:
        (ix, iy, iw, ih), to_in = got
        probe_arrows(ax, E, to_in)
        lab = dict(fontsize=S.LABEL_PT, zorder=7, linespacing=1.05)
        lead = dict(arrowstyle="-", lw=0.4, shrinkA=1, shrinkB=2.0)
        mods = E["parent"]["modules"]
        mk = max(range(len(mods)), key=lambda k: to_in(f"module{k}")[0])          # rightmost module
        mx, my = to_in(f"module{mk}")
        ax.annotate("one module", (mx, my), xytext=(XA + 1.80, my + 0.30), ha="right", va="bottom", color=S.INK,
                    arrowprops={**lead, "color": S.INK2, "relpos": (0.5, 0)}, **lab)
        fw = {f["wheel"] for f in E["faults"]}
        wk = min((w for w in range(len(E["parent"]["wheel_pos"])) if w not in fw),
                 key=lambda w: to_in(f"wheel{w}")[1])                              # lowest (front) healthy wheel
        wx, wy = to_in(f"wheel{wk}")
        ax.annotate(f"mecanum wheel:\n{E['rollers']} rollers at {E['roller_angle_deg']:g}\u00b0\nlet it drive\n"
                    "sideways", (wx, wy), xytext=(wx - 0.17, wy - 0.13), ha="left", va="top", color=S.INK,
                    arrowprops={**lead, "color": S.INK2, "relpos": (0, 1)}, **lab)   # ~37 deg leader
        f = E["faults"][0]
        fx, fy = to_in(f"wheel{f['wheel']}")
        ax.annotate(f"{MODE_WORD[f['mode']]} wheel\n(unknown to\nthe robot)", (fx, fy), xytext=(XA, 0.62),
                    ha="left", va="top", color=S.ORANGE_TX,
                    arrowprops={**lead, "color": S.ORANGE_TX, "relpos": (1, 1)}, **lab)
    ax.text(XA, 0.12, "blue arrows: the test drive\n(forward/back, sideways, turn)", ha="left", va="center",
            fontsize=S.LABEL_PT, color=S.BLUE, linespacing=1.1)
    if V is None:
        ax.text(XB + 1.3, 1.0, "awaiting " + DEC, ha="center", va="center", color=S.ORANGE_TX)
        S.save(fig, out)
        return E, V, srcs, None
    # (b)
    bv = V["body"]
    nb = {v: int((bv == v).sum()) for v in ("certify", "probe", "refuse")}
    ax.text(XB, top2 + 0.02, f"{nb['certify']} certified \u00b7 {nb['probe']} test first \u00b7 {nb['refuse']} refused",
            ha="left", va="top", fontsize=S.LABEL_PT, color=S.INK, style="italic")
    yb, _ = panel_bodies(ax, E, V, XB, top2 - 0.16, XC - XB - 0.14)
    counts = key_line(ax, E, V, XB, max(0.07, min(yb - 0.10, 0.25)))
    # (c)
    kinds = [k for k in LEG_ORDER if any(l.startswith(k) for l in E["job"]["taus"])]
    ax.text(XC, top2, f"{E['job']['domain']} job: {', '.join(PLAIN_LEG[k] for k in kinds)}",
            ha="left", va="top", fontsize=S.LABEL_PT, color=S.INK2)
    yk = top2 - 0.19
    for i, (ext, lab) in enumerate(((False, "observed by the test drive"), (True, "extrapolated by the motor model"))):
        yy = yk - i * 0.105
        ax.add_patch(Rectangle((XC + 0.02, yy - 0.04), 0.16, 0.08, lw=0.6, ec=S.BLUE, fc=S.SURFACE if ext else S.BLUE,
                               hatch="//////" if ext else None))
        ax.text(XC + 0.24, yy, lab, ha="left", va="center", fontsize=S.LABEL_PT, color=S.INK2)
    yk2 = yk - 0.21
    ax.plot([XC + 0.10] * 2, [yk2 - 0.04, yk2 + 0.04], color=S.INK, lw=1.0, solid_capstyle="butt")
    ax.text(XC + 0.24, yk2, "what the job needs", ha="left", va="center", fontsize=S.LABEL_PT,
            color=S.INK2)
    yk3 = yk2 - 0.105                         # bar key: thick start = lower bound, end tick = estimate
    bx0, bx1 = XC + 0.02, XC + 0.18
    ax.add_patch(Rectangle((bx0, yk3 - 0.04), bx1 - bx0, 0.08, lw=0.6, ec=S.BLUE, fc=S.BLUE))
    ax.plot([bx0] * 2, [yk3 - 0.055, yk3 + 0.055], color=S.BLUE, lw=S.LW_OURS, solid_capstyle="butt")
    ax.plot([bx1] * 2, [yk3 - 0.055, yk3 + 0.055], color=S.INK2, lw=0.9)
    ax.text(XC + 0.24, yk3, "start: lower bound, end tick: estimate", ha="left", va="center",
            fontsize=S.LABEL_PT, color=S.INK2)
    panel_legs(ax, E, V, XC, W - 0.01, yk3 - 0.09, None)
    S.save(fig, out)
    return E, V, srcs, counts


def probe_arrows(ax, E, to_in):
    """The test drive drawn on the render, in its ground plane: a forward/back and a sideways double arrow crossing
    on the rear half of the body, and a turn arc around its rear (the render's empty top-left corner). Cell
    coordinates (i forward, j left) are mapped to inches through the projected module centres."""
    cells = [tuple(c) for c in E["parent_cells"]]
    m = {cells[k]: to_in(f"module{k}") for k in range(len(cells))}
    imax, jmax = max(c[0] for c in cells), max(c[1] for c in cells)
    o = m[(0, 0)]
    df = (m[(imax, 0)] - o) / imax
    dl = (m[(0, jmax)] - o) / jmax
    P = lambda i, j: o + i * df + j * dl
    halo = [pe.withStroke(linewidth=3.0, foreground=S.SURFACE)]
    kw = dict(arrowstyle="<|-|>,head_length=3.2,head_width=1.8", mutation_scale=1, lw=1.2, color=S.BLUE,
              shrinkA=0, shrinkB=0, zorder=6, path_effects=halo)
    ci, cj = 0.5, jmax / 2                               # rear half: clear of the faulted wheel
    ax.add_patch(FancyArrowPatch(tuple(P(ci - 0.5, cj)), tuple(P(ci + 0.5, cj)), **kw))    # forward / back
    ax.add_patch(FancyArrowPatch(tuple(P(ci, cj - 0.5)), tuple(P(ci, cj + 0.5)), **kw))    # sideways
    gi, gj = imax / 2, jmax / 2
    ph = np.radians(np.linspace(120, 235, 80))
    pts = np.array([P(gi + (imax / 2 + 0.75) * np.cos(a), gj + (jmax / 2 + 0.85) * np.sin(a)) for a in ph])
    ax.plot(pts[:-2, 0], pts[:-2, 1], color=S.BLUE, lw=1.2, zorder=6, path_effects=halo)
    ax.add_patch(FancyArrowPatch(tuple(pts[-4]), tuple(pts[-1]), **{**kw, "arrowstyle":
                                                                   "-|>,head_length=3.2,head_width=1.8"}))


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
