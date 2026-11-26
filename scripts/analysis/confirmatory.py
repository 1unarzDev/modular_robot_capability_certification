"""Confirmatory scorer: every claim of docs/CONFIRMATORY.md (C-MAIN, C-MERGE, C-GATE, C-ENV, C-SCALE, C-HET), scored
exactly as pre-registered. Committed and unit-tested on mock data (tests/test_confirmatory.py) BEFORE any confirmatory
shard is pulled; the scoring window opens 2026-11-26 01:45Z and the script refuses real inputs before then.

    source env.sh
    $PY scripts/analysis/confirmatory.py                       # real inputs (results/*/confirmatory/), after 01:45Z
    $PY scripts/analysis/confirmatory.py --mock /tmp/cm        # write mock inputs and score them (any time)

Outputs: results/confirmatory/verdicts.csv (one row per bar, with the git sha of docs/CONFIRMATORY.md),
results/confirmatory/cells.csv (every descriptive cell), results/confirmatory/CONFIRMATORY_RESULTS.md (power checks
first, then the verdicts, then the Holm table for P1-P7).

Input contracts (docs/board/paper.md §5): decision rows results/<role>/confirmatory/dec_*.csv[.gz] (column `claim`
routes rows), missions results/<role>/confirmatory/missions_*.csv[.gz], the C-ENV prerequisite
results/env/buildings_ks.csv.

Rules implemented (section numbers of docs/CONFIRMATORY.md):
  §0.1  canary shards (shard 0) and the pre-touched conf_test indices 0-39 are excluded from every claim; C-MAIN uses
        conf_test 40-3,039, C-GATE 3,040-5,039 (paired across severities). Exclusions are counted and printed.
  §0.2  matched operating points are fitted only on rows with split == 'cal' (exploratory cal); nothing is refitted.
  §0.3  95 % percentile cluster bootstrap, B = 2000, by the stated unit (episode; assembly pair; building).
        Superiority: paired cluster bootstrap on the same clusters (same weights), one-sided p = (1 + #{d* <= 0}) /
        (B + 1), Holm over P1-P7 at family-wise 0.05 (a missing P keeps its slot: m = 7).
        Validity bars "x <= b": where the document states a separate upper bound ("<= b (upper 95 % <= u)") both the
        point <= b and the one-sided 95 % upper (95th bootstrap percentile) <= u are required; where it states only
        b, the one-sided upper must be <= b (the general rule of §0.3). "Lower 95 %" = the 5th bootstrap percentile.
        Labels: `truth` = rho_dec with the FIXED T1 reference for every plant; `truth_own` (own-plant reference) is
        secondary and only reported.
        Headline stratum: faulted candidate or |rho_dec - tau| < 0.10, tau in {0.6, 0.7, 0.8}.
  §0.4  E_ep = share of clusters with >= 1 unsafe commit (all decisions of the cell, the method's own certified
        bound); E_dec = unsafe / incapable stratum decisions. Power rule for every E_dec bar, printed before any
        verdict: n_incapable >= 20,000 AND 0.01 + 1.96 sqrt(deff * 0.01 * 0.99 / n) <= 0.013, where deff is the
        cluster design effect of the cell's ratio estimator (linearised; at an observed rate of 0 the Kish
        unequal-cluster-size deff is used). z = 1.96 reproduces the document's "deff <= ~4.5 at 20k". A failing cell
        is UNDERPOWERED and its E_dec bar does not count; E_ep decides where the document says so.
Amendments of docs/CONFIRMATORY.md §9 (e176756): (1) the E_ep validity bars of C-MAIN bar 1 and C-MERGE bar 2 use
the `all` block (block-restricted E_ep is descriptive); (2) ">=" thresholds (C-SCALE coverage, P4 CF, C-HET drop
ID - split <= 0.05) are judged on the point estimate with the one-sided bound printed; (3) P5's >=3-module dominated
cell, when underpowered, is contrast-judged: one-sided upper <= 2 %; when powered, <= 1 % with upper <= 1.3 %.
Amendments A-GATE / A-HET (727f995): C-GATE H13 and P6 are withdrawn; P6' (Holm, m stays 7) = paired
E_ep(par) - E_ep(robust) > 0 at s = 3 on >=3-module children AND robust's undominated E_dec <= 1 % (power rule, else
E_ep <= 0.10); robust validity at every severity (not Holm); price of robustness and lawcal at the held-out s = 3
(descriptive, pre-stated); robust = v2+clamp rows, the others v2. C-HET's H1 bars use the faulted-candidate block.
A-ENV (0dc1aec): conf-ID mission 1499 excluded; amortized_comp E_mis = empirical transfer bars (not guarantees);
amortized_comp_envcal scored under the same bars (guarantee on ID); raw = plug-in median, descriptive.
c2b97cd: C-EXPLAIN (§6b, secondary, not in Holm; models' wheels_conf_models.npz + point = q50; truth regenerated with
make_episode) and an exploratory C-MAIN row with tau from the task-library histogram.
A-SEED (03387d0, §10): model rows (C-MAIN, C-SCALE, C-ENV, C-EXPLAIN) are filtered to model_tag == st-n6-2; other
tags (st-n6-1) go to a descriptive seed-variation column, never to a bar.
Coordinator 00:35Z: credible_lower comes from models (seed-filtered like every model row); always_go, never_go and
metadata_nf are computed by the scorer (common.trivial_methods on amortized_comp's labels), never read from rows.
Which bound each bar uses: E_dec at the matched point (U* = 1 %, margin from cal) for C-MAIN bar 1 and C-MERGE P4;
each certifier's own bound for E_ep, for P3 (raw vs ours: calibration is what is being tested) and for C-GATE
(par / raw / pi / gate are each a complete certifier).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROOT = C.ROOT
OUT = ROOT / "results/confirmatory"
B = 2000
SEED = 92599
TAUS = (0.6, 0.7, 0.8)
BAND = 0.10
U_STAR = 0.01
ALPHA_FWER = 0.05
SCORING_OPENS = dt.datetime(2026, 11, 26, 1, 45, tzinfo=dt.timezone.utc)
CANARY_SHARDS = {0}
PRE_TOUCHED = {"conf_test": set(range(0, 40))}           # §0.1 amendment ac543b8
CMAIN_IDX, CGATE_IDX = (40, 3040), (3040, 5040)
SMC_FIRST = 200                                          # §1: SMC on the first 200 indices, descriptive
POWER_N, POWER_UP, POWER_Z = 20000, 0.013, 1.96
HOLM = ("P1", "P2", "P3", "P4", "P5", "P6'", "P7")     # P6' replaces P6 (A-GATE, 727f995); m stays 7

OURS = "amortized_comp"                                  # models export exactly one variant under this name
TRIVIAL_P2 = ("metadata_nf", "detector_gated", "ghat_physics", "copy_parent", "min_over_members", "dominance_pi")
SCALE_BINS = ((7, 8), (9, 10), (11, 13), (14, 16))
DIRECT = ("st_direct", "blackbox_gnn")

DEC_REQ = ("claim", "method", "test_plant", "split", "status", "eid", "shard", "cand", "kind", "cand_size",
           "has_fault", "leg", "truth", "lower")
DEC_OPT = ("point", "scale", "truth_own", "severity", "probe", "dominated", "parent_n", "seconds", "lower_marg",
           "arm", "variant", "model_tag")
MIS_REQ = ("claim", "method", "family", "building", "mission", "shard", "committed", "failed", "success",
           "oracle_success")
MIS_OPT = ("reconfig", "time_s", "status", "model_tag")
KS_REQ = ("stat", "ks_p", "ks_d")
# A-SEED (03387d0, §10): every confirmatory model row of C-MAIN / C-SCALE / C-ENV / C-EXPLAIN is st-n6-2. Rows of any
# other model tag (st-n6-1 if it finishes) are a descriptive seed-variation column only: never in a bar, no
# best-of-seeds. C-HET uses its own het models and is not filtered.
PRIMARY_TAG = "st-n6-2"
SEED_CLAIMS = ("C-MAIN", "C-SCALE", "C-ENV")             # C-EXPLAIN reads C-MAIN rows
MODEL_METHODS = {"amortized_comp", "amortized_comp_blind", "amortized_comp_envcal", "raw", "credible_lower",
                 "blackbox_gnn",
                 "st_direct", "amortized_marg", "amortized_sim"}


# ------------------------------------------------------------------------------------------------- loading

def _load(pattern, req, opt, res):
    files = sorted(glob.glob(str(res / pattern)))
    frames = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        miss = [c for c in req if c not in d.columns]
        if miss:
            raise C.SchemaError(f"{f}: missing {miss} (docs/board/paper.md §5)")
        for c in opt:
            if c not in d.columns:
                d[c] = np.nan
        d["_src"] = C.rel(f)
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=list(req) + list(opt) + ["_src"]), []
    return pd.concat(frames, ignore_index=True), [C.rel(f) for f in files]


def exclusions(d, log, what):
    """§0.1: drop canary shards and the pre-touched conf_test indices from test streams (cal rows untouched)."""
    if d.empty:
        return d
    test = d["split"].astype(str) != "cal"
    can = test & d["shard"].isin(CANARY_SHARDS)
    pre = np.zeros(len(d), bool)
    for sp, idx in PRE_TOUCHED.items():
        pre |= test.to_numpy() & (d["split"] == sp).to_numpy() & d["eid"].isin(idx).to_numpy()
    log.append(f"{what}: excluded {int(can.sum())} rows of canary shards {sorted(CANARY_SHARDS)} "
               f"({d[can].eid.nunique()} clusters), {int(pre.sum())} rows of pre-touched conf_test 0-39 "
               f"({d[pre].eid.nunique()} clusters)")
    return d[~(can.to_numpy() | pre)]


# ------------------------------------------------------------------------------------------------- per-cluster stats

def stratum_mask(d, tau):
    return (d["has_fault"].to_numpy(float) > 0.5) | (np.abs(d["truth"].to_numpy(float) - tau) < BAND)


def cluster_stats(d, lower, sel=None, cluster="eid"):
    """Per-cluster decision counts over TAUS. sel(d, tau) -> bool mask restricting the E_dec/CF counts (e.g. the
    stratum); unsafe commits for E_ep are counted on every decision of `d` (the cell). Returns DataFrame indexed by
    cluster with cap, cert, inc, uns (in sel) and uns_all, ep_unsafe (0/1), n_rows."""
    tr = d["truth"].to_numpy(float)
    lo = np.asarray(lower, float)
    ok = np.isfinite(tr)
    acc = {k: np.zeros(len(d)) for k in ("cap", "cert", "inc", "uns", "uns_all")}
    for tau in TAUS:
        g = np.isfinite(lo) & (lo >= tau) & ok
        s = ok & (sel(d, tau) if sel is not None else True)
        acc["cap"] += s & (tr >= tau)
        acc["cert"] += s & (tr >= tau) & g
        acc["inc"] += s & (tr < tau)
        acc["uns"] += s & (tr < tau) & g
        acc["uns_all"] += ok & (tr < tau) & g
    out = pd.DataFrame(acc).groupby(d[cluster].to_numpy()).sum()
    out["ep_unsafe"] = (out["uns_all"] > 0).astype(float)
    out["one"] = 1.0
    return out


def cluster_cov(d, lower, sel=None, cluster="eid"):
    """Tau-free coverage per cluster: entries (in sel(d) if given) and whether all are covered (miscoverage)."""
    tr = d["truth"].to_numpy(float)
    lo = np.nan_to_num(np.asarray(lower, float), nan=-np.inf)
    ok = np.isfinite(tr) & (sel(d) if sel is not None else True)
    cov = (lo <= tr) & ok
    df = pd.DataFrame(dict(n=ok.astype(float), cov=cov.astype(float))).groupby(d[cluster].to_numpy()).sum()
    df["miscov"] = (df["cov"] < df["n"]).astype(float)
    df["one"] = (df["n"] > 0).astype(float)
    return df


def cluster_flips(d, cluster="eid"):
    """Decision flips of a point prediction vs truth (delta_comp): (point >= tau) != (truth >= tau)."""
    tr, p = d["truth"].to_numpy(float), d["point"].to_numpy(float)
    ok = np.isfinite(tr) & np.isfinite(p)
    f = np.zeros(len(d))
    n = np.zeros(len(d))
    for tau in TAUS:
        f += ok & ((p >= tau) != (tr >= tau))
        n += ok
    return pd.DataFrame(dict(flip=f, n=n)).groupby(d[cluster].to_numpy()).sum()


# ------------------------------------------------------------------------------------------------- bootstrap

class Boot:
    """Cluster bootstrap with one weight matrix per cluster set, reused for paired comparisons. strata: optional
    array of stratum labels (independent resampling within each, e.g. size bins or splits)."""

    def __init__(self, clusters, strata=None, B=None, seed=SEED):
        B = B or globals()["B"]
        self.idx = pd.Index(clusters)
        m = len(self.idx)
        rng = np.random.default_rng(seed)
        if m == 0:
            self.W = np.zeros((B, 0))
            return
        if strata is None:
            self.W = rng.multinomial(m, np.full(m, 1 / m), size=B).astype(float)
        else:
            strata = np.asarray(strata)
            self.W = np.zeros((B, m))
            for s in pd.unique(strata):
                j = np.flatnonzero(strata == s)
                self.W[:, j] = rng.multinomial(len(j), np.full(len(j), 1 / len(j)), size=B)

    def align(self, df):
        return df.reindex(self.idx).fillna(0.0)

    def ratio(self, df, num, den):
        a = self.align(df)
        n, dd = a[num].to_numpy(float), a[den].to_numpy(float)
        est = n.sum() / dd.sum() if dd.sum() > 0 else np.nan
        Dn = self.W @ dd
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = np.where(Dn > 0, (self.W @ n) / Dn, np.nan)
        return est, reps


def ci(reps):
    r = reps[np.isfinite(reps)]
    if not len(r):
        return dict(ci_lo=np.nan, ci_hi=np.nan, upper95=np.nan, lower95=np.nan)
    return dict(ci_lo=float(np.quantile(r, 0.025)), ci_hi=float(np.quantile(r, 0.975)),
                upper95=float(np.quantile(r, 0.95)), lower95=float(np.quantile(r, 0.05)))


def p_greater(reps, null=0.0):
    """One-sided bootstrap p for H0: parameter <= null."""
    r = reps[np.isfinite(reps)]
    if not len(r):
        return np.nan
    return float((1 + np.sum(r <= null)) / (len(r) + 1))


def power(cs):
    """§0.4 power rule on a per-cluster table with inc / uns columns."""
    i, u = cs["inc"].to_numpy(float), cs["uns"].to_numpy(float)
    n = i.sum()
    m = int((i > 0).sum())
    if n <= 0 or m < 2:
        return dict(n_incapable=int(n), deff=np.nan, up_at_1pct=np.nan, power_ok=False)
    R = u.sum() / n
    if R > 0:
        var_lin = m / (m - 1) * np.sum((u - R * i) ** 2) / n ** 2
        deff = var_lin / (R * (1 - R) / n)
    else:
        ii = i[i > 0]
        deff = 1 + (ii.std() / ii.mean()) ** 2               # Kish unequal-cluster-size design effect
    up = 0.01 + POWER_Z * math.sqrt(max(deff, 0) * 0.01 * 0.99 / n)
    return dict(n_incapable=int(n), deff=float(deff), up_at_1pct=float(up),
                power_ok=bool(n >= POWER_N and up <= POWER_UP))


# ------------------------------------------------------------------------------------------------- recording

class Scorer:
    def __init__(self):
        self.rows, self.cells, self.powers, self.log, self.p = [], [], [], [], {}

    def bar(self, claim, bar, metric, est, reps=None, *, le=None, up_le=None, ge=None, lo_gt=None, method="",
            cell="", plant="", n_clusters=np.nan, power_row=None, note="", status="", descriptive=False):
        c = ci(reps) if reps is not None else dict(ci_lo=np.nan, ci_hi=np.nan, upper95=np.nan, lower95=np.nan)
        conds = []
        if le is not None:
            conds.append(est <= le)
        if up_le is not None:
            conds.append(c["upper95"] <= up_le)
        if ge is not None:
            conds.append(est >= ge)
        if lo_gt is not None:
            conds.append(c["lower95"] > lo_gt)
        if status:
            v = status
        elif descriptive:
            v = "DESCRIPTIVE"
        elif est is None or not np.isfinite(est):
            v = "MISSING"
        elif power_row is not None and not power_row["power_ok"]:
            v = "UNDERPOWERED"
        else:
            v = "PASS" if all(bool(x) for x in conds) else "FAIL"
        thr = "; ".join(s for s in (f"<= {le}" if le is not None else "", f"upper95 <= {up_le}" if up_le is not None
                                    else "", f">= {ge}" if ge is not None else "",
                                    f"lower95 > {lo_gt}" if lo_gt is not None else "") if s)
        pr = power_row or {}
        self.rows.append(dict(claim=claim, bar=bar, metric=metric, method=method, cell=cell, plant=plant,
                              estimate=est, **c, threshold=thr, n_clusters=n_clusters,
                              n_incapable=pr.get("n_incapable", np.nan), deff=pr.get("deff", np.nan),
                              power_ok=pr.get("power_ok", np.nan), verdict=v, note=note))
        return v

    def power_check(self, claim, cell, method, plant, cs):
        pw = power(cs)
        self.powers.append(dict(claim=claim, cell=cell, method=method, plant=plant, **pw))
        return pw

    def cell(self, **kw):
        self.cells.append(kw)


# ------------------------------------------------------------------------------------------------- matched point

def matched(cal_m, ev):
    """Matched lower at U* on the stratum, margin fitted on the method's T1 cal rows only (§0.2). Without cal rows the
    method has no matched point (returns None)."""
    fit = cal_m[cal_m["test_plant"] == "T1"] if len(cal_m) else cal_m
    if fit.empty:
        return None
    lo, _ = C.matched_lower(fit, ev, u_star=U_STAR, taus=TAUS, stratum="rel")
    return lo


def _sel(block):
    """Block masks within the stratum (tau-dependent); 'all' = every decision."""
    def f(d, tau):
        s = stratum_mask(d, tau)
        k, n = d["kind"].to_numpy(), d["cand_size"].to_numpy(float)
        hf = d["has_fault"].to_numpy(float) > 0.5
        if block == "stratum":
            return s
        if block == "lone-faulted":
            return (k == "child") & (n == 1) & hf
        if block == "multi-child":
            return s & (k == "child") & (n >= 2)
        if block == "self":
            return s & (k == "self")
        return np.ones(len(d), bool)
    return f


BLOCKS = ("stratum", "lone-faulted", "multi-child", "self", "all")


# ------------------------------------------------------------------------------------------------- C-MAIN

def score_main(S, dec, cal):
    d = dec[(dec.claim == "C-MAIN") & (dec.split == "conf_test") & dec.eid.between(CMAIN_IDX[0], CMAIN_IDX[1] - 1)]
    cal = cal[cal.claim == "C-MAIN"]
    S.log.append(f"C-MAIN: {d.eid.nunique()} episodes after exclusions (conf_test {CMAIN_IDX[0]}-{CMAIN_IDX[1] - 1})")
    methods = sorted(d.method.unique())
    stats = {}          # (method, plant, block, kind) -> cluster table
    plants = ["T1", "T2"] + sorted(set(d.test_plant.astype(str)) - {"T1", "T2"})   # e.g. T2 all+motor: cells only
    for plant in plants:
        dp = d[d.test_plant == plant]
        bt = Boot(np.sort(dp.eid.unique()))
        for m in methods:
            ev = dp[dp.method == m].reset_index(drop=True)
            if ev.empty:
                continue
            if m.startswith("smc") or m == "nuisance_smc":
                ev = ev[ev.eid < SMC_FIRST].reset_index(drop=True)
            # cfBH: lower = largest selected tau (or -inf), commit-at-tau <=> lower >= tau; no matched point
            mlo = None if m in C.SELECTION_METHODS else matched(cal[cal.method == m], ev)
            for blk in BLOCKS:
                own = cluster_stats(ev, ev["lower"], _sel(blk))
                stats[m, plant, blk, "own"] = own
                row = dict(claim="C-MAIN", method=m, plant=plant, block=blk, n_ep=len(own),
                           E_ep=own.ep_unsafe.mean(), E_dec_own=C._ratio(own.uns.sum(), own.inc.sum()),
                           CF_own=C._ratio(own.cert.sum(), own.cap.sum()),
                           sec=float(np.nanmedian(ev.drop_duplicates("eid").seconds)) if ev.seconds.notna().any()
                           else np.nan)
                if mlo is not None:
                    mt = cluster_stats(ev, mlo, _sel(blk))
                    stats[m, plant, blk, "matched"] = mt
                    row.update(E_dec_matched=C._ratio(mt.uns.sum(), mt.inc.sum()),
                               CF_matched=C._ratio(mt.cert.sum(), mt.cap.sum()))
                if m == "cfBH":
                    row["FDR"] = C._ratio(own.uns_all.sum(), (own.uns_all + own.cert).sum()) if blk == "all" else np.nan
                if blk == "stratum" and "truth_own" in ev and ev.truth_own.notna().any():
                    e2 = ev.assign(truth=ev.truth_own)
                    o2 = cluster_stats(e2, e2["lower"], _sel(blk))
                    row["E_dec_own_ownref"] = C._ratio(o2.uns.sum(), o2.inc.sum())
                S.cell(**row)
        if plant not in ("T1", "T2"):
            continue
        # ---- bar 1: validity of ours
        bars = {"T1": dict(ep=(0.10, 0.12), dec=(0.01, POWER_UP)), "T2": dict(ep=(0.15, 0.15), dec=(0.02, 0.02))}[plant]
        own = stats.get((OURS, plant, "all", "own"))
        if own is None:
            S.bar("C-MAIN", "1 validity", "E_ep", np.nan, method=OURS, plant=plant, status="MISSING")
            S.bar("C-MAIN", "1 validity", "E_dec@U*", np.nan, method=OURS, plant=plant, status="MISSING")
        else:
            est, reps = bt.ratio(own, "ep_unsafe", "one")
            S.bar("C-MAIN", "1 validity", "E_ep", est, reps, le=bars["ep"][0], up_le=bars["ep"][1], method=OURS,
                  cell="all decisions", plant=plant, n_clusters=len(own),
                  note="all block (§9.1); block-restricted E_ep is descriptive (cells.csv)")
            mt = stats.get((OURS, plant, "stratum", "matched"))
            if mt is None:
                S.bar("C-MAIN", "1 validity", "E_dec@U*", np.nan, method=OURS, plant=plant, status="MISSING",
                      note="no cal rows for the matched point")
            else:
                pw = S.power_check("C-MAIN", "stratum", OURS, plant, mt)
                est, reps = bt.ratio(mt, "uns", "inc")
                S.bar("C-MAIN", "1 validity", "E_dec@U*", est, reps, le=bars["dec"][0], up_le=bars["dec"][1],
                      method=OURS, cell="stratum", plant=plant, n_clusters=len(mt), power_row=pw)
        if plant == "T1":
            # ---- P1: vs capxfer, lone-faulted, CF@U*
            a, b = stats.get((OURS, "T1", "lone-faulted", "matched")), stats.get(("capxfer", "T1", "lone-faulted", "matched"))
            _paired_diff(S, bt, "C-MAIN", "P1", "CF@U* (lone-faulted)", a, b, "cert", "cap", 0.10, OURS, "capxfer", "T1")
            # ---- P2: vs best trivial picked on cal
            best, cf_best = None, -1
            for m in TRIVIAL_P2:
                cm = cal[(cal.method == m) & (cal.test_plant == "T1")].reset_index(drop=True)
                if cm.empty:
                    continue
                lo = matched(cm, cm)
                t = cluster_stats(cm, lo, _sel("stratum"))
                cf = C._ratio(t.cert.sum(), t.cap.sum())
                if np.isfinite(cf) and cf > cf_best:
                    best, cf_best = m, cf
            a = stats.get((OURS, "T1", "stratum", "matched"))
            b = stats.get((best, "T1", "stratum", "matched")) if best else None
            _paired_diff(S, bt, "C-MAIN", "P2", "CF@U* (stratum)", a, b, "cert", "cap", 0.05, OURS,
                         best or "best trivial (none with cal rows)", "T1", note=f"best trivial on cal: {best}")
        else:
            # ---- P3: calibration vs raw posterior in T2 (own bounds)
            r_o, o_o = stats.get(("raw", "T2", "stratum", "own")), stats.get((OURS, "T2", "stratum", "own"))
            r_a, o_a = stats.get(("raw", "T2", "all", "own")), stats.get((OURS, "T2", "all", "own"))
            if r_o is None or o_o is None:
                S.bar("C-MAIN", "P3", "E_dec own (raw)", np.nan, method="raw", plant="T2", status="MISSING")
                S.p["P3"] = np.nan
            else:
                er, rr = bt.ratio(r_o, "uns", "inc")
                v_raw = S.bar("C-MAIN", "P3", "E_dec own (raw) > 2 %", er, rr, ge=0.02 + 1e-12, lo_gt=0.01,
                              method="raw", cell="stratum", plant="T2", n_clusters=len(r_o))
                eo, ro = bt.ratio(o_o, "uns", "inc")
                S.bar("C-MAIN", "P3", "E_dec own (ours) <= 2 %", eo, ro, le=0.02, up_le=0.02, method=OURS,
                      cell="stratum", plant="T2", n_clusters=len(o_o))
                if np.isfinite(er) and er <= 0.02:
                    S.bar("C-MAIN", "P3", "claim", er, status="DROPPED", method="raw", plant="T2",
                          note="raw is <= 2 % in T2: calibration buys the guarantee, not safety (pre-stated)")
                ea, ra = bt.ratio(r_a, "ep_unsafe", "one")
                eb, rb = bt.ratio(o_a, "ep_unsafe", "one")
                S.p["P3"] = p_greater(ra - rb)
                S.bar("C-MAIN", "P3", "E_ep(raw) - E_ep(ours), paired", ea - eb, ra - rb, lo_gt=0.0, method=OURS,
                      cell="all decisions", plant="T2", n_clusters=len(o_a), status="HOLM",
                      note=f"raw E_dec condition: {v_raw}")
    # cfBH and SMC are descriptive (cells.csv)
    score_task_library(S, d)


# ------------------------------------------------------------------------------------------------- task-library tau

TASK_LIBRARY = ROOT / "results/annotations/task_library.csv"


def task_library_taus(path=None):
    """leg index -> array of requirement values (rho_dec units) from the task library's r_<leg> columns."""
    path = pathlib.Path(path or TASK_LIBRARY)
    if not path.exists():
        return None
    t = pd.read_csv(path)
    out = {}
    for li, leg in enumerate(C.LEGS):
        col = f"r_{leg}"
        if col in t:
            v = t[col].dropna().to_numpy(float)
            if len(v):
                out[li] = v
    return out or None


def task_library_stats(d, lower, taus_by_leg):
    """Per-episode weighted counts with tau drawn from the leg's task-library histogram (expectation: every library
    tau of that leg with weight 1/n_leg). Stratum = faulted or |rho_dec - tau| < BAND; trivially capable = capable
    and outside the stratum."""
    tr = d["truth"].to_numpy(float)
    lo = np.nan_to_num(np.asarray(lower, float), nan=-np.inf)
    hf = d["has_fault"].to_numpy(float) > 0.5
    leg = d["leg"].to_numpy(int)
    acc = {k: np.zeros(len(d)) for k in ("cap", "cert", "inc", "uns", "w", "triv", "uns_all")}
    for li, taus in taus_by_leg.items():
        i = np.flatnonzero((leg == li) & np.isfinite(tr))
        if not len(i):
            continue
        T = taus[None, :]
        w = 1.0 / len(taus)
        cap = tr[i, None] >= T
        go = lo[i, None] >= T
        st = hf[i, None] | (np.abs(tr[i, None] - T) < BAND)
        acc["w"][i] += w * len(taus)
        acc["cap"][i] += w * (cap & st).sum(1)
        acc["cert"][i] += w * (cap & st & go).sum(1)
        acc["inc"][i] += w * (~cap & st).sum(1)
        acc["uns"][i] += w * (~cap & st & go).sum(1)
        acc["triv"][i] += w * (cap & ~st).sum(1)
        acc["uns_all"][i] += w * (~cap & go).sum(1)
    out = pd.DataFrame(acc).groupby(d["eid"].to_numpy()).sum()
    out["ep_unsafe"] = (out["uns_all"] > 0).astype(float)
    out["one"] = 1.0
    return out


def score_task_library(S, d, path=None):
    """Exploratory secondary C-MAIN row (c2b97cd): decisions re-thresholded at tau from the task-library histogram."""
    tb = task_library_taus(path)
    if tb is None:
        S.bar("C-MAIN (task-library tau)", "exploratory", "CF / E_dec / trivially capable", np.nan, status="MISSING",
              note=f"{C.rel(TASK_LIBRARY)} missing")
        return
    for plant in sorted(d.test_plant.astype(str).unique()):
        dp = d[d.test_plant == plant]
        bt = Boot(np.sort(dp.eid.unique()))
        for m in sorted(dp.method.unique()):
            ev = dp[dp.method == m].reset_index(drop=True)
            t = task_library_stats(ev, ev["lower"], tb)
            cf, rcf = bt.ratio(t, "cert", "cap")
            ed, red = bt.ratio(t, "uns", "inc")
            tv, rtv = bt.ratio(t, "triv", "w")
            S.cell(claim="C-MAIN (task-library tau)", method=m, plant=plant, block="stratum", n_ep=len(t), CF_own=cf,
                   E_dec_own=ed, trivially_capable_share=tv, status="exploratory")
            if m == OURS:
                note = "exploratory; tau from results/annotations/task_library.csv; own bound"
                S.bar("C-MAIN (task-library tau)", "exploratory", "CF (stratum)", cf, rcf, method=m, plant=plant,
                      descriptive=True, note=note)
                S.bar("C-MAIN (task-library tau)", "exploratory", "E_dec (stratum)", ed, red, method=m, plant=plant,
                      descriptive=True, note=note)
                S.bar("C-MAIN (task-library tau)", "exploratory", "share of trivially capable decisions", tv, rtv,
                      method=m, plant=plant, descriptive=True, note="capable and outside the stratum")


# ------------------------------------------------------------------------------------------------- C-EXPLAIN

WHEELS_GLOB = "models/confirmatory/wheels_*.npz"
P_BLAME = 0.5


def episode_truth(split, index, plate="plated"):
    """True per-candidate wheel faults, regenerated from the episode index (cfcap.gen.episode.make_episode; latents
    only, no simulation). -> {cand: [(role, wheel, faulty)]}."""
    from cfcap.gen.episode import make_episode
    ep = make_episode(plate, split, int(index))
    return {ci: [(r, 4 * k + j, int(ep.asms[r].mode[4 * k + j] != 0)) for (r, k) in c.modules for j in range(4)]
            for ci, c in enumerate(ep.cands)}


def load_wheels(res):
    """models' C-EXPLAIN input (commit 1ffcc69, docs/board/models.md): results/models/confirmatory/
    wheels_conf_models.npz, one row per wheel sorted by eid then ep_wheel, arrays eid, asm_role (0 parent, 1+ merge
    partners), wheel (4*module + corner within the assembly), ep_wheel, p (W x 4: P(ok, weak, coast, brake)), g_mean,
    classes. -> long frame eid, role, wheel, p_fault = 1 - P(ok)."""
    files = sorted(glob.glob(str(res / WHEELS_GLOB)))
    if not files:
        return None, []
    parts = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        miss = [k for k in ("eid", "asm_role", "wheel", "p") if k not in z.files]
        if miss:
            raise C.SchemaError(f"{f}: missing arrays {miss} (docs/board/paper.md §5d, models 1ffcc69)")
        P = np.asarray(z["p"], float)
        if P.ndim != 2 or P.shape[1] < 2:
            raise C.SchemaError(f"{f}: p must be (W, n_classes)")
        ok = 0
        if "classes" in z.files:
            cl = [str(c).lower() for c in z["classes"]]
            if "ok" not in cl:
                raise C.SchemaError(f"{f}: classes {cl} has no 'ok'")
            ok = cl.index("ok")
        parts.append(pd.DataFrame(dict(eid=z["eid"].astype(int), role=z["asm_role"].astype(int),
                                       wheel=z["wheel"].astype(int), p_fault=1.0 - P[:, ok])))
    return pd.concat(parts, ignore_index=True), [C.rel(f) for f in files]


def explain_stats(d, wheels, truth_fn):
    """Per-episode explainer counts over refused stratum decisions of one method (rows d, own lower, point = q50)."""
    pf = {(int(e), int(r), int(w)): float(p) for e, r, w, p in wheels[["eid", "role", "wheel", "p_fault"]].itertuples(
        index=False)}
    blame = {}
    for eid in d.eid.unique():
        tr = truth_fn("conf_test", int(eid))
        for ci, ws in tr.items():
            b = [(r, w, f) for r, w, f in ws if pf.get((int(eid), r, w), 0.0) >= P_BLAME]
            blame[int(eid), ci] = (len(b), sum(f for _, _, f in b))
    key = list(zip(d.eid.astype(int), d.cand.astype(int)))
    nb = np.array([blame.get(k, (0, 0))[0] for k in key], float)
    nc = np.array([blame.get(k, (0, 0))[1] for k in key], float)
    tr = d["truth"].to_numpy(float)
    lo = np.nan_to_num(d["lower"].to_numpy(float), nan=-np.inf)
    med = d["point"].to_numpy(float)
    acc = {k: np.zeros(len(d)) for k in ("ref", "blamed", "blamed_ok", "deficit", "deficit_ok", "margin", "inc_ref",
                                         "inc_informative", "cap_ref", "cap_wrong_reason")}
    for tau in TAUS:
        ref = np.isfinite(tr) & stratum_mask(d, tau) & ~(lo >= tau)
        deficit = ref & (med < tau)
        acc["ref"] += ref
        acc["blamed"] += ref * nb
        acc["blamed_ok"] += ref * nc
        acc["deficit"] += deficit
        acc["deficit_ok"] += deficit & (tr < tau)
        acc["margin"] += ref & ~(med < tau)
        inc = ref & (tr < tau)
        acc["inc_ref"] += inc
        acc["inc_informative"] += inc & ((nb > 0) | deficit)
        cap = ref & (tr >= tau)
        acc["cap_ref"] += cap
        acc["cap_wrong_reason"] += cap & deficit
    return pd.DataFrame(acc).groupby(d["eid"].to_numpy()).sum()


def score_explain(S, dec, res, truth_fn=None):
    """C-EXPLAIN (§6b, c2b97cd; secondary, not in Holm): point-judged, episode-clustered CI printed."""
    claim = "C-EXPLAIN"
    wheels, srcs = load_wheels(res)
    d = dec[(dec.claim == "C-MAIN") & (dec.split == "conf_test") & (dec.test_plant == "T1") & (dec.method == OURS)
            & dec.eid.between(CMAIN_IDX[0], CMAIN_IDX[1] - 1)].reset_index(drop=True)
    if wheels is None or d.empty:
        S.bar(claim, "all bars", "", np.nan, status="MISSING",
              note="needs models' wheels_conf_models.npz and C-MAIN T1 amortized_comp rows")
        return
    if d["point"].isna().all():
        raise C.SchemaError("C-EXPLAIN: amortized_comp rows need point = posterior median (q50)")
    t = explain_stats(d, wheels, truth_fn or episode_truth)
    bt = Boot(t.index)
    for metric, num, den, kw in (("wheel precision", "blamed_ok", "blamed", dict(ge=0.95)),
                                 ("deficit-claim precision", "deficit_ok", "deficit", dict(ge=0.90)),
                                 ("informative share (incapable refusals)", "inc_informative", "inc_ref",
                                  dict(ge=0.80)),
                                 ("false-reason rate (capable refusals)", "cap_wrong_reason", "cap_ref", dict(le=0.05))):
        e, r = bt.ratio(t, num, den)
        S.bar(claim, metric, metric, e, r, method=OURS, cell="C-MAIN T1 refusals (stratum)", plant="T1",
              n_clusters=len(t), note="point-judged; secondary, not in Holm", **kw)
    e, r = bt.ratio(t, "margin", "ref")
    S.bar(claim, "margin share", "share of refusals with a margin claim", e, r, method=OURS, descriptive=True,
          plant="T1", note="margin claims make no physical assertion")


def _paired_diff(S, bt, claim, pid, metric, a, b, num, den, min_pts, ma, mb, plant, note=""):
    if a is None or b is None:
        S.p[pid] = np.nan
        S.bar(claim, pid, metric, np.nan, method=ma, plant=plant, status="MISSING", note=f"vs {mb}; " + note)
        return
    ea, ra = bt.ratio(a, num, den)
    eb, rb = bt.ratio(b, num, den)
    S.p[pid] = p_greater(ra - rb)
    S.bar(claim, pid, f"{metric}: {ma} - {mb}", ea - eb, ra - rb, ge=min_pts, lo_gt=0.0, method=ma, plant=plant,
          n_clusters=len(bt.idx), status="HOLM", note=note)


# ------------------------------------------------------------------------------------------------- C-MERGE

def score_merge(S, dec, cal):
    d = dec[dec.claim == "C-MERGE"].copy()
    cal = cal[cal.claim == "C-MERGE"]
    if d.empty:
        for b in ("1 delta_comp", "2 validity", "P4"):
            S.bar("C-MERGE", b, "", np.nan, status="MISSING")
        S.p["P4"] = np.nan
        return
    d["pair"] = d["arm"].astype(str) + ":" + d["eid"].astype(str)        # resampling unit: the assembly pair
    S.log.append(f"C-MERGE: {d.pair.nunique()} pairs after exclusions")
    for plant, (fl, ep_mc, mc_up) in (("T1", (0.01, 0.10, 0.13)), ("T2", (0.02, 0.15, 0.18))):
        dp = d[d.test_plant == plant]
        bt = Boot(np.sort(dp.pair.unique()))
        o = dp[dp.method == "comp_oracle"]
        if o.empty or o.point.isna().all():
            S.bar("C-MERGE", "1 delta_comp", "flip share", np.nan, method="comp_oracle", plant=plant, status="MISSING")
        else:
            f = cluster_flips(o.reset_index(drop=True), "pair")
            est, reps = bt.ratio(f, "flip", "n")
            c = ci(reps)
            S.bar("C-MERGE", "1 delta_comp", "decision flips", est, reps, up_le=fl - 1e-12, method="comp_oracle",
                  plant=plant, n_clusters=len(f), note=f"bar: upper95 < {fl}")
        m = dp[dp.method == "comp_enum_size"].reset_index(drop=True)
        if m.empty:
            S.bar("C-MERGE", "2 validity", "miscoverage per pair", np.nan, method="comp_enum_size", plant=plant,
                  status="MISSING")
        else:
            cv = cluster_cov(m, m["lower"], cluster="pair")
            est, reps = bt.ratio(cv, "miscov", "one")
            S.bar("C-MERGE", "2 validity", "miscoverage per pair", est, reps, le=ep_mc, up_le=mc_up,
                  method="comp_enum_size", plant=plant, n_clusters=len(cv))
            cs = cluster_stats(m, m["lower"], cluster="pair")
            est, reps = bt.ratio(cs, "ep_unsafe", "one")
            S.bar("C-MERGE", "2 validity", "E_ep per pair", est, reps, up_le=0.10, method="comp_enum_size",
                  note="all block: any unsafe merge decision of the pair (§9.1)",
                  plant=plant, n_clusters=len(cs))
        # P4 (T1 decides; T2 reported against the same numbers)
        tabs = {}
        for meth in ("comp_enum_size", "copy_parent", "comp_enum", "detector_gated", "comp_oracle"):
            ev = dp[dp.method == meth].reset_index(drop=True)
            if ev.empty:
                continue
            lo = matched(cal[cal.method == meth], ev)
            note = ""
            if lo is None:
                lo, note = ev["lower"].to_numpy(float), "own bound (no cal rows)"
            t = cluster_stats(ev, lo, _sel("stratum"), cluster="pair")
            tabs[meth] = t
            S.cell(claim="C-MERGE", method=meth, plant=plant, block="stratum", n_ep=len(t),
                   CF_matched=C._ratio(t.cert.sum(), t.cap.sum()), E_dec_matched=C._ratio(t.uns.sum(), t.inc.sum()),
                   note=note)
        a = tabs.get("comp_enum_size")
        if a is None:
            S.bar("C-MERGE", "P4", "CF@U* comp_enum_size >= 85 %", np.nan, plant=plant, status="MISSING")
        else:
            pw = S.power_check("C-MERGE", "stratum merges", "comp_enum_size", plant, a)
            e, r = bt.ratio(a, "uns", "inc")
            S.bar("C-MERGE", "P4" if plant == "T1" else "P4 (T2, reported)", "realised E_dec@U*", e, r,
                  method="comp_enum_size", plant=plant, power_row=pw, descriptive=not pw["power_ok"] or plant == "T2",
                  note="descriptive unless the power rule holds")
            e, r = bt.ratio(a, "cert", "cap")
            S.bar("C-MERGE", "P4" if plant == "T1" else "P4 (T2, reported)", "CF@U* comp_enum_size", e, r, ge=0.85,
                  note="point-judged (§9.2)",
                  method="comp_enum_size", plant=plant, n_clusters=len(a), descriptive=plant == "T2")
        if plant == "T1":
            _paired_diff(S, bt, "C-MERGE", "P4", "CF@U* (stratum merges)", a, tabs.get("copy_parent"), "cert", "cap",
                         0.10, "comp_enum_size", "copy_parent", "T1")


# ------------------------------------------------------------------------------------------------- C-GATE

PROBE_OF = {"robust": ("v2+clamp",), "lawcal": ("v2",), "par": ("v2",), "raw": ("v2",), "pi": ("v2",)}


def _robust_validity(S, bt, t, bar, cell, dc):
    """E_dec(robust) <= 1 % (upper <= 1.3 %) when powered; else E_ep(robust) <= 0.10 (one-sided upper)."""
    if t is None:
        S.bar("C-GATE", bar, "E_dec(robust)", np.nan, method="robust", status="MISSING")
        return "MISSING"
    pw = S.power_check("C-GATE", cell, "robust", "T2", t)
    e, r = bt.ratio(t, "uns", "inc")
    if pw["power_ok"]:
        return S.bar("C-GATE", bar, f"E_dec(robust, {dc}) <= 1 %", e, r, le=0.01, up_le=POWER_UP, method="robust",
                     cell=cell, plant="T2", n_clusters=len(t), power_row=pw)
    S.bar("C-GATE", bar, f"E_dec(robust, {dc}) <= 1 %", e, r, le=0.01, up_le=POWER_UP, method="robust", cell=cell,
          plant="T2", power_row=pw, note="E_ep decides (next row)")
    e, r = bt.ratio(t, "ep_unsafe", "one")
    return S.bar("C-GATE", bar, "E_ep(robust) <= 0.10 (E_dec underpowered)", e, r, up_le=0.10, method="robust",
                 cell=cell, plant="T2", n_clusters=len(t))


def score_gate(S, dec):
    d = dec[(dec.claim == "C-GATE") & dec.eid.between(CGATE_IDX[0], CGATE_IDX[1] - 1)].copy()
    if d.empty:
        for b in ("P5", "P6'"):
            S.bar("C-GATE", b, "", np.nan, status="MISSING")
        S.p["P5"] = S.p["P6'"] = np.nan
        return
    S.log.append(f"C-GATE: {d.eid.nunique()} episodes after exclusions (conf_test {CGATE_IDX[0]}-{CGATE_IDX[1] - 1}), "
                 f"severities {sorted(d.severity.unique())}")
    bt = Boot(np.sort(d.eid.unique()))       # paired across severities and certifiers (same episodes)
    prim = lambda x: (x.kind == "child") & (x.cand_size >= 3)                       # noqa: E731
    sec = lambda x: (x.kind == "child") & (x.cand_size <= 2)                        # noqa: E731
    dom = lambda flag: (lambda x, tau: stratum_mask(x, tau) & (x["dominated"].to_numpy(float) == flag))  # noqa
    T = {}
    for (m, s, pr), g in d.groupby(["method", "severity", "probe"]):
        for size, fsel in (("ge3", prim), ("le2", sec)):
            e = g[fsel(g)].reset_index(drop=True)
            if e.empty:
                continue
            for dc, sel in (("undominated", dom(0)), ("dominated", dom(1)), ("stratum", _sel("stratum"))):
                t = cluster_stats(e, e["lower"], sel)
                T[m, s, pr, size, dc] = t
                S.cell(claim="C-GATE", method=m, severity=s, probe=pr, size=size, dominance=dc, n_ep=len(t),
                       E_dec_own=C._ratio(t.uns.sum(), t.inc.sum()), CF_own=C._ratio(t.cert.sum(), t.cap.sum()),
                       E_ep=t.ep_unsafe.mean(), n_incapable=int(t.inc.sum()))
    # ---- bar 1 / P5: silent extrapolation at s = 3, v2, >= 3-module children
    ps, killed = [], True
    for m in ("raw", "par"):
        t = T.get((m, 3, "v2", "ge3", "undominated"))
        if t is None:
            S.bar("C-GATE", "P5", f"E_dec undominated ({m}) > 2 %", np.nan, method=m, status="MISSING")
            ps.append(np.nan)
            continue
        pw = S.power_check("C-GATE", "s3 v2 ge3 undominated", m, "T2", t)
        e, r = bt.ratio(t, "uns", "inc")
        killed &= bool(np.isfinite(e) and e <= 0.01)
        S.bar("C-GATE", "P5", f"E_dec undominated ({m}) > 2 %", e, r, ge=0.02 + 1e-12, lo_gt=0.01, method=m,
              cell="s=3 v2 >=3-module", plant="T2", n_clusters=len(t), status="HOLM",
              note=f"power: {'ok' if pw['power_ok'] else 'UNDERPOWERED'}")
        ps.append(p_greater(r, 0.01))
        td = T.get((m, 3, "v2", "ge3", "dominated"))
        if td is not None:
            pwd = S.power_check("C-GATE", "s3 v2 ge3 dominated", m, "T2", td)
            e, r = bt.ratio(td, "uns", "inc")
            if pwd["power_ok"]:
                S.bar("C-GATE", "1 dominated", f"E_dec dominated ({m}) <= 1 %", e, r, le=0.01, up_le=POWER_UP,
                      method=m, cell="s=3 v2 >=3-module", plant="T2", n_clusters=len(td), power_row=pwd)
            else:   # CONFIRMATORY §9 amendment 3: contrast-judged against the undominated excess threshold
                S.bar("C-GATE", "1 dominated", f"E_dec dominated ({m}) upper95 <= 2 %", e, r, up_le=0.02, method=m,
                      cell="s=3 v2 >=3-module", plant="T2", n_clusters=len(td),
                      note="dominated: UNDERPOWERED, contrast-judged (§9.3)")
    S.p["P5"] = max(ps) if ps and all(np.isfinite(ps)) else np.nan          # intersection-union: both must hold
    if killed:
        S.bar("C-GATE", "kill", "par and raw <= 1 % undominated at s=3", 0.0, status="KILLED",
              note="report 'extrapolation is safe under these misspecifications'")
    # ---- A-GATE (727f995): H13 and P6 withdrawn. robust = dominance + clamp (v2+clamp), lawcal = law-family
    # calibrated physics posterior (v2, margin fitted on s in {0,1,2}: s = 3 held out). gate: cells.csv only.
    def tab(m, s, dc, size="ge3"):
        for pr in PROBE_OF.get(m, ("v2",)) + ("v2", "v2+clamp"):
            if (m, s, pr, size, dc) in T:
                return T[m, s, pr, size, dc], pr
        return None, None

    # P6' (Holm): paired E_ep(par) - E_ep(robust) > 0 at s = 3, >=3-module children; AND robust undominated E_dec
    a, _ = tab("par", 3, "stratum")
    b_, prb = tab("robust", 3, "stratum")
    _paired_diff(S, bt, "C-GATE", "P6'", "E_ep (s=3, >=3-module)", a, b_, "ep_unsafe", "one", -np.inf, "par",
                 "robust", "T2", note="paired episode bootstrap; condition 2 in 'P6' robust undominated'")
    tu, _ = tab("robust", 3, "undominated")
    _robust_validity(S, bt, tu, "P6' robust undominated", "s=3 >=3-module undominated", "undominated")
    # validity of robust at every severity (not Holm), on the stratum
    for s_ in sorted(d.severity.unique()):
        t, pr = tab("robust", s_, "stratum")
        _robust_validity(S, bt, t, f"robust validity s={s_:g}", f"s={s_:g} >=3-module stratum ({pr})", "stratum")
    # price of robustness (descriptive, pre-stated): CF(par, v2) - CF(robust, v2+clamp) at s = 0 <= 10 pts (point)
    a, _ = tab("par", 0, "stratum")
    b_, _ = tab("robust", 0, "stratum")
    if a is not None and b_ is not None:
        ea, ra = bt.ratio(a, "cert", "cap")
        eb, rb = bt.ratio(b_, "cert", "cap")
        S.bar("C-GATE", "price of robustness", "CF(par, v2) - CF(robust, v2+clamp) at s=0", ea - eb, ra - rb,
              le=0.10, method="robust", cell="s=0 >=3-module stratum", plant="T2", n_clusters=len(bt.idx),
              note="descriptive, pre-stated; point-judged")
    else:
        S.bar("C-GATE", "price of robustness", "CF(par) - CF(robust) at s=0", np.nan, status="MISSING")
    # lawcal at the held-out severity (descriptive but pre-stated; a safety bar, so upper-bound-judged)
    t, _ = tab("lawcal", 3, "stratum")
    if t is None:
        S.bar("C-GATE", "lawcal held-out s=3", "E_dec(lawcal) <= 2 %", np.nan, method="lawcal", status="MISSING")
    else:
        e, r = bt.ratio(t, "uns", "inc")
        v = S.bar("C-GATE", "lawcal held-out s=3", "E_dec(lawcal) <= 2 %", e, r, up_le=0.02, method="lawcal",
                  cell="s=3 >=3-module stratum", plant="T2", n_clusters=len(t), note="descriptive, pre-stated")
        if v == "FAIL":
            S.rows[-1]["note"] += "; paper: calibrating over a law family protects only inside that family"
    # ---- dose response (descriptive): Spearman of E_dec(par, undominated) over s, paired bootstrap
    sev = sorted(s for s in d.severity.unique() if ("par", s, "v2", "ge3", "undominated") in T)
    if len(sev) >= 3:
        ests, reps = [], []
        for s in sev:
            e, r = bt.ratio(T["par", s, "v2", "ge3", "undominated"], "uns", "inc")
            ests.append(e)
            reps.append(r)
        sp = lambda v: pd.Series(v).rank().corr(pd.Series(sev).rank())          # noqa: E731
        R = np.vstack(reps).T
        S.bar("C-GATE", "dose response", "Spearman(E_dec par undominated, s)", sp(ests),
              np.array([sp(x) for x in R]), method="par", descriptive=True, note="pre-stated direction: >= 0")


# ------------------------------------------------------------------------------------------------- C-ENV

ENV_EXCLUDED = {("ID", 1499)}                            # A-ENV: constructed before scoring
ENVCAL = "amortized_comp_envcal"
ENV_CERTIFIERS = {                                       # A-ENV: same bars, different status of the bar
    OURS: "primary; empirical transfer bar, NOT a guarantee (T1-cal constant outside its calibration distribution)",
    ENVCAL: "secondary; env-calibrated on exploratory ID missions, guarantee on ID by exchangeability (A-ENV)",
}


def score_env(S, mis, ks):
    if ks is None or ks.empty:
        S.bar("C-ENV", "prerequisite", "HM3D statistics", np.nan, status="NOT SCORED",
              note="results/env/buildings_ks.csv missing: no confirmatory scoring without it (§4)")
        S.p["P7"] = np.nan
        return
    # env role's file (scripts/env_score.py side): stat, ks_p, ks_d [, n_hm3d, n_gen, family, passed]. The rule is
    # recomputed here from ks_p / ks_d (its `passed` column is informational); §4 also needs >= 500 generated buildings.
    ok_stat = (ks.ks_p > 0.05) | (ks.ks_d < 0.1)
    enough = bool((ks.n_gen >= 500).all()) if "n_gen" in ks else True
    ok = bool(ok_stat.all()) and enough
    S.bar("C-ENV", "prerequisite", "KS p > 0.05 or D < 0.1 for every statistic (>= 500 generated buildings)",
          float(ok), ge=1.0, note=f"{len(ks)} statistics" + ("" if enough else "; fewer than 500 generated buildings")
          + ("" if ok_stat.all() else "; failing: " + ", ".join(ks.stat[~ok_stat].astype(str))))
    if not ok:
        S.bar("C-ENV", "all bars", "", np.nan, status="NOT SCORED", note="prerequisite failed")
        S.p["P7"] = np.nan
        return
    m = mis[mis.claim == "C-ENV"].copy() if not mis.empty else mis
    if not m.empty and (m["shard"] < 0).any():
        S.log.append(f"C-ENV: {int((m['shard'] < 0).sum())} mission rows carry shard = -1 (env_score: no shard column "
                     "in the runner rows), so the canary-shard exclusion cannot apply to them")
    if m.empty:
        S.bar("C-ENV", "1 validity", "E_mis", np.nan, status="MISSING")
        S.p["P7"] = np.nan
        return
    # A-ENV (0dc1aec): conf-ID mission 1499 was constructed before scoring -> excluded from every cell
    exc = np.zeros(len(m), bool)
    for fam, mi in ENV_EXCLUDED:
        exc |= (m.family == fam).to_numpy() & (m.mission == mi).to_numpy()
    S.log.append(f"C-ENV: excluded {int(exc.sum())} rows of conf-ID mission 1499 (A-ENV); ID = missions 0-1,498")
    m = m[~exc].copy()
    m["fail_c"] = (m.committed > 0.5) & (m.failed > 0.5)
    for meth, note in ENV_CERTIFIERS.items():
        for fam in sorted(m.family.unique()):
            e = m[(m.method == meth) & (m.family == fam)]
            if e.empty:
                if fam == "ID":
                    S.bar("C-ENV", "1 validity", "E_mis", np.nan, method=meth, cell=fam, status="MISSING")
                continue
            cs = e.groupby("building").agg(num=("fail_c", "sum"), den=("committed", "sum"))
            bt = Boot(cs.index)
            est, reps = bt.ratio(cs, "num", "den")
            if fam == "ID":
                S.bar("C-ENV", "1 validity", "E_mis", est, reps, le=0.10, up_le=0.13, method=meth, cell=fam,
                      n_clusters=len(cs), note=note)
            else:
                S.bar("C-ENV", "1 validity", "E_mis", est, reps, up_le=0.15, method=meth, cell=fam,
                      n_clusters=len(cs), note=note)
    # P7: ID success among oracle-successful missions, paired by building
    idm = m[(m.family == "ID") & (m.oracle_success > 0.5)]
    bt = Boot(np.sort(idm.building.unique()))
    tab = {meth: g.groupby("building").agg(s=("success", "sum"), n=("success", "size"))
           for meth, g in idm.groupby("method")}
    base = [b for b in ("nominal", "biggest_body", "greedy_feasible") if b in tab]
    if OURS not in tab or not base:
        S.bar("C-ENV", "P7", "success - best baseline", np.nan, status="MISSING")
        S.p["P7"] = np.nan
    else:
        best = max(base, key=lambda b: tab[b].s.sum() / tab[b].n.sum())
        _paired_diff(S, bt, "C-ENV", "P7", "ID success (oracle-successful missions)", tab[OURS], tab[best], "s", "n",
                     0.05, OURS, best, "")
    # price of the T1 constant (A-ENV, descriptive): useful commits, envcal - amortized_comp, ID, paired by building
    if ENVCAL in tab and OURS in tab:
        ea, ra = bt.ratio(tab[ENVCAL], "s", "n")
        eb, rb = bt.ratio(tab[OURS], "s", "n")
        S.bar("C-ENV", "price of the T1 constant", f"useful commits: {ENVCAL} - {OURS} (ID)", ea - eb, ra - rb,
              method=ENVCAL, cell="ID", descriptive=True, note="A-ENV, descriptive")
    # generalisation (descriptive) and reconfiguration necessity (descriptive)
    for fam in sorted(f for f in m.family.unique() if f != "ID"):
        for meth in (OURS, "raw"):
            a = m[(m.method == meth) & (m.oracle_success > 0.5)]
            if a.empty:
                continue
            si, sf = a[a.family == "ID"].success.mean(), a[a.family == fam].success.mean()
            S.bar("C-ENV", "3 generalisation", f"success drop ID -> {fam}", si - sf, le=0.10 if meth == OURS else None,
                  method=meth, cell=fam, descriptive=meth != OURS,
                  note="raw = plug-in median, not a bound (A-ENV)" if meth == "raw" else "")
    o = m[(m.method == "oracle") & (m.family == "ID") & (m.oracle_success > 0.5)]
    if len(o) and o.reconfig.notna().any():
        S.bar("C-ENV", "4 reconfiguration", "share of oracle plans that merge/split", o.reconfig.mean(), ge=0.20,
              method="oracle", cell="ID", descriptive=True)


# ------------------------------------------------------------------------------------------------- C-SCALE

def score_scale(S, dec):
    d = dec[(dec.claim == "C-SCALE") & dec.split.isin(["conf_H4", "conf_H4b"])].copy()
    if d.empty:
        S.bar("C-SCALE", "1 validity", "", np.nan, status="MISSING")
        return
    d["bin"] = np.nan
    for i, (a, b) in enumerate(SCALE_BINS):
        d.loc[d.parent_n.between(a, b), "bin"] = i
    d = d[d.bin.notna()]
    d["ck"] = d.split.astype(str) + ":" + d.eid.astype(str)          # conf_H4 and conf_H4b indices both start at 0
    eps = d.drop_duplicates("ck")[["ck", "bin"]].sort_values("ck")
    bt = Boot(eps.ck.to_numpy(), strata=eps.bin.to_numpy())
    direct = next((x for x in DIRECT if x in set(d.method)), None)
    covs = {}
    for meth in (OURS, direct):
        if meth is None:
            continue
        for i, (a, b) in enumerate(SCALE_BINS):
            e = d[(d.method == meth) & (d.bin == i)].reset_index(drop=True)
            if e.empty:
                continue
            lm = e["lower_marg"] if e.lower_marg.notna().any() else e["lower"]
            cv = cluster_cov(e, lm, sel=lambda x: C.rel_mask(x), cluster="ck")
            covs[meth, i] = cv
            if meth == OURS:
                cs = cluster_stats(e, e["lower"], cluster="ck")
                est, reps = bt.ratio(cs, "ep_unsafe", "one")
                S.bar("C-SCALE", "1 validity", "E_ep", est, reps, le=0.15, up_le=0.18, method=meth,
                      cell=f"N {a}-{b}", n_clusters=len(cs))
                est, reps = bt.ratio(cv, "cov", "n")
                S.bar("C-SCALE", "1 validity", "stratum coverage", est, reps, ge=0.85, method=meth, cell=f"N {a}-{b}",
                      note="point-judged (§9.2)",
                      n_clusters=len(cv))
    # bar 2: coverage slope over bin index, comp - direct, paired (secondary, not in Holm)
    if direct and all((m_, i) in covs for m_ in (OURS, direct) for i in range(len(SCALE_BINS))):
        x = np.arange(len(SCALE_BINS), dtype=float)

        def slope(meth):
            e, r = zip(*(bt.ratio(covs[meth, i], "cov", "n") for i in range(len(SCALE_BINS))))
            e, r = np.array(e), np.vstack(r).T
            f = lambda y: np.polyfit(x, y, 1)[0]                                    # noqa: E731
            return f(e), np.array([f(y) if np.isfinite(y).all() else np.nan for y in r])
        ea, ra = slope(OURS)
        eb, rb = slope(direct)
        S.bar("C-SCALE", "2 comp vs direct", "coverage slope comp - direct", ea - eb, ra - rb, lo_gt=0.0,
              method=OURS, cell="bins", note=f"direct = {direct}; secondary (not in Holm)")
    else:
        S.bar("C-SCALE", "2 comp vs direct", "coverage slope comp - direct", np.nan, status="MISSING")
    S.bar("C-SCALE", "3 extrapolated share", "closed-form share vs N", np.nan, status="NOT SCORED HERE",
          note="theory role, deterministic T0 (not a stochastic claim)")


# ------------------------------------------------------------------------------------------------- C-HET

def score_het(S, dec):
    S.bar("C-HET", "sanity gate", "conventional module |rho_sway| < 0.02", np.nan, status="SEE TEST",
          note="generator unit test test_conventional_module_cannot_sway, run before generation")
    d = dec[(dec.claim == "C-HET") & dec.split.isin(["conf_het_test", "conf_H1", "conf_H2"])].copy()
    if d.empty:
        S.bar("C-HET", "1 validity", "", np.nan, status="MISSING")
        return
    eps = d.drop_duplicates(["split", "eid"])
    keys = (eps.split + ":" + eps.eid.astype(str)).to_numpy()
    d["ck"] = d.split + ":" + d.eid.astype(str)
    bt = Boot(np.sort(keys), strata=np.array([k.split(":")[0] for k in np.sort(keys)]))
    direct = next((x for x in DIRECT if x in set(d.method)), None)
    # A-HET (727f995): H1 tests new FAULT behaviour of an unseen wheel type, so its bars use the faulted-candidate
    # block (decisions of candidates holding a faulted module); the all-decisions block is descriptive. H2 unchanged.
    JUDGE = {"conf_H1": "faulted", "conf_H2": "all"}
    res = {}
    for meth in (OURS, "amortized_comp_blind", direct, "metadata_nf", "raw"):
        if meth is None:
            continue
        for sp in ("conf_het_test", "conf_H1", "conf_H2"):
            e0 = d[(d.method == meth) & (d.split == sp)].reset_index(drop=True)
            if e0.empty:
                continue
            for blk in ("all", "faulted"):
                e = e0 if blk == "all" else e0[e0.has_fault > 0.5].reset_index(drop=True)
                if e.empty:
                    continue
                cs = cluster_stats(e, e["lower"], _sel("stratum"), cluster="ck")
                cv = cluster_cov(e, e["lower"], sel=lambda x: C.rel_mask(x), cluster="ck")
                res[meth, sp, blk] = (cs, cv)
                S.cell(claim="C-HET", method=meth, split=sp, block=blk, n_ep=len(cs), E_ep=cs.ep_unsafe.mean(),
                       CF_own=C._ratio(cs.cert.sum(), cs.cap.sum()), cov=C._ratio(cv["cov"].sum(), cv["n"].sum()))
    for meth in (OURS, "amortized_comp_blind"):
        for sp in ("conf_H1", "conf_H2"):
            for blk in ("faulted", "all") if sp == "conf_H1" else ("all",):
                judged = blk == JUDGE[sp]
                if (meth, sp, blk) not in res:
                    S.bar("C-HET", "1 validity", f"E_ep ({blk})", np.nan, method=meth, cell=sp, status="MISSING")
                    continue
                cs = res[meth, sp, blk][0]
                est, reps = bt.ratio(cs, "ep_unsafe", "one")
                note = "" if meth == OURS else "type withheld (interpretation, §6.3)"
                if sp == "conf_H1":
                    note = (note + "; " if note else "") + ("faulted-candidate block (A-HET)" if judged
                                                            else "all-decisions block: descriptive (A-HET)")
                S.bar("C-HET", "1 validity", f"E_ep ({blk})", est, reps, le=0.15, up_le=0.18, method=meth, cell=sp,
                      n_clusters=len(cs), descriptive=(meth != OURS) or not judged, note=note)
    # bar 2: degradation ID -> H1/H2 <= 5 pts (point, §9.2) and smaller than direct's (secondary); same block in ID
    for sp in ("conf_H1", "conf_H2"):
        blk = JUDGE[sp]
        for metric, which, num, den in (("coverage", 1, "cov", "n"), ("CF", 0, "cert", "cap")):
            if (OURS, "conf_het_test", blk) not in res or (OURS, sp, blk) not in res:
                S.bar("C-HET", "2 degradation", f"{metric} drop ID -> {sp}", np.nan, status="MISSING")
                continue
            ei, ri = bt.ratio(res[OURS, "conf_het_test", blk][which], num, den)
            eh, rh = bt.ratio(res[OURS, sp, blk][which], num, den)
            S.bar("C-HET", "2 degradation", f"{metric} drop ID -> {sp} ({blk})", ei - eh, ri - rh, le=0.05,
                  method=OURS, note="ID - split, point-judged (§9.2)" + ("; faulted block (A-HET)"
                                                                          if blk == "faulted" else ""), cell=sp)
            if direct and (direct, "conf_het_test", blk) in res and (direct, sp, blk) in res:
                di, rdi = bt.ratio(res[direct, "conf_het_test", blk][which], num, den)
                dh, rdh = bt.ratio(res[direct, sp, blk][which], num, den)
                S.bar("C-HET", "2 degradation", f"{metric} drop: direct - ours ({sp}, {blk})", (di - dh) - (ei - eh),
                      (rdi - rdh) - (ri - rh), lo_gt=0.0, method=OURS, cell=sp, note="secondary")
    told = [r for r in S.rows if r["claim"] == "C-HET" and r["bar"] == "1 validity" and r["method"] == OURS
            and r["verdict"] != "DESCRIPTIVE"]
    blind = [r for r in S.rows if r["claim"] == "C-HET" and r["bar"] == "1 validity"
             and r["method"] == "amortized_comp_blind" and "descriptive (A-HET)" not in r["note"]]
    if told and blind:
        tp = all(r["verdict"] == "PASS" for r in told)
        bp = all(r["estimate"] <= 0.15 and r["upper95"] <= 0.18 for r in blind if np.isfinite(r["estimate"]))
        if tp and not bp:
            S.bar("C-HET", "3 interpretation", "passes only when told the type", 1.0, descriptive=True,
                  note="claim: composition transfers to new module types whose wheel channel model is known")


# ------------------------------------------------------------------------------------------------- Holm + report

def holm(S):
    ps = {k: S.p.get(k, np.nan) for k in HOLM}
    order = sorted((p, k) for k, p in ps.items() if np.isfinite(p))
    m = len(HOLM)
    rej, stop = {}, False
    for i, (p, k) in enumerate(order):
        lvl = ALPHA_FWER / (m - i)
        rej[k] = (not stop) and p <= lvl
        stop |= not rej[k]
    table = []
    for k in HOLM:
        table.append(dict(P=k, p=ps[k], holm_reject=bool(rej.get(k, False)),
                          level=next((ALPHA_FWER / (m - i) for i, (_, kk) in enumerate(order) if kk == k), np.nan)))
    # resolve HOLM-status rows: PASS iff Holm rejects AND every point bar of that row holds
    for r in S.rows:
        if r["verdict"] != "HOLM":
            continue
        pid = r["bar"]
        thr_ok = True
        est = r["estimate"]
        for part in r["threshold"].split("; "):
            if part.startswith(">= "):
                thr_ok &= bool(np.isfinite(est) and est >= float(part[3:]))
            if part.startswith("lower95 > "):
                thr_ok &= bool(np.isfinite(r["lower95"]) and r["lower95"] > float(part[10:]))
        r["holm_p"] = ps.get(pid, np.nan)
        r["verdict"] = "PASS" if (rej.get(pid, False) and thr_ok) else "FAIL"
        r["note"] = (r["note"] + "; " if r["note"] else "") + f"Holm {'rejects' if rej.get(pid) else 'does not reject'}"
    return pd.DataFrame(table)


def h12(S):
    """C-GATE bar 1 (H12) overall: both undominated excess rows (P5, Holm-resolved) and both dominated rows pass."""
    und = [r for r in S.rows if r["claim"] == "C-GATE" and r["bar"] == "P5"]
    dom = [r for r in S.rows if r["claim"] == "C-GATE" and r["bar"] == "1 dominated"]
    if not und:
        return
    if any(r["verdict"] == "MISSING" for r in und) or len(dom) < 2:
        v = "MISSING"
    else:
        v = "PASS" if all(r["verdict"] == "PASS" for r in und + dom) else "FAIL"
    S.rows.append(dict(claim="C-GATE", bar="1 H12 overall", metric="undominated excess (P5) and dominated condition",
                       method="raw, par", cell="s=3 v2 >=3-module", plant="T2", estimate=np.nan, ci_lo=np.nan,
                       ci_hi=np.nan, upper95=np.nan, lower95=np.nan, threshold="all component rows PASS",
                       n_clusters=np.nan, n_incapable=np.nan, deff=np.nan, power_ok=np.nan, verdict=v,
                       note="dominated rows: " + ", ".join(f"{r['method']} {r['verdict']}" for r in dom)))


def p6_overall(S):
    """A-GATE P6' = Holm-resolved paired E_ep difference AND robust's undominated validity (E_dec if powered, else
    the E_ep row that replaces it)."""
    hd = [r for r in S.rows if r["claim"] == "C-GATE" and r["bar"] == "P6'"]
    cond = [r for r in S.rows if r["claim"] == "C-GATE" and r["bar"] == "P6' robust undominated"]
    if not hd:
        return
    decisive = [r for r in cond if r["verdict"] != "UNDERPOWERED"]
    if any(r["verdict"] == "MISSING" for r in hd + cond) or not decisive:
        v = "MISSING"
    else:
        v = "PASS" if all(r["verdict"] == "PASS" for r in hd + decisive) else "FAIL"
    S.rows.append(dict(claim="C-GATE", bar="P6' overall", metric="E_ep(par) - E_ep(robust) > 0 (Holm) AND robust "
                       "undominated validity", method="robust", cell="s=3 >=3-module", plant="T2", estimate=np.nan,
                       ci_lo=np.nan, ci_hi=np.nan, upper95=np.nan, lower95=np.nan, threshold="component rows PASS",
                       n_clusters=np.nan, n_incapable=np.nan, deff=np.nan, power_ok=np.nan, verdict=v,
                       note="; ".join(f"{r['metric']}: {r['verdict']}" for r in hd + decisive)))


def conf_sha():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", "docs/CONFIRMATORY.md"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:            # noqa: BLE001
        return "unknown"


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100 * x:.2f} %" if pct else f"{x:.3g}"


def report(S, H, out, sha, srcs, mock):
    V = pd.DataFrame(S.rows)
    V["conf_sha"] = sha
    out.mkdir(parents=True, exist_ok=True)
    V.to_csv(out / "verdicts.csv", index=False)
    pd.DataFrame(S.cells).to_csv(out / "cells.csv", index=False)
    P = pd.DataFrame(S.powers)
    L = [f"# Confirmatory results{' (MOCK DATA - never cite)' if mock else ''}", "",
         f"Scored by `scripts/analysis/confirmatory.py` against `docs/CONFIRMATORY.md` @ `{sha}` on "
         f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%MZ}. Inputs: {', '.join(srcs) or 'none'}.", "",
         "## Exclusions (§0.1)", ""] + [f"- {x}" for x in S.log] + ["", "## Power checks (§0.4), before any verdict", "",
         "| claim | cell | method | plant | incapable stratum decisions | deff | upper bound at 1 % | powered |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in P.iterrows():
        L.append(f"| {r.claim} | {r.cell} | {r.method} | {r.plant} | {r.n_incapable:,} | {fmt(r.deff, False)} | "
                 f"{fmt(r.up_at_1pct)} | {'yes' if r.power_ok else '**UNDERPOWERED**'} |")
    L += ["", "## Verdicts", "", "| claim | bar | metric | method | cell | plant | estimate | 95 % CI | one-sided "
          "upper / lower | threshold | verdict | note |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in V.iterrows():
        L.append(f"| {r.claim} | {r.bar} | {r.metric} | {r.method} | {r.cell} | {r.plant} | {fmt(r.estimate)} | "
                 f"[{fmt(r.ci_lo)}, {fmt(r.ci_hi)}] | {fmt(r.upper95)} / {fmt(r.lower95)} | {r.threshold} | "
                 f"**{r.verdict}** | {r.note} |")
    L += ["", "## Holm table, P1–P7 (family-wise 0.05, m = 7)", "", "| P | one-sided p | Holm level | rejects |",
          "|---|---|---|---|"]
    for _, r in H.iterrows():
        L.append(f"| {r.P} | {fmt(r.p, False)} | {fmt(r.level, False)} | {'yes' if r.holm_reject else 'no'} |")
    (out / "CONFIRMATORY_RESULTS.md").write_text("\n".join(L) + "\n")
    return V


def split_seeds(d, S, what):
    """A-SEED: keep model rows of PRIMARY_TAG for scoring; return (scored rows, seed-variation rows). Untagged model rows
    are accepted only if the file carries no tags at all for that claim (logged); mixing tagged and untagged raises."""
    if d.empty:
        return d, d.iloc[:0]
    mm = d["method"].isin(MODEL_METHODS) & d["claim"].isin(SEED_CLAIMS)
    tag = d["model_tag"].astype(object)
    has = tag.notna()
    for cl in SEED_CLAIMS:
        c = mm & (d["claim"] == cl)
        if c.any() and has[c].any() and (~has[c]).any():
            raise C.SchemaError(f"{what}: {cl} model rows mix tagged and untagged model_tag (A-SEED needs the tag)")
        if c.any() and not has[c].any():
            S.log.append(f"{what}: {cl} model rows carry no model_tag; assumed {PRIMARY_TAG} (A-SEED) - check the export")
    other = mm & has & (tag.astype(str) != PRIMARY_TAG)
    if other.any():
        S.log.append(f"{what}: {int(other.sum())} model rows of tag(s) {sorted(set(tag[other].astype(str)))} moved to "
                     "the descriptive seed-variation column (A-SEED; never in a bar)")
    return d[~other], d[other]


def seed_variation(S, sv):
    """Descriptive cells + DESCRIPTIVE rows for non-primary model tags (A-SEED): own-bound E_ep and stratum CF/E_dec."""
    if sv.empty:
        return
    for (cl, m, tg, pl), g in sv.groupby(["claim", "method", "model_tag", "test_plant"]):
        g = g.reset_index(drop=True)
        if cl == "C-MAIN":
            g = g[g.eid.between(CMAIN_IDX[0], CMAIN_IDX[1] - 1)].reset_index(drop=True)
        if g.empty:
            continue
        key = "ck"
        g[key] = g["split"].astype(str) + ":" + g["eid"].astype(str)
        t = cluster_stats(g, g["lower"], _sel("stratum"), cluster=key)
        bt = Boot(t.index)
        e, rr = bt.ratio(t, "ep_unsafe", "one")
        S.cell(claim=cl, method=m, model_tag=tg, plant=pl, block="seed variation", n_ep=len(t), E_ep=e,
               CF_own=C._ratio(t.cert.sum(), t.cap.sum()), E_dec_own=C._ratio(t.uns.sum(), t.inc.sum()))
        S.bar(cl, "seed variation (A-SEED)", f"E_ep, {tg}", e, rr, method=m, plant=pl, n_clusters=len(t),
              descriptive=True, note=f"descriptive only; the bars use {PRIMARY_TAG}; no best-of-seeds")


SCORER_TRIVIAL = ("always_go", "never_go", "metadata_nf")   # computed here (coordinator 00:35Z), never read from rows


def add_trivial(d, S, what):
    """C-MAIN constant and shape-metadata baselines computed by the scorer from the shared labels of OURS's rows
    (common.trivial_methods: always_go finite huge bound, never_go -inf, metadata_nf 2-fold cross-fit over episodes
    within each (plant, split) block). Role-exported rows of these names are dropped (logged)."""
    if d.empty:
        return d
    ex = d["method"].isin(SCORER_TRIVIAL) & (d["claim"] == "C-MAIN")
    if ex.any():
        S.log.append(f"{what}: dropped {int(ex.sum())} role-exported rows of {sorted(set(d.method[ex]))}; the scorer "
                     "computes them")
        d = d[~ex]
    base = d[(d.claim == "C-MAIN") & (d.method == OURS)]
    if base.empty:
        return d
    t = C.trivial_methods(base)
    t = t[t.method.isin(SCORER_TRIVIAL)].copy()
    return pd.concat([d, t[d.columns.intersection(t.columns)]], ignore_index=True)


def run(res, out, mock=False, now=None, truth_fn=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    if not mock and not (res / "MOCK.txt").exists() and now < SCORING_OPENS:
        raise SystemExit(f"refusing: the confirmatory scoring window opens {SCORING_OPENS:%Y-%m-%d %H:%MZ} "
                         "(docs/CONFIRMATORY.md §0.1); use --mock for a dry run")
    dec, s1 = _load("*/confirmatory/dec_*.csv*", DEC_REQ, DEC_OPT, res)
    mis, s2 = _load("*/confirmatory/missions_*.csv*", MIS_REQ, MIS_OPT, res)
    ksf = res / "env/buildings_ks.csv"
    ks = None
    if ksf.exists():
        ks = pd.read_csv(ksf)
        if any(c not in ks for c in KS_REQ):
            raise C.SchemaError(f"{ksf}: needs {KS_REQ}")
    S = Scorer()
    dec, sv_dec = split_seeds(dec, S, "decision rows")
    if not mis.empty:
        mis, sv_mis = split_seeds(mis.assign(test_plant="env"), S, "mission rows")
    cal = add_trivial(dec[dec.split.astype(str) == "cal"], S, "cal rows")
    dec = add_trivial(exclusions(dec[dec.split.astype(str) != "cal"], S.log, "decision rows"), S, "decision rows")
    mis = exclusions(mis.assign(split="missions", eid=mis.get("mission", pd.Series(dtype=float))), S.log,
                     "mission rows") if not mis.empty else mis
    seed_variation(S, exclusions(sv_dec[sv_dec.split.astype(str) != "cal"], [], "seed variation")
                   if not sv_dec.empty else sv_dec)
    score_main(S, dec, cal)
    score_merge(S, dec, cal)
    score_gate(S, dec)
    score_env(S, mis, ks)
    score_scale(S, dec)
    score_het(S, dec)
    score_explain(S, dec, res, truth_fn)
    H = holm(S)
    h12(S)
    p6_overall(S)
    V = report(S, H, out, conf_sha(), s1 + s2 + ([C.rel(ksf)] if ks is not None else []), mock or
               (res / "MOCK.txt").exists())
    return V, H, S


# ------------------------------------------------------------------------------------------------- mock data

def cfbh_rows(base, rng):
    """Baselines' encoding of conformal selection: lower = the largest tau at which the entry was selected (FDR 0.1),
    -inf if never selected; no point/scale. Mock selection: selected at tau iff a noisy score clears tau + 0.01."""
    x = base.copy()
    score = x.truth + 0.02 * rng.standard_normal(len(x))
    lo = np.full(len(x), -np.inf)
    for tau in TAUS:
        lo = np.where(score >= tau + 0.01, tau, lo)
    x["method"], x["point"], x["scale"], x["lower"], x["seconds"] = "cfBH", np.nan, np.nan, lo, 0.01
    return x


def write_mock(out, good=True, seed=0, n_main=160, n_gate=120, n_pairs=120, knee_bites=True, ks_ok=True):
    """Contract-conforming mock inputs. good=True: every bar passes by construction (well calibrated 'ours', weak
    baselines, a raw posterior that is unsafe in T2, a knee that bites raw/par on undominated legs, a gate that
    stays safe). good=False: 'ours' is miscalibrated (unsafe)."""
    rng = np.random.default_rng(seed)
    out = pathlib.Path(out)
    (out / "MOCK.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "MOCK.txt").write_text("mock confirmatory inputs; never cite\n")

    def body(claim, split, eids, plant, n_c=8, extra=None):
        rows = []
        for e in eids:
            for c in range(n_c):
                kind = "self" if c == 0 else ("merge" if claim == "C-MERGE" else "child")
                size = 1 if c in (1, 2, 3) else (2 if c in (4, 5) else 3)
                hf = int(c in (1, 4, 6))
                for leg in range(10):
                    truth = float(rng.uniform(0.55, 1.0) - (0.35 * rng.random() if hf else 0.0))
                    r = dict(claim=claim, test_plant=plant, split=split, status="confirmatory", eid=e,
                             shard=e // 50, cand=c, kind=kind, cand_size=size, has_fault=hf, leg=leg, truth=truth,
                             truth_own=truth, dominated=int(leg < 4))
                    if extra:
                        r.update(extra(e, c, leg))
                    rows.append(r)
        return pd.DataFrame(rows)

    def methods(base, spec):
        out_ = []
        for m, (bias, sd, q) in spec.items():
            b = base.copy()
            noise = rng.standard_normal(len(b))
            b["method"] = m
            b["point"] = b.truth + bias + sd * noise
            b["scale"] = max(sd, 1e-3)
            b["lower"] = b.point - q
            b["seconds"] = 0.01
            out_.append(b)
        return pd.concat(out_, ignore_index=True)

    ours_q = 0.08 if good else -0.05
    main_spec = {OURS: (0.0, 0.02, ours_q), "raw": (0.0, 0.02, 0.0), "capxfer": (-0.15, 0.05, 0.10),
                 "metadata_nf": (-0.05, 0.15, 0.25), "detector_gated": (-0.10, 0.10, 0.20),
                 "ghat_physics": (-0.05, 0.08, 0.15), "copy_parent": (-0.10, 0.15, 0.25),
                 "min_over_members": (-0.10, 0.10, 0.2), "dominance_pi": (-0.08, 0.05, 0.10),
                 "oracle": (0.0, 0.0, 0.0), "nuisance_smc": (0.0, 0.02, 0.06)}
    frames = []
    test = range(0, 40 + n_main)                    # 0-39 are excluded; shard 0 (eids 0-49) is the canary
    for plant in ("T1", "T2"):
        base = body("C-MAIN", "conf_test", test, plant)
        spec = dict(main_spec)
        if plant == "T2":
            spec["raw"] = (0.06, 0.02, 0.0)          # misspecified plant: raw posterior over-predicts
            spec[OURS] = (0.01, 0.02, ours_q + 0.02)
        frames.append(methods(base, spec))
        frames.append(cfbh_rows(base, rng))
    cal = body("C-MAIN", "cal", range(100000, 100000 + n_main), "T1")
    cal["status"] = "exploratory"
    frames.append(methods(cal, main_spec))
    # C-MERGE
    for plant in ("T1", "T2"):
        for arm in ("P0", "P5"):
            base = body("C-MERGE", "conf_merge", range(50, 50 + n_pairs), plant, n_c=6,
                        extra=lambda e, c, l, a=arm: dict(arm=a))
            frames.append(methods(base, {"comp_enum_size": (0.0, 0.02, 0.07), "copy_parent": (-0.1, 0.15, 0.25),
                                          "comp_oracle": (0.0, 0.001, 0.0), "comp_enum": (0.0, 0.02, 0.05)}))
    mc = body("C-MERGE", "cal", range(200000, 200000 + n_pairs), "T1", n_c=6, extra=lambda e, c, l: dict(arm="P0"))
    mc["status"] = "exploratory"
    frames.append(methods(mc, {"comp_enum_size": (0.0, 0.02, 0.07), "copy_parent": (-0.1, 0.15, 0.25)}))
    # C-GATE: same episodes at every severity; raw/par over-predict on undominated legs as s grows
    gb = body("C-GATE", "conf_test", range(CGATE_IDX[0], CGATE_IDX[0] + n_gate), "T2", n_c=8)
    for s in (0, 1, 2, 3):
        for probe in ("v2", "v2+clamp"):
            b = gb.copy()
            b["severity"], b["probe"] = s, probe
            und = b.dominated.to_numpy() == 0
            for m, q in (("raw", 0.0), ("par", 0.02), ("pi", 0.15), ("gate", 0.05), ("detector_gated", 0.2),
                         ("robust", 0.04), ("lawcal", 0.03)):
                if m == "robust" and probe != "v2+clamp" or m == "lawcal" and probe != "v2":
                    continue
                x = b.copy()
                x["method"] = m
                bias = (0.05 * s * und) if m in ("raw", "par") and knee_bites else 0.0
                bias = bias + (0.012 * s * und if m == "lawcal" else 0.0)
                x["point"] = x.truth + bias + 0.01 * rng.standard_normal(len(x))
                x["scale"] = 0.01
                x["lower"] = x.point - q - ((0.05 * s * und) if m == "gate" else 0.0)
                frames.append(x)
    # C-SCALE
    sb = []
    for i, (a, b_) in enumerate(SCALE_BINS):
        sb.append(body("C-SCALE", "conf_H4" if a < 11 else "conf_H4b", range(1000 * (i + 1), 1000 * (i + 1) + 60),
                       "T1", extra=lambda e, c, l, n=a: dict(parent_n=n)))
    sb = pd.concat(sb, ignore_index=True)
    sm = methods(sb, {OURS: (0.0, 0.02, 0.07), "blackbox_gnn": (0.0, 0.02, 0.05)})
    big = (sm.method == "blackbox_gnn") & (sm.parent_n >= 11)
    sm.loc[big, "lower"] += 0.05 * (sm.loc[big, "parent_n"] - 10) / 4        # direct head degrades with size
    frames.append(sm)
    # C-HET
    for sp in ("conf_het_test", "conf_H1", "conf_H2"):
        hb = body("C-HET", sp, range(60, 160), "T1")
        frames.append(methods(hb, {OURS: (0.0, 0.02, 0.07), "amortized_comp_blind": (0.0, 0.02, 0.07),
                                    "blackbox_gnn": (0.0, 0.02, 0.05 if sp == "conf_het_test" else 0.0),
                                    "metadata_nf": (-0.05, 0.15, 0.25), "raw": (0.0, 0.02, 0.0)}))
    D = pd.concat(frames, ignore_index=True)
    for role, claims in (("models", ("C-MAIN", "C-SCALE", "C-HET")), ("stress", ("C-MERGE", "C-GATE"))):
        p = out / role / "confirmatory"
        p.mkdir(parents=True, exist_ok=True)
        D[D.claim.isin(claims)].to_csv(p / f"dec_conf_{role}.csv.gz", index=False)
    # C-ENV
    rows = []
    for fam, nb in (("ID", 60), ("E-H1", 30), ("E-H2", 30)):
        for b in range(nb):
            for mi in range(5):
                os_ = int(rng.random() < 0.8)
                for meth, ps, pf in ((OURS, 0.9, 0.03), ("raw", 0.85, 0.12), ("nominal", 0.7, 0.2),
                                     ("biggest_body", 0.6, 0.1), ("greedy_feasible", 0.72, 0.15),
                                     ("oracle", 1.0, 0.0)):
                    succ = int(os_ and rng.random() < ps)
                    rows.append(dict(claim="C-ENV", method=meth, family=fam, building=f"{fam}-{b}", mission=mi,
                                     shard=1 + b % 3, committed=1, failed=int(rng.random() < pf), success=succ,
                                     oracle_success=os_, reconfig=int(rng.random() < 0.4), time_s=100.0,
                                     status="confirmatory"))
    p = out / "env" / "confirmatory"
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p / "missions_conf_env.csv", index=False)
    pd.DataFrame(dict(stat=["route_len", "steps", "passages", "width", "sites"], ks_p=[0.3, 0.2, 0.5, 0.06, 0.01],
                      ks_d=[0.05, 0.06, 0.04, 0.08, 0.09 if ks_ok else 0.2])).to_csv(out / "env/buildings_ks.csv", index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(C.RES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--mock", default=None, help="write mock inputs to this dir and score them there")
    ap.add_argument("--bad", action="store_true", help="mock: miscalibrated 'ours'")
    a = ap.parse_args(argv)
    if a.mock:
        res = write_mock(pathlib.Path(a.mock), good=not a.bad)
        out = pathlib.Path(a.out) if a.out else res / "confirmatory_out"
        V, H, _ = run(res, out, mock=True)
    else:
        res = pathlib.Path(a.results)
        out = pathlib.Path(a.out) if a.out else OUT
        if out.resolve() == OUT.resolve() and (res / "MOCK.txt").exists():
            raise SystemExit("refusing: mock inputs would write results/confirmatory/")
        V, H, _ = run(res, out)
    print((out / "CONFIRMATORY_RESULTS.md").read_text())
    return V, H


if __name__ == "__main__":
    main()
