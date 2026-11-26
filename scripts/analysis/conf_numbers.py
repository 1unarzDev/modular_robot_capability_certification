"""Confirmatory run-1 numbers for the paper (macros \\cfConf...), read from the scored files only.

Sources: results/confirmatory/verdicts.csv and cells.csv (run 77fc39b, scorer fb51ccb, scored once 03:17Z 11-26) and
the post-hoc pairing correction results/confirmatory/posthoc_p1p2/p1p2_paired.csv (reported as post-hoc). Nothing is
re-scored here; the script only formats numbers. Bar rows carry status 'confirmatory'; descriptive cells of the same
run carry 'confirmatory-descriptive'; the post-hoc rows 'posthoc'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

V = "results/confirmatory/verdicts.csv"
CELLS = "results/confirmatory/cells.csv"
PH = "results/confirmatory/posthoc_p1p2/p1p2_paired.csv"


def _p(N, key, x, src, st, small=False):
    d = 2 if (small or (x is not None and np.isfinite(x) and abs(x) < 0.01)) else 1
    N.pct(key, x, src, st, d=d)


def _pts(N, key, x, src, st):
    if x is None or not np.isfinite(x):
        return N.ph(key, f"no value in {src}")
    N.set(key, f"{100 * x:.1f}", src, st)


def run(res, paper):
    N = C.Numbers("conf_numbers.py")
    vp, cp, pp = res / "confirmatory/verdicts.csv", res / "confirmatory/cells.csv", res / "confirmatory/posthoc_p1p2/p1p2_paired.csv"
    if not vp.exists():
        N.ph("conf.missing", "results/confirmatory/verdicts.csv missing")
        return N
    v = pd.read_csv(vp)
    c = pd.read_csv(cp)
    st, sd = "confirmatory", "confirmatory-descriptive"

    def bar(claim, b, metric=None, method=None, cell=None, plant=None):
        r = v[(v.claim == claim) & (v.bar == b)]
        if metric is not None:
            r = r[r.metric == metric]
        if method is not None:
            r = r[r.method == method]
        if cell is not None:
            r = r[r.cell == cell]
        if plant is not None:
            r = r[r.plant == plant]
        if len(r) != 1:
            raise ValueError(f"expected one verdict row for {claim}/{b}/{metric}/{method}/{cell}/{plant}, got {len(r)}")
        return r.iloc[0]

    def put(key, r, small=False, pts=False, status=st):
        f = _pts if pts else (lambda N_, k, x, s, t: _p(N_, k, x, s, t, small))
        f(N, key, r.estimate, V, status)
        f(N, key + ".lo", r.ci_lo, V, status)
        f(N, key + ".hi", r.ci_hi, V, status)

    # ---- C-MAIN (T1)
    r = bar("C-MAIN", "1 validity", "E_ep", plant="T1")
    put("conf.main.eep", r, small=True)
    _p(N, "conf.main.eep.up", r.upper95, V, st, small=True)
    N.num("conf.main.neps", r.n_clusters, V, st)
    r = bar("C-MAIN", "1 validity", "E_dec@U*", plant="T1")
    put("conf.main.edec.matched", r)
    N.num("conf.main.nincap", r.n_incapable, V, st)
    for kind, lab in (("merge", "merge"), ("child >=2 modules", "multi"), ("lone faulted child", "lone")):
        put(f"conf.main.kind.{lab}.matched", bar("C-MAIN", "1b per-kind validity", cell=kind))
    put("conf.main.p1", bar("C-MAIN", "P1"), pts=True)
    put("conf.main.p2", bar("C-MAIN", "P2"), pts=True)
    r = bar("C-MAIN (task-library tau)", "exploratory", "CF (stratum)")
    _p(N, "conf.main.task.cf", r.estimate, V, sd)
    r = bar("C-MAIN (task-library tau)", "exploratory", "E_dec (stratum)")
    _p(N, "conf.main.task.edec", r.estimate, V, sd, small=True)

    # own-bound and matched cells (descriptive, same run)
    m = c[(c.claim == "C-MAIN") & (c.plant == "T1")]
    names = {"amortized_comp": "ours", "capxfer": "capxfer", "ghat_physics": "ghat", "copy_parent": "copy",
             "detector_gated": "detector", "min_over_members": "minmem", "credible_lower": "credible", "oracle": "oracle",
             "metadata_nf": "metadata"}
    for meth, lab in names.items():
        for blk, bl in (("stratum", "stratum"), ("lone-faulted", "lone"), ("merge", "merge"), ("multi-child", "multi")):
            rr = m[(m.method == meth) & (m.block == blk)]
            if len(rr) != 1:
                continue
            rr = rr.iloc[0]
            _p(N, f"conf.cell.{lab}.{bl}.cfown", rr.CF_own, CELLS, sd)
            _p(N, f"conf.cell.{lab}.{bl}.edecown", rr.E_dec_own, CELLS, sd, small=True)
            _p(N, f"conf.cell.{lab}.{bl}.cfmatched", rr.CF_matched, CELLS, sd)
            _p(N, f"conf.cell.{lab}.{bl}.edecmatched", rr.E_dec_matched, CELLS, sd, small=True)
            if bl == "stratum":
                _p(N, f"conf.cell.{lab}.eep", rr.E_ep, CELLS, sd, small=True)
                N.num(f"conf.cell.{lab}.neps", rr.n_ep, CELLS, sd)

    # post-hoc paired correction
    if pp.exists():
        ph = pd.read_csv(pp)
        for b in ("P1", "P2"):
            r = ph[ph.bar == b].iloc[0]
            k = f"conf.posthoc.{b.lower()}"
            _pts(N, k, r["diff"], PH, "posthoc")
            _pts(N, k + ".lo", r.lo95, PH, "posthoc")
            _pts(N, k + ".hi", r.hi95, PH, "posthoc")
            _p(N, k + ".ours", r.ours, PH, "posthoc")
            _p(N, k + ".other", r.other, PH, "posthoc")
        N.num("conf.posthoc.ncommon", int(ph.n_common_episodes.iloc[0]), PH, "posthoc")

    # ---- C-MERGE
    for plant in ("T1", "T2"):
        p = plant.lower()
        r = bar("C-MERGE", "1 delta_comp", plant=plant)
        put(f"conf.merge.flip.{p}", r, small=True)
        _p(N, f"conf.merge.flip.{p}.up", r.upper95, V, st, small=True)
        put(f"conf.merge.miscov.{p}", bar("C-MERGE", "2 validity", "miscoverage per pair", plant=plant), small=True)
        put(f"conf.merge.eep.{p}", bar("C-MERGE", "2 validity", "E_ep per pair", plant=plant), small=True)
    r = bar("C-MERGE", "P4", "CF@U* comp_enum_size")
    put("conf.merge.cf", r)
    N.num("conf.merge.npairs", r.n_clusters, V, st)
    put("conf.merge.p4", bar("C-MERGE", "P4", "CF@U* (stratum merges): comp_enum_size - copy_parent"), pts=True)
    _p(N, "conf.merge.cf.ttwo", bar("C-MERGE", "P4 (T2, reported)", "CF@U* comp_enum_size").estimate, V, sd)

    # ---- C-GATE (T2 knee law, severity 3, >= 3-module children)
    put("conf.gate.raw.undom", bar("C-GATE", "P5", method="raw"))
    put("conf.gate.par.undom", bar("C-GATE", "P5", method="par"), small=True)
    r = bar("C-GATE", "P6' robust undominated", "E_ep(robust) <= 0.10 (E_dec underpowered)")
    put("conf.gate.robust.eep.sthree", r)
    N.num("conf.gate.neps", r.n_clusters, V, st)
    put("conf.gate.lawcal.sthree", bar("C-GATE", "lawcal held-out s=3"), small=True)
    _p(N, "conf.gate.robust.eep.szero", bar("C-GATE", "robust validity s=0", "E_ep(robust) <= 0.10 (E_dec underpowered)").estimate, V, st, small=True)

    # ---- C-SCALE
    for cell, lab in (("N 7-8", "seven"), ("N 9-10", "nine")):
        put(f"conf.scale.eep.{lab}", bar("C-SCALE", "1 validity", "E_ep", cell=cell))
        put(f"conf.scale.cov.{lab}", bar("C-SCALE", "1 validity", "stratum coverage", cell=cell))

    # ---- C-HET
    put("conf.het.eep.hone", bar("C-HET", "1 validity", "E_ep (faulted)", method="amortized_comp", cell="conf_H1"), small=True)
    put("conf.het.eep.htwo", bar("C-HET", "1 validity", "E_ep (all)", method="amortized_comp", cell="conf_H2"))
    put("conf.het.covdrop.hone", bar("C-HET", "2 degradation", "coverage drop ID -> conf_H1 (faulted)"), pts=True)
    put("conf.het.covdrop.htwo", bar("C-HET", "2 degradation", "coverage drop ID -> conf_H2 (all)"), pts=True)
    put("conf.het.cfdrop.htwo", bar("C-HET", "2 degradation", "CF drop ID -> conf_H2 (all)"), pts=True)
    put("conf.het.cfdirect.hone", bar("C-HET", "2 degradation", "CF drop: direct - ours (conf_H1, faulted)"), pts=True)
    put("conf.het.covgain.hone", bar("C-HET", "2 degradation", "coverage drop: direct - ours (conf_H1, faulted)"), pts=True)

    # ---- C-EXPLAIN
    for b, lab in (("wheel precision", "wheel"), ("deficit-claim precision", "deficit"),
                   ("false-reason rate (capable refusals)", "falsereason"),
                   ("informative share (incapable refusals)", "informative")):
        put(f"conf.explain.{lab}", bar("C-EXPLAIN", b))

    # ---- C-ENV prerequisite / descriptive
    _p(N, "conf.env.reconfig", bar("C-ENV", "4 reconfiguration").estimate, V, sd)

    # ---- post-hoc: episode miscoverage (Eq. 2 coverage event), ours, C-MAIN T1 (not a pre-registered bar)
    dm = res / "models/confirmatory/dec_conf_models.csv.gz"
    if dm.exists():
        d = pd.read_csv(dm, usecols=["claim", "method", "test_plant", "split", "eid", "lower", "truth", "shard"])
        d = d[(d.claim == "C-MAIN") & (d.method == "amortized_comp") & (d.test_plant == "T1") &
              (d.split == "conf_test") & (d.shard != 0)]
        ep = (d.lower > d.truth + 1e-12).groupby(d.eid).any().to_numpy(float)
        rng = np.random.default_rng(92599)
        bs = rng.choice(ep, size=(2000, len(ep))).mean(1)
        src = "results/models/confirmatory/dec_conf_models.csv.gz"
        _p(N, "conf.posthoc.miscov", ep.mean(), src, "posthoc")
        _p(N, "conf.posthoc.miscov.lo", np.percentile(bs, 2.5), src, "posthoc")
        _p(N, "conf.posthoc.miscov.hi", np.percentile(bs, 97.5), src, "posthoc")
        # calibration-conditional spread of split-conformal miscoverage with n = 100, alpha = 0.1: Beta(n+1-k, k)
        n, a = 100, 0.10
        k = int(np.ceil((n + 1) * (1 - a)))
        A, Bb = n + 1 - k, k
        sd = np.sqrt(A * Bb / ((A + Bb) ** 2 * (A + Bb + 1)))
        _pts(N, "conf.posthoc.calsd", sd, "Beta(n+1-k,k), n=100 composed cal episodes (FACTS.md)", "posthoc")
    oc = res / "confirmatory/posthoc_p1p2/ownbound_common.csv"
    if oc.exists():
        o = pd.read_csv(oc)
        for _, r in o.iterrows():
            if r.block == "merge":
                continue
            k = f"conf.posthoc.common.{r.method}.{'lone' if r.block == 'lone-faulted' else 'stratum'}"
            src = "results/confirmatory/posthoc_p1p2/ownbound_common.csv"
            _p(N, k + ".cf", r.CF_own, src, "posthoc")
            _p(N, k + ".edec", r.E_dec_own, src, "posthoc", small=True)
            N.num(k + ".nuns", int(round(r.E_dec_own * r.n_incapable)), src, "posthoc")
            N.num(k + ".ninc", int(r.n_incapable), src, "posthoc")
    return N
