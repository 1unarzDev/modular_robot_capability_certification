"""T5, never-co-assembled merges (E2, regime 3 "bodies that never existed"; docs/board/research.md E2).

Inputs:
  contract 'merge' = results/stress/A4_merge.csv: composition defect delta_comp per payload arm x plant x group
    (flip share of decisions between the composed prediction from true latents and the plant's merged truth, with an
    episode-cluster CI; |d rho| quantiles; simultaneous miscoverage per pair of the Thm 5' certificate; CF@U*).
  contract 'dec' rows from results/stress/paper/dec_merge.csv (kind = merge; methods compositional_enum, oracle, ...):
    scored here with the T1 machinery (decision-relevant stratum headline, margin from T1 cal), kept out of T1-T3.
Tables: t5_merge_defect.tex (delta_comp, groups all and faulted), t5_merge_cert.tex (certificate on merges).
Success rule (pre-written in research.md E2): flip CI upper bound < 1 % in T1 and < 2 % in T2; (c) miscoverage per
pair <= alpha; CF on capable decision-stratum merges >= 85 % at U* = 1 %. Kill: T1 delta_comp flip > 2 %.
Macros cfTFive*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
from t1_certificate import name, sfx

KEYS = ("t5.flip.T1", "t5.fliphi.T1", "t5.flip.T2", "t5.fliphi.T2", "t5.miscov", "t5.verdict", "t5.cert.comp",
        "t5.cert.comp.all", "t5.unsafe.comp")
COMP = "compositional_enum"


def _defect(M, srcs, st, path, N):
    A = M[M.group.isin(["all", "faulted"])]
    lines = [r"\begin{tabular}{lllrrrrr}", r"\toprule",
             r"Plant & payload & merges & decision flips (\%) [95\,\% CI] & $|\Delta\rho|$ q95 & q99 & "
             r"sim.\ miscov./pair (\%) & decisions\\", r"\midrule"]
    for _, r in A.sort_values(["plant", "arm", "group"]).iterrows():
        lines.append(f"{r.plant} & {r.arm} & {r.group} & {C.fmt_pct(r.flip, 2)} [{C.fmt_pct(r.flip_lo, 2)}, "
                     f"{C.fmt_pct(r.flip_hi, 2)}] & {r.absd_q95:.3f} & {r.absd_q99:.3f} & {C.fmt_pct(r.miscov_pair)} & "
                     f"{int(r.decisions):,}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(path, "\n".join(lines), srcs, f"composition defect of never-co-assembled merges [{st}]")
    A = M[M.group == "all"]
    p0 = A[A.arm == A.arm.min()] if len(A) else A          # P0 (no payload) when present
    for plant in ("T1", "T2"):
        r = p0[p0.plant == plant]
        N.pct(f"t5.flip.{plant}", r.flip.iloc[0] if len(r) else np.nan, srcs, st, d=2)
        N.pct(f"t5.fliphi.{plant}", r.flip_hi.iloc[0] if len(r) else np.nan, srcs, st, d=2)
    N.num("t5.pairs", float(p0.pairs.max()) if len(p0) else np.nan, srcs, st)
    N.num("t5.merges", float(p0.merges.max()) if len(p0) else np.nan, srcs, st)
    mc = A.miscov_pair.max() if A.miscov_pair.notna().any() else np.nan
    N.pct("t5.miscov", mc, srcs, st)
    return A


def _cert(D, srcs, st, path, N):
    D = D[D.kind == "merge"]
    rows = []
    for m, dm in D.groupby("method"):
        fit = dm[(dm.test_plant == "T1") & (dm.split == "cal")]
        dagger = fit.empty
        if dagger:
            fit = dm[(dm.test_plant == "T1") & (dm.split == "test")]
        for plant in ("T1", "T2"):
            ev = dm[(dm.test_plant == plant) & (dm.split == "test")].reset_index(drop=True)
            if ev.empty or fit.empty:
                continue
            row = dict(method=m, plant=plant, dagger=dagger)
            for s in C.STRATA:
                lo, _ = C.matched_lower(fit, ev, stratum=s)
                cnt = C.ep_counts(ev, lo, stratum=s)
                row[f"cf_{s}"] = C._ratio(cnt["cert"].sum(), cnt["cap"].sum())
                nom = C.decision_counts(ev, stratum=s)
                row[f"unsafe_{s}"] = nom["unsafe_commit"]
                row[f"epcov_{s}"] = C.coverage(ev, stratum=s)["ep_cov"]
            rows.append(row)
    R = pd.DataFrame(rows)
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Method & plant & cert.\ @U$^*$ (\%) & unsafe, nominal (\%) & sim.\ cov.\ (\%) & cert.\ @U$^*$ all (\%)\\",
             r"\midrule"]
    for _, r in R.iterrows():
        lines.append(f"{name(r.method)}{'$^\\dagger$' if r.dagger else ''} & {r.plant} & {C.fmt_pct(r.cf_rel)} & "
                     f"{C.fmt_pct(r.unsafe_rel, 2)} & {C.fmt_pct(r.epcov_rel)} & {C.fmt_pct(r.cf_all)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(path, "\n".join(lines), srcs, f"certificates on never-co-assembled merges, decision-relevant [{st}]")
    for _, r in R.iterrows():
        for s in C.STRATA:
            N.pct(f"t5.cert.{r.method}.{r.plant}{sfx(s)}", r[f"cf_{s}"], srcs, st)
            N.pct(f"t5.unsafe.{r.method}.{r.plant}{sfx(s)}", r[f"unsafe_{s}"], srcs, st, d=2)
    c = R[(R.method == COMP) & (R.plant == "T1")]
    N.pct("t5.cert.comp", c.cf_rel.iloc[0] if len(c) else np.nan, srcs, st)
    N.pct("t5.cert.comp.all", c.cf_all.iloc[0] if len(c) else np.nan, srcs, st)
    N.pct("t5.unsafe.comp", c.unsafe_rel.iloc[0] if len(c) else np.nan, srcs, st, d=2)
    return R


def run(res, paper):
    N = C.Numbers("merge_table.py")
    out_d, out_c = paper / "tables/t5_merge_defect.tex", paper / "tables/t5_merge_cert.tex"
    M, msrc = C.load("merge", res)
    dec, dsrc = C.load("dec", res)
    _, D = C.split_merge_rows(dec)
    A = None
    if M.empty:
        C.placeholder_table(out_d, "tFiveDefect", 8, "awaiting results/stress/A4_merge.csv (E2)")
    else:
        st = C.status_of(M) if M["status"].notna().any() else "exploratory"
        A = _defect(M, msrc, st, out_d, N)
    R = None
    if D.empty:
        C.placeholder_table(out_c, "tFiveCert", 6, "awaiting results/stress/paper/dec_merge.csv (E2)")
    else:
        R = _cert(D, sorted(D["_src"].unique()), C.status_of(D), out_c, N)
    # success rule
    if A is not None and len(A):
        p0 = A[A.arm == A.arm.min()]
        t1, t2 = p0[p0.plant == "T1"], p0[p0.plant == "T2"]
        if len(t1) and t1.flip.iloc[0] > 0.02:
            v = "E2 killed: T1 composition defect > 2 %"
        else:
            ok1 = len(t1) and t1.flip_hi.iloc[0] < 0.01
            ok2 = len(t2) and t2.flip_hi.iloc[0] < 0.02
            mc = A.miscov_pair.max() if A.miscov_pair.notna().any() else np.nan
            ok3 = not np.isfinite(mc) or mc <= C.ALPHA
            cf = R[(R.method == COMP) & (R.plant == "T1")].cf_rel if R is not None and len(R) else pd.Series(dtype=float)
            defect_ok = bool(ok1 and ok2 and ok3)
            if not len(cf):   # the certificate half has not landed: never call that a failure
                v = ("E2 delta-comp part passes" if defect_ok else "E2 delta-comp part fails") \
                    + "; certificate part pending (dec_merge.csv)"
            else:
                v = "E2 passes" if defect_ok and cf.iloc[0] >= 0.85 else "E2 fails its pre-written success rule"
        N.set("t5.verdict", v, msrc + (sorted(D["_src"].unique()) if not D.empty else []), "exploratory")
    for k in KEYS:
        if C.macro_name(k) not in N.items:
            N.ph(k)
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
