"""T4, decision metrics per probe policy (fixed v2, random, EIG, EPIG, decision-EVOI, always split, ...).

Input: contract 'probe' (one row per episode x policy, same episodes for every policy; pairs = (candidate, task),
labels rho_dec). Per (test_plant, policy): certified fraction and unsafe-commit rate (CI over episodes) in the
decision-relevant stratum (HEADLINE; columns n_*_rel: pairs whose candidate holds a faulted module or whose label is
within REL_BAND of the task requirement) and over all pairs; share of v2-ambiguous pairs resolved; mean probe time;
ambiguity resolved per extra probe-minute vs fixed_v2; abstain rate. The table shows the T1 plant if present, else
the first plant found (e.g. the decision role's T0 pilot), and says which.
Stage A3 found too few ambiguous decisions to test probing (results/stageA/A3_verdict.json), so this table may land
in the appendix (docs/board/paper.md, claim C4).
Outputs: paper/tables/t4_probing.tex, t4_probing.csv, macros cfTFour* (unsuffixed = rel).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
from t1_certificate import sfx

NAMES = {"fixed_v2": "Fixed v2", "fixed_all": "Fixed full library", "random": "Random probe",
         "eig": "EIG (parameters)", "epig": "EPIG (predictions)", "evoi": "Decision-EVOI (ours)",
         "always_split": "Always split first", "dad": "Amortized policy (DAD/TNDP)",
         "oracle_probe": "Oracle probe choice"}
ORDER = list(NAMES)
KEY = ("t4.resolved.evoi", "t4.resolved.epig", "t4.unsafe.evoi", "t4.cert.evoi", "t4.permin.evoi",
       "t4.namb.fixedv2", "t4.plant")


def _ratio_ci(d, num, den):
    if num not in d or d[num].isna().all():
        return np.nan, (np.nan, np.nan)
    return C._ratio(d[num].sum(), d[den].sum()), C.boot_ratio(d[num].to_numpy(float), d[den].to_numpy(float))


def run(res, paper):
    N = C.Numbers("t4_probing.py")
    P, srcs = C.load("probe", res)
    out = paper / "tables/t4_probing.tex"
    if not P.empty:
        P = P[P.split == "test"].copy()
        P["test_plant"] = P["test_plant"].fillna("T1").astype(str)
    if P.empty:
        C.placeholder_table(out, "tFourTable", 8, "awaiting probe_*.csv (decision role)")
        for k in KEY:
            N.ph(k)
        N.ph("t4.cert.evoi.all")
        return N
    plants = sorted(P.test_plant.unique(), key=lambda p: (p != "T1", p))
    rows = []
    for (plant, pol), d in P.groupby(["test_plant", "policy"]):
        base_t = P[(P.test_plant == plant) & (P.policy == "fixed_v2")].probe_time_s.mean()
        amb_b, amb_a = d.n_amb_before.sum(), d.n_amb_after.sum()
        res_ = 1 - amb_a / amb_b if amb_b else np.nan
        extra_min = (d.probe_time_s.mean() - base_t) / 60 if np.isfinite(base_t) else np.nan
        cert_r, _ = _ratio_ci(d, "n_certified_rel", "n_capable_rel")
        uns_r, uns_r_ci = _ratio_ci(d, "n_unsafe_rel", "n_incapable_rel")
        rows.append(dict(plant=plant, policy=pol, n_ep=len(d), status=C.status_of(d),
                         cert=cert_r, unsafe=uns_r, unsafe_ci=uns_r_ci,
                         cert_all=C._ratio(d.n_certified.sum(), d.n_capable.sum()),
                         unsafe_all=C._ratio(d.n_unsafe.sum(), d.n_incapable.sum()),
                         resolved=res_, n_amb=amb_b, t=d.probe_time_s.mean(),
                         per_min=(res_ / extra_min) if extra_min and extra_min > 0 and np.isfinite(res_) else np.nan,
                         utility=d.utility.mean(), abstain=(d.action == "abstain").mean()))
    R = pd.DataFrame(rows)
    R["o"] = R.policy.map(lambda p: ORDER.index(p) if p in ORDER else 99)
    R = R.sort_values(["plant", "o", "policy"])
    R.drop(columns="o").to_csv(paper / "tables/t4_probing.csv", index=False)
    plant = plants[0]
    Rp = R[R.plant == plant]
    st = C.status_of(P[P.test_plant == plant])
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & & & & \multicolumn{2}{c}{decision-relevant} & all & \\",
             r"Policy & probe time (s) & ambiguity resolved (\%) & per extra min & cert.\ (\%) & unsafe (\%) & "
             r"cert.\ (\%) & abstain (\%)\\", r"\midrule"]
    for _, r in Rp.iterrows():
        nm = NAMES.get(r.policy, r.policy.replace("_", r"\_"))
        pm = f"{100 * r.per_min:.0f}" if np.isfinite(r.per_min) else r"\textemdash"
        lines.append(f"{nm} & {r.t:.1f} & {C.fmt_pct(r.resolved)} & {pm} & {C.fmt_pct(r.cert)} & "
                     f"{C.fmt_pct(r.unsafe, 2)} & {C.fmt_pct(r.cert_all)} & {C.fmt_pct(r.abstain)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(out, "\n".join(lines), srcs, f"plant {plant}; same test episodes for every policy [{st}]")
    N.set("t4.plant", plant, srcs, st)
    for _, r in Rp.iterrows():
        p = r.policy
        N.pct(f"t4.resolved.{p}", r.resolved, srcs, st)
        N.pct(f"t4.unsafe.{p}", r.unsafe, srcs, st, d=2)
        N.pct(f"t4.cert.{p}", r.cert, srcs, st)
        N.pct(f"t4.unsafe.{p}{sfx('all')}", r.unsafe_all, srcs, st, d=2)
        N.pct(f"t4.cert.{p}{sfx('all')}", r.cert_all, srcs, st)
        N.num(f"t4.time.{p}", r.t, srcs, st, fmt="{:.1f}")
        if p != "fixed_v2":
            N.pct(f"t4.permin.{p}", r.per_min, srcs, st, d=0)
    v2 = Rp[Rp.policy == "fixed_v2"]
    N.num("t4.namb.fixedv2", float(v2.n_amb.iloc[0]) if len(v2) else np.nan, srcs, st)
    for k in KEY:
        if C.macro_name(k) not in N.items:
            N.ph(k)
    if C.macro_name("t4.cert.evoi.all") not in N.items:
        N.ph("t4.cert.evoi.all")
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
