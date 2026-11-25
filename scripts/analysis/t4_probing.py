"""T4, decision metrics per probe policy (fixed v2, random, EIG, EPIG, decision-EVOI, always split, ...).

Input: contract 'probe' (one row per episode x policy, same episodes for every policy). Per policy: certified fraction
of capable decisions, unsafe-commit rate (CI over episodes), share of v2-ambiguous decisions resolved, mean probe
time, ambiguity resolved per extra probe-minute vs fixed_v2, mean utility. Stage A3 found too few ambiguous decisions
to test probing (results/stageA/A3_verdict.json), so this table is expected to land in the appendix unless the
nuisance role's T1 ambiguity brings probing back (docs/board/paper.md, claim C4).
Outputs: paper/tables/t4_probing.tex, macros cfTFour*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

NAMES = {"fixed_v2": "Fixed v2", "random": "Random probe", "eig": "EIG (parameters)", "epig": "EPIG (predictions)",
         "evoi": "Decision-EVOI (ours)", "always_split": "Always split first", "dad": "Amortized policy (DAD/TNDP)",
         "oracle_probe": "Oracle probe choice"}
ORDER = list(NAMES)


def run(res, paper):
    N = C.Numbers("t4_probing.py")
    P, srcs = C.load("probe", res)
    out = paper / "tables/t4_probing.tex"
    if P.empty:
        C.placeholder_table(out, "tFourTable", 7, "awaiting probe_*.csv (probe/decision role)")
        for k in ("t4.resolved.evoi", "t4.resolved.epig", "t4.unsafe.evoi", "t4.permin.evoi", "t4.namb.fixedv2"):
            N.ph(k)
        return N
    P = P[P.split == "test"]
    st = C.status_of(P)
    base_t = P[P.policy == "fixed_v2"].probe_time_s.mean() if (P.policy == "fixed_v2").any() else np.nan
    rows = []
    for pol, d in P.groupby("policy"):
        amb_b, amb_a = d.n_amb_before.sum(), d.n_amb_after.sum()
        res_ = 1 - amb_a / amb_b if amb_b else np.nan
        extra_min = (d.probe_time_s.mean() - base_t) / 60
        rows.append(dict(policy=pol, n_ep=len(d), cert=d.n_certified.sum() / max(d.n_capable.sum(), 1),
                         unsafe=d.n_unsafe.sum() / max(d.n_incapable.sum(), 1),
                         unsafe_ci=C.boot_ratio(d.n_unsafe.to_numpy(), d.n_incapable.to_numpy()),
                         resolved=res_, n_amb=amb_b, t=d.probe_time_s.mean(),
                         per_min=(res_ / extra_min) if extra_min and extra_min > 0 and np.isfinite(res_) else np.nan,
                         utility=d.utility.mean(), abstain=(d.action == "abstain").mean()))
    R = pd.DataFrame(rows)
    R["o"] = R.policy.map(lambda p: ORDER.index(p) if p in ORDER else 99)
    R = R.sort_values(["o", "policy"])
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Policy & probe time (s) & ambiguity resolved (\%) & per extra min & certified (\%) & unsafe (\%) & abstain (\%)\\",
             r"\midrule"]
    for _, r in R.iterrows():
        nm = NAMES.get(r.policy, r.policy.replace("_", r"\_"))
        pm = f"{100 * r.per_min:.0f}" if np.isfinite(r.per_min) else r"\textemdash"
        lines.append(f"{nm} & {r.t:.1f} & {C.fmt_pct(r.resolved)} & {pm} & {C.fmt_pct(r.cert)} & "
                     f"{C.fmt_pct(r.unsafe, 2)} & {C.fmt_pct(r.abstain)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(out, "\n".join(lines), srcs, f"same test episodes for every policy [{st}]")
    for _, r in R.iterrows():
        p = r.policy
        N.pct(f"t4.resolved.{p}", r.resolved, srcs, st)
        N.pct(f"t4.unsafe.{p}", r.unsafe, srcs, st, d=2)
        N.pct(f"t4.cert.{p}", r.cert, srcs, st)
        N.num(f"t4.time.{p}", r.t, srcs, st, fmt="{:.1f}")
        if p != "fixed_v2":
            N.pct(f"t4.permin.{p}", r.per_min, srcs, st, d=0)
    v2 = R[R.policy == "fixed_v2"]
    N.num("t4.namb.fixedv2", float(v2.n_amb.iloc[0]) if len(v2) else np.nan, srcs, st)
    for k in ("evoi", "epig"):
        if not (R.policy == k).any():
            N.ph(f"t4.resolved.{k}")
            N.ph(f"t4.unsafe.{k}")
            N.ph(f"t4.permin.{k}")
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
