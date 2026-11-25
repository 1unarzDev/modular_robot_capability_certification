"""Stage A numbers used in the intro and the 'what is actually hidden' paragraph (all exploratory, all T0/T1/T2 sim).

Inputs (existing): results/stageA/{A1_summary.csv, A2_verdict.json, A3_verdict.json, A6_summary.csv},
results/stress/A5_verdict.json, results/theory/coverage_dominance.csv. Macros cfSA*.
"""
from __future__ import annotations

import json

import pandas as pd

import common as C


def run(res, paper):
    N = C.Numbers("stageA_numbers.py")
    f = res / "stageA/A2_verdict.json"
    if f.exists():
        v = json.loads(f.read_text())
        s = C.rel(f)
        N.pct("sa.amb.vtwo", v["amb_share_v2"], s, d=2)
        N.pct("sa.oracle.cert", v["oracle_cert_frac_all"], s, d=2)
        N.pct("sa.oracle.cert.lone", v["oracle_cert_frac_lone_faulted"], s, d=1)
    else:
        for k in ("sa.amb.vtwo", "sa.oracle.cert", "sa.oracle.cert.lone"):
            N.ph(k)
    f = res / "stageA/A3_verdict.json"
    if f.exists():
        v = json.loads(f.read_text())
        N.num("sa.amb.n", v["v2_ambiguous_decisions"], C.rel(f))
    else:
        N.ph("sa.amb.n")
    f = res / "stageA/A1_summary.csv"
    if f.exists():
        a = pd.read_csv(f)
        a = a[(a.detector == "d2") & (a.het == "on")]
        hi = a[a.g_bin.astype(str).str.startswith("0.75")]
        if len(hi):
            hi = hi[hi["shape"].isin(["pair", "square"])]
            N.pct("sa.det.highg.min", hi.detection.min(), C.rel(f), d=0)
            N.pct("sa.det.highg.max", hi.detection.max(), C.rel(f), d=0)
        else:
            N.ph("sa.det.highg.min"), N.ph("sa.det.highg.max")
    else:
        N.ph("sa.det.highg.min"), N.ph("sa.det.highg.max")
    f = res / "stress/A5_verdict.json"
    if f.exists():
        j = json.loads(f.read_text())
        v = j.get("P0/rho_dec", j.get("P0"))   # rho_dec is the decision label (coordinator 2026-11-25)
        N.pct("sa.tTwo.flip", v["flip"], C.rel(f), d=1)
        N.pct("sa.tTwo.fliphi", v["flip_ci"][1], C.rel(f), d=1)
        N.pct("sa.tTwo.dfive", v["share_d05"], C.rel(f), d=1)
    else:
        for k in ("sa.tTwo.flip", "sa.tTwo.fliphi", "sa.tTwo.dfive"):
            N.ph(k)
    f = res / "stageA/A6_summary.csv"
    if f.exists():
        a = pd.read_csv(f)
        N.num("sa.enum.sec", a.sec_per_episode_median.iloc[0], C.rel(f), fmt="{:.1f}")
    else:
        N.ph("sa.enum.sec")
    f = res / "theory/coverage_dominance.csv"
    if f.exists():
        a = pd.read_csv(f).set_index("leg_kind")
        if "diag" in a.index:
            N.pct("sa.diag.undom", a.loc["diag", "share_legs_with_undominated_wheel_v2"], C.rel(f), d=1)
        else:
            N.ph("sa.diag.undom")
    else:
        N.ph("sa.diag.undom")
    N.pct("sa.capxfer.lo", 0.58, "capxfer paper_rss (58-75 % certified, sim)", d=0)
    N.pct("sa.capxfer.hi", 0.75, "capxfer paper_rss (58-75 % certified, sim)", d=0)
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
