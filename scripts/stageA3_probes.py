"""Stage A3 (exploratory): value of richer probes for decision ambiguity (T0 idealised oracle).

Same episodes (shards) as stageA2_ambiguity.py, which computes every probe set in one pass (the A2 run IS the A3
rerun: identical seeds, parents, faults and candidates; extra legs are solved on the v2-consistent hypotheses).
Probe sets: v2 | v2+clamp | v2+moddiff | v2+tug | v2+clamp+moddiff+tug | diag_split (observe the candidate itself
with its own v2 after splitting/merging). Targets: scripts/stageA_targets.md.

  python scripts/stageA3_probes.py            -> results/stageA/A3_probes.csv, A3_probes_by.csv, A3_verdict.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from stageA2_ambiguity import COUNTS, RES, load, metrics  # noqa: E402

from cfcap.t0 import probesets as PS  # noqa: E402

ORDER = ["v2", "v2+clamp", "v2+moddiff", "v2+tug", "v2+clamp+moddiff+tug", "diag_split"]


def probe_cost_s(name):
    if name == "diag_split":
        return PS.set_time("v2") + PS.T_SPLIT_S + PS.probe_time("v2")
    return PS.set_time(name)


def per_seed(D):
    return D.groupby(["seed", "probe_set"])[COUNTS].sum().reset_index()


def boot_reduction(D, ps, B=1000, seed=0):
    """Cluster-bootstrap CI of the relative reduction of the ambiguous count vs v2 (paired by parent)."""
    s = D.groupby(["seed", "probe_set"])["n_amb"].sum().unstack("probe_set").fillna(0)
    if ps not in s or s["v2"].sum() == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(s), size=(B, len(s)))
    a0 = s["v2"].to_numpy()[idx].sum(1)
    a1 = s[ps].to_numpy()[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        red = 1 - a1 / a0
    return tuple(np.nanpercentile(red, [2.5, 97.5]))


def summarize(D, extra_by=()):
    out = []
    for keys, g in D.groupby(list(extra_by), dropna=False) if extra_by else [((), D)]:
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = g[g.probe_set == "v2"][COUNTS].sum()
        for ps in ORDER:
            sub = g[g.probe_set == ps]
            if sub.empty:
                continue
            m = metrics(sub[COUNTS].sum())
            red = 1 - sub["n_amb"].sum() / base["n_amb"] if base["n_amb"] else np.nan
            row = dict(zip(extra_by, keys), probe_set=ps, n_parents=sub.seed.nunique(), n_amb=int(sub.n_amb.sum()),
                       **m, rel_reduction_vs_v2=red, probe_time_s=probe_cost_s(ps),
                       extra_time_s=probe_cost_s(ps) - PS.set_time("v2"))
            if not extra_by:
                row["red_lo"], row["red_hi"] = boot_reduction(g, ps)
            out.append(row)
    return pd.DataFrame(out)


def main():
    E, D = load()
    prim = D[(D.noise == "sensor") & (D.rho_model == "known")]
    S = summarize(prim)
    S.insert(0, "cell", "primary")
    S2 = summarize(D[(D.noise == "floor") & (D.rho_model == "known")])
    S2.insert(0, "cell", "floor_known")
    S3 = summarize(D[(D.het == 1) & (D.noise == "sensor") & (D.rho_model == "prior")])
    S3.insert(0, "cell", "het_sensor_prior")
    S4 = summarize(prim[prim.het == 1])
    S4.insert(0, "cell", "primary_het_on")
    A3 = pd.concat([S, S4, S2, S3], ignore_index=True)
    A3["ambiguity_removed_per_extra_min"] = np.where(A3.extra_time_s > 0,
                                                   A3.rel_reduction_vs_v2 / (A3.extra_time_s / 60), np.nan)
    A3.to_csv(RES / "A3_probes.csv", index=False)
    B = pd.concat([summarize(prim, ("het",)), summarize(prim, ("kind",)), summarize(prim, ("fault_type",)),
                   summarize(prim, ("tau",))], ignore_index=True)
    B.to_csv(RES / "A3_probes_by.csv", index=False)
    v2_amb = int(S.loc[S.probe_set == "v2", "n_amb"].iloc[0])
    best = S[S.probe_set != "v2"].sort_values("rel_reduction_vs_v2", ascending=False).iloc[0]
    verdict = dict(v2_ambiguous_decisions=v2_amb, testable=bool(v2_amb >= 50),
                   best_probe_set=best.probe_set, best_rel_reduction=float(best.rel_reduction_vs_v2),
                   best_ci=[float(best.red_lo), float(best.red_hi)],
                   any_probe_cuts_30pct=bool((S[S.probe_set != "v2"].rel_reduction_vs_v2 >= 0.30).any()),
                   reductions={r.probe_set: float(r.rel_reduction_vs_v2) for r in S.itertuples()})
    verdict["A3_drop_probing_headline"] = (not verdict["testable"]) or (not verdict["any_probe_cuts_30pct"])
    json.dump(verdict, open(RES / "A3_verdict.json", "w"), indent=1)
    cols = ["cell", "probe_set", "n_amb", "amb_share", "rel_reduction_vs_v2", "red_lo", "red_hi", "probe_time_s",
            "cert_frac_of_capable", "unsafe_go_share"]
    print(A3[cols].to_string())
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
