"""Stage A2 (exploratory): decision ambiguity of the idealised-oracle consistent set after probe v2 (T0).

Targets: scripts/stageA_targets.md (written before the run). Engine: cfcap/t0/stagea.py.

  run      python scripts/stageA2_ambiguity.py run --seeds 0:100        (one shard; raw tables -> $CFCAP_OUT/stageA)
  analyze  python scripts/stageA2_ambiguity.py analyze                  (all shards -> results/stageA/A2_*.csv)

The run computes every probe set at once (extra legs are solved only on the v2-consistent hypotheses, so they are
nearly free); stageA3_probes.py analyses the same shards. Local rule: nice -n 15, 4 threads, one process.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", os.path.expanduser("~/.cache/modular-runs/2026-11/cfcap"))) / "stageA"
RES = ROOT / "results" / "stageA"
COUNTS = ["n_dec", "n_amb", "n_go", "n_cap", "n_unsafe", "n_go_cap"]
CAPXFER_CERT = (0.58, 0.75)        # capxfer RESULTS_CORPUS.md: certified 58-75 % of capable plated modules


def cmd_run(a):
    import torch
    torch.set_num_threads(a.threads)
    from cfcap.t0 import qs, stagea
    P, th = qs.load_params(a.plate), qs.load_theta(a.plate)
    s0, s1 = map(int, a.seeds.split(":"))
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.plate}_{s0:05d}_{s1:05d}"
    recs, rows = [], []
    t0 = time.time()
    for seed in range(s0, s1):
        rec, rr = stagea.run_episode(seed, P, th)
        recs.append(rec)
        rows += rr
        if (seed - s0) % 10 == 0:
            print(f"seed {seed} size {rec['parent_size']} het {rec['het']} {rec['fault_type']} |U| {rec['n_union']} "
                  f"t {rec['t_total_s']}s elapsed {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(recs).to_csv(OUT / f"episodes_{tag}.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / f"decisions_{tag}.csv.gz", index=False)
    json.dump(dict(seeds=[s0, s1], plate=a.plate, wall_s=time.time() - t0, threads=a.threads,
                   host=os.uname().nodename), open(OUT / f"manifest_{tag}.json", "w"))
    print("wrote", tag, f"{time.time() - t0:.0f}s")


def load(plate="plated"):
    E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / f"episodes_{plate}_*.csv")))], ignore_index=True)
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / f"decisions_{plate}_*.csv.gz")))],
                  ignore_index=True)
    E = E.drop_duplicates("seed")
    D = D.drop_duplicates(["seed", "kind", "cand_size", "has_fault", "probe_set", "noise", "rho_model", "tau"])
    D = D.merge(E[["seed", "parent_size", "het", "fault_type", "n_faults"]], on="seed")
    return E, D


def metrics(c):
    """counts (dict/Series of sums) -> metric dict."""
    nd, na, ng, nc, nu, ngc = (float(c[k]) for k in COUNTS)
    return dict(n_dec=int(nd), amb_share=na / nd if nd else np.nan, go_share=ng / nd if nd else np.nan,
                cap_share=nc / nd if nd else np.nan, cert_frac_of_capable=ngc / nc if nc else np.nan,
                unsafe_go_share=nu / nd if nd else np.nan, unsafe_given_go=nu / ng if ng else np.nan)


def boot_ci(D, B=1000, seed=0, key="amb_share"):
    """Percentile CI over episodes (cluster bootstrap by parent seed)."""
    per = D.groupby("seed")[COUNTS].sum()
    if len(per) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(per), size=(B, len(per)))
    arr = per.to_numpy(float)
    tot = arr[idx].sum(1)                                              # (B, 6)
    nd, na, ng, nc, nu, ngc = tot.T
    with np.errstate(invalid="ignore", divide="ignore"):
        v = dict(amb_share=na / nd, cert_frac_of_capable=ngc / nc, unsafe_go_share=nu / nd)[key]
    return tuple(np.nanpercentile(v, [2.5, 97.5]))


def table(D, by, ci=True):
    out = []
    for keys, g in D.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        m = metrics(g[COUNTS].sum())
        row = dict(zip(by, keys), n_parents=g["seed"].nunique(), **m)
        if ci:
            row["amb_lo"], row["amb_hi"] = boot_ci(g, key="amb_share")
            row["cert_lo"], row["cert_hi"] = boot_ci(g, key="cert_frac_of_capable")
        out.append(row)
    return pd.DataFrame(out)


def cmd_analyze(a):
    E, D = load(a.plate)
    RES.mkdir(parents=True, exist_ok=True)
    E.to_csv(RES / "A2_episodes.csv", index=False)
    v2 = D[D.probe_set == "v2"]
    prim = v2[(v2.noise == "sensor") & (v2.rho_model == "known")]
    rows = []
    for name, sub, by in [
        ("overall", prim, ["tau"]),
        ("by_het", prim, ["het", "tau"]),
        ("by_parent_size", prim, ["parent_size", "tau"]),
        ("by_cand_size", prim, ["kind", "cand_size", "tau"]),
        ("by_kind", prim, ["kind", "tau"]),
        ("by_fault_type", prim, ["fault_type", "tau"]),
        ("by_has_fault", prim, ["has_fault", "tau"]),
        ("lone_faulted_module", prim[(prim.kind == "child") & (prim.cand_size == 1) & (prim.has_fault == 1)],
         ["het", "tau"]),
        ("noise_x_model", v2, ["noise", "rho_model", "het", "tau"]),
        ("pooled_tau", prim, []),
        ("pooled_tau_het", prim, ["het"]),
    ]:
        if by:
            t = table(sub, by)
        else:
            m = metrics(sub[COUNTS].sum())
            lo, hi = boot_ci(sub)
            clo, chi = boot_ci(sub, key="cert_frac_of_capable")
            t = pd.DataFrame([dict(n_parents=sub.seed.nunique(), **m, amb_lo=lo, amb_hi=hi, cert_lo=clo,
                                   cert_hi=chi)])
        t.insert(0, "breakdown", name)
        rows.append(t)
    S = pd.concat(rows, ignore_index=True)
    S.to_csv(RES / "A2_summary.csv", index=False)
    # set sizes and cost
    cost = dict(n_parents=len(E), mean_t_total_s=E.t_total_s.mean(), mean_t_parent_s=E.t_parent_s.mean(),
                p90_t_total_s=E.t_total_s.quantile(0.9), mean_qs_parent=E.qs_parent.mean(),
                mean_qs_child=E.qs_child.mean(), mean_n_hyp=E.n_hyp.mean(),
                share_singleton_v2_sensor=(E["setsize_v2_sensor"] == 1).mean(),
                mean_setsize_v2_sensor=E["setsize_v2_sensor"].mean(),
                mean_nonconv_child=E.nonconv_child.mean())
    pd.DataFrame([cost]).to_csv(RES / "A2_cost.csv", index=False)
    by_size = E.groupby("parent_size")[["t_total_s", "t_parent_s", "n_cands", "setsize_v2_sensor"]].mean()
    by_size.to_csv(RES / "A2_cost_by_parent_size.csv")
    # verdicts
    pooled = S[S.breakdown == "pooled_tau"].iloc[0]
    lone = metrics(prim[(prim.kind == "child") & (prim.cand_size == 1) & (prim.has_fault == 1)][COUNTS].sum())
    verdict = dict(
        amb_share_v2=pooled.amb_share, amb_ci=[pooled.amb_lo, pooled.amb_hi],
        A2_probing_secondary=bool(pooled.amb_share < 0.05),
        oracle_cert_frac_all=pooled.cert_frac_of_capable, oracle_cert_frac_lone_faulted=lone["cert_frac_of_capable"],
        gain_vs_capxfer_hi_pts=100 * (lone["cert_frac_of_capable"] - CAPXFER_CERT[1]),
        A2_attack_conformal_slack=bool(100 * (lone["cert_frac_of_capable"] - CAPXFER_CERT[1]) < 10),
        oracle_unsafe_go_share=pooled.unsafe_go_share,
    )
    json.dump(verdict, open(RES / "A2_verdict.json", "w"), indent=1, default=float)
    print(S[S.breakdown.isin(["pooled_tau", "pooled_tau_het", "overall", "lone_faulted_module"])].to_string())
    print(json.dumps(verdict, indent=1, default=float))
    print(pd.DataFrame([cost]).T)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--seeds", default="0:100")
    r.add_argument("--plate", default="plated")
    r.add_argument("--threads", type=int, default=4)
    an = sp.add_parser("analyze")
    an.add_argument("--plate", default="plated")
    a = ap.parse_args()
    dict(run=cmd_run, analyze=cmd_analyze)[a.cmd](a)
