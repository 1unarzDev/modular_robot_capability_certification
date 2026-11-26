"""Critic-A audit of the held-out-severity check (exploratory).

  (1) K plumbing: 48 severity-3 seeds (7000+, seed % 4 = 3) at K in {15, 26.9, 28.7, 60}: screened-set size, empty
      screen share, kept-set identity, env_cal EVOI useful (certificate from the external sev 0-2 calibration pickle).
  (2) held-out result with the non-converged severity-3 episodes counted as worst case.
Writes results/decision/hos_audit_K.csv, hos_audit_worstcase.csv.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = Path(os.environ.get("CFCAP_OUT", "/tmp")) / "decision"


def _ep(args):
    seed, ks = args
    import torch
    torch.set_num_threads(1)
    from cfcap.t0 import qs
    from cfcap.decide import envelope_post as EP
    P, th = qs.load_params(), qs.load_theta()
    return pd.DataFrame(EP.episode(seed, P, th, k_screen=ks)).assign(K=ks)


def main():
    from cfcap.decide import envelope as En
    from cfcap.t0 import qs
    cal = pd.read_pickle(OUT / "gatepost_v2_0.99_K26.92_cal012_6000_6800.pkl")
    cal = cal.groupby("seed").filter(lambda g: bool(g.conv.all()))
    cal_eps = En.episodes_from_gate(cal, qs.LEG_NAMES)
    seeds = [s for s in range(7000, 8200) if s % 4 == 3][:48]
    Ks = (15.0, 26.924, 28.694, 60.0)
    with get_context("spawn").Pool(4) as p:
        dfs = p.map(_ep, [(s, k) for k in Ks for s in seeds], chunksize=2)
    df = pd.concat(dfs)
    rows = []
    kept = {}
    for k, g in df.groupby("K"):
        e = g.groupby("seed").agg(ns=("n_screen", "first"), conv=("conv", "all"), nk=("n_kept_v2", "first"))
        gc = g[g.seed.isin(e.index[e.conv])]
        r = En.run(En.episodes_from_gate(gc, qs.LEG_NAMES), cal_eps=cal_eps)
        ev = r[(r.predictor == "env_cal") & (r.policy == "evoi")]
        kept[k] = gc.sort_values(["seed", "cand", "leg"]).rho_pi_v2.to_numpy()
        rows.append(dict(K=k, n_seeds=len(e), n_conv=int(e.conv.sum()), n_screen_median=float(e.ns.median()),
                         n_screen_mean=float(e.ns.mean()), empty_share=float((e[e.conv].ns == 0).mean()),
                         n_kept_mean=float(e.nk.mean()), evoi_useful=ev.n_served.sum() / max(ev.oracle_served.sum(), 1),
                         evoi_ep_unsafe=float((ev.n_unsafe > 0).mean())))
    A = pd.DataFrame(rows)
    for k in Ks:
        A.loc[A.K == k, "bounds_equal_K26.9"] = bool(np.array_equal(np.nan_to_num(kept[k], nan=-9),
                                                                   np.nan_to_num(kept[26.924], nan=-9)))
    A.to_csv(REPO / "results/decision/hos_audit_K.csv", index=False)
    print(A.round(4).to_string(index=False))
    # (2) worst case: all 300 severity-3 test episodes, non-converged ones scored as abstain everywhere
    full = pd.read_pickle(OUT / "gatepost_v2_0.99_K26.92_L3_7000_8200.pkl")
    conv = full.groupby("seed").conv.all()
    r = En.run(En.episodes_from_gate(full, qs.LEG_NAMES), cal_eps=cal_eps)
    r["dropped"] = r.seed.map(~conv)
    w = r[r.predictor == "env_cal"].copy()
    m = w.dropped
    w.loc[m, "regret"] = w.loc[m, "regret"] + w.loc[m, "U"]
    w.loc[m, ["n_unsafe", "n_served"]] = 0
    w.loc[m, "probe_s"] = 0.0
    out = []
    for pol, g in w.groupby("policy"):
        out.append(dict(policy=pol, n_ep=len(g), n_dropped=int(g.dropped.sum()), ep_unsafe=float((g.n_unsafe > 0).mean()),
                        useful=g.n_served.sum() / max(g.oracle_served.sum(), 1), regret=g.regret.mean(),
                        probe_s=g.probe_s.mean()))
    W = pd.DataFrame(out)
    for b in ("clamp", "dom"):
        W.loc[W.policy == "evoi", f"paired_vs_{b}"] = str([round(x, 2) for x in En.paired(w, "env_cal", "evoi", b)[:3]])
    W.to_csv(REPO / "results/decision/hos_audit_worstcase.csv", index=False)
    print(W.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
