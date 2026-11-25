"""Per-episode robot-side compute time (from the noisy observation: screen + fault posterior + per-hypothesis bounds;
label simulation excluded) of the envelope_post pipeline, one
thread per process, for the knee-tolerant screen (K from results/decision/screen_cal_knee_5000_5600.csv) and K = 5.

    scripts/heavy.sh $PY scripts/decide_latency.py --seeds 3900:3948 --workers 4

Writes results/decision/paper/latency_decision.csv (contract 1e: method, shape, n_wheels, sec_per_decision (median),
hardware, status, n) and results/decision/latency_envpost_episodes.csv (per episode).
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _one(args):
    seed, ks = args
    import torch
    torch.set_num_threads(1)
    from cfcap.t0 import qs
    from cfcap.decide import envelope_post as EP
    P, th = qs.load_params(), qs.load_theta()
    t0 = time.perf_counter()
    r = EP.episode(seed, P, th, k_screen=ks)
    dt = time.perf_counter() - t0
    return dict(seed=seed, k_screen=ks, episode_wall_s=dt, wall_s=float(r["robot_s"][0]),
                parent_size=int(r["parent_size"][0]),
                n_screen=int(r["n_screen"][0]) if "n_screen" in r else 0, n_cand=len(set(r["cand"])),
                law=int(r["law"][0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3900:3948")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from cfcap.conformal.core import split_quantile
    zd = pd.read_csv(REPO / "results/decision/screen_cal_knee_5000_5600.csv")
    K = split_quantile(zd[zd.conv].z_true.to_numpy(), 0.01)
    s0, s1 = map(int, a.seeds.split(":"))
    jobs = [(s, K) for s in range(s0, s1)] + [(s, 5.0) for s in range(s0, s1)]
    with get_context("spawn").Pool(a.workers) as p:
        rows = p.map(_one, jobs, chunksize=1)
    d = pd.DataFrame(rows)
    from cfcap.t0 import stagea  # noqa: F401  (parent size from the episode's candidates is enough)
    d.to_csv(REPO / "results/decision/latency_envpost_episodes.csv", index=False)
    hw = f"{platform.processor() or platform.machine()} 1 thread (local, shared)"
    out = []
    for ks, g in d.groupby("k_screen"):
        name = "envpost_ktol" if ks > 5 else "envpost_k5"
        for ps, gg in g.groupby("parent_size"):
            out.append(dict(method=name, shape=f"{ps}mod", n_wheels=4 * ps, sec_per_decision=float(gg.wall_s.median()),
                            hardware=hw, status="exploratory", n=len(gg)))
        out.append(dict(method=name, shape="all", n_wheels=np.nan, sec_per_decision=float(g.wall_s.median()),
                        hardware=hw, status="exploratory", n=len(g)))
    o = pd.DataFrame(out)
    (REPO / "results/decision/paper").mkdir(parents=True, exist_ok=True)
    o.to_csv(REPO / "results/decision/paper/latency_decision.csv", index=False)
    print(o.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
