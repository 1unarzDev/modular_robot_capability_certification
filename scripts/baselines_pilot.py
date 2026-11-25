"""Baselines pilot (exploratory): six baselines on ~300 T1 episodes, each feeding conformal.core.SimultaneousLower.

  python scripts/baselines_pilot.py gen  [--per-shape 100]           # episodes -> $CFCAP_OUT/baselines/episodes.pt
  python scripts/baselines_pilot.py run  --baseline NAME [--max-min 14] [--shapes pair,square,2x3]
  python scripts/baselines_pilot.py eval                              # -> results/baselines/*.csv

Episodes: parents pair / square / 2x3 (plated, heterogeneity HET_FULL, 0-2 faults), candidates = every connected
proper child, labels = T1 closed-loop rho_rel on 10 decision legs (cfcap.baselines.episode).
Baselines (cfcap.baselines): nominal, rls_interval, capxfer, enum_known, nuisance_smc, oracle.
Evaluation: 2-fold cross-fit split conformal (folds alternate within each shape), alpha = 0.1 simultaneous over every
(candidate, leg) of an episode; decisions commit iff lower >= tau, tau in {0.6, 0.7, 0.8}. capxfer's raw certificate
(its own Theorem 5 levels, no recalibration) is reported as `capxfer_raw`.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch

from cfcap.baselines import base
from cfcap.baselines import episode as E

OUT = pathlib.Path(os.environ.get("CFCAP_OUT", pathlib.Path.home() / ".cache/modular-runs/2026-11/cfcap")) / "baselines"
RES = pathlib.Path(__file__).resolve().parents[1] / "results/baselines"
SHAPES = ("pair", "square", "2x3")
ALPHA = 0.1


def make(name):
    if name == "nominal":
        from cfcap.baselines.simple import Nominal
        return Nominal()
    if name == "oracle":
        from cfcap.baselines.simple import Oracle
        return Oracle()
    if name == "rls_interval":
        from cfcap.baselines.rls import RLSInterval
        return RLSInterval()
    if name == "capxfer":
        from cfcap.baselines.capxfer_cert import CapxferCert
        return CapxferCert(alpha=ALPHA)
    if name == "enum_known":
        from cfcap.baselines.bayes import Enumeration
        return Enumeration()
    if name == "nuisance_smc":
        from cfcap.baselines.bayes import NuisanceSMC
        return NuisanceSMC()
    raise KeyError(name)


def cmd_gen(a):
    OUT.mkdir(parents=True, exist_ok=True)
    eps = []
    for si, s in enumerate(SHAPES):
        t0 = time.time()
        eps += E.generate(s, a.per_shape, seed=9000 + si, eid0=len(eps))
        print(s, len(eps), f"{time.time() - t0:.1f}s", flush=True)
    E.save(eps, OUT / "episodes.pt")


def cmd_run(a):
    eps = E.load(OUT / "episodes.pt")
    shapes = a.shapes.split(",")
    eps = [e for e in eps if e.shape in shapes]
    if a.per_shape:
        eps = [e for s in shapes for e in [x for x in eps if x.shape == s][:a.per_shape]]
    b = make(a.baseline)
    f = OUT / f"pred_{a.baseline}.pt"
    done = torch.load(f, weights_only=False) if f.exists() and not a.fresh else {}
    t_start = time.time()
    # interleave shapes so a time-limited run keeps every shape represented
    by = {s: [e for e in eps if e.shape == s] for s in shapes}
    order = [e for k in range(max(len(v) for v in by.values())) for s in shapes if k < len(by[s]) for e in [by[s][k]]]
    for e in order:
        if e.eid in done:
            continue
        if time.time() - t_start > a.max_min * 60:
            print("time limit", flush=True)
            break
        t0 = time.perf_counter()
        p = b(e)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        info = {k: v for k, v in p.info.items() if isinstance(v, (int, float, str))}
        done[e.eid] = dict(point=p.point, scale=p.scale, lower=p.lower, seconds=sec, info=info)
        if len(done) % 10 == 0:
            torch.save(done, f)
            print(a.baseline, len(done), e.shape, f"{sec:.2f}s", info, flush=True)
    torch.save(done, f)
    print("saved", f, len(done), flush=True)


def _arrays(eps, preds, key="point"):
    W = max(e.n_cand for e in eps) * base.L
    pt = base.pad([preds[e.eid][key] if preds[e.eid][key] is not None else np.full((e.n_cand, base.L), np.nan)
                   for e in eps], W)
    return pt, W


def cmd_eval(a):
    RES.mkdir(parents=True, exist_ok=True)
    eps_all = E.load(OUT / "episodes.pt")
    rows, timing = [], []
    for name in ("nominal", "rls_interval", "capxfer", "enum_known", "nuisance_smc", "oracle"):
        f = OUT / f"pred_{name}.pt"
        if not f.exists():
            continue
        preds = torch.load(f, weights_only=False)
        eps = [e for e in eps_all if e.eid in preds]
        W = max(e.n_cand for e in eps) * base.L
        truth = base.pad([e.rho_rel for e in eps], W)
        hf = base.pad([np.repeat(e.cand_has_fault()[:, None], base.L, 1).astype(float) for e in eps], W)
        point = base.pad([preds[e.eid]["point"] for e in eps], W)
        scale = base.pad([preds[e.eid]["scale"] for e in eps], W)
        point = np.where(np.isnan(truth), np.nan, point)
        shapes = np.array([e.shape for e in eps])
        folds = np.zeros(len(eps), int)
        for s in SHAPES:
            idx = np.flatnonzero(shapes == s)
            folds[idx] = np.arange(len(idx)) % 2
        variants = {name: base.cross_fit_lower(point, scale, truth, folds, ALPHA)}
        if name == "capxfer":
            raw = base.pad([preds[e.eid]["lower"] for e in eps], W)
            variants["capxfer_raw"] = (np.where(np.isnan(truth), np.nan, raw), {})
        for vname, (lower, q) in variants.items():
            for shp in ("all",) + SHAPES:
                m = np.ones(len(eps), bool) if shp == "all" else shapes == shp
                if not m.any():
                    continue
                for r in base.metrics(lower, truth, hf, ep_mask=m):
                    rows.append(dict(baseline=vname, shape=shp, alpha=ALPHA, q_fold0=q.get(0, np.nan),
                                     q_fold1=q.get(1, np.nan), **r))
        sec = np.array([preds[e.eid]["seconds"] for e in eps])
        for shp in SHAPES:
            m = shapes == shp
            if m.any():
                nc = eps[int(np.flatnonzero(m)[0])].n_cand
                timing.append(dict(baseline=name, shape=shp, n_ep=int(m.sum()), n_cand=nc,
                                   sec_per_decision_median=float(np.median(sec[m])),
                                   sec_per_decision_mean=float(sec[m].mean()),
                                   sec_per_candidate_median=float(np.median(sec[m]) / nc)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "pilot_metrics.csv", index=False)
    pd.DataFrame(timing).to_csv(RES / "pilot_timing.csv", index=False)
    cols = ["baseline", "tau", "n_ep", "cert_frac", "cert_frac_faulted", "unsafe_commit_rate", "ep_any_unsafe",
            "ep_miscover", "q_fold0", "q_fold1"]
    print(df[df["shape"] == "all"][cols].round(4).to_string(index=False))
    print(pd.DataFrame(timing).round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "run", "eval"])
    ap.add_argument("--per-shape", type=int, default=100)
    ap.add_argument("--baseline")
    ap.add_argument("--max-min", type=float, default=14.0)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run" and a.per_shape == 100:
        a.per_shape = 0
    dict(gen=cmd_gen, run=cmd_run, eval=cmd_eval)[a.cmd](a)


if __name__ == "__main__":
    main()
