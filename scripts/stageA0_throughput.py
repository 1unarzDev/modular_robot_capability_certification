"""Stage A0 (exploratory): T1 throughput on one GPU.

Trials/s for (a) 3 s decision legs (cfcap.t1.capability, 10 legs per episode) and (b) the 12.5 s v2 probe routine, at
assembly sizes 1, 2, 4, 6, 8 modules, with heterogeneity HET_FULL, random faults (<= 2 per assembly) and the v2
self-model's DR around theta. Timed region = plant.simulate_bucket (latent prep + CUDA-graph capture + integration),
after one warm-up run per shape; peak VRAM = torch.cuda.max_memory_reserved over the timed runs.

Self-contained: needs torch + numpy, PYTHONPATH with cfcap and capxfer, and CAPXFER_THETA_DIR holding
theta_{plated,plateless}.json. No corpus, pandas or PyYAML needed.

  python scripts/stageA0_throughput.py --out results/stageA/A0_throughput.csv

Rented RTX 3090 shard (inputs pushed to /workspace/inputs/{cfcap, capxfer (repo root), theta/theta_*.json}; tested
locally with PyYAML, pandas and CAPXFER_ROOT absent):
  cd /workspace/inputs && PYTHONPATH=/workspace/inputs/cfcap:/workspace/inputs/capxfer \
    CAPXFER_THETA_DIR=/workspace/inputs/theta python cfcap/scripts/stageA0_throughput.py \
    --engines graph,compile --wheels 65536 --max-vram-gb 22 --out /workspace/out/cfcap/$CRANE_JOB_ID/A0_throughput.csv \
  && python cfcap/scripts/stageA0_throughput.py --engines compile --wheels 262144 --max-vram-gb 22 \
    --out /workspace/out/cfcap/$CRANE_JOB_ID/A0_throughput.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import platform
import time

import numpy as np
import torch

from cfcap.t1 import capability as cap
from cfcap.t1 import params as P
from cfcap.t1 import plant

COLS = ("gpu", "kind", "modules", "wheels", "B", "engine", "T_s", "sim_seconds", "trials_per_s", "trial_seconds_per_s",
        "episodes_per_s", "peak_vram_gb", "reps")


def run_case(kind, size, B, engine, reps, par, th, rng):
    asm = P.named_assembly(P.SIZE_SHAPE[size], par)
    dr = P.theta_dr(th)
    if kind == "legs":
        nep = max(1, B // len(cap.LEGS))
        lat = plant.sample_latents(asm, nep, rng, plant.HET_FULL, dr=dr,
                                   faults=plant.random_faults(nep, asm.n, rng))
        job, _ = cap.job(asm, par, lat)
        T = cap.T_LEG
        per_ep = len(cap.LEGS)
    else:
        lat = plant.sample_latents(asm, B, rng, plant.HET_FULL, dr=dr, faults=plant.random_faults(B, asm.n, rng))
        tab, T = plant.v2_probe(asm, par, B, jitter=False)
        job = dict(asm=asm, lat=lat, cmd=tab)
        per_ep = 1
    Bt = job["lat"]["mode"].shape[0]
    plant.simulate_bucket([job], par, T, engine=engine)            # warm-up (compile / allocator)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        R = plant.simulate_bucket([job], par, T, engine=engine)[0]
        del R
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    return dict(gpu=torch.cuda.get_device_name(0), kind=kind, modules=size, wheels=asm.n, B=Bt, engine=engine,
                T_s=round(T, 3), sim_seconds=round(dt, 4), trials_per_s=round(Bt / dt, 1),
                trial_seconds_per_s=round(Bt * T / dt, 1), episodes_per_s=round(Bt / dt / per_ep, 1),
                peak_vram_gb=round(torch.cuda.max_memory_reserved() / 1e9, 3), reps=reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,2,4,6,8")
    ap.add_argument("--kinds", default="legs,v2")
    ap.add_argument("--engines", default="graph,compile")
    ap.add_argument("--wheels", type=int, default=65536, help="B * n per run (sets B per size)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--plate", default="plated")
    ap.add_argument("--max-vram-gb", type=float, default=7.5, help="refuse a case whose warm-up exceeds this")
    ap.add_argument("--out", default="results/stageA/A0_throughput.csv")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    par, th = P.base_params(a.plate), P.load_theta(a.plate)
    rng = np.random.default_rng(20261125)
    rows = []
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        for engine in a.engines.split(","):
            for kind in a.kinds.split(","):
                for size in [int(s) for s in a.sizes.split(",")]:
                    B = max(len(cap.LEGS), a.wheels // (4 * size))
                    r = run_case(kind, size, B, engine, a.reps, par, th, rng)
                    print(json.dumps(r), flush=True)
                    if r["peak_vram_gb"] > a.max_vram_gb:
                        print(f"WARNING peak VRAM {r['peak_vram_gb']} GB > {a.max_vram_gb}", flush=True)
                    w.writerow(r)
                    fh.flush()
                    rows.append(r)
                    torch.cuda.empty_cache()
    job = os.environ.get("CRANE_JOB_ID")
    if job:
        (out.parent / "manifest.json").write_text(json.dumps(dict(job=job, host=platform.node(), rows=len(rows),
                                                                  args=vars(a)), indent=1))


if __name__ == "__main__":
    main()
