"""T2 throughput: 3 s decision legs per second, numba (CPU threads = NUMBA_NUM_THREADS) and JAX (GPU, float32).

    NUMBA_NUM_THREADS=4 scripts/heavy.sh $PY scripts/stress_throughput.py numba
    scripts/heavy.sh --gpu $PY scripts/stress_throughput.py jax
Appends to results/stress/T2_throughput.csv.
"""
import csv
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main(backend):
    from cfcap.t1 import capability as cap1, params as P, plant
    from cfcap.t2 import build, kernel
    par = P.base_params("plated")
    dr = P.theta_dr(P.load_theta("plated"))
    K = 150
    rows = []
    sizes = {"numba": (256, 1024), "jax": (4096, 16384)}[backend]
    for name in (("pair", "square", "2x3") if backend == "numba" else ("square", "2x3")):
        asm = P.named_assembly(name, par)
        tabs, _ = cap1.leg_tables(asm, par)
        for E in sizes:
            rng = np.random.default_rng(0)
            lat = plant.sample_latents(asm, E, rng, plant.HET_FULL, dr=dr, faults=plant.random_faults(E, asm.n, rng),
                                       device="cpu")
            for eff in ("base", "all"):
                S = build.episode_specs(asm, lat, par, build.ABLATIONS[eff], seed=0)
                which = np.arange(E) % 10
                kernel.run(dict(S, **{k: v[:8] for k, v in S.items() if k not in build.FLAGS}), tabs.astype(float),
                           which[:8], K, backend) if backend == "numba" else None   # compile
                if backend == "jax":
                    kernel.run(S, tabs.astype(float), which, K, backend)            # compile at this shape
                t0 = time.time()
                kernel.run(S, tabs.astype(float), which, K, backend)
                dt = time.time() - t0
                r = dict(backend=backend, threads=os.environ.get("NUMBA_NUM_THREADS", "") if backend == "numba" else "gpu",
                         shape=name, wheels=asm.n, E=E, effects=eff, seconds=round(dt, 3), legs_per_s=round(E / dt, 1))
                print(r, flush=True)
                rows.append(r)
    f = ROOT / "results/stress/T2_throughput.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
