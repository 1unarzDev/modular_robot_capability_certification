"""Runner task `smc` (cfcap.runner contract): GPU nuisance-SMC baseline (and optionally the known-nuisance enumeration)
over T1 episodes generated from seeds on the box.

    python -m cfcap.runner smc --shard K --n-shards N --split test --episodes 500 [--per-unit 2] [--with-enum]

Episodes. Episode j of a split is a pure function of (split, j): parent shape SHAPES[j % 3], seed
unit_seed("cfcap-baselines-episode", split, j), generated with `cfcap.baselines.episode.generate` (T1, plated,
HET_FULL heterogeneity, 0-2 faults, v2 probe, connected children as candidates, T1 rho_rel labels). The same (split, j)
gives the same episode on any host, so the local pilot's conformal calibration split can be regenerated anywhere.
Splits: "test" and "cal" use disjoint seed streams (the split name is part of the seed).

One unit = `per_unit` consecutive episodes of this shard's slice. Within a unit the state (finished episodes' rows)
is checkpointed after every episode; ctx.should_stop() is checked between episodes.

Rows (columnar, one per episode x candidate x leg): eid, shape_id (0 pair, 1 square, 2 2x3), cand, leg, truth
(rho_rel), has_fault, smc_point, smc_scale, smc_q05 (+ enum_point, enum_scale, enum_q05 with --with-enum), and
per-episode columns repeated on each row: smc_s, smc_sims, smc_beta, smc_stages, enum_s, peak_vram_gb.
The conformal layer (SimultaneousLower) is applied offline (scripts/baselines_pilot.py) to these columns.
"""
from __future__ import annotations

import time

import numpy as np

from cfcap.runner import Stop, Task, chunk_units, shard_range, unit_seed

SHAPES = ("pair", "square", "2x3")


def _args(p):
    p.add_argument("--split", default="test")
    p.add_argument("--episodes", type=int, default=500, help="episodes in the split (all shards)")
    p.add_argument("--per-unit", type=int, default=2)
    p.add_argument("--shapes", default=",".join(SHAPES))
    p.add_argument("--K", type=int, default=96, help="fault hypotheses kept after the screen")
    p.add_argument("--M", type=int, default=24, help="nuisance draws per kept hypothesis")
    p.add_argument("--N", type=int, default=128, help="particles after resampling")
    p.add_argument("--mh-steps", type=int, default=2)
    p.add_argument("--max-stages", type=int, default=12)
    p.add_argument("--n-prop", type=int, default=32, help="particles pushed to the candidates")
    p.add_argument("--with-enum", action="store_true", help="also the known-nuisance exact enumeration")


def _units(a, shard, n_shards):
    return chunk_units(f"{a.split}", shard_range(a.episodes, shard, n_shards), a.per_unit)


def episode(split, j, shapes=SHAPES, device="cuda"):
    from cfcap.baselines import episode as E
    shape = shapes[j % len(shapes)]
    seed = unit_seed("cfcap-baselines-episode", split, j) % (1 << 62)
    return E.generate(shape, 1, seed=seed, device=device, eid0=j)[0]


def _one(ep, a, shapes):
    import torch
    from cfcap.baselines.bayes import Enumeration, NuisanceSMC
    torch.cuda.reset_peak_memory_stats()
    smc = NuisanceSMC(K=a.K, M=a.M, N=a.N, mh_steps=a.mh_steps, n_prop=a.n_prop, seed=unit_seed("smc", ep.eid))
    smc.kw["max_stages"] = a.max_stages
    t0 = time.perf_counter()
    p = smc(ep)
    torch.cuda.synchronize()
    smc_s = time.perf_counter() - t0
    C, L = ep.rho_rel.shape
    hf = np.repeat(ep.cand_has_fault()[:, None], L, 1)
    rows = dict(eid=np.full(C * L, ep.eid), shape_id=np.full(C * L, shapes.index(ep.shape)),
                cand=np.repeat(np.arange(C), L), leg=np.tile(np.arange(L), C), truth=ep.rho_rel.ravel(),
                has_fault=hf.ravel().astype(np.int8), smc_point=p.point.ravel(), smc_scale=p.scale.ravel(),
                smc_q05=p.lower.ravel())
    per = dict(smc_s=smc_s, smc_sims=p.info.get("n_sims", 0), smc_beta=p.info.get("beta", np.nan),
               smc_stages=p.info.get("n_stages", 0), enum_s=np.nan)
    if a.with_enum:
        t0 = time.perf_counter()
        q = Enumeration()(ep)
        torch.cuda.synchronize()
        per["enum_s"] = time.perf_counter() - t0
        rows.update(enum_point=q.point.ravel(), enum_scale=q.scale.ravel(), enum_q05=q.lower.ravel())
    per["peak_vram_gb"] = torch.cuda.max_memory_reserved() / 1e9
    for k, v in per.items():
        rows[k] = np.full(C * L, float(v))
    return rows


def _run(unit, ctx):
    from cfcap.runner import parse_chunk
    a = ctx.args
    shapes = tuple(a.shapes.split(","))
    r = parse_chunk(unit)
    lo, hi = r.start, r.stop
    st = ctx.load_state() or {"done": {}}
    for j in range(lo, hi):
        if j in st["done"]:
            continue
        if ctx.should_stop():
            ctx.save_state(st)
            raise Stop()
        ctx.progress((j - lo) / max(hi - lo, 1), phase=f"smc episode {j}")
        ep = episode(a.split, j, shapes)
        st["done"][j] = _one(ep, a, shapes)
        ctx.save_state(st)
    parts = [st["done"][j] for j in sorted(st["done"])]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ctx.meta.update(episodes=len(parts), smc_s_mean=float(np.mean([p["smc_s"][0] for p in parts])),
                    peak_vram_gb=float(max(p["peak_vram_gb"][0] for p in parts)))
    return out


def _summary(recs, a):
    m = [r.get("meta", {}) for r in recs]
    n = sum(x.get("episodes", 0) for x in m)
    return {"smc.episodes": n,
            "smc.s_per_episode_mean": sum(x.get("smc_s_mean", 0) * x.get("episodes", 0) for x in m) / max(n, 1),
            "smc.peak_vram_gb": max([x.get("peak_vram_gb", 0) for x in m] or [0])}


TASK = Task("smc", _units, _run, _args, _summary, ("numpy", "torch", "capxfer"),
            "nuisance-SMC (+ optional known-nuisance enumeration) baseline over seeded T1 episodes")
