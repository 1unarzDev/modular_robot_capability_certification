"""cfcap.runner Task for the nuisance-marginalised T1 posterior ambiguity study (role nuisance; docs/board/nuisance.md).

    python -m cfcap.runner nuisance --task-spec cfcap.nuisance_task:TASK --shard K --n-shards N \
        --shapes pair,square,2x3 --probe-sets v2,v2+clamp,v2+tug,v2+moddiff,v2+clamp+moddiff+tug \
        --episodes pair=96,square=64,2x3=48 --ep-per-unit pair=8,square=8,2x3=4 --nseed 1909 \
        --sweep-deltas 0.02,0.04,0.12,0.16

Work items = (shape, probe set, episode block); one runner unit = one item. Items are ordered episode-block-major
(all probe sets of a block next to each other) and dealt round-robin to shards, so every shard carries a balanced mix
of shapes and probe sets and an interrupted campaign still has matched probe comparisons. Episode truth, merges and
observation noise derive from numpy seeds of (nseed, shape, episode) only: identical on every host and probe set.
Unit rows = one per (episode, candidate, decision leg) with the 64-draw posterior q05/q50/q95 and the truth of the
primary label rho_dec = rho_abs / rho_ref(shape, leg) (columns truth, q05, q50, q95) and of rho_rel (*_rel columns); per-episode SMC diagnostics and timings go to the unit sidecar (meta.episodes).
"""
from __future__ import annotations

from cfcap.runner import Stop, Task

SHAPE_ORDER = ("pair", "square", "2x3")


def _kv(s, cast=int):
    return {k: cast(v) for k, v in (x.split("=") for x in s.split(",") if x)}


def _args(p):
    p.add_argument("--shapes", default="pair,square,2x3")
    p.add_argument("--probe-sets", default="v2,v2+clamp,v2+tug")
    p.add_argument("--episodes", default="pair=96,square=64,2x3=48", help="episodes per shape")
    p.add_argument("--ep-per-unit", default="pair=8,square=8,2x3=4")
    p.add_argument("--nseed", type=int, default=1909, help="episode seed (truth / merges / noise)")
    p.add_argument("--particles", type=int, default=256)
    p.add_argument("--draws", type=int, default=64)
    p.add_argument("--max-stages", type=int, default=150)
    p.add_argument("--engine", default="graph")
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--delta", type=float, default=0.08, help="per-wheel motor spread half-width of the main grid")
    p.add_argument("--sweep-deltas", default="", help="extra deltas run with probe set v2 only (F1 curve)")
    p.add_argument("--sweep-episodes", default="", help="episodes per shape for the delta sweep (default --episodes)")


def _items(a):
    eps, per = _kv(a.episodes), _kv(a.ep_per_unit)
    sw = _kv(a.sweep_episodes) if a.sweep_episodes else eps
    out = []
    shapes = [s for s in SHAPE_ORDER if s in a.shapes.split(",")]
    d0 = f"d{round(a.delta * 100):02d}"
    for sh in shapes:
        for b0 in range(0, eps[sh], per[sh]):
            for ps in a.probe_sets.split(","):
                out.append(f"{sh}.{ps.replace('+', 'P')}.{d0}.e{b0:05d}-{min(eps[sh], b0 + per[sh]):05d}")
    for dl in [float(x) for x in a.sweep_deltas.split(",") if x]:
        for sh in shapes:
            for b0 in range(0, sw[sh], per[sh]):
                out.append(f"{sh}.v2.d{round(dl * 100):02d}.e{b0:05d}-{min(sw[sh], b0 + per[sh]):05d}")
    return out


def _units(a, shard, n_shards):
    return _items(a)[shard::n_shards]


def _run(unit, ctx):
    from cfcap import nuisance_post as NP
    a = ctx.args
    sh, ps, dd, rng_ = unit.split(".", 3)
    ps = ps.replace("P", "+")
    delta = int(dd[1:]) / 100
    lo, hi = (int(x) for x in rng_[1:].split("-"))
    if ctx.should_stop():
        raise Stop("before unit")
    ctx.progress(0.0, phase=f"{sh} {ps} episodes {lo}-{hi}")
    dec, eps = NP.run_batch(sh, ps, list(range(lo, hi)), seed=a.nseed, particles=a.particles, draws=a.draws,
                            max_stages=a.max_stages, engine=a.engine, chunk=a.chunk, delta=delta)
    ctx.meta["episodes"] = eps
    ctx.meta["wall_per_ep"] = sum(e["total_s_per_ep"] for e in eps) / max(len(eps), 1)
    return dec


def _summary(recs, a):
    eps = [e for r in recs for e in r["meta"].get("episodes", [])]
    out = {"nuis.units": len(recs), "nuis.episodes": len(eps)}
    if eps:
        out["nuis.s_per_ep_mean"] = sum(e["total_s_per_ep"] for e in eps) / len(eps)
        out["nuis.found_share"] = sum(e["ll_true"] - e["ll_best"] <= 10 for e in eps) / len(eps)
        out["nuis.forced"] = sum(e["forced"] for e in eps)
    return out


TASK = Task("nuisance", _units, _run, _args, _summary, ("numpy", "torch"),
            "nuisance-marginalised T1 SMC posterior -> rho_dec decision ambiguity (rho_rel secondary) (role nuisance)")


# ----------------------------------------------------------------------------- T0 A2/A3 scale-up (seed blocks)
def _a2_args(p):
    p.add_argument("--seeds", default="1000:3000", help="T0 parent seeds lo:hi (A2 pilot used 0:100)")
    p.add_argument("--per-unit", type=int, default=10)
    p.add_argument("--plate", default="plated")
    p.add_argument("--threads", type=int, default=1)


def _a2_units(a, shard, n_shards):
    lo, hi = map(int, a.seeds.split(":"))
    items = [f"a2-{s:06d}-{min(hi, s + a.per_unit):06d}" for s in range(lo, hi, a.per_unit)]
    return items[shard::n_shards]


def _a2_run(unit, ctx):
    import numpy as np
    import pandas as pd
    import torch
    from cfcap.t0 import qs, stagea
    a = ctx.args
    torch.set_num_threads(a.threads)
    P, th = qs.load_params(a.plate), qs.load_theta(a.plate)
    _, lo, hi = unit.split("-")
    recs, rows = [], []
    for k, seed in enumerate(range(int(lo), int(hi))):
        if ctx.should_stop():
            raise Stop("between episodes")
        rec, rr = stagea.run_episode(seed, P, th)
        recs.append(rec)
        rows += rr
        ctx.progress((k + 1) / (int(hi) - int(lo)))
    ctx.meta["episodes"] = recs
    df = pd.DataFrame(rows)
    return {c: (df[c].to_numpy() if df[c].dtype != object else df[c].astype(str).to_numpy()) for c in df.columns}


def _a2_summary(recs, a):
    eps = [e for r in recs for e in r["meta"].get("episodes", [])]
    return {"a2.units": len(recs), "a2.episodes": len(eps),
            "a2.s_per_ep_mean": sum(float(e.get("t_total_s", 0)) for e in eps) / max(len(eps), 1)}


A2_TASK = Task("t0_a2", _a2_units, _a2_run, _a2_args, _a2_summary, ("numpy", "torch", "pandas"),
               "Stage A2/A3 T0 oracle episodes by seed block (stagea.run_episode); analyse with stageA2/A3 scripts")
