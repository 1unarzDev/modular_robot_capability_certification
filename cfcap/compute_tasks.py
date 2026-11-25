"""Runner adapters (cfcap.runner Tasks) over other roles' public entry points, so those roles need not know the runner.

gen_shard: the generator's `cfcap.gen.shard.write_shard` (its own atomic npz + sidecar + resume). One runner unit is
one generator shard; runner shard k of N takes generator shards shard_range(G, k, N) of G. The data land under
OUT/gen/<plate>/<split>/shard_<j>.npz in the generator's schema (docs/board/generator.md), so directories from many
boxes merge by copying.

    python -m cfcap.runner gen_shard --shard 0 --n-shards 8 --split test --gen-shards 32 --episodes 2000
"""
from __future__ import annotations

from cfcap.runner import Task, shard_range


def _gen_args(p):
    p.add_argument("--split", required=True)
    p.add_argument("--plate", default="plated")
    p.add_argument("--gen-shards", type=int, required=True, help="generator shards in the split (G)")
    p.add_argument("--episodes", type=int, required=True, help="episodes per generator shard (fixed per split)")
    p.add_argument("--probes", default="v2")
    p.add_argument("--engine", default="compile")
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--max-wheels", type=int, default=1 << 17)


def _gen_units(a, shard, n_shards):
    return [f"{a.plate}-{a.split}-g{j:05d}" for j in shard_range(a.gen_shards, shard, n_shards)]


def _gen_run(unit, ctx):
    from cfcap.gen import shard as SH          # the generator role's module (torch, capxfer)
    a = ctx.args
    j = int(unit.rsplit("-g", 1)[1])
    ctx.progress(0.0, phase=f"gen {a.split} shard {j}")
    info = SH.write_shard(ctx.out_dir / "gen", a.plate, a.split, j, a.episodes, tuple(a.probes.split(",")),
                          engine=a.engine, block=a.block, max_wheels=a.max_wheels, log=None)
    ctx.meta.update({k: v for k, v in info.items() if k in ("file", "episodes", "n_cand", "n_asm", "n_wheel",
                                                            "bytes", "sha256", "gen_seconds", "write_seconds")})
    return None                                 # the generator wrote its own columnar file


def _gen_summary(recs, a):
    m = [r["meta"] for r in recs]
    return {"gen.episodes": sum(x.get("episodes", 0) for x in m), "gen.candidates": sum(x.get("n_cand", 0) for x in m),
            "gen.bytes": sum(x.get("bytes", 0) for x in m),
            "gen.seconds_mean": sum(x.get("gen_seconds", 0) for x in m) / max(len(m), 1)}


GEN_SHARD = Task("gen_shard", _gen_units, _gen_run, _gen_args, _gen_summary, ("numpy", "torch"),
                 "generator shards via cfcap.gen.shard.write_shard")
