"""cfcap.runner task `stress_merge` (role stress): confirmatory C-MERGE pairs (docs/CONFIRMATORY.md §2).

    python -m cfcap.runner stress_merge --task-spec cfcap.t2.merge_task:TASK --shard K --n-shards N \
        --arm P0 --pairs 1000 --per-unit 25

Seed root [92599, 21] (claim code 21): pair p of arm a draws from default_rng([92599, 21, a == P5, p]) (cfcap.t2.merge).
One unit = `--per-unit` consecutive pairs of one arm; checkpoint after every pair. Rows = one per (pair, merge) with
per-leg columns: truth_T1.l, truth_T2.l, comp_oracle.l, enum_point.l, enum_lower.l, enum_scale.l, copy_parent.l,
detector_gated.l (fixed T1 reference), plus arm, pair, merge, n_mod, n_fault. No margin is applied here: the frozen
exploratory margins are applied by scripts/stress_conf_merge.py. Requires torch + CUDA (T1, enum_known) and numba
(T2; backend auto falls back to jax, then numpy), input crane_ml.
"""
from __future__ import annotations

import numpy as np

from cfcap.runner import Stop, Task, shard_range
from cfcap.t2 import deps  # noqa: F401

ROOT = (92599, 21)


def _args(p):
    p.add_argument("--arm", default="P0", choices=("P0", "P5"))
    p.add_argument("--pairs", type=int, default=1000)
    p.add_argument("--per-unit", type=int, default=25)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", default="auto")
    p.add_argument("--root", default="92599,21", help="seed root; anything but 92599,21 is a smoke/exploratory run")


def _units(a, shard, n_shards):
    nu = -(-a.pairs // a.per_unit)
    return [f"merge-{a.arm}-u{j:04d}" for j in shard_range(nu, shard, n_shards)]


def _run(unit, ctx):
    from cfcap.t1 import params as P
    from cfcap.t2 import merge as MG
    a = ctx.args
    j = int(unit.rsplit("-u", 1)[1])
    par = P.base_params("plated")
    dr = P.theta_dr(P.load_theta("plated"))
    pairs = list(range(j * a.per_unit, min(a.pairs, (j + 1) * a.per_unit)))
    st = ctx.load_state() or {"done": [], "rows": []}
    for k, p in enumerate(pairs):
        if p in st["done"]:
            continue
        if ctx.should_stop():
            ctx.save_state(st)
            raise Stop("stress_merge checkpoint")
        rec = MG.gen_pair(tuple(int(x) for x in a.root.split(",")), a.arm, p, par, dr)
        o = MG.score_pair(rec, par, a.device, t2_seed=92599 * 1000 + 21 * 10 + int(a.arm == "P5"), backend=a.backend)
        for i in range(len(o["merge"])):
            r = dict(arm=a.arm, pair=p, merge=int(o["merge"][i]), n_mod=int(o["n_mod"][i]), n_fault=int(o["n_fault"][i]))
            for key in ("truth_T1", "truth_T2", "comp_oracle", "enum_point", "enum_lower", "enum_scale", "copy_parent",
                        "detector_gated"):
                for l, v in enumerate(o[key][i]):
                    r[f"{key}.{l}"] = float(v)
            st["rows"].append(r)
        st["done"].append(p)
        ctx.save_state(st)
        ctx.progress((k + 1) / len(pairs), phase=f"merge {a.arm} pair {p}")
    ctx.meta.update(pairs=len(pairs), merges=len(st["rows"]))
    rows = st["rows"]
    if not rows:
        return None
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def _summary(recs, a):
    m = [r.get("meta", {}) for r in recs]
    return {"merge.pairs": sum(x.get("pairs", 0) for x in m), "merge.merges": sum(x.get("merges", 0) for x in m)}


TASK = Task("stress_merge", _units, _run, _args, _summary, ("numpy", "torch"),
            "confirmatory C-MERGE pairs: truths T1/T2, comp_oracle, composed enum_known, copy_parent, detector_gated")
