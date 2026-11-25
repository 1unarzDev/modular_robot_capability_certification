"""cfcap.runner task `stress_gate` (role stress): C-GATE rows in T2 (cfcap/t2/gate.py; docs/CONFIRMATORY.md §3, A-GATE).

    python -m cfcap.runner stress_gate --task-spec cfcap.t2.gate_task:TASK --shard K --n-shards N \
        --split conf_test --start 3040 --stop 5040 --sevs 0,1,2,3 --root 92599,22 --per-unit 10 --spare-shard0

One unit = `--per-unit` consecutive episode indices, every severity (paired). Rows = (index, severity, candidate, leg)
with truth (T2, fixed T1 reference), rho_nom and per probe set (v2, v2clamp) rho_par, rho_pi, dom, undom_speed,
undom_torque, misfit. Exploratory calibration episodes: the same task with --split cal --root 92599,97 (any index
range). --spare-shard0: shard 0 runs one exploratory unit (split test, indices 90000+, root 92599,98), never scored.
CPU only (numba T2 + numpy T0); torch is imported for the generator / T1 reference.
"""
from __future__ import annotations

import numpy as np

from cfcap.runner import Stop, Task, shard_range
from cfcap.t2 import deps  # noqa: F401

L = 10


def _args(p):
    p.add_argument("--split", default="conf_test")
    p.add_argument("--start", type=int, default=3040)
    p.add_argument("--stop", type=int, default=5040)
    p.add_argument("--sevs", default="0,1,2,3")
    p.add_argument("--root", default="92599,22")
    p.add_argument("--per-unit", type=int, default=10)
    p.add_argument("--backend", default="auto")
    p.add_argument("--spare-shard0", action="store_true")


def _units(a, shard, n_shards):
    nu = -(-(a.stop - a.start) // a.per_unit)
    if a.spare_shard0:
        if shard == 0:
            return ["gate-spare"]
        return [f"gate-{a.split}-u{j:04d}" for j in shard_range(nu, shard - 1, n_shards - 1)]
    return [f"gate-{a.split}-u{j:04d}" for j in shard_range(nu, shard, n_shards)]


def _run(unit, ctx):
    from cfcap.t2 import gate as GT
    a = ctx.args
    sevs = tuple(int(x) for x in a.sevs.split(","))
    if unit == "gate-spare":
        split, root, idx = "test", (92599, 98), list(range(90000, 90000 + a.per_unit))
    else:
        j = int(unit.rsplit("-u", 1)[1])
        split, root = a.split, tuple(int(x) for x in a.root.split(","))
        idx = list(range(a.start + j * a.per_unit, min(a.stop, a.start + (j + 1) * a.per_unit)))
    st = ctx.load_state() or {"done": [], "rows": []}
    for k, i in enumerate(idx):
        if i in st["done"]:
            continue
        if ctx.should_stop():
            ctx.save_state(st)
            raise Stop("stress_gate checkpoint")
        for r in GT.gate_episode(i, sevs, split, root, backend=a.backend):
            for l in range(L):
                row = {kk: (v[l] if isinstance(v, np.ndarray) else v) for kk, v in r.items()}
                row["leg"] = l
                st["rows"].append(row)
        st["done"].append(i)
        ctx.save_state(st)
        ctx.progress((k + 1) / len(idx), phase=f"gate {split} {i}")
    ctx.meta.update(episodes=len(idx), split=split, root=list(root))
    rows = st["rows"]
    if not rows:
        return None
    return {kk: np.array([r[kk] for r in rows]) for kk in rows[0]}


def _summary(recs, a):
    m = [r.get("meta", {}) for r in recs]
    return {"gate.episodes": sum(x.get("episodes", 0) for x in m)}


TASK = Task("stress_gate", _units, _run, _args, _summary, ("numpy", "torch"),
            "C-GATE rows in T2 (knee severities): truth, rho_par / rho_pi per probe set, dominance flags, misfit")
