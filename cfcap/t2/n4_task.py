"""cfcap.runner task `stress_n4` (role stress, EXPLORATORY): N4 merge certificate with inferred latents on the E2 pairs.

    python -m cfcap.runner stress_n4 --task-spec cfcap.t2.n4_task:TASK --shard K --n-shards N [--exclude UNIT,...]

20 units = (arm P0/P5) x (E2 shard 0/1) x (pairs 0-59, 60-119, ..., 240-249): scripts/stress_N4.py run on that chunk
(E2 replay of scripts/stress_A4_merge.py, enum_known / copy_parent / detector_gated on each G_j's own v2 probe).
Input `a4_e2/` = the E2 outputs A4_<arm>_s<shard>.npz (truths, fixed-reference factors). Rows = one per merge, per-leg
columns <key>.<l>; scripts/stress_N4.py analyze-runs turns the units back into N4_<arm>_s<shard>_<start>.npz.
Requires torch + CUDA and numba (backend auto), inputs crane_ml and a4_e2.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

from cfcap.runner import Task, shard_range
from cfcap.t2 import deps  # noqa: F401

REPO = pathlib.Path(__file__).resolve().parents[2]
UNITS = [f"n4-{a}-s{s}-{st:03d}" for a in ("P0", "P5") for s in (0, 1) for st in (0, 60, 120, 180, 240)]


def _args(p):
    p.add_argument("--exclude", default="", help="comma list of unit ids already done locally")
    p.add_argument("--device", default="cuda")


def _units(a, shard, n_shards):
    ex = set(x for x in a.exclude.split(",") if x)
    us = [u for u in UNITS if u not in ex]
    return [us[i] for i in shard_range(len(us), shard, n_shards)]


def _mod():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("stress_N4", REPO / "scripts" / "stress_N4.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(unit, ctx):
    _, arm, sh, st = unit.split("-")
    shard, start = int(sh[1:]), int(st)
    f = ctx.input(f"a4_e2/A4_{arm}_s{shard}.npz")
    import os
    lim = int(os.environ.get("CFCAP_N4_LIMIT", "0")) or None          # smoke tests only
    out = _mod().run(arm, shard, device=ctx.args.device, start=start, stop=start + 60, a4_file=f, save=False,
                     limit=lim)
    cols = {}
    for k, v in out.items():
        v = np.asarray(v)
        if v.ndim == 2:
            for l in range(v.shape[1]):
                cols[f"{k}.{l}"] = v[:, l].astype(float)
        else:
            cols[k] = v
    cols["arm"] = np.array([arm] * len(out["pair"]))
    cols["e2_shard"] = np.full(len(out["pair"]), shard)
    cols["start"] = np.full(len(out["pair"]), start)
    ctx.meta.update(merges=len(out["pair"]))
    return cols


TASK = Task("stress_n4", _units, _run, _args, None, ("numpy", "torch"),
            "N4 (exploratory): composed enum_known merge certificate on the E2 pairs, 60-pair chunks")
