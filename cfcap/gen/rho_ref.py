"""Nominal healthy reference rho_ref(shape, leg) for the decision label rho_dec = rho_abs / rho_ref (coordinator
decision 2026-11-25, docs/board/README.md).

rho_ref of a morphology = closed-loop rho on the 10 decision legs of cfcap.t1.capability for the *nominal* plant: the
plate state's identified theta as a point domain (P.point_domain: no DR, no cg offset, no mounting-yaw error), no
heterogeneity (motor multipliers 1, no V_batt / mu offsets, least-norm normal loads), no faults, no payload. It is a
deterministic function of (plate, shape key), so it is a fixed table: computed once per key, cached in-process and in
an optional JSON table (`load_table` / `save_table`), and identical in every shard.

Shape key = translation-normalised module cells with module yaws, "i,j,yaw;..." sorted by cell. Yaws are kept (the
roller pattern seen by the body depends on them); no symmetry reduction is assumed. A non-mecanum module appends its
wheel type, "i,j,yaw,type" (cfcap.t1.plant CONVENTIONAL 1, OMNI 2), so all-mecanum keys are unchanged and every
module type has its own nominal reference.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from cfcap.t1 import capability as cap
from cfcap.t1 import params as P

_CACHE: dict = {}


def key(asm, types=None):
    t = [0] * len(asm.modules) if types is None else [int(x) for x in types]
    mods = [(int(i), int(j), int(y) % 4, tt) for (_, i, j, y), tt in zip(asm.modules, t)]
    mi, mj = min(m[0] for m in mods), min(m[1] for m in mods)
    return ";".join(f"{i - mi},{j - mj},{y}" + (f",{tt}" if tt else "") for i, j, y, tt in sorted(mods))


def nominal_latents(asm, plate, B, types=None):
    n, m = asm.n, len(asm.modules)
    th = P.load_theta(plate)
    one = lambda *s: torch.ones(*s)
    zero = lambda *s: torch.zeros(*s)
    return dict(dom=P.point_domain(th, B, "cpu"), mode=torch.zeros(B, n, dtype=torch.long), g=one(B, n),
                stall_w=one(B, n), noload_w=one(B, n), vbatt_off=zero(B, m), mu_mod=one(B, m), yaw_err=zero(B, n),
                load_z=zero(B, n), gyro_bias=zero(B), acc_bias=zero(B, 2), drop_p=0.0, payload=None) | (
        {} if types is None or not np.any(types) else
        dict(wtype=torch.as_tensor(np.repeat(np.asarray(types, np.int64), 4)).expand(B, n).clone()))


def get(plate, k):
    return _CACHE.get((plate, k))


def put(plate, k, rho):
    _CACHE[(plate, k)] = np.asarray(rho, np.float32)


def jobs_for(plate, asms, params):
    """Label jobs (one per missing key) for the nominal references of `asms` (Assembly or (Assembly, module types));
    returns (jobs, keys, tws, Lcs)."""
    jobs, keys, tws, lcs, seen = [], [], [], [], set()
    for a in asms:
        a, types = a if isinstance(a, tuple) else (a, None)
        k = key(a, types)
        if get(plate, k) is not None or k in seen:
            continue
        seen.add(k)
        tabs, tw = cap.leg_tables(a, params)
        jobs.append(dict(asm=a, lat=nominal_latents(a, plate, len(cap.LEGS), types), cmd=tabs, plate=plate))
        keys.append(k)
        tws.append(tw)
        lcs.append(a.char_length())
    return jobs, keys, tws, lcs


def table_path(directory, plate):
    return pathlib.Path(directory) / f"rho_ref_{plate}.json"


def load_table(path, plate):
    p = pathlib.Path(path)
    if p.exists():
        for k, v in json.loads(p.read_text())["rho_ref"].items():
            put(plate, k, v)


def save_table(path, plate):
    rows = {k: [round(float(x), 6) for x in v] for (pl, k), v in sorted(_CACHE.items()) if pl == plate}
    pathlib.Path(path).write_text(json.dumps(dict(schema="cfcap.gen.rho_ref/v1", plate=plate, legs=cap.LEGS,
                                                  definition=__doc__.split("\n\n")[1], rho_ref=rows), indent=0))
