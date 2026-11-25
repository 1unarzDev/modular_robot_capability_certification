"""Nominal healthy reference rho_ref(shape, leg) for the primary decision label rho_dec = rho_abs / rho_ref
(docs/board/README.md, coordinator decision 2026-11-25).

rho_ref is the closed-loop T1 rho (cfcap.t1.capability, the 10 decision legs) of the assembly with every wheel
healthy, the identified theta of its plate state as a point domain (params.point_domain: no DR), no heterogeneity, no
mounting-yaw error, least-norm normal loads and no payload. It is deterministic, so "mean over draws" is one rollout.
The table is keyed by the assembly's canonical shape (module cells translated to the origin, in module order, with
module yaws) and plate state, and cached in memory and on disk ($CFCAP_OUT/rho_ref_<plate>.json, safe to delete).

Additive module (new file); nothing else in cfcap.t1 changes.
"""
from __future__ import annotations

import json
import os
import pathlib
import threading

import numpy as np

_MEM: dict = {}
_LOCK = threading.Lock()


def shape_key(asm):
    cells = [(int(i), int(j), int(y)) for _, i, j, y in asm.modules]
    x0, y0 = min(c[0] for c in cells), min(c[1] for c in cells)
    return ";".join(f"{i - x0},{j - y0},{y}" for i, j, y in cells)


def _disk(plate):
    d = os.environ.get("CFCAP_OUT")
    return pathlib.Path(d) / f"rho_ref_{plate}.json" if d else None


def _load(plate):
    if plate in _MEM:
        return _MEM[plate]
    tab = {}
    f = _disk(plate)
    if f is not None and f.exists():
        try:
            tab = {k: np.asarray(v, float) for k, v in json.loads(f.read_text()).items()}
        except (OSError, ValueError):
            tab = {}
    _MEM[plate] = tab
    return tab


def _save(plate):
    f = _disk(plate)
    if f is None:
        return
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({k: list(map(float, v)) for k, v in _MEM[plate].items()}))
        tmp.replace(f)
    except OSError:
        pass


def rho_ref_many(asms, plate="plated", device="cuda"):
    """(len(asms), L) nominal healthy reference rho for each assembly (computes the missing ones in one batch)."""
    import torch
    from cfcap.t1 import capability as cap
    from cfcap.t1 import params as P
    from cfcap.t1 import plant
    with _LOCK:
        tab = _load(plate)
        keys = [shape_key(a) for a in asms]
        need, seen = [], set()
        for k, a in zip(keys, asms):
            if k not in tab and k not in seen:
                need.append((k, a))
                seen.add(k)
        if need:
            par = P.base_params(plate)
            th = P.load_theta(plate)
            rng = np.random.default_rng(0)
            targets = []
            for _, a in need:
                lat = plant.sample_latents(a, 1, rng, plant.HET_OFF, domain=P.point_domain(th, 1, device=device),
                                           device=device)
                lat["yaw_err"] = torch.zeros_like(lat["yaw_err"])
                targets.append((a, lat))
            outs = cap.capability(targets, par, device=device)
            for (k, _), o in zip(need, outs):
                tab[k] = o["rho"][0].cpu().numpy().astype(float)
            _save(plate)
        return np.stack([tab[k] for k in keys]) if keys else np.zeros((0, len(cap.LEGS)))


def rho_ref(asm, plate="plated", device="cuda"):
    return rho_ref_many([asm], plate, device)[0]
