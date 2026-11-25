"""Nominal healthy reference rho_ref(shape, leg) for the primary decision label rho_dec = rho_abs / rho_ref
(docs/board/README.md, coordinator decision 2026-11-25; docs/CONFIRMATORY.md 0.3: the FIXED T1 table for every plant).

Single source: this module delegates to the generator's table `cfcap.gen.rho_ref` (the one its labels use), so every
baselines prediction and label divides by exactly the same numbers. The reference is the mecanum nominal reference of
the shape (types=None: the absolute requirement scale; a per-type reference would be ~0 on legs a type cannot do).
Missing keys are simulated with the generator's own label path (simrun._run_by_plate, window_mean, rho_rows) and put
into its cache.
"""
from __future__ import annotations

import numpy as np


def shape_key(asm):
    from cfcap.gen import rho_ref as RR
    return RR.key(asm, None)


def rho_ref_many(asms, plate="plated", device="cuda"):
    """(len(asms), L) fixed T1 nominal healthy reference rho of each assembly (mecanum reference of the shape)."""
    import torch
    from cfcap.gen import episode as GE
    from cfcap.gen import rho_ref as RR
    from cfcap.gen import simrun as S
    from cfcap.t1 import capability as cap
    jobs, keys, tws, lcs = RR.jobs_for(plate, list(asms), GE.params(plate))
    if jobs:
        res = S._run_by_plate(jobs, cap.T_LEG, ("twist",), "graph" if device == "cuda" else "eager", 1 << 17, None,
                              reduce=S.window_mean) if device == "cuda" else None
        if res is None:                                   # CPU: plain capability path on the same nominal latents
            outs = cap.capability([(j["asm"], {k: v for k, v in j["lat"].items()})
                                   for j in [dict(asm=jj["asm"], lat=RR.nominal_latents(jj["asm"], plate, 1))
                                             for jj in jobs]], GE.params(plate), device="cpu")
            for k, o in zip(keys, outs):
                RR.put(plate, k, o["rho"][0].numpy())
        else:
            L = len(cap.LEGS)
            tw = torch.as_tensor(np.concatenate(tws), dtype=torch.float32)
            lc = torch.as_tensor(np.concatenate([np.full(len(t), l) for t, l in zip(tws, lcs)]), dtype=torch.float32)
            rho, _ = S.rho_rows(torch.cat([r["xr"] for r in res]), tw, lc)
            rho = rho.numpy()
            for i, k in enumerate(keys):
                RR.put(plate, k, rho[i * L:(i + 1) * L])
    if not len(asms):
        from cfcap.t1 import capability as cap
        return np.zeros((0, len(cap.LEGS)))
    return np.stack([RR.get(plate, RR.key(a, None)) for a in asms]).astype(float)


def rho_ref(asm, plate="plated", device="cuda"):
    return rho_ref_many([asm], plate, device)[0]
