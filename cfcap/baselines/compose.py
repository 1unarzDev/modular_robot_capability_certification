"""Merge baselines from separately probed assemblies (research NEXT N4; one implementation, for role stress and the
baselines runner).

Setting: several assemblies were each probed alone (their own v2 trial); a merge candidate is built from modules of
some of them. Every function returns a `Prediction` (point, scale, lower) per (merge candidate, leg) on the primary
label rho_dec = rho_abs / rho_ref with rho_ref the FIXED T1 nominal healthy reference (`cfcap.t1.reference`,
`episode.ref_table`); a different test plant (T2) must divide its rho_abs by this same table, never by its own.

  merge_enum_known(probed, cands)    the A6 exact enumeration (every nuisance known, <= 2 faults per assembly) run per
                                     assembly; the posteriors are independent across assemblies, so a merge's
                                     posterior is their product. Each assembly keeps hypotheses to 1 - 1e-3 mass (cap
                                     64); the product is kept to 1 - DROP_TOL (cap 1024); if the mass left out exceeds
                                     DROP_TOL the candidate abstains (lower = point = -inf). Rows are pushed through T1
                                     with the merged geometry, the candidate's own load split and payload.
  merge_copy_parent(probed, cands)   per leg the min over the constituent assemblies of their own measured v2 rho
                                     (cfcap.baselines.trivial.parent_rho_v2): a merge is as capable as its weakest
                                     piece; scale 1.
  merge_detector_gated(probed, cands)  1 unless a constituent wheel is flagged by the RLS detector (then 0); scale 1.

Inputs:
  Probed(asm, cmd, T, obs, lat, plate): one probed assembly; lat = its true latents (B = 1; enum_known reads the
      nuisances, as the single-assembly baseline does; copy_parent / detector_gated read only obs).
  MergeCand(asm, modules, load_z, payload): modules = [(probed index, module index in that assembly)] in the
      candidate's module order.
`gen_merge_inputs(plate, split, index)` builds both from a generator episode (parent + other assemblies, the
generator's own probes and labels), with the merge candidates' T1 rho_dec labels.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import posterior as Po
from .base import L, Prediction
from .episode import _MOD_KEYS, _WHEEL_KEYS, lat_to, ref_table

DROP_TOL = 1e-3
SCALE_FLOOR = 0.02


@dataclass
class Probed:
    asm: object
    cmd: np.ndarray
    T: float
    obs: dict
    lat: dict
    plate: str = "plated"


@dataclass
class MergeCand:
    asm: object
    modules: list
    load_z: np.ndarray | None = None
    payload: tuple | None = None

    @property
    def size(self):
        return len(self.asm.modules)


def merge_latents(rows_per_asm, c, device):
    """Latents (B rows) of merge candidate c; rows_per_asm[i] = latent rows (B, on device) of probed assembly i
    (aligned: row b of every assembly belongs to joint hypothesis b)."""
    first = rows_per_asm[c.modules[0][0]]
    B = first["mode"].shape[0]
    out = dict(first)
    for k in _WHEEL_KEYS:
        out[k] = torch.cat([rows_per_asm[r][k][:, 4 * m:4 * m + 4] for r, m in c.modules], 1)
    for k in _MOD_KEYS:
        out[k] = torch.cat([rows_per_asm[r][k][:, m:m + 1] for r, m in c.modules], 1)
    if any("wtype" in rows_per_asm[r] for r, _ in c.modules):        # module wheel types: 0 = mecanum default
        z = lambda r: torch.zeros(B, rows_per_asm[r]["mode"].shape[1], dtype=torch.long, device=device)
        out["wtype"] = torch.cat([rows_per_asm[r].get("wtype", z(r))[:, 4 * m:4 * m + 4].long() for r, m in c.modules], 1)
    else:
        out.pop("wtype", None)
    n = out["mode"].shape[1]
    lz = np.zeros(n) if c.load_z is None else np.asarray(c.load_z)
    out["load_z"] = torch.as_tensor(lz, dtype=torch.float32, device=device).expand(B, n).contiguous()
    out["payload"] = None
    if c.payload is not None:
        f = lambda v: torch.full((B,), float(v), device=device)
        out["payload"] = dict(mass=f(c.payload[0]), x=f(c.payload[1]), y=f(c.payload[2]))
    return out


def _empty(C):
    z = np.zeros((C, L))
    return Prediction(z, z.copy(), z.copy(), dict(refuse="empty candidate set"))


def merge_enum_known(probed, cands, device="cuda", grid=Po.GRID, cap_asm=64, cap_joint=1024):
    from cfcap.t1 import capability as cap
    from cfcap.t1 import params as P
    if not cands:
        return _empty(0)
    t0 = time.perf_counter()
    par = P.base_params(probed[0].plate)
    posts = []
    for p in probed:
        lat = lat_to(p.lat, device)
        post = Po.enumerate_known(p.asm, par, p.cmd, p.T, lat, p.obs, grid)
        k = Po.top_mass(post["w"], 1 - DROP_TOL, cap_asm)
        posts.append(dict(lat=lat, HM=post["HM"][k], HG=post["HG"][k], w=post["w"][k],
                          dropped=float(max(0.0, 1 - post["w"][k].sum()))))
    point = np.full((len(cands), L), -np.inf)
    lower, scale = point.copy(), np.ones((len(cands), L))
    info = dict(dropped=[], abstain=0)
    for ci, c in enumerate(cands):
        used = sorted({r for r, _ in c.modules})
        combos = list(itertools.product(*[range(len(posts[r]["w"])) for r in used]))
        w = np.array([np.prod([posts[r]["w"][j] for r, j in zip(used, cb)]) for cb in combos])
        keep = Po.top_mass(w / max(w.sum(), 1e-300), 1 - DROP_TOL, cap_joint)
        prior_kept = float(np.prod([1 - posts[r]["dropped"] for r in used]))   # mass of the per-assembly sets
        dropped = 1.0 - prior_kept * float(w[keep].sum() / max(w.sum(), 1e-300))
        info["dropped"].append(dropped)
        if dropped > DROP_TOL:
            info["abstain"] += 1
            continue
        B = len(keep)
        rows = {}
        for i_r, r in enumerate(used):
            idx = [combos[j][i_r] for j in keep]
            rr = Po.rows_from(posts[r]["lat"], B)
            rows[r] = Po.set_faults(rr, posts[r]["HM"][idx], posts[r]["HG"][idx])
        lat_c = merge_latents(rows, c, device)
        o = Po.with_fallback(lambda e: cap.capability([(c.asm, lat_c)], par, device=device, engine=e,
                                                      max_wheels=Po.CAP_MAX_WHEELS), "graph")[0]
        rho = o["rho"].cpu().numpy() / ref_table([c], probed[0].plate, device)[0][None]
        m, sd, q05 = Po.summarise(rho[:, None, :], w[keep])
        point[ci], scale[ci], lower[ci] = m[0], sd[0] + SCALE_FLOOR, q05[0]
    info["seconds"] = time.perf_counter() - t0
    return Prediction(point, scale, lower, info)


def merge_copy_parent(probed, cands):
    from .trivial import LEG_DIR, parent_rho_v2
    from cfcap.t1 import capability as cap
    if not cands:
        return _empty(0)
    per = []
    for p in probed:
        d = dict(zip(LEG_DIR, parent_rho_v2(p.asm, p.cmd, p.obs)))
        row = []
        for l in cap.LEGS:
            if l in d:
                row.append(d[l])
            else:
                sx = "surge+" if l[4] == "+" else "surge-"
                sy = "sway+" if l[5] == "+" else "sway-"
                row.append(np.nanmin([d[sx], d[sy]]))
        per.append(np.nan_to_num(np.asarray(row, float), nan=0.0))
    pt = np.stack([np.min([per[r] for r in sorted({r for r, _ in c.modules})], 0) for c in cands])
    return Prediction(pt, np.ones_like(pt), pt.copy(), {})


def merge_detector_gated(probed, cands):
    from .rls import PASSIVE_G, rls_gains
    from .trivial import DET_G
    if not cands:
        return _empty(0)
    flags = []
    for p in probed:
        gh, _, _ = rls_gains(p.asm, p.cmd, p.obs)
        flags.append((gh < DET_G) | (gh < PASSIVE_G))
    pt = np.ones((len(cands), L))
    for ci, c in enumerate(cands):
        if any(flags[r][4 * m:4 * m + 4].any() for r, m in c.modules):
            pt[ci] = 0.0
    return Prediction(pt, np.ones_like(pt), pt.copy(), {})


METHODS = dict(enum_known=merge_enum_known, copy_parent=merge_copy_parent, detector_gated=merge_detector_gated)


def gen_merge_inputs(plate, split, index, engine="compile"):
    """(probed list [parent, other1, ...], merge cands, rho_dec labels (C, L), candidate indices in the generator
    episode) for the merge candidates of generator episode (plate, split, index)."""
    from cfcap.gen import episode as GE
    from cfcap.gen import simrun as S
    from cfcap.t1 import params as P
    from cfcap.t1 import plant
    gep = GE.make_episode(plate, split, index)
    par = GE.params(plate)
    obs_all = S.probe_episodes([gep], ("v2",), engine=engine)["v2"][0]
    probed = []
    for a in gep.asms:
        o = obs_all[a.role]
        rng = GE.episode_rng(plate, split, index, stream=1 + 16 * a.role + S.PROBE_KINDS.index("v2"))
        cmd, _, _ = S.probe_tables("v2", a.asm, par, rng)
        speed = o["dcnt"].astype(np.float32) / P.CPR * 2 * np.pi / plant.DT_OBS
        obs = dict(pre_duty=torch.as_tensor(o["duty"], dtype=torch.float32), speed=torch.as_tensor(speed),
                   gz=torch.as_tensor(o["imu"][:, 0], dtype=torch.float32),
                   ax=torch.as_tensor(o["imu"][:, 1], dtype=torch.float32),
                   ay=torch.as_tensor(o["imu"][:, 2], dtype=torch.float32))
        pay = GE.parent_payload(gep) if a.role == 0 else None
        lat = GE.plant_latents(gep, [(a.role, k) for k in range(a.m)], a.load_z, pay, gyro=(a.gyro_bias, a.acc_bias))
        probed.append(Probed(a.asm, cmd, S.probe_T("v2"), obs, lat, plate))
    idx = [ci for ci, c in enumerate(gep.cands) if c.kind == GE.KIND["merge"]]
    cands = [MergeCand(gep.cands[ci].asm, list(gep.cands[ci].modules), gep.cands[ci].load_z, gep.cands[ci].payload)
             for ci in idx]
    lab = S.label_episodes([gep], engine=engine)[0]
    truth = lab["rho"][idx] / ref_table(cands, plate) if idx else np.zeros((0, L))
    return probed, cands, truth, idx
