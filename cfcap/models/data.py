"""Pilot episode generation and batching for the amortized posterior (models role).

An episode: a parent assembly (named shape) with T1 latents (heterogeneity on), the jittered v2 probe, robot-visible
observations, and C candidate children (connected module subsets, the parent itself included) whose rho_rel on the
10 decision legs comes from closed-loop T1 rollouts (`t1.capability`, relative=True). Stand-in until the generator
role's `cfcap/gen` is ready; the on-disk format is one npz per (split, shape).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from cfcap.models import posterior as M
from cfcap.t1 import capability as cap
from cfcap.t1 import params as P
from cfcap.t1 import plant

KIND_HEALTHY, KIND_WEAK1, KIND_RANDOM = 0, 1, 2


def connected_subsets(cells):
    """All non-empty 4-connected subsets of the cell list, as sorted index tuples."""
    m = len(cells)
    adj = {i: {j for j in range(m) if abs(cells[i][0] - cells[j][0]) + abs(cells[i][1] - cells[j][1]) == 1}
           for i in range(m)}
    out = []
    for r in range(1, m + 1):
        for s in itertools.combinations(range(m), r):
            seen, todo = {s[0]}, [s[0]]
            while todo:
                k = todo.pop()
                for j in adj[k]:
                    if j in s and j not in seen:
                        seen.add(j)
                        todo.append(j)
            if len(seen) == r:
                out.append(s)
    return out


def morph_cells(shape):
    """A named shape (P.SHAPES, yaws 0) or an explicit cell list [(i, j[, yaw_k])] -> [(i, j, yaw_k)]."""
    cells = P.SHAPES[shape] if isinstance(shape, str) else shape
    return [tuple(int(v) for v in c) + ((0,) if len(c) == 2 else ()) for c in cells]


def build_morph(shape, par, subset=None, name="asm"):
    """Assembly of the morphology (or of the module subset, in subset order); module k keeps id MODULE_IDS[k]."""
    from capxfer import assembly as asmlib
    cells = morph_cells(shape)
    ks = range(len(cells)) if subset is None else subset
    return asmlib.build(name, [(P.MODULE_IDS[k], *cells[k]) for k in ks], par)


def child_assembly(shape, subset, par):
    return build_morph(shape, par, subset, "child")


def sample_faults(B, n, rng, mix=(0.35, 0.35, 0.30)):
    """Mixture: healthy / one weak wheel g~U(0.3,0.9) / capxfer-style random faults (1-2 of weak, coast, brake)."""
    kind = rng.choice(3, size=B, p=np.asarray(mix) / np.sum(mix))
    mode = np.zeros((B, n), np.int64)
    g = np.ones((B, n), np.float32)
    m1, g1 = plant.weak_or_healthy(B, n, rng, p_healthy=0.0)
    m2, g2 = plant.random_faults(B, n, rng, k_max=2, p_none=0.0)
    mode[kind == 1], g[kind == 1] = m1[kind == 1], g1[kind == 1]
    mode[kind == 2], g[kind == 2] = m2[kind == 2], g2[kind == 2]
    return kind, mode, g


def _nuisance(lat, asm):
    """(B, m, 3) log mu, log stall, log no-load per module (trial DR x module factor x mean wheel multiplier)."""
    wm = torch.as_tensor(np.asarray(asm.wheel_module), device=lat["mode"].device)
    m = len(asm.modules)
    oh = torch.nn.functional.one_hot(wm, m).float()
    mean_w = lambda x: (x @ oh) / oh.sum(0)
    d = lat["dom"]
    mu = d["mu"][:, None] * lat["mu_mod"]
    st = d["stall_scale"][:, None] * mean_w(lat["stall_w"])
    nl = d["noload_scale"][:, None] * mean_w(lat["noload_w"])
    return torch.stack([mu.log(), st.log(), nl.log()], -1)


def generate(shape, B, rng, n_child=2, het=plant.HET_FULL, batch=2000, baseline=None, plate="plated", par=None,
             dr=None):
    """-> dict of numpy arrays for B episodes on `shape` (a P.SHAPES name or a cell list with yaws)."""
    par = par or P.base_params(plate)
    dr = dr or P.theta_dr(P.load_theta(plate))
    asm = build_morph(shape, par)
    n, m = asm.n, len(asm.modules)
    subsets = connected_subsets(morph_cells(shape))
    child_asm = {s: child_assembly(shape, s, par) for s in subsets}
    outs = []
    for b0 in range(0, B, batch):
        b = min(batch, B - b0)
        kind, mode, g = sample_faults(b, n, rng)
        lat = plant.sample_latents(asm, b, rng, het, dr=dr, faults=(mode, g))
        tab, T = plant.v2_probe(asm, par, b, rng)
        R = plant.simulate_bucket([dict(asm=asm, lat=lat, cmd=tab)], par, T, seed=int(rng.integers(1 << 30)))[0]
        obs = plant.observe(R, lat, seed=int(rng.integers(1 << 30)))
        Tn = obs["pre_duty"].shape[1]
        cmd = tab[:, :Tn]
        o = dict(kind=kind.astype(np.int8), mode=mode.astype(np.int8), g=g.astype(np.float32),
                 wf=M.wheel_features(asm, cmd, obs["pre_duty"].cpu().numpy(), obs["speed"].cpu().numpy()
                                     ).astype(np.float16),
                 af=M.asm_features(cmd, obs["gz"].cpu().numpy(), obs["ax"].cpu().numpy(),
                                   obs["ay"].cpu().numpy()).astype(np.float16),
                 nuis=_nuisance(lat, asm).cpu().numpy().astype(np.float32))
        if baseline is not None:
            p2, gh2 = baseline(asm, cmd, obs)
            o.update(base_p=p2.astype(np.float32), base_g=gh2.astype(np.float32))
        # children: C random connected subsets per episode, rho_rel from closed-loop rollouts
        pick = rng.integers(len(subsets), size=(b, n_child))
        member = np.zeros((b, n_child, m), bool)
        rho_rel = np.full((b, n_child, len(cap.LEGS)), np.nan, np.float32)
        rho = np.full_like(rho_rel, np.nan)
        targets, where = [], []
        for si, s in enumerate(subsets):
            for c in range(n_child):
                idx = np.nonzero(pick[:, c] == si)[0]
                if len(idx) == 0:
                    continue
                li = plant.subset(plant._index(lat, torch.as_tensor(idx, device=lat["mode"].device)), s)
                targets.append((child_asm[s], li))
                where.append((idx, c))
        res = cap.capability(targets, par, relative=True)
        for (idx, c), r in zip(where, res):
            rho_rel[idx, c] = r["rho_rel"].cpu().numpy()
            rho[idx, c] = r["rho"].cpu().numpy()
        for c in range(n_child):
            for i in range(b):
                member[i, c] = np.isin(np.arange(m), subsets[pick[i, c]])
        o.update(child=pick.astype(np.int16), member=member, rho_rel=rho_rel, rho=rho)
        outs.append(o)
    out = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    out["subsets"] = np.array([np.isin(np.arange(m), s) for s in subsets])
    return out


# ---------------------------------------------------------------------- batching
class ShapeSet:
    """One morphology's episodes (an npz path or a `generate` dict) on the device, with child geometry precomputed
    per connected subset."""

    def __init__(self, path, shape, device="cuda", par=None):
        z = np.load(path) if not isinstance(path, dict) else path
        par = par or P.base_params("plated")
        self.shape = shape
        self.asm = build_morph(shape, par)
        self.n, self.m = self.asm.n, len(self.asm.modules)
        t = lambda a, dt=None: torch.as_tensor(a, device=device, dtype=dt)
        self.wf, self.af = t(z["wf"]), t(z["af"])
        self.mode, self.g, self.kind = t(z["mode"], torch.long), t(z["g"]), t(z["kind"], torch.long)
        self.nuis = t(z["nuis"])
        self.child, self.rho_rel = t(z["child"], torch.long), t(z["rho_rel"])
        self.base = (t(z["base_p"]), t(z["base_g"])) if "base_p" in z else None
        subs = [tuple(np.nonzero(r)[0]) for r in z["subsets"]]
        self.subsets = subs
        wm = np.asarray(self.asm.wheel_module)
        self.geom = t(M.wheel_geometry(self.asm))
        self.wm = t(wm, torch.long)
        cg = np.zeros((len(subs), self.n, M.GEOM_DIM), np.float32)
        wmem = np.zeros((len(subs), self.n), bool)
        for i, s in enumerate(subs):
            ca = child_assembly(shape, s, par)
            gi = M.wheel_geometry(ca)
            for k, mod in enumerate(s):
                cg[i, wm == mod] = gi[4 * k:4 * k + 4]
                wmem[i, wm == mod] = True
        self.cgeom, self.wmember = t(cg), t(wmem)
        self.N = self.wf.shape[0]

    def batch(self, idx):
        B = len(idx)
        ch = self.child[idx]
        return dict(wf=self.wf[idx].float(), af=self.af[idx].float(), geom=self.geom.expand(B, -1, -1),
                    wm=self.wm.expand(B, -1), mask=torch.ones(B, self.n, dtype=torch.bool, device=self.wf.device),
                    n_mod=self.m, mode=self.mode[idx], g=self.g[idx], nuis=self.nuis[idx],
                    mod_mask=torch.ones(B, self.m, device=self.wf.device), member=self.wmember[ch],
                    cgeom=self.cgeom[ch], rho_rel=self.rho_rel[idx])
