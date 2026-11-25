"""Generator episodes (cfcap.gen, Stage B) as model batches.

`episodes(plate, split, n, index0, keep)` runs the generator's own pipeline (make_episode -> v2 probes of every
assembly -> closed-loop labels of every candidate -> columnar arrays) on the episode indices it keeps, so episode ids
(`ep_index`) are the generator's and match every other role's rows. `keep(ep)` filters on the parent before any
simulation (training-diversity restriction); eval sets use keep=None.

`GenSet` holds a columnar dict on the GPU and makes padded batches:
  * assemblies (parent and others, each observed under its own v2 probe) -> wheel tokens with wheel features
    (IK ref/5, pre_duty, speed/5), assembly channels, geometry, module index, per-wheel and per-module truth;
  * candidates (self / child / merge) -> the global wheel rows they are made of (a merge spans two assemblies), their
    wheel geometry in the candidate frame, rho_rel truth.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from cfcap.gen import episode as E
from cfcap.gen import families as F
from cfcap.gen import shard as SH
from cfcap.gen import simrun as S
from cfcap.models import posterior as M
from cfcap.t1 import params as P

DT = 0.02
W_REL = 3.0
IK_RADIUS = 0.0501


def episodes(plate, split, n, index0=0, keep=None, block=256, engine="compile", log=None, max_scan=None,
             indices=None, cand_cap=None, time_budget=None):
    """Columnar dict of n kept episodes starting the scan at index0 (or exactly `indices`); returns (D, next_index,
    stats). cand_cap: label only the parent itself plus cand_cap - 1 random other candidates per episode (training
    pools; the subset is drawn from the episode's own rng stream 99, so it is reproducible). time_budget: stop after
    the block that crosses it (stats["n"] = episodes made)."""
    parts, i, got = [], int(index0), 0
    st = dict(make_s=0.0, probe_s=0.0, label_s=0.0, scanned=0)
    if indices is not None:
        indices = list(indices)
        n, index0, max_scan = len(indices), 0, len(indices)
        i = 0
    max_scan = max_scan or 100 * n
    while got < n and i < index0 + max_scan:
        t0 = time.perf_counter()
        eps = []
        while len(eps) < min(block, n - got) and i < index0 + max_scan:
            ep = E.make_episode(plate, split, indices[i] if indices is not None else i)
            i += 1
            st["scanned"] += 1
            if indices is not None or keep is None or keep(ep):
                if cand_cap is not None and len(ep.cands) > cand_cap:
                    r = E.episode_rng(plate, split, ep.index, stream=99)
                    rest = [c for c in ep.cands[1:]]
                    pick = sorted(r.choice(len(rest), cand_cap - 1, replace=False))
                    ep.cands = [ep.cands[0]] + [rest[k] for k in pick]
                eps.append(ep)
        if not eps:
            break
        t1 = time.perf_counter()
        obs = S.probe_episodes(eps, ("v2",), engine=engine)
        t2 = time.perf_counter()
        labs = S.label_episodes(eps, engine=engine)
        t3 = time.perf_counter()
        parts.append(SH._assemble(eps, obs, labs, ("v2",)))
        st["make_s"] += t1 - t0
        st["probe_s"] += t2 - t1
        st["label_s"] += t3 - t2
        got += len(eps)
        st["n"] = got
        if log:
            log(f"[models.gen] {plate}/{split} {got}/{n} (scan {i - index0}) probe {t2 - t1:.1f}s label {t3 - t2:.1f}s")
        if time_budget is not None and st["make_s"] + st["probe_s"] + st["label_s"] > time_budget:
            break
    return SH.concat(parts), i, st


def keep_fn(max_n=None, topologies=None):
    """Parent filter for a training-diversity level: parent modules <= max_n and topology in `topologies`."""
    def keep(ep):
        a = ep.parent
        if max_n is not None and a.m > max_n:
            return False
        return topologies is None or a.topo in topologies
    return keep


def _asm_geometry(cells, yaws, plate):
    a = E.build("x", cells, yaws, ["M1"] * len(cells), plate)
    J = a.ik(np.eye(3), IK_RADIUS).astype(np.float32)           # (3, n): ref = cmd @ J (IK is linear in the twist)
    return M.wheel_geometry(a), J.T, np.asarray(a.wheel_module)


class GenSet:
    """A generator columnar dict as padded model batches (on `device`). label: "dec" (primary, rho_abs / rho_ref of
    the nominal healthy morphology; coordinator decision 2026-11-25) or "rel" (rho / own fault-free twin)."""

    def __init__(self, D, plate="plated", device="cuda", label="dec"):
        self.D, self.plate, self.dev = D, plate, device
        A = len(D["asm_ep"])
        self.A, self.E, self.C = A, len(D["ep_index"]), len(D["cand_ep"])
        mo = D["asm_mod_off"]
        self.asm_m = np.diff(mo).astype(int)
        self.asm_n = 4 * self.asm_m
        self.asm_w0 = 4 * mo[:-1]
        W = int(4 * mo[-1])
        geom = np.zeros((W, M.GEOM_DIM), np.float32)
        J = np.zeros((W, 3), np.float32)
        wmod = np.zeros(W, np.int64)
        for a in range(A):
            m0, m1 = mo[a], mo[a + 1]
            g, j, wm = _asm_geometry(D["mod_cell"][m0:m1], D["mod_yaw"][m0:m1], plate)
            geom[4 * m0:4 * m1], J[4 * m0:4 * m1], wmod[4 * m0:4 * m1] = g, j, wm
        t = lambda x, dt=None: torch.as_tensor(np.asarray(x), device=device, dtype=dt)
        self.geom, self.J, self.wmod = t(geom), t(J), t(wmod, torch.long)
        self.duty = t(D["obs_v2_duty"])                                       # (W, T) fp16
        self.speed = t(D["obs_v2_dcnt"].astype(np.float16) * np.float16(2 * math.pi / P.CPR / DT / 5.0))
        self.cmd = t(D["obs_v2_cmd"]).float()                                 # (A, T, 3)
        self.imu = t(D["obs_v2_imu"]).float()                                 # (A, T, 3)
        self.mode = t(D["wheel_mode"], torch.long)
        self.g = t(D["wheel_g"])
        dom = D["ep_dom"]
        K = {k: i for i, k in enumerate(E.DOM_KEYS)}
        ep_of_mod = D["asm_ep"][np.repeat(np.arange(A), self.asm_m)]
        mean4 = lambda x: x.reshape(-1, 4).mean(1)
        nu = np.stack([np.log(dom[ep_of_mod, K["mu"]] * D["mod_mu"]),
                       np.log(dom[ep_of_mod, K["stall_scale"]] * mean4(D["wheel_stall"])),
                       np.log(dom[ep_of_mod, K["noload_scale"]] * mean4(D["wheel_noload"]))], 1)
        self.nuis = t(nu.astype(np.float32))
        self.mod_asm = D["mod_asm"]
        # candidates: global wheel rows and candidate-frame geometry, padded to the widest candidate
        co = D["cand_cm_off"]
        self.cand_n = 4 * np.diff(co).astype(int)
        nmax = int(self.cand_n.max()) if self.C else 4
        rows = np.full((self.C, nmax), -1, np.int64)
        cg = np.zeros((self.C, nmax, M.GEOM_DIM), np.float32)
        cache = {}
        for c in range(self.C):
            k0, k1 = co[c], co[c + 1]
            mods = D["cm_mod"][k0:k1]
            r = (4 * mods[:, None] + np.arange(4)).reshape(-1)
            rows[c, :len(r)] = r
            key = (D["cm_cell"][k0:k1].tobytes(), D["cm_yaw"][k0:k1].tobytes())
            if key not in cache:
                cache[key] = _asm_geometry(D["cm_cell"][k0:k1], D["cm_yaw"][k0:k1], plate)[0]
            cg[c, :len(r)] = cache[key]
        self.cand_rows, self.cand_geom = t(rows), t(cg)
        from cfcap.t1 import reference
        asms, built = [], {}
        for c in range(self.C):
            k0, k1 = co[c], co[c + 1]
            key = (D["cm_cell"][k0:k1].tobytes(), D["cm_yaw"][k0:k1].tobytes())
            if key not in built:
                built[key] = E.build("cand", D["cm_cell"][k0:k1], D["cm_yaw"][k0:k1], ["M1"] * (k1 - k0), plate)
            asms.append(built[key])
        if "cand_rho_ref" in D and "cand_rho_dec" in D:                        # generator schema v2
            self.ref = D["cand_rho_ref"].astype(np.float32)
        else:
            self.ref = reference.rho_ref_many(asms, plate).astype(np.float32)  # (C, L) nominal healthy rho
            D["cand_rho_ref"] = self.ref
            D["cand_rho_dec"] = (D["cand_rho"] / self.ref).astype(np.float32)
        fin = lambda x: np.nan_to_num(np.clip(x, 0.0, 2.0), nan=1.0).astype(np.float32)
        self.label = label
        self.target = t(fin(D["cand_rho_dec"] if label == "dec" else D["cand_rho_rel"]))
        self.rho_rel = t(fin(D["cand_rho_rel"]))
        self.cand_ep = D["cand_ep"]
        self.ep_asm = [np.arange(D["ep_asm_off"][e], D["ep_asm_off"][e + 1]) for e in range(self.E)]
        self.ep_cand = [np.arange(D["ep_cand_off"][e], D["ep_cand_off"][e + 1]) for e in range(self.E)]
        self.ep_wheels = np.array([self.asm_n[a].sum() for a in self.ep_asm])
        # decision-relevant oversampling (training only): candidates holding a faulted wheel, or with a label within
        # 0.1 of a requirement tau in {0.6, 0.7, 0.8} on any leg, weigh W_REL x in the direct-head loss
        y = D["cand_rho_dec"] if label == "dec" else D["cand_rho_rel"]
        rel = (D["cand_n_fault"] > 0) | ((y > 0.5) & (y < 0.9)).any(1)
        self.cand_rel = rel
        self.cand_w = np.where(rel, W_REL, 1.0)
        self.cand_w_t = t(self.cand_w.astype(np.float32))
        self.N = self.E

    # ---------------------------------------------------------------- batches
    def batch(self, eps, max_cand=None, rng=None):
        """Episodes -> dict of padded tensors: assemblies (B_a, T, n_max, ...) and candidates (B_c, n_c, ...)."""
        asms = np.concatenate([self.ep_asm[e] for e in eps])
        cands = [self.ep_cand[e] for e in eps]
        if max_cand is not None and rng is not None:
            cands = [c if len(c) <= max_cand else rng.choice(c, max_cand, replace=False) for c in cands]
        cands = np.concatenate(cands) if cands else np.zeros(0, int)
        return self._batch(asms, cands)

    def _batch(self, asms, cands):
        dev = self.dev
        n = self.asm_n[asms]
        nmax = int(n.max())
        Ba = len(asms)
        loc = np.arange(nmax)[None, :]
        valid = loc < n[:, None]
        rows = np.where(valid, self.asm_w0[asms][:, None] + loc, 0)
        rows_t = torch.as_tensor(rows, device=dev)
        mask = torch.as_tensor(valid, device=dev)
        at = torch.as_tensor(asms, device=dev)
        cmd = self.cmd[at]                                                  # (Ba, T, 3)
        ref = torch.einsum("btk,bnk->btn", cmd, self.J[rows_t]) / 5.0
        wf = torch.stack([ref, self.duty[rows_t].transpose(1, 2).float(), self.speed[rows_t].transpose(1, 2).float()],
                         -1) * mask[:, None, :, None]
        af = torch.cat([cmd[..., :2] / 0.3, cmd[..., 2:], self.imu[at]], -1)
        mmax = int(self.asm_m[asms].max())
        mod_rows = self.D["asm_mod_off"][asms][:, None] + np.arange(mmax)[None, :]
        mod_valid = np.arange(mmax)[None, :] < self.asm_m[asms][:, None]
        mrt = torch.as_tensor(np.where(mod_valid, mod_rows, 0), device=dev)
        b = dict(wf=wf, af=af, geom=self.geom[rows_t] * mask[..., None], wm=torch.where(mask, self.wmod[rows_t], 0),
                 mask=mask, n_mod=mmax, mode=torch.where(mask, self.mode[rows_t], 0), g=self.g[rows_t],
                 nuis=self.nuis[mrt], mod_mask=torch.as_tensor(mod_valid, device=dev).float(), asms=asms, cands=cands)
        # candidate tokens: global wheel row -> flat position (assembly in batch, local wheel)
        pos = np.full(int(4 * self.D["asm_mod_off"][-1]), -1, np.int64)
        for bi, a in enumerate(asms):
            pos[self.asm_w0[a]:self.asm_w0[a] + self.asm_n[a]] = bi * nmax + np.arange(self.asm_n[a])
        cr = self.cand_rows[torch.as_tensor(cands, device=dev)]
        cmask = cr >= 0
        cpos = torch.as_tensor(pos, device=dev)[cr.clamp(min=0)]
        b.update(cand_pos=torch.where(cmask, cpos, 0), cand_mask=cmask,
                 cand_geom=self.cand_geom[torch.as_tensor(cands, device=dev)],
                 rho_rel=self.rho_rel[torch.as_tensor(cands, device=dev)],
                 target=self.target[torch.as_tensor(cands, device=dev)],
                 cf_w=self.cand_w_t[torch.as_tensor(cands, device=dev)][:, None])
        return b


def forward(model, b):
    """Model outputs for a GenSet batch: per-wheel / per-module heads on the assemblies, direct-head quantiles for
    every candidate (tokens gathered across assemblies, so merges see both parents)."""
    zw, za = model.encode(b["wf"], b["af"], b["geom"], b["mask"])
    logits, ab, nu = model.latent_heads(zw, za, b["wm"], b["mask"], b["n_mod"])
    out = dict(logits=logits, ab=ab, nuis=nu)
    if len(b["cands"]):
        zf = zw.reshape(-1, zw.shape[-1])
        zc = zf[b["cand_pos"]]                                              # (Bc, nc, h)
        out["cf"] = model.counterfactual(zc, b["cand_mask"][:, None], b["cand_geom"][:, None])[:, 0]
    return out


def loss(out, b, **w):
    """posterior.loss_fn on a GenSet batch (cf is (Bc, L, Q) against rho_rel (Bc, L))."""
    return M.loss_fn(out, b, **w)
