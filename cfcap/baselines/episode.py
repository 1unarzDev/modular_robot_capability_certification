"""Baseline episodes: one parent assembly observed under the v2 probe, its candidate set, and closed-loop T1 labels.

An `Episode` is what every baseline consumes:
  * public (robot-visible): parent shape + morphology, the v2 command table, observations (pre_duty with N(0, 0.005)
    duty noise as A6, quantised encoder speed, gz/ax/ay with IMU noise and bias, wheel-sample dropout), and the
    candidate set (connected proper children of the parent, by parent module indices);
  * privileged (only the oracle and the known-nuisance enumeration may read it): the true latents `lat` (faults and
    every nuisance) -- `Episode.privileged()` returns them;
  * labels (evaluation only): `rho_rel` (C, L) and `rho` (C, L) of every candidate from T1 closed-loop leg rollouts
    (cfcap.t1.capability, relative=True: divided by the same latents' fault-free rollout, capxfer D17).

Generation is batched per parent shape (`generate`): heterogeneity on (plant.HET_FULL), plated theta and the v2
self-model DR, 0-2 faults (weak g ~ U(0.3, 0.9) / coast / brake, p_none = 0.2). Children inherit the parent's per-wheel
and per-module latents through `plant.subset` (normal-load split and payload do not transfer).
This stands in for role generator's `cfcap/gen` until that exists; the Episode fields are the contract.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from cfcap.t0 import candidates as Cands
from cfcap.t1 import capability as cap
from cfcap.t1 import params as P
from cfcap.t1 import plant

LEGS = cap.LEGS
DUTY_NOISE = 0.005


@dataclass
class Episode:
    eid: int
    shape: str
    plate: str
    cmd: np.ndarray                    # (K+1, 3) v2 command tick table (50 Hz)
    T: float
    obs: dict                          # pre_duty/speed (T, n), gz/ax/ay (T,) cpu tensors
    cand_modules: list                 # list of tuples of parent module indices (candidate set)
    lat: dict = field(repr=False, default=None)        # PRIVILEGED true latents (B = 1, cpu)
    rho_rel: np.ndarray = field(repr=False, default=None)  # LABEL (C, L)
    rho: np.ndarray = field(repr=False, default=None)      # LABEL (C, L)
    meta: dict = field(default_factory=dict)

    @property
    def n_cand(self):
        return len(self.cand_modules)

    def privileged(self):
        return self.lat

    def faults(self):
        return self.lat["mode"][0].numpy(), self.lat["g"][0].numpy()

    def cand_has_fault(self):
        m = self.lat["mode"][0].numpy()
        return np.array([bool((m.reshape(-1, 4)[list(c)] != 0).any()) for c in self.cand_modules], bool)


_CACHE = {}


def context(shape, plate="plated"):
    """(params, parent asm, candidate list) for a named parent shape (cached)."""
    key = (shape, plate)
    if key not in _CACHE:
        par = P.base_params(plate)
        asm = P.named_assembly(shape, par)
        cands = Cands.children(asm, par)
        _CACHE[key] = (par, asm, cands)
    return _CACHE[key]


def lat_to(lat, device):
    out = {}
    for k, v in lat.items():
        if k == "dom":
            out[k] = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        elif k == "payload":
            out[k] = None if v is None else {kk: vv.to(device) for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def candidate_truth(shape, lat, plate="plated", device="cuda"):
    """T1 labels for every candidate of `shape` and B latents rows -> rho_rel, rho (B, C, L) numpy."""
    par, asm, cands = context(shape, plate)
    lat = lat_to(lat, device)
    targets = [(c.asm, plant.subset(lat, c.modules)) for c in cands]
    outs = cap.capability(targets, par, relative=True, device=device)
    rr = np.stack([o["rho_rel"].cpu().numpy() for o in outs], 1)
    r = np.stack([o["rho"].cpu().numpy() for o in outs], 1)
    return rr, r


def observe_parent(shape, lat, plate="plated", device="cuda", seed=0, duty_noise=DUTY_NOISE):
    par, asm, _ = context(shape, plate)
    B = lat["mode"].shape[0]
    tab, T = plant.v2_probe(asm, par, 1, jitter=False)
    R = plant.simulate_bucket([dict(asm=asm, lat=lat_to(lat, device), cmd=tab[0])], par, T, device=device)[0]
    obs = plant.observe(R, lat_to(lat, device), seed=seed)
    gen = torch.Generator(device=obs["pre_duty"].device).manual_seed(int(seed) + 1)
    obs["pre_duty"] = obs["pre_duty"] + duty_noise * torch.randn(obs["pre_duty"].shape, generator=gen,
                                                                 device=obs["pre_duty"].device)
    return tab[0], T, {k: v.cpu() for k, v in obs.items()}


def generate(shape, B, seed, plate="plated", het=plant.HET_FULL, p_none=0.2, k_max=2, device="cuda", eid0=0):
    """B episodes of one parent shape."""
    t0 = time.time()
    par, asm, cands = context(shape, plate)
    rng = np.random.default_rng(seed)
    dr = P.theta_dr(P.load_theta(plate))
    faults = plant.random_faults(B, asm.n, rng, k_max=k_max, p_none=p_none)
    lat = plant.sample_latents(asm, B, rng, het, dr=dr, faults=faults, device=device)
    tab, T, obs = observe_parent(shape, lat, plate, device, seed=seed)
    rr, r = candidate_truth(shape, lat, plate, device)
    lat_c = lat_to(lat, "cpu")
    eps = []
    for b in range(B):
        idx = torch.tensor([b])
        lb = plant._index(lat_c, idx)
        eps.append(Episode(eid=eid0 + b, shape=shape, plate=plate, cmd=tab, T=T,
                           obs={k: v[b] for k, v in obs.items()},
                           cand_modules=[tuple(c.modules) for c in cands], lat=lb, rho_rel=rr[b], rho=r[b],
                           meta=dict(seed=seed, gen_s=(time.time() - t0) / B)))
    return eps


def batch_obs(eps):
    """Stack the observations of same-shape episodes -> dict of (B, T, ...) tensors."""
    return {k: torch.stack([e.obs[k] for e in eps]) for k in eps[0].obs}


def save(eps, path):
    torch.save([e.__dict__ for e in eps], path)


def load(path):
    return [Episode(**d) for d in torch.load(path, weights_only=False)]
