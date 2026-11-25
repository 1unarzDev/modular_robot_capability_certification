"""Simulator-likelihood Bayes baselines: exact enumeration with nuisances known (A6) and nuisance-SMC.

Both output per (candidate, leg): point = posterior mean rho_dec, scale = posterior sd + SCALE_FLOOR, and the posterior
5 % quantile as the raw lower bound. Posterior particles reach the candidates through `posterior.propagate`.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import posterior as Po
from .base import Baseline, Prediction
from .episode import lat_to

SCALE_FLOOR = 0.02


def _pred(rho, w, info):
    m, sd, q05 = Po.summarise(rho, w)
    return Prediction(m, sd + SCALE_FLOOR, q05, info)


class Enumeration(Baseline):
    """A6 exact enumeration over <= 2-fault hypotheses, every nuisance at its true value (oracle-ish)."""
    name = "enum_known"
    privileged = True

    def __init__(self, grid=Po.GRID, cover=0.999, cap=32, device="cuda"):
        self.grid, self.cover, self.cap, self.device = grid, cover, cap, device

    def predict(self, ep):
        par, asm, cands = ep.params, ep.parent_asm, ep.cands
        lat = lat_to(ep.privileged(), self.device)
        t0 = time.perf_counter()
        post = Po.enumerate_known(asm, par, ep.cmd, ep.T, lat, ep.obs, self.grid)
        t_post = time.perf_counter() - t0
        k = Po.top_mass(post["w"], self.cover, self.cap)
        rows = Po.set_faults(Po.rows_from(lat, len(k)), post["HM"][k], post["HG"][k])
        rho = Po.propagate(par, cands, rows, post["w"][k], self.device, ep.plate)
        tm, tg = ep.faults()
        hit = int(np.flatnonzero((post["HM"] == tm).all(1))[0]) if ((post["HM"] == tm).all(1)).any() else -1
        return _pred(rho, post["w"][k], dict(t_posterior=t_post, n_sims=post["n_sims"], n_prop=len(k),
                                             map_mass=float(post["w"].max()),
                                             mode_mass_true=float(post["w"][(post["HM"] == tm).all(1)].sum())))


class NuisanceSMCIS(Baseline):
    """Superseded by NuisanceSMC (role nuisance's sampler); kept for comparison: faults x sampled nuisances
    (heterogeneity, mu, load split) with the rest of the domain known; see posterior.nuisance_smc."""
    name = "nuisance_smc_is"
    privileged = False      # reads only the known part of the domain and the mounting-yaw errors (as A6)

    def __init__(self, grid=Po.GRID, K=96, M=24, N=128, mh_steps=2, n_prop=32, seed=0, device="cuda"):
        self.kw = dict(grid=grid, K=K, M=M, N=N, mh_steps=mh_steps)
        self.n_prop, self.seed, self.device = n_prop, seed, device

    def predict(self, ep):
        par, asm, cands = ep.params, ep.parent_asm, ep.cands
        lat_known = lat_to(ep.privileged(), self.device)  # only domain (minus mu) and yaw errors are used
        rng = np.random.default_rng(self.seed + 7919 * ep.eid)
        t0 = time.perf_counter()
        post = Po.nuisance_smc(asm, par, ep.cmd, ep.T, lat_known, ep.obs, rng, **self.kw)
        t_post = time.perf_counter() - t0
        N = post["lat"]["mode"].shape[0]
        sel = rng.choice(N, size=min(self.n_prop, N), replace=False)
        rows = Po.plant._index(post["lat"], torch.as_tensor(sel, device=self.device))
        rho = Po.propagate(par, cands, rows, np.ones(len(sel)), self.device, ep.plate)
        tm, _ = ep.faults()
        return _pred(rho, np.ones(len(sel)), dict(t_posterior=t_post, n_sims=post["n_sims"], beta=post["beta"], n_stages=post["n_stages"],
                                                  n_prop=len(sel),
                                                  mode_mass_true=float((post["HM"] == tm).all(1).mean())))


class NuisanceSMC(Baseline):
    """Nuisance-SMC over faults x every nuisance, run by role nuisance's sampler (`cfcap.nuisance_post`, used read-only):
    standard-normal reparametrised joint over (mode vector, g, global DR, per-wheel motor spread, mounting yaw,
    normal-load split, per-module V_batt and mu), single-fault screen proposal, adaptive tempering, random-walk +
    discrete + ridge-Gibbs moves, on 0.5-s bin means of duty / speed / gz (its SD table). This baseline adds:
      * the observation of a baselines Episode (any parent, any v2 table; generator episodes included), binned as
        nuisance_post.feats_from does, gz minus the gyro bias estimated on the leading rest;
      * a known payload (task property) copied onto every particle;
      * propagation of n_prop posterior draws to every candidate (children and merges with known partners) as
        rho_dec, summarised as posterior mean / sd + SCALE_FLOOR / 5 % quantile.
    Nothing privileged is read except the payload."""
    name = "nuisance_smc"
    privileged = False

    def __init__(self, particles=256, max_stages=150, n_prop=32, seed=0, device="cuda"):
        self.particles, self.max_stages, self.n_prop, self.seed, self.device = particles, max_stages, n_prop, seed, device
        self._setup = {}

    def _ctx(self, ep, asm, par):
        import types
        from cfcap import nuisance_post as NP
        from cfcap.t1 import params as P
        key = (id(asm), ep.T, ep.cmd.shape)
        if key not in self._setup:
            if len(self._setup) > 64:
                self._setup.clear()
            dr = P.theta_dr(P.load_theta(ep.plate))
            probe = types.SimpleNamespace(tw=np.asarray(ep.cmd, np.float32), pw=None, T=float(ep.T))
            self._setup[key] = (NP.Space(asm, dr), NP.Sim(asm, par, probe, "graph", 8192))
        return self._setup[key]

    def predict(self, ep):
        from cfcap import nuisance_post as NP
        par, asm, cands = ep.params, ep.parent_asm, ep.cands
        space, simr = self._ctx(ep, asm, par)
        dev = self.device
        gb = Po.gyro_bias_rest(ep.obs, ep.cmd)
        T = (ep.obs["pre_duty"].shape[0] // NP.BIN) * NP.BIN
        b = lambda x: x[None, :T].reshape(1, T // NP.BIN, NP.BIN, *x.shape[1:]).mean(2).to(dev).float()
        y = dict(duty=b(ep.obs["pre_duty"]), speed=b(ep.obs["speed"]), gz=b(ep.obs["gz"] - gb))
        gen = torch.Generator(device=dev).manual_seed(int(self.seed + 7919 * ep.eid) % (1 << 62))
        t0 = time.perf_counter()
        simr.n_sims = 0
        q = NP.Q(NP.screen(space, simr, y, 1), asm.n)
        modes, z, ll, info = NP.smc(space, simr, y, q, 1, self.particles, gen, max_stages=self.max_stages)
        t_post = time.perf_counter() - t0
        pick = torch.randperm(self.particles, device=dev, generator=gen)[:self.n_prop]
        rows = space.to_lat(modes[0, pick], z[0, pick])
        pay = ep.privileged().get("payload")
        if pay is not None:
            rows["payload"] = {k: v.to(dev).expand(len(pick)).contiguous() for k, v in pay.items()}
        rho = Po.propagate(par, cands, rows, np.ones(len(pick)), dev, ep.plate)
        tm, _ = ep.faults()
        mt = torch.as_tensor(tm, device=dev)
        return _pred(rho, np.ones(len(pick)), dict(t_posterior=t_post, n_sims=int(simr.n_sims), beta=1.0,
                                                   n_stages=int(info["stages"]), forced=int(info["forced"]),
                                                   n_prop=len(pick),
                                                   mode_mass_true=float((modes[0] == mt).all(-1).float().mean())))
