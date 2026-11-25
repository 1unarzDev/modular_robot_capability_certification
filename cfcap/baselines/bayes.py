"""Simulator-likelihood Bayes baselines: exact enumeration with nuisances known (A6) and nuisance-SMC.

Both output per (candidate, leg): point = posterior mean rho_rel, scale = posterior sd + SCALE_FLOOR, and the posterior
5 % quantile as the raw lower bound. Posterior particles reach the candidates through `posterior.propagate`.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import posterior as Po
from .base import Baseline, Prediction
from .episode import context, lat_to

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
        par, asm, cands = context(ep.shape, ep.plate)
        lat = lat_to(ep.privileged(), self.device)
        t0 = time.perf_counter()
        post = Po.enumerate_known(asm, par, ep.cmd, ep.T, lat, ep.obs, self.grid)
        t_post = time.perf_counter() - t0
        k = Po.top_mass(post["w"], self.cover, self.cap)
        rows = Po.set_faults(Po.rows_from(lat, len(k)), post["HM"][k], post["HG"][k])
        rho = Po.propagate(par, cands, rows, post["w"][k], self.device)
        tm, tg = ep.faults()
        hit = int(np.flatnonzero((post["HM"] == tm).all(1))[0]) if ((post["HM"] == tm).all(1)).any() else -1
        return _pred(rho, post["w"][k], dict(t_posterior=t_post, n_sims=post["n_sims"], n_prop=len(k),
                                             map_mass=float(post["w"].max()),
                                             mode_mass_true=float(post["w"][(post["HM"] == tm).all(1)].sum())))


class NuisanceSMC(Baseline):
    """Faults x sampled nuisances (heterogeneity, mu, load split); see posterior.nuisance_smc."""
    name = "nuisance_smc"
    privileged = False      # reads only the known part of the domain and the mounting-yaw errors (as A6)

    def __init__(self, grid=Po.GRID, K=96, M=24, N=128, mh_steps=2, n_prop=32, seed=0, device="cuda"):
        self.kw = dict(grid=grid, K=K, M=M, N=N, mh_steps=mh_steps)
        self.n_prop, self.seed, self.device = n_prop, seed, device

    def predict(self, ep):
        par, asm, cands = context(ep.shape, ep.plate)
        lat_known = lat_to(ep.privileged(), self.device)  # only domain (minus mu) and yaw errors are used
        rng = np.random.default_rng(self.seed + 7919 * ep.eid)
        t0 = time.perf_counter()
        post = Po.nuisance_smc(asm, par, ep.cmd, ep.T, lat_known, ep.obs, rng, **self.kw)
        t_post = time.perf_counter() - t0
        N = post["lat"]["mode"].shape[0]
        sel = rng.choice(N, size=min(self.n_prop, N), replace=False)
        rows = Po.plant._index(post["lat"], torch.as_tensor(sel, device=self.device))
        rho = Po.propagate(par, cands, rows, np.ones(len(sel)), self.device)
        tm, _ = ep.faults()
        return _pred(rho, np.ones(len(sel)), dict(t_posterior=t_post, n_sims=post["n_sims"], beta=post["beta"], n_stages=post["n_stages"],
                                                  n_prop=len(sel),
                                                  mode_mass_true=float((post["HM"] == tm).all(1).mean())))
