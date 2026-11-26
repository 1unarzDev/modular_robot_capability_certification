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
    """A6 exact enumeration over <= 2-fault hypotheses, every nuisance at its true value (oracle-ish).

    No screen: the likelihood of every hypothesis is computed. Propagation to the candidates takes the highest-mass
    hypotheses until 1 - DROP_TOL of the posterior is covered (at most `cap`); the mass left out is recorded, and if it
    exceeds DROP_TOL the episode abstains (lower = -inf everywhere: unknown mass is never certified; codex screen-bug
    review, docs/codex/screen_bug.md). Truth retention (info): `truth_in_support` (the true mode vector has <= 2
    faults; generator episodes can have 3), `truth_retained` (it is among the propagated hypotheses with the grid's
    nearest gains or any gains), `mass_true` (posterior mass on the true mode vector)."""
    name = "enum_known"
    privileged = True
    DROP_TOL = 1e-3

    def __init__(self, grid=Po.GRID, cover=None, cap=512, device="cuda"):
        self.grid, self.cap, self.device = grid, cap, device
        self.cover = 1 - self.DROP_TOL if cover is None else cover

    def predict(self, ep):
        par, asm, cands = ep.params, ep.parent_asm, ep.cands
        lat = lat_to(ep.privileged(), self.device)
        t0 = time.perf_counter()
        post = Po.enumerate_known(asm, par, ep.cmd, ep.T, lat, ep.obs, self.grid)
        t_post = time.perf_counter() - t0
        k = Po.top_mass(post["w"], self.cover, self.cap)
        dropped = float(max(0.0, 1.0 - post["w"][k].sum()))
        rows = Po.set_faults(Po.rows_from(lat, len(k)), post["HM"][k], post["HG"][k])
        rho = Po.propagate(par, cands, rows, post["w"][k], self.device, ep.plate)
        tm, tg = ep.faults()
        same = (post["HM"] == tm).all(1)
        info = dict(t_posterior=t_post, n_sims=post["n_sims"], n_prop=len(k), map_mass=float(post["w"].max()),
                    mass_true=float(post["w"][same].sum()), truth_in_support=int(same.any()),
                    truth_retained=int(same[k].any()), dropped_mass=dropped, abstain=int(dropped > self.DROP_TOL))
        info["mode_mass_true"] = info["mass_true"]
        pred = _pred(rho, post["w"][k], info)
        if dropped > self.DROP_TOL:
            pred.lower = np.full_like(pred.lower, -np.inf)
            pred.point = np.full_like(pred.point, -np.inf)       # the simultaneous layer then refuses too
        return pred


class NuisanceSMCIS(Baseline):
    """NOT REPORTED. Superseded by NuisanceSMC (role nuisance's sampler). Its screen keeps only the top K hypotheses at
    the mean nuisance (sd x 3) and excluded faults never return (the screen-truncation flaw of docs/codex/screen_bug.md),
    so it is not in the runner's method list and no result uses it. Faults x sampled nuisances
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
    Nothing privileged is read except the payload (and the true faults, only to record truth retention).
    Screen: nuisance_post's q(H) mixes the single-fault screen with the prior (0.3) and the discrete moves reach every
    mode vector, so no hypothesis is excluded for good; truth retention is still recorded per episode
    (`truth_retained`: the true mode vector is among the propagated draws; `mass_true`: share of particles on it).
    Its prior allows <= 2 faults; generator episodes with 3 faults are outside the support (`truth_in_support` 0)."""
    name = "nuisance_smc"
    privileged = False

    SIMS_PER_STAGE = 7          # nuisance_post.smc: per stage 3 sweeps x (continuous + discrete) evals + resample

    def __init__(self, particles=256, max_stages=150, n_prop=32, seed=0, device="cuda", wall_budget_s=None):
        self.particles, self.max_stages, self.n_prop, self.seed, self.device = particles, max_stages, n_prop, seed, device
        self.wall_budget_s = wall_budget_s
        self._setup = {}
        self._s_per_sim = {}

    def _stage_cap(self, key, t_screen, n_screen, t_prop):
        """Matched compute: the largest stage count whose predicted wall (screen + stages + propagation) fits
        wall_budget_s. Seconds per simulated row are measured on this episode's screen, then refined per parent
        shape from the previous episodes' SMC walls. nuisance_post.smc jumps to beta = 1 at its stage cap (the
        posterior then is the cheaper, less mixed one); the cap is at least 1."""
        sps = self._s_per_sim.get(key, t_screen / max(n_screen, 1))
        per_stage = sps * self.SIMS_PER_STAGE * self.particles
        left = self.wall_budget_s - t_screen - t_prop
        return int(max(1, min(self.max_stages, np.floor(left / max(per_stage, 1e-9) + 1e-9))))

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
            self._setup[key] = (NP.Space(asm, dr), NP.Sim(asm, par, probe, Po.pick_engine("graph"),
                                                          max(64, Po.CHUNK_WHEELS // asm.n)))
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
        t_screen, n_screen = time.perf_counter() - t0, simr.n_sims
        cap_ = self.max_stages
        key = (id(asm), ep.cmd.shape)
        if self.wall_budget_s is not None:
            cap_ = self._stage_cap(key, t_screen, n_screen, self._t_prop.get(key, 0.0) if hasattr(self, "_t_prop")
                                   else 0.0)
        t1 = time.perf_counter()
        modes, z, ll, info = NP.smc(space, simr, y, q, 1, self.particles, gen, max_stages=cap_)
        t_post = time.perf_counter() - t0
        if simr.n_sims > n_screen:
            self._s_per_sim[key] = (time.perf_counter() - t1) / (simr.n_sims - n_screen)
        pick = torch.randperm(self.particles, device=dev, generator=gen)[:self.n_prop]
        rows = space.to_lat(modes[0, pick], z[0, pick])
        pay = ep.privileged().get("payload")
        if pay is not None:
            rows["payload"] = {k: v.to(dev).expand(len(pick)).contiguous() for k, v in pay.items()}
        t2 = time.perf_counter()
        rho = Po.propagate(par, cands, rows, np.ones(len(pick)), dev, ep.plate)
        if not hasattr(self, "_t_prop"):
            self._t_prop = {}
        self._t_prop[key] = time.perf_counter() - t2
        tm, _ = ep.faults()
        mt = torch.as_tensor(tm, device=dev)
        hit = (modes[0] == mt).all(-1)
        hit_p = (modes[0, pick] == mt).all(-1)
        return _pred(rho, np.ones(len(pick)), dict(t_posterior=t_post, n_sims=int(simr.n_sims), beta=1.0,
                                                   n_stages=int(info["stages"]), forced=int(info["forced"]),
                                                   n_prop=len(pick), mass_true=float(hit.float().mean()),
                                                   mode_mass_true=float(hit.float().mean()),
                                                   truth_in_support=int(int((tm != 0).sum()) <= 2),
                                                   truth_retained=int(bool(hit_p.any())), dropped_mass=0.0,
                                                   abstain=0, stage_cap=int(cap_),
                                                   wall_budget_s=float(self.wall_budget_s or np.nan)))
