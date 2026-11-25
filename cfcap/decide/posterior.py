"""Posterior-sample interface the decision layer consumes (agreed contract; docs/board/decision.md).

A `DecisionPosterior` is a weighted particle set over the parent's latents, already pushed to the candidates:
  w       (P,)        normalised weights;
  rho     (P, C, L)   capability of every particle on every candidate and decision leg (NaN = absent entry);
  latent  (P,)        int id of the particle's discrete latent (fault mode vector incl. weak-g bin), for EIG;
  probes  {name: Probe} the probes still available: per-particle predicted signal `sig` (P, D), its noise sd
          (`sd` (P, D) or a callable sd(y) -> (D,)), and the time cost `t_s` (s). A diagnostic split is a probe like
          any other (name "split:<cand>", cost T_SPLIT_S + its v2 time).
Producers:
  * T0 (`cfcap.decide.t0data`): enumeration of <= 2-fault hypotheses x prior nuisance draws, importance-weighted by
    the parent's v2 likelihood;
  * baselines (`cfcap.baselines.bayes.NuisanceSMC` / `Enumeration`): posterior.propagate gives rho (P, C, L) and w;
    `from_baseline` wraps them (signals for extra probes are optional; without them the only action is decide);
  * models (amortized posterior): samples of latents pushed through the same propagation, or direct quantile heads
    turned into P pseudo-samples (no probes).
Updates are exact importance reweighting with the probe's Gaussian likelihood; `predictive` draws y from the mixture.
`summary` gives the (point, scale) the conformal layer (`SimultaneousLower`) consumes: weighted mean and
sd + SCALE_FLOOR.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCALE_FLOOR = 0.02        # as cfcap.baselines.bayes


@dataclass
class Probe:
    sig: np.ndarray                   # (P, D)
    sd: object                        # (P, D) array, or callable y (D,) -> sd (D,)
    t_s: float
    y_true: np.ndarray | None = None  # (D,) the observation the episode will actually return (evaluation only)

    def sd_at(self, y):
        return self.sd(y) if callable(self.sd) else None


@dataclass
class DecisionPosterior:
    w: np.ndarray
    rho: np.ndarray
    latent: np.ndarray
    probes: dict = field(default_factory=dict)
    unknown: bool = False             # support was pruned without a coverage guarantee (empty screen, budget, underflow):
                                      # every bound is -inf, so every decision abstains

    def __post_init__(self):
        self.w = np.asarray(self.w, float)
        s = self.w.sum()
        self.w = self.w / s if s > 0 and np.isfinite(s) else np.full(self.w.shape, 1.0 / max(len(self.w), 1))

    @property
    def P(self):
        return len(self.w)

    # ------------------------------------------------------------------ summaries
    def summary(self, w=None):
        w = self.w if w is None else w
        x = np.nan_to_num(self.rho, nan=0.0)
        m = np.tensordot(w, x, 1)
        sd = np.sqrt(np.maximum(np.tensordot(w, (x - m) ** 2, 1), 0.0))
        absent = np.isnan(self.rho).all(0)
        m[absent] = np.nan
        if self.unknown:
            m[:] = np.nan
        return m, sd + SCALE_FLOOR

    def bayes_lower(self, q=0.05, w=None):
        w = self.w if w is None else w
        x = np.where(np.isnan(self.rho), -np.inf, self.rho)
        o = np.argsort(x, 0)
        xs = np.take_along_axis(x, o, 0)
        cw = np.cumsum(w[o], 0)
        k = np.minimum((cw < q - 1e-12).sum(0), self.P - 1)
        lo = np.take_along_axis(xs, k[None], 0)[0]
        return np.full_like(lo, -np.inf) if self.unknown else lo

    # ------------------------------------------------------------------ probe likelihood
    def loglik(self, name, y):
        pr = self.probes[name] if isinstance(name, str) else name
        sd = pr.sd_at(y)
        if sd is None:
            sd = pr.sd
        sd = np.broadcast_to(np.asarray(sd, float), pr.sig.shape)
        z = (np.asarray(y)[None] - pr.sig) / sd
        return -0.5 * np.nansum(z * z, axis=1) - np.nansum(np.where(np.isnan(z), 0.0, np.log(sd)), axis=1)

    def reweight(self, name, y, w=None):
        """Posterior weights after observing y from probe `name` (a key or a Probe)."""
        w = self.w if w is None else w
        lw = np.log(np.maximum(w, 1e-300)) + self.loglik(name, y)
        if not np.isfinite(lw.max()):
            return np.full(len(w), np.nan)                     # underflow: caller marks the support unknown
        lw -= lw.max()
        nw = np.exp(lw)
        return nw / nw.sum()

    def predictive(self, name, rng, S):
        """S draws y ~ sum_i w_i N(sig_i, sd_i) -> (S, D)."""
        pr = self.probes[name]
        idx = rng.choice(self.P, size=S, p=self.w)
        mu = pr.sig[idx]
        if callable(pr.sd):
            sd = np.stack([pr.sd(m) for m in mu])
        else:
            sd = np.broadcast_to(pr.sd, pr.sig.shape)[idx]
        return mu + sd * rng.standard_normal(mu.shape)

    def observe(self, name):
        """Apply the probe's true observation (if any) and consume the probe. Returns its time cost."""
        pr = self.probes.pop(name)
        if pr.y_true is not None:
            nw = self.reweight(pr, pr.y_true)
            if not np.isfinite(nw).all():
                self.unknown = True
            else:
                self.w = nw
        return pr.t_s

    def copy(self):
        return DecisionPosterior(self.w.copy(), self.rho, self.latent, dict(self.probes), self.unknown)


def degenerate(rho, latent=0):
    """A single-particle 'posterior' (oracle / nominal predictors): no probe can change it."""
    rho = np.asarray(rho, float)[None]
    return DecisionPosterior(np.ones(1), rho, np.array([latent]))


def from_baseline(rho, w, latent=None, probes=None):
    """Wrap baselines.posterior.propagate output rho (P, C, L) and weights w (P,)."""
    rho = np.asarray(rho, float)
    lat = np.arange(rho.shape[0]) if latent is None else np.asarray(latent)
    return DecisionPosterior(np.asarray(w, float), rho, lat, dict(probes or {}))
