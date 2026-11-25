"""Reference baselines: nominal (assume every module healthy) and the full-state oracle."""
from __future__ import annotations

import numpy as np

from .base import L, Baseline, Prediction


class Nominal(Baseline):
    """Every candidate is assumed healthy: rho_rel = 1 on every leg (relative capability of a fault-free body),
    unit scale (the conformal q is then an additive margin)."""
    name = "nominal"

    def predict(self, ep):
        C = ep.n_cand
        return Prediction(np.ones((C, L)), np.ones((C, L)), np.ones((C, L)))


class Oracle(Baseline):
    """Full-state oracle: the T1 label of the candidate under the true latents (what a perfect self-model with every
    fault and nuisance known would compute). Upper bound on certifiable fraction; zero unsafe commits."""
    name = "oracle"
    privileged = True

    def __init__(self, eps_scale=1e-3):
        self.eps_scale = eps_scale

    def predict(self, ep):
        t = np.asarray(ep.rho_rel, float)
        return Prediction(t.copy(), np.full(t.shape, self.eps_scale), t.copy())
