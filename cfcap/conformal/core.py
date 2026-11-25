"""Conformal layers for counterfactual capability certificates.

Guarantees (all marginal over exchangeable generator episodes; see docs/PLAN.md section 3):
- `split_quantile`: standard split-conformal quantile with the (n+1) correction.
- `SimultaneousLower`: one score per episode = max over (candidate, leg) of the normalised under-prediction,
  so the lower bounds hold for every candidate in the episode at once (picking the best-looking candidate
  keeps the guarantee).
- `risk_control_margin`: conformal risk control (Angelopoulos et al. 2024) for a monotone loss in a margin
  lambda, e.g. the unsafe-commit indicator of a commit rule `lower - lambda >= requirement`.
- `weighted_quantile`: covariate-shift weighted conformal (Tibshirani et al. 2019) for known density ratios.
Refusal is the safe default: missing, non-finite or empty inputs never produce a "go".
"""
from __future__ import annotations

import math
import numpy as np


def split_quantile(scores, alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th smallest score; +inf if n is too small for that level."""
    s = np.asarray(scores, dtype=float).ravel()
    s = np.sort(np.where(np.isnan(s), np.inf, s))  # a NaN score counts as the worst case
    n = s.size
    k = math.ceil((n + 1) * (1.0 - alpha))
    if n == 0 or k > n:
        return math.inf
    return float(s[k - 1])


def weighted_quantile(scores, w_cal, w_test: float, alpha: float) -> float:
    """Weighted conformal quantile: calibration weights w_cal, test weight w_test (mass at +inf)."""
    s = np.asarray(scores, dtype=float).ravel()
    w = np.asarray(w_cal, dtype=float).ravel()
    if s.size == 0 or s.size != w.size:
        return math.inf
    order = np.argsort(s)
    s, w = s[order], w[order]
    p = w / (w.sum() + w_test)
    cum = np.cumsum(p)
    idx = np.searchsorted(cum, 1.0 - alpha - 1e-12)
    return float(s[idx]) if idx < s.size else math.inf


class SimultaneousLower:
    """Joint lower bounds over all (candidate, leg) entries of an episode.

    pred, scale, truth: arrays (episodes, entries) with NaN for absent entries (ragged candidate sets).
    Score per episode = max_j (pred_j - truth_j) / scale_j over present entries.
    Lower bound = pred - q * scale.
    """

    def __init__(self, alpha: float, min_scale: float = 1e-6):
        self.alpha, self.min_scale, self.q = alpha, min_scale, math.inf

    def scores(self, pred, scale, truth):
        pred, truth = np.asarray(pred, float), np.asarray(truth, float)
        scale = np.maximum(np.asarray(scale, float), self.min_scale)
        r = (pred - truth) / scale
        r = np.where(np.isnan(r), -np.inf, r)
        s = r.max(axis=1)
        # an episode with no present entries carries no information; drop it
        return s[np.isfinite(s) | (s == np.inf)]

    def fit(self, pred, scale, truth):
        self.q = split_quantile(self.scores(pred, scale, truth), self.alpha)
        return self

    def lower(self, pred, scale):
        scale = np.maximum(np.asarray(scale, float), self.min_scale)
        lo = np.asarray(pred, float) - self.q * scale
        return np.where(np.isfinite(lo), lo, -np.inf)


def commit(lower, requirement):
    """Commit (True) only if a finite lower bound meets the requirement; NaN/inf/empty refuse."""
    lo = np.asarray(lower, float)
    req = np.asarray(requirement, float)
    ok = np.isfinite(lo) & np.isfinite(req) & (lo >= req)
    return ok


def risk_control_margin(loss_fn, lambdas, alpha: float, B: float = 1.0):
    """Smallest lambda (lambdas sorted so loss is non-increasing) with
    (n/(n+1)) * mean_loss(lambda) + B/(n+1) <= alpha. loss_fn(lambda) -> per-episode losses in [0, B].
    Returns +inf if none qualifies (always refuse)."""
    for lam in lambdas:
        L = np.asarray(loss_fn(lam), float)
        n = L.size
        if n == 0:
            return math.inf
        if (n / (n + 1)) * L.mean() + B / (n + 1) <= alpha:
            return float(lam)
    return math.inf
