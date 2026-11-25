"""Mission-level certificates on cfcap.env.missions records (env role's interface: plan_options / certified / plan_ok).

The env record gives per mission: candidates C, segments S (leg, condition, phase), conditions K, req (C, S) in
rho_dec units, geometry fp_ok / load_ok (C, S), truth rho_dec (C, K, L). A predictor gives point / scale (C, K, L)
for the mission's body (posterior draws pushed through T1 under each condition).

Simultaneous mission score = max over geometry-feasible (c, s) of (point - truth) / scale at (c, seg_cond[s],
seg_leg[s]); q = split quantile over calibration missions, cross-fitted by BUILDING (missions of one building are
not exchangeable with each other: clusters stay on one side). lower = point - q x scale certifies every feasible
segment of every candidate of the mission at once, so whichever plan (a, b) is then chosen keeps P(any committed
segment truly fails) <= alpha marginally over missions. `lower_rows` writes the (family, mission, cand, seg, lower)
rows that scripts/env_score.py consumes.
"""
from __future__ import annotations

import numpy as np

from cfcap.conformal import core


def seg_values(rec, x):
    """(C, K, L) -> (C, S) at each segment's condition and leg."""
    return np.asarray(x, float)[:, rec["seg_cond"], rec["seg_leg"]]


def mission_score(rec, point, scale, min_scale=1e-6):
    feas = rec["fp_ok"] & rec["load_ok"]
    p, t = seg_values(rec, point), np.asarray(rec["seg_truth"], float)
    sc = np.maximum(seg_values(rec, scale), min_scale)
    r = np.where(feas & ~np.isnan(p), (p - t) / sc, -np.inf)
    r = np.where(np.isnan(r), np.inf, r)                     # a present but undefined residual is the worst case
    return float(r.max()) if r.size else -np.inf


def fit_q(recs, points, scales, alpha=0.1):
    s = np.array([mission_score(r, p, c) for r, p, c in zip(recs, points, scales)])
    return core.split_quantile(s[s > -np.inf], alpha)


def crossfit_q(recs, points, scales, alpha=0.1, n_folds=3):
    """Per-mission q from the other folds; folds by building id (hash) so a building never straddles cal / test."""
    b = np.array([hash((r["family"], r["building"])) % n_folds for r in recs])
    q = np.empty(len(recs))
    for f in range(n_folds):
        ca = np.flatnonzero(b != f)
        qf = fit_q([recs[i] for i in ca], [points[i] for i in ca], [scales[i] for i in ca], alpha)
        q[b == f] = qf
    return q


def lower_rows(rec, point, scale, q):
    """Rows (family, mission, cand, seg, lower) for env_score; non-finite -> -inf (certifies nothing)."""
    lo = seg_values(rec, point) - (q * seg_values(rec, scale) if np.isfinite(q) else np.inf)
    lo = np.where(np.isfinite(lo), lo, -np.inf)
    C, S = lo.shape
    cc, ss = np.meshgrid(np.arange(C), np.arange(S), indexing="ij")
    return dict(family=np.full(C * S, rec["family"]), mission=np.full(C * S, rec["mission"]), cand=cc.ravel(),
                seg=ss.ravel(), lower=lo.ravel())
