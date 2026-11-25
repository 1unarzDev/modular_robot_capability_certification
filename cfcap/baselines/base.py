"""Common baseline interface and the conformal evaluation every baseline feeds.

Interface. A baseline is any object with `name` and `predict(ep: Episode) -> Prediction`:
  input  = one Episode (parent observations, parent morphology, candidate set; privileged latents only for the
           oracle-type baselines that say so);
  output = per (candidate, leg): a point rho_rel `point` and a `scale` (> 0), optionally a raw lower bound `lower`
           (the baseline's own certificate, e.g. capxfer's), all (C, L).
Conformal layer: `conformal.core.SimultaneousLower` on (point, scale) with one max-normalised score per episode, so
the lower bounds hold for every candidate and leg of an episode at once (P(any entry under-covered) <= alpha).
Decisions: commit (candidate, leg, tau) iff lower >= tau. Empty candidate sets, missing or non-finite predictions
refuse (no commit).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from cfcap.conformal import core

L = 10
TAUS = (0.6, 0.7, 0.8)


@dataclass
class Prediction:
    point: np.ndarray                 # (C, L)
    scale: np.ndarray                 # (C, L) > 0
    lower: np.ndarray | None = None   # (C, L) raw lower bound of the baseline's own certificate (optional)
    info: dict = field(default_factory=dict)

    @classmethod
    def empty(cls):
        z = np.zeros((0, L))
        return cls(z, z.copy(), z.copy(), dict(refuse="empty candidate set"))

    def check(self, n_cand):
        assert self.point.shape == (n_cand, L) and self.scale.shape == (n_cand, L), (self.point.shape, n_cand)
        return self


class Baseline:
    name = "base"
    privileged = False                 # True: reads Episode.privileged()

    def predict(self, ep) -> Prediction:
        raise NotImplementedError

    def __call__(self, ep):
        if ep.n_cand == 0:
            return Prediction.empty()
        t0 = time.perf_counter()
        p = self.predict(ep).check(ep.n_cand)
        p.info.setdefault("seconds", time.perf_counter() - t0)
        return p


def refuse_lower(n_cand):
    return np.full((n_cand, L), -np.inf)


def decide(lower, tau):
    """Go mask (C, L) for requirement tau; empty -> empty (refuse); NaN/inf -> no go."""
    lower = np.asarray(lower, float)
    if lower.size == 0:
        return np.zeros(lower.shape, bool)
    return core.commit(lower, np.full(lower.shape, tau))


# ----------------------------------------------------------------------------- evaluation
def pad(arrs, width):
    out = np.full((len(arrs), width), np.nan)
    for i, a in enumerate(arrs):
        a = np.asarray(a, float).ravel()
        out[i, :a.size] = a
    return out


def cross_fit_lower(point, scale, truth, folds, alpha):
    """2+-fold split conformal: each fold's lower bounds use the q fitted on the other folds. Returns lower, q per
    fold."""
    lower = np.full_like(point, np.nan)
    qs = {}
    for f in np.unique(folds):
        te, ca = folds == f, folds != f
        sl = core.SimultaneousLower(alpha).fit(point[ca], scale[ca], truth[ca])
        lower[te] = sl.lower(point[te], scale[te])
        qs[int(f)] = sl.q
    lower = np.where(np.isnan(point), np.nan, lower)
    return lower, qs


def metrics(lower, truth, has_fault, taus=TAUS, ep_mask=None):
    """lower/truth/has_fault (E, W) padded with NaN. Per tau: certified fraction of capable entries (all, and those of
    candidates holding a faulted wheel), unsafe-commit rate among incapable entries, and the episode-level rate of
    any unsafe commit (the simultaneous guarantee bounds it by alpha)."""
    rows = []
    E = lower.shape[0]
    m = np.ones(E, bool) if ep_mask is None else ep_mask
    lo, tr, hf = lower[m], truth[m], has_fault[m]
    present = ~np.isnan(tr)
    for tau in taus:
        go = np.where(present, np.nan_to_num(lo, nan=-np.inf) >= tau, False)
        cap_ = present & (tr >= tau)
        inc = present & (tr < tau)
        f = present & (hf > 0.5)
        unsafe = go & inc
        rows.append(dict(tau=tau, n_ep=int(m.sum()), n_capable=int(cap_.sum()), n_incapable=int(inc.sum()),
                         cert_frac=_r(go & cap_, cap_), cert_frac_faulted=_r(go & cap_ & f, cap_ & f),
                         n_capable_faulted=int((cap_ & f).sum()),
                         unsafe_commit_rate=_r(unsafe, inc), n_unsafe=int(unsafe.sum()),
                         ep_any_unsafe=float(unsafe.any(1).mean()) if len(go) else np.nan,
                         ep_miscover=float((present & (np.nan_to_num(lo, nan=-np.inf) > tr)).any(1).mean())
                         if len(go) else np.nan))
    return rows


def _r(a, b):
    nb = int(b.sum())
    return float(a.sum() / nb) if nb else np.nan
