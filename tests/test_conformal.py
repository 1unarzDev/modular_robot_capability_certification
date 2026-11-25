import math
import numpy as np

from cfcap.conformal.core import (SimultaneousLower, commit, risk_control_margin, split_quantile,
                                  weighted_quantile)


def test_split_quantile_small_n_refuses():
    assert split_quantile([0.1, 0.2], 0.1) == math.inf
    assert split_quantile([], 0.1) == math.inf


def test_split_quantile_nan_is_worst():
    s = np.r_[np.zeros(98), np.nan]
    assert split_quantile(s, 0.01) == math.inf


def test_simultaneous_coverage_exchangeable():
    rng = np.random.default_rng(0)
    alpha, E, J, reps = 0.1, 400, 12, 200
    miss = 0
    for _ in range(reps):
        truth = rng.uniform(0, 1, (E + 1, J))
        scale = rng.uniform(0.05, 0.2, (E + 1, J))
        pred = truth + scale * rng.standard_normal((E + 1, J))
        pred[rng.random((E + 1, J)) < 0.2] = np.nan  # ragged candidate sets
        m = SimultaneousLower(alpha).fit(pred[:E], scale[:E], truth[:E])
        lo = m.lower(pred[E], scale[E])
        present = ~np.isnan(pred[E])
        miss += np.any(lo[present] > truth[E][present])
    assert miss / reps <= alpha + 0.06


def test_commit_refuses_nonfinite_and_empty():
    assert not commit(np.nan, 0.5)
    assert not commit(-np.inf, 0.5)
    assert not commit(np.inf, np.nan)
    assert commit(0.9, 0.8)
    assert commit(np.array([]), np.array([])).size == 0


def test_risk_control_bound():
    rng = np.random.default_rng(1)
    lo = rng.uniform(0, 1, 1000)
    truth = lo + rng.normal(0, 0.1, 1000)
    req = 0.5
    lams = np.linspace(0, 0.5, 51)
    lam = risk_control_margin(lambda l: ((lo - l >= req) & (truth < req)).astype(float), lams, 0.05)
    assert np.isfinite(lam)
    # unsafe commits on fresh data stay near alpha
    lo2 = rng.uniform(0, 1, 20000)
    t2 = lo2 + rng.normal(0, 0.1, 20000)
    assert ((lo2 - lam >= req) & (t2 < req)).mean() <= 0.06


def test_risk_control_impossible_refuses():
    assert risk_control_margin(lambda l: np.ones(10), [0.0, 1.0], 0.05) == math.inf


def test_weighted_equals_split_with_unit_weights():
    s = np.arange(1, 100, dtype=float)
    assert weighted_quantile(s, np.ones_like(s), 1.0, 0.1) == split_quantile(s, 0.1)
