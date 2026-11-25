"""Exploratory audit: baseline scorers, cross-fit isolation and projected consistent sets.

Intended predictors: nominal = healthy rho_dec; oracle = privileged true rho_dec;
RLS = duty/speed regression, relative gain interval propagated through QS;
enumeration/SMC = weighted posterior mean, sd + floor and lower quantile;
capxfer = worst band over globally consistent projected mode vectors, empty -> refuse.
All feed episode-max SimultaneousLower, calibrated on other folds. Baseline
unsafe rate is unsafe / incapable (different from decide's unsafe / assigned).
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cfcap.baselines import base as B
from cfcap.baselines.capxfer_cert import SC, child_vectors
from cfcap.baselines.posterior import summarise, wquantile
from cfcap.baselines.simple import Nominal, Oracle
from cfcap.baselines.rls import rls_gains


def test_empty_baseline_never_calls_predict():
    for baseline in (Nominal(), Oracle()):
        p = baseline(SimpleNamespace(n_cand=0))
        assert p.point.shape == (0, B.L)
        assert not B.decide(p.lower, .7).any()


def test_oracle_nominal_and_quantile_hand_example():
    truth = np.full((1, B.L), .7)
    ep = SimpleNamespace(n_cand=1, truth=truth)
    np.testing.assert_equal(Oracle()(ep).lower, truth)
    assert B.decide(Oracle()(ep).lower, .7).all()
    assert B.decide(Nominal()(ep).lower, .7).all()
    rho = np.array([[[.4]], [[1.]]])
    mean, sd, lo = summarise(rho, np.array([.25, .75]))
    np.testing.assert_allclose(mean, [[.85]])
    np.testing.assert_allclose(sd ** 2, [[.0675]])
    np.testing.assert_equal(lo, [[.4]])
    np.testing.assert_equal(wquantile(rho, np.array([.25, .75]), .25), [[.4]])


def test_crossfit_own_labels_do_not_change_own_lower():
    p = np.ones((20, 2)); s = np.ones_like(p)
    tr = np.full_like(p, .8); folds = np.arange(20) % 2
    lo, _ = B.cross_fit_lower(p, s, tr, folds, .1)
    tr[folds == 0] = -100
    changed, _ = B.cross_fit_lower(p, s, tr, folds, .1)
    np.testing.assert_equal(lo[folds == 0], changed[folds == 0])
    assert np.all(changed[folds == 1] < lo[folds == 1])


def test_baseline_metrics_refuse_infinite_lower():
    r = B.metrics(np.array([[np.inf]]), np.array([[.2]]), np.ones((1, 1)), taus=(.7,))[0]
    assert r['n_unsafe'] == 0, r


def test_baseline_metrics_ignore_nonfinite_truth():
    r = B.metrics(np.ones((1, 3)), np.array([[np.inf, -np.inf, np.nan]]), np.ones((1, 3)), taus=(.7,))[0]
    assert r['n_dec'] == 0, r


def test_baseline_relevant_stratum_is_faulted_or_near():
    r = B.metrics(np.ones((1, 3)), np.array([[1., .75, .2]]), np.array([[1, 0, 0]]),
                  taus=(.7,), band=.1)[0]
    assert r['n_dec'] == 2, r


def test_projected_empty_set_inside_refuses():
    assert child_vectors([set(), {0}], [1, 1], [0], 2) == []


def test_projected_empty_set_outside_refuses():
    assert child_vectors([{0}, set()], [1, 1], [0], 2) == []


def test_projection_budget_cap_and_known_partner():
    assert child_vectors([{1}, {1}, {0, 1}], [.7] * 3, [2], 3, k=2) == [((0,), (1.,))]
    assert child_vectors([{0, 1}], [.7], [0], 1, cap_=1) is None
    assert child_vectors([{0}], [1], [0, -1], 1, fixed=([0, 3], [1, 1])) == [((0, 3), (1., 1.))]


def test_mondrian_absent_class_and_threshold_tie():
    q = SC.mondrian_thresholds(np.tile([1, 0, 0, 0], (9, 1)), np.zeros(9), .1)
    assert q[0] == 0 and np.isinf(q[1:]).all()
    assert SC.wheel_sets([[.7, .3, 0, 0]], [1 - .7, 0, 0, 0])[0, 0]


def test_rls_singular_observations_have_unbounded_uncertainty():
    asm = SimpleNamespace(n=2, ik=lambda cmd, radius: np.zeros((len(cmd), 2)))
    obs = dict(pre_duty=torch.zeros((4, 2)), speed=torch.zeros((4, 2)))
    gh, se, k = rls_gains(asm, np.zeros((4, 3)), obs)
    np.testing.assert_equal(gh, [0, 0])
    assert np.isinf(se).all()


def test_rls_exact_relative_gain_and_no_truth_input():
    # Full rank duty/sign design; known slopes in a noiseless synthetic motor.
    u = np.array([[.1, .2], [.4, .5], [-.2, -.1], [-.6, -.4]])
    cmd = np.array([[.1, 0, 0], [.1, 0, 0], [-.1, 0, 0], [-.1, 0, 0]])
    ref = np.tile(np.sign(cmd[:, :1]), (1, 2))
    asm = SimpleNamespace(n=2, ik=lambda cmd, radius: ref)
    obs = dict(pre_duty=torch.tensor(u), speed=torch.tensor(u * [2., 4.] + .1 * ref))
    gh, se, k = rls_gains(asm, cmd, obs)
    np.testing.assert_allclose(k, [2, 4])
    np.testing.assert_allclose(gh, [2 / 3, 4 / 3])
    assert (se < 1e-12).all()
