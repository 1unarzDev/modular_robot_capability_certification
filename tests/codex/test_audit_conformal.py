"""Exploratory scorer audit: joint lower coverage, finite refusal, CRC and weighted quantiles.

Observed final full-suite run, 2026-11-25 (CPU, CUDA_VISIBLE_DEVICES=''):
    $PY -m pytest -q tests/codex
    19 failed, 47 passed, 2 warnings in 12.77s; exit status 1.
    Audit files: 17 failed, 40 passed. Existing test_screen_bug.py: 2 failed, 7 passed.
Environment: PYTHONDONTWRITEBYTECODE=1, PYTEST_ADDOPTS='-p no:cacheprovider --tb=short',
OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1.
Only audit test files were written by this worker; source changed concurrently
during the task. This record describes the final run, including simF/simA code.
Malformed weights/losses and a present prediction with a missing truth label
are deliberate input-validation challenges, not claims about observed corpus data.
"""
import math

import numpy as np
import pytest

from cfcap.conformal import core as C


@pytest.mark.parametrize("scores,alpha,expected", [([], .1, math.inf), ([1, 2], .1, math.inf),
    ([1, 2, 3], .5, 2), ([0] * 8 + [np.nan], .1, math.inf)])
def test_split_order_statistic(scores, alpha, expected):
    assert C.split_quantile(scores, alpha) == expected


def test_commit_ties_empty_and_nonfinite():
    assert C.commit([.7, np.nan, np.inf, -np.inf], .7).tolist() == [True, False, False, False]
    assert not C.commit([], []).any()
    assert not C.commit([1, 1], [np.nan, np.inf]).any()


def test_ragged_joint_max_is_per_episode():
    p = [[2, 4, np.nan], [3, np.nan, np.nan], [np.nan] * 3]
    t = [[1, 0, np.nan], [2, np.nan, np.nan], [np.nan] * 3]
    s = [[1, 2, 1], [1, 1, 1], [1, 1, 1]]
    np.testing.assert_equal(C.SimultaneousLower(.5).scores(p, s, t), [2, 1])


def test_zero_width_joint_set_refuses():
    z = np.empty((3, 0))
    cert = C.SimultaneousLower(.1).fit(z, z, z)
    assert cert.q == math.inf
    assert not C.commit(cert.lower(z, z), .7).any()


def test_missing_label_for_present_prediction_is_worst_case():
    cert = C.SimultaneousLower(.1)
    scores = cert.scores([[1, 1]], [[1, 1]], [[1, np.nan]])
    assert scores.tolist() == [math.inf], scores


def test_small_calibration_and_unfit_lower_refuse():
    cert = C.SimultaneousLower(.1).fit([[1]], [[1]], [[1]])
    assert cert.q == math.inf
    assert not C.commit(cert.lower([1, np.nan, np.inf], [1, 1, 1]), .7).any()


@pytest.mark.parametrize("weights,test_weight", [([1, -1], 1), ([1, np.nan], 1), ([1, 1], -2)])
def test_invalid_weights_cannot_supply_finite_quantile(weights, test_weight):
    try:
        q = C.weighted_quantile([0, 1], weights, test_weight, .1)
    except ValueError:
        return
    assert q == math.inf, q


def test_weighted_nan_score_is_worst():
    assert C.weighted_quantile([0] * 8 + [np.nan], np.ones(9), 1, .1) == math.inf


def test_unit_weight_quantile_and_crc_exact_bound():
    x = np.arange(19.)
    assert C.weighted_quantile(x, np.ones(19), 1, .1) == C.split_quantile(x, .1)
    assert C.risk_control_margin(lambda l: np.zeros(9), [0, 1], .1) == 0
    assert C.risk_control_margin(lambda l: np.zeros(8), [0, 1], .1) == math.inf
    assert C.risk_control_margin(lambda l: [], [0], .1) == math.inf
    assert C.risk_control_margin(lambda l: [np.nan] * 20, [0], .1) == math.inf


def test_crc_negative_infinite_loss_refuses():
    try:
        q = C.risk_control_margin(lambda l: [-np.inf] * 20, [0], .1)
    except ValueError:
        return
    assert q == math.inf
