"""Exploratory audit: candidate-task safety, useful work, utility/regret, policy scores and cross-fit."""
import numpy as np
import pytest

from cfcap.decide import evaluate as E, policies as P, posterior as D, problem as B


def instance():
    return B.Instance(['parent'], np.array([1]), np.array([1]), [(0, 0)])


def episode():
    return dict(inst=instance(), tasks=[B.Task(np.array([.7]))], rho_true=np.array([[.7]]),
                has_fault=np.array([False]))


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, .69])
def test_required_nonfinite_or_below_refuses(value):
    ch, cost = B.decide(instance(), episode()['tasks'], np.array([[value]]))
    assert ch.tolist() == [-1] and cost == 0


def test_empty_candidates_and_no_tasks_abstain():
    inst = B.Instance([], np.array([], int), np.array([], int), [])
    ch, cost = B.decide(inst, episode()['tasks'], np.empty((0, 1)))
    assert ch.tolist() == [-1] and cost == 0
    ch, cost = B.decide(instance(), [], np.array([[1.]]))
    assert ch.size == 0 and cost == 0


def test_tie_capable_unsafe_and_regret_hand_example():
    inst, tasks = instance(), episode()['tasks']
    ch, tc = B.decide(inst, tasks, np.array([[.7]]))
    safe = B.outcome(inst, tasks, ch, tc, np.array([[.7]]))
    unsafe = B.outcome(inst, tasks, ch, tc, np.array([[.69]]), probe_time=5)
    assert (safe['n_served'], safe['n_unsafe'], safe['U']) == (1, 0, B.V_TASK)
    assert (unsafe['n_served'], unsafe['n_unsafe'], unsafe['U']) == (0, 1, -B.C_UNSAFE - 5)
    oracle = B.oracle(inst, tasks, np.array([[.69]]))[2]
    assert oracle['U'] - unsafe['U'] == B.C_UNSAFE + 5
    assert B.expected_utility(inst, tasks, ch, tc, np.array([[[.7]], [[.69]]]), [.25, .75]) == .25 * B.V_TASK - .75 * B.C_UNSAFE


def test_nonrequired_nan_ignored_and_size_enforced():
    R = np.array([[.7, np.nan]])
    assert B.capable([[.7, np.nan]], R, [1], [1]).item()
    assert not B.capable([[1, 1]], R, [2], [1]).item()


def test_pair_counts_and_ambiguity_probability_boundaries():
    ep = episode()
    rho = np.array([[[.7]], [[.2]]])
    counts = E.pair_counts(ep, [[.7]], [.05, .95], [.5, .5], rho)
    assert (counts['n_capable'], counts['n_certified'], counts['n_unsafe_cert']) == (1, 1, 0)
    assert (counts['n_amb_before'], counts['n_amb_after']) == (0, 1)
    assert counts['n_capable_rel'] == 1


def test_ragged_fit_q_and_crossfit_label_isolation():
    eps = [episode(), episode()]
    eps[1]['rho_true'] = np.array([[.7], [.7]])
    points = [np.array([[.8]]), np.array([[.9], [.8]])]
    scales = [np.ones_like(p) for p in points]
    assert E.fit_q(points, scales, eps, .5) == pytest.approx(.2)
    folds = np.array([0, 1, 0, 1])
    labels = np.array([1., 2., 3., 4.])
    a = E.crossfit(None, folds, lambda idx: labels[idx].sum())
    labels[folds == 0] = 999
    b = E.crossfit(None, folds, lambda idx: labels[idx].sum())
    np.testing.assert_equal(a[folds == 0], b[folds == 0])
    assert E.fit_q([], [], [], .1) == np.inf


def test_crc_curve_dominates_raw_and_is_monotone():
    ep = episode(); ep['rho_true'][:] = .2
    curve = E.episode_loss_curve(ep, np.array([[1.]]), np.array([[.1]]), [0, 1, 4])
    np.testing.assert_equal(curve, [1, 1, 0])


def test_gaussian_probe_likelihood_includes_particle_sd_normalizer():
    post = D.DecisionPosterior(np.ones(2), np.array([[[1.]], [[0.]]]), np.arange(2),
        {'x': D.Probe(np.zeros((2, 1)), np.array([[1.], [10.]]), 1)})
    # At the common mean, density is proportional to 1/sd.
    np.testing.assert_allclose(post.reweight('x', [0]), [10 / 11, 1 / 11])


def test_policy_information_scores_identical_signal_zero():
    ep = episode()
    post = D.DecisionPosterior(np.ones(2), np.array([[[1.]], [[0.]]]), np.arange(2),
        {'x': D.Probe(np.zeros((2, 1)), np.ones((2, 1)), 1)})
    for policy in ('eig', 'epig'):
        scores, _ = P.scores(policy, ep['inst'], ep['tasks'], post, P.Certifier(), np.random.default_rng(0), S=4)
        assert scores['x'] == pytest.approx(0, abs=1e-12)


def test_all_missing_posterior_refuses_even_negative_q():
    post = D.DecisionPosterior(np.ones(2), np.full((2, 1, 1), np.nan), np.arange(2))
    assert np.isneginf(P.Certifier(-10).lower(post)).all()


def test_adaptive_end_to_end_crossfit_excludes_test_labels():
    # Test-fold labels must not affect its certificate through the OTHER fold's
    # in-loop q and consequent adaptive probe stopping decision.
    def bank(test_truth):
        eps = []
        for i in range(40):
            ep = episode()
            ep.update(fold=i % 2, rho_nom=np.ones((1, 1)),
                meta=dict(seed=i, parent_size=1, n_faults=0))
            ep['rho_true'] = np.array([[test_truth if i % 2 == 0 else .9]])
            ep['post'] = D.DecisionPosterior(np.ones(2), np.array([[[1.]], [[0.]]]), np.arange(2),
                {'x': D.Probe(np.array([[0.], [1.]]), np.full((2, 1), .01), 1, np.array([0.]))})
            eps.append(ep)
        return eps
    a = E.run_policies(bank(.9), ['evoi'], alpha=.1, alpha_crc=(), max_probes=1, S=4)
    b = E.run_policies(bank(.1), ['evoi'], alpha=.1, alpha_crc=(), max_probes=1, S=4)
    select = lambda rows: [(r['q'], r['n_assigned']) for r in rows
                          if r['fold'] == 0 and r['predictor'] == 'posterior' and r['cert'] == 'sim0.1']
    assert select(a) == select(b), (select(a), select(b))


def test_absolute_margin_keeps_infinite_present_error():
    eps = [episode() for _ in range(10)]
    points = [e['rho_true'].copy() for e in eps]
    points[-1][:] = np.inf
    assert E.margin_abs(points, eps, .1) == np.inf


def test_scale_floor_fit_splits_its_calibration_data():
    eps = [episode() for _ in range(40)]
    points = [np.array([[.8]]) for _ in eps]
    scales = [np.array([[.02]]) for _ in eps]
    ca = np.arange(20)
    q, floor = E.fit_floor(points, scales, eps, ca, .1)
    assert floor == pytest.approx(.1) and q == pytest.approx(1)
    # Held-out episode labels affect neither calibration half.
    eps[39]['rho_true'][:] = -100
    assert E.fit_floor(points, scales, eps, ca, .1) == (q, floor)
    # q-half labels must not be used to learn sigma_min.
    for i in ca[1::2]:
        eps[i]['rho_true'][:] = 0
    assert E.fit_floor(points, scales, eps, ca, .1)[1] == floor
