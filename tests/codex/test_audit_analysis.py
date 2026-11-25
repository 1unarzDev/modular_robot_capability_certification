"""Exploratory audit of T1--T4. All table output is intercepted; no corpus or output files are touched.

Semantics: unsafe / incapable; useful certified / capable; episode-cluster bootstrap;
faulted OR near-threshold stratum; T1 matched margin fit on cal only and frozen for T2;
T2 lower coverage; T3 robustness AND latency verdict; T4 ratios undefined with no denominator.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts/analysis'))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts'))
import common as C
import t1_certificate as T1
import t2_heldout as T2
import t3_grid as T3
import t4_probing as T4
import nuisance_pilot as N


def frame(truth=(.2, .9), point=None, lower=None):
    n = len(truth)
    return pd.DataFrame(dict(truth=truth, point=np.full(n, .8) if point is None else point,
        scale=np.ones(n), lower=np.full(n, .7) if lower is None else lower, eid=np.arange(n),
        has_fault=np.ones(n), method='audit', route='simultaneous', posterior='smc', train='T1',
        test_plant='T1', split='test', status='exploratory', shape='pair', kind='child', cand=0,
        cand_size=1, leg=0, seconds=1.))


@pytest.fixture
def capture(monkeypatch):
    saved = {}
    monkeypatch.setattr(pathlib.Path, 'mkdir', lambda *a, **k: None)
    monkeypatch.setattr(pd.DataFrame, 'to_csv', lambda self, path, **kw: saved.update({str(path): self.copy()}))
    monkeypatch.setattr(C, 'write_table', lambda *a, **kw: None)
    return saved


def test_empty_and_all_capable_all_incapable_metrics():
    empty = C.decision_counts(frame(()), taus=(.7,))
    assert empty['n_dec'] == 0 and np.isnan(empty['cert_frac']) and np.isnan(empty['unsafe_commit'])
    good = C.decision_counts(frame((.7, .9)), taus=(.7,))
    bad = C.decision_counts(frame((.2, .3)), taus=(.7,))
    assert good['cert_frac'] == 1 and np.isnan(good['unsafe_commit'])
    assert bad['unsafe_commit'] == 1 and np.isnan(bad['cert_frac'])


def test_zero_row_stratum_keeps_unconditional_episode_denominator():
    d = frame((.69, 1.2)); d['has_fault'] = 0
    c = C.decision_counts(d, taus=(.7,), stratum='rel')
    # Decision risk is marginal over ALL generator episodes; coverage explicitly
    # uses a different convention (drops episodes without relevant entries).
    assert c['n_ep'] == 2 and c['ep_any_unsafe'] == .5, c
    d['truth'] = 1.2
    c = C.decision_counts(d, taus=(.7,), stratum='rel')
    assert c['n_dec'] == 0 and np.isnan(c['cert_frac']) and c['ep_any_unsafe'] == 0
    assert C.coverage(d, stratum='rel')['n_ep'] == 0


def test_coverage_refusal_and_simultaneity():
    d = frame((.7, .7, .7), lower=[.7, np.nan, .8]); d['eid'] = [0, 0, 1]
    c = C.coverage(d)
    assert c['entry_cov'] == pytest.approx(2 / 3) and c['ep_cov'] == .5
    assert not C.go_mask([np.inf, np.nan, -np.inf], .7).any()


def test_matched_margin_ties_and_small_n():
    for z in (np.zeros(100), np.arange(99.), np.arange(100.), np.array([1., 1., 0., 0.])):
        k = C.matched_margin(z)
        assert (z >= k).mean() <= C.U_STAR
    assert C.matched_margin(np.arange(100.)) < 99


def test_matched_all_capable_does_not_refuse_everything():
    d = frame((.9, .9))
    lower, k = C.matched_lower(d, d, taus=(.7,))
    assert C.go_mask(lower, .7).all(), (lower, k)


def test_matched_no_fit_data_refuses():
    lower, _ = C.matched_lower(frame(()), frame(), taus=(.7,))
    assert np.isneginf(lower).all()


def test_test_labels_cannot_change_frozen_matched_lower():
    fit = frame((.2,) * 100, point=np.linspace(.5, 1., 100))
    ev = frame(); lower, k = C.matched_lower(fit, ev, stratum='rel')
    ev['truth'] = [100, -100]; ev['has_fault'] = 0
    changed, kk = C.matched_lower(fit, ev, stratum='rel')
    assert kk == k
    np.testing.assert_equal(changed, lower)


def test_trivial_always_go_really_goes():
    d = C.trivial_methods(frame())
    c = C.decision_counts(d[d.method == 'always_go'], taus=(.7,))
    assert c['n_certified'] == 1 and c['n_unsafe'] == 1, c


def test_t1_cal_only_margin_is_frozen_across_plants(monkeypatch, capture):
    fit = frame((.2,) * 100, point=np.linspace(.5, 1., 100)); fit['split'] = 'cal'
    te = frame(); stress = te.copy(); stress['test_plant'] = 'T2'; stress['truth'] = [.1, .1]
    monkeypatch.setattr(C, 'load', lambda *a: (pd.concat([fit, te, stress], ignore_index=True), ['synthetic']))
    monkeypatch.setattr(C, 'trivial_methods', lambda d: d.iloc[:0])
    T1.run(pathlib.Path('/unused'), pathlib.Path('/unused'))
    r = capture['/unused/tables/t1_certificate.csv']
    assert not r.dagger.any() and r.groupby('stratum').k.nunique().eq(1).all()
    expected = C.matched_lower(fit, te)[1]
    assert r.k.eq(expected).all()


def test_t2_heldout_coverage_excludes_cal(monkeypatch, capture):
    d = frame((.7, .7), lower=[.7, .8]); d['split'] = 'H3'
    cal = frame((.7,), lower=[100]); cal['split'] = 'cal'
    monkeypatch.setattr(C, 'load', lambda *a: (pd.concat([d, cal]), ['synthetic']))
    T2.run(pathlib.Path('/unused'), pathlib.Path('/unused'))
    r = capture['/unused/tables/t2_heldout.csv']
    assert set(r.split) == {'H3'} and r.ep_cov.eq(.5).all()


def test_t3_does_not_declare_smc_win_when_amortized_is_more_useful(monkeypatch, capture):
    smc = frame(); smc['test_plant'] = 'T2'; smc['lower'] = [.1, .1]
    am = smc.copy(); am['posterior'] = 'amortized'; am['method'] = 'amortized_sim'
    am['lower'] = [.7, .7]; am['seconds'] = 2
    monkeypatch.setattr(C, 'load', lambda *a: (pd.concat([smc, am]), ['synthetic']))
    n = T3.run(pathlib.Path('/unused'), pathlib.Path('/unused'))
    verdict = n.items[C.macro_name('t3.verdict')][0]
    assert "SMC wins on both" not in verdict, verdict


def test_t4_zero_denominator_is_undefined():
    d = pd.DataFrame(dict(n_certified_rel=[0], n_capable_rel=[0]))
    rate, _ = T4._ratio_ci(d, 'n_certified_rel', 'n_capable_rel')
    assert np.isnan(rate), rate


def test_t4_policy_ratios_and_ambiguity_hand_example(monkeypatch, capture):
    d = pd.DataFrame(dict(policy=['fixed_v2', 'evoi'], split='test', test_plant='T1', status='exploratory',
        eid=0, probe_time_s=[12, 72], n_amb_before=10, n_amb_after=[10, 4], n_certified_rel=[1, 2],
        n_capable_rel=2, n_unsafe_rel=0, n_incapable_rel=2, n_certified=[1, 2], n_capable=2,
        n_unsafe=0, n_incapable=2, utility=1, action='commit'))
    monkeypatch.setattr(C, 'load', lambda *a: (d, ['synthetic']))
    T4.run(pathlib.Path('/unused'), pathlib.Path('/unused'))
    r = capture['/unused/tables/t4_probing.csv'].set_index('policy').loc['evoi']
    assert r.cert == 1 and r.unsafe == 0 and r.resolved == .6 and r.per_min == .6


def test_nuisance_ambiguity_ties_use_decision_label(capture, monkeypatch):
    # q05 == tau: already capable; q95 == tau with q05 < tau: ambiguous.
    d = pd.DataFrame(dict(probe='v2', delta=.08, has_fault=1, ekey=['a', 'b', 'c'],
        q05=[.7, .6, .6], q95=[.9, .7, .69], truth=[.1, .99, .7], q05_rel=0, q95_rel=1))
    monkeypatch.setattr(N, 'TAUS', (.7,))
    N.amb_het(d)
    r = next(iter(capture.values()))
    assert r.amb_share.eq(1 / 3).all() and r.n_dec.eq(3).all()
