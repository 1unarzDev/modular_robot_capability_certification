"""Exploratory E1 audit. Run: scripts/heavy.sh $PY -m pytest -s tests/codex/test_gate_audit.py"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import theory_gate as gate

ROOT = Path(__file__).resolve().parents[2]
CERT = ('par', 'pi', 'gate', 'raw')

@pytest.fixture(scope='module')
def data():
    files = sorted(glob.glob(str(gate.OUT / 'gate_*.pkl')))
    if not files:
        pytest.skip('E1 episode pickle files unavailable')
    d = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    return d.drop_duplicates(['seed', 'cand', 'leg']).reset_index(drop=True)

@pytest.fixture(scope='module')
def bounds(data):
    result = {}
    for probe in gate.PROBES:
        result[probe] = {c: np.empty(len(data)) for c in CERT}
        for fold in (0, 1):
            test = (data.seed // 4) % 2 == fold
            cal = data.loc[~test]
            assert set(cal.seed).isdisjoint(data.loc[test, 'seed'])
            assert set(cal.loc[(cal.seed // 8) % 2 == 0, 'seed']).isdisjoint(
                cal.loc[(cal.seed // 8) % 2 == 1, 'seed'])
            lo, _ = gate.lower_bounds(cal, data.loc[test], probe)
            for cert in CERT:
                result[probe][cert][test] = lo[cert]
    return result

def test_crossfit_inputs(data, monkeypatch):
    # Trace every residual quantile: calibration episodes only; gate's final
    # quantile is on B, while candidate thresholds are scored on A.
    original = gate.q_sim
    for probe in gate.PROBES:
        for fold in (0, 1):
            mask = (data.seed // 4) % 2 == fold
            cal, test = data.loc[~mask], data.loc[mask]
            calls = []
            def traced(frame, col, ep_mask=None):
                used = frame if ep_mask is None else frame[ep_mask]
                calls.append((col, set(used.seed), set(used.law)))
                return original(frame, col, ep_mask)
            with monkeypatch.context() as m:
                m.setattr(gate, 'q_sim', traced)
                _, info = gate.lower_bounds(cal, test, probe)
            cal_ids, test_ids = set(cal.seed), set(test.seed)
            assert all(ids <= cal_ids and ids.isdisjoint(test_ids) for _, ids, _ in calls)
            assert calls[0][2] == {0}
            a = set(cal.loc[(cal.seed // 8) % 2 == 0, 'seed'])
            b = set(cal.loc[(cal.seed // 8) % 2 == 1, 'seed'])
            gcalls = [ids for col, ids, _ in calls if col == 'g_l0']
            assert all(ids == a for ids in gcalls[:-1]) and gcalls[-1] == b
            assert np.isfinite([info['q_par'], info['q_gate_margin']]).all()

def test_recompute_and_episode_bootstrap(data, bounds):
    saved = pd.read_csv(ROOT / 'results/theory/gate_cert.csv')
    rng = np.random.default_rng(907)
    print('\nExploratory decision stratum: CF [95% CI], unsafe [95% CI]; bootstrap resamples episodes, fixed cross-fit bounds')
    for probe in gate.PROBES:
        for sev in range(4):
            d = data.loc[data.law == sev]
            seeds, inv = np.unique(d.seed, return_inverse=True)
            draws = rng.integers(0, len(seeds), size=(2000, len(seeds)))
            weights = np.stack([np.bincount(x, minlength=len(seeds)) for x in draws])
            for cert in CERT:
                lo = bounds[probe][cert][d.index]
                counts = np.zeros((len(seeds), 4))
                for t in gate.TAUS:
                    m = (d.has_fault.to_numpy() == 1) | (np.abs(d.rho_true.to_numpy() - t) < gate.REL_BAND)
                    cap = d.rho_true.to_numpy() >= t
                    go = lo >= t
                    for j, v in enumerate((m & cap & go, m & cap, m & ~cap & go, m & ~cap)):
                        counts[:, j] += np.bincount(inv, weights=v, minlength=len(seeds))
                total = counts.sum(axis=0)
                assert total[1] and total[3]
                point = total[[0, 2]] / total[[1, 3]]
                boot = weights @ counts
                ratios = np.divide(boot[:, [0, 2]], boot[:, [1, 3]], out=np.full((len(boot), 2), np.nan), where=boot[:, [1, 3]] != 0)
                ci = np.nanpercentile(ratios, [2.5, 97.5], axis=0)
                row = saved[(saved.probe == probe) & (saved.severity == sev) & (saved.certifier == cert) & (saved.stratum == 'decision')].iloc[0]
                np.testing.assert_allclose(point, [row.cert_frac_capable, row.unsafe_rate], atol=1e-12)
                print(f'{probe:7s} sev={sev} {cert:4s} CF {point[0]:.4f} [{ci[0,0]:.4f}, {ci[1,0]:.4f}] unsafe {point[1]:.4f} [{ci[0,1]:.4f}, {ci[1,1]:.4f}]')

def test_preregistered_bars_and_verdict(tmp_path, monkeypatch):
    r = pd.read_csv(ROOT / 'results/theory/gate_cert.csv')
    q = pd.read_csv(ROOT / 'results/theory/gate_thresholds.csv')
    def get(st, p, c, s, col):
        x = r[(r.stratum == st) & (r.probe == p) & (r.certifier == c) & (r.severity == s)]
        assert len(x) == 1
        return x.iloc[0][col]
    assert (r[(r.stratum == 'all') & (r.certifier == 'par')].n_episodes >= 100).all()
    kill = all(get('undominated', p, 'par', s, 'unsafe_rate') <= .01 for p in gate.PROBES for s in (1, 2, 3))
    s1 = get('decision', 'v2', 'gate', 0, 'cert_frac_capable') >= get('decision', 'v2', 'par', 0, 'cert_frac_capable') - .02
    bad = [s for s in (1, 2, 3) if get('decision', 'v2', 'par', s, 'unsafe_rate') > .02]
    worst = max(bad) if bad else None
    s2 = worst is not None and get('decision', 'v2', 'gate', worst, 'unsafe_rate') <= .01 and get('decision', 'v2', 'gate', worst, 'cert_frac_capable') >= get('decision', 'v2', 'pi', worst, 'cert_frac_capable') + .05
    assert (kill, s1, worst, s2) == (False, False, 3, False)
    monkeypatch.setattr(gate, 'RES', tmp_path)
    gate.verdict(r, q)
    assert (tmp_path / 'gate_verdict.txt').read_text() == (ROOT / 'results/theory/gate_verdict.txt').read_text()
