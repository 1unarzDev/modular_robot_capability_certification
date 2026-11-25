"""Unit tests of the confirmatory scorer (scripts/analysis/confirmatory.py) on mock data only. No conf_* output is read."""
import datetime as dt
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts/analysis"))
import confirmatory as K  # noqa: E402


@pytest.fixture(autouse=True)
def small_B(monkeypatch):
    monkeypatch.setattr(K, "B", 300)


def _run(tmp_path, **kw):
    res = K.write_mock(tmp_path / "in", n_main=80, n_gate=60, n_pairs=60, **kw)
    V, H, S = K.run(res, tmp_path / "out", mock=True)
    return V, H, S, tmp_path / "out"


def _v(V, claim, bar, metric_sub="", method=None, plant=None):
    q = V[(V.claim == claim) & (V.bar == bar) & V.metric.str.contains(metric_sub, regex=False)]
    if method is not None:
        q = q[q.method == method]
    if plant is not None:
        q = q[q.plant == plant]
    assert len(q), (claim, bar, metric_sub)
    return q


def test_refuses_real_inputs_before_the_window(tmp_path):
    (tmp_path / "models/confirmatory").mkdir(parents=True)
    with pytest.raises(SystemExit):
        K.run(tmp_path, tmp_path / "out", now=dt.datetime(2026, 11, 26, 1, 44, tzinfo=dt.timezone.utc))
    # after the window opens, an empty real dir scores to MISSING rows without crashing
    V, H, _ = K.run(tmp_path, tmp_path / "out", now=dt.datetime(2026, 11, 26, 1, 46, tzinfo=dt.timezone.utc))
    assert (V.verdict.isin(["MISSING", "NOT SCORED", "NOT SCORED HERE", "SEE TEST"])).all()
    assert not H.holm_reject.any()


def test_exclusions_shard0_and_pretouched():
    d = pd.DataFrame(dict(split=["conf_test"] * 5 + ["cal"], eid=[0, 39, 40, 60, 3045, 0], shard=[0, 0, 0, 1, 2, 0]))
    log = []
    out = K.exclusions(d, log, "x")
    assert sorted(out.eid.tolist()) == [0, 60, 3045]           # cal row 0 kept; shard-0 40 dropped; 0-39 dropped
    assert (out[out.eid == 0].split == "cal").all() and "canary" in log[0]


def test_holm_step_down():
    S = K.Scorer()
    S.p = dict(P1=0.001, P2=0.009, P3=0.0105, P4=0.2, P5=np.nan, P6=0.004, P7=0.03)
    H = K.holm(S).set_index("P")
    # sorted: P1 .001 <= .05/7; P6 .004 <= .05/6; P2 .009 <= .05/5=.01; P3 .0105 > .05/4=.0125? no: .0105 <= .0125
    assert H.loc["P1"].holm_reject and H.loc["P6"].holm_reject and H.loc["P2"].holm_reject and H.loc["P3"].holm_reject
    # P7 .03 vs .05/3 = .0167 -> stop; P4 after it fails too; missing P5 never rejects
    assert not H.loc["P7"].holm_reject and not H.loc["P4"].holm_reject and not H.loc["P5"].holm_reject


def test_paired_bootstrap_uses_same_clusters():
    bt = K.Boot(np.arange(50), B=500)
    a = pd.DataFrame(dict(x=np.ones(50), n=np.ones(50)), index=np.arange(50))
    b = pd.DataFrame(dict(x=np.r_[np.ones(25), np.zeros(25)], n=np.ones(50)), index=np.arange(50))
    ea, ra = bt.ratio(a, "x", "n")
    eb, rb = bt.ratio(b, "x", "n")
    assert ea - eb == 0.5 and np.all(ra - rb >= 0) and K.p_greater(ra - rb) < 0.01


def test_power_rule():
    m = 3000
    inc = np.full(m, 9.0)
    rng = np.random.default_rng(0)
    uns = rng.binomial(9, 0.01, m).astype(float)
    pw = K.power(pd.DataFrame(dict(inc=inc, uns=uns)))
    assert pw["n_incapable"] == 27000 and pw["power_ok"] and pw["deff"] < 2
    small = K.power(pd.DataFrame(dict(inc=inc[:1000], uns=uns[:1000])))
    assert not small["power_ok"]                              # 9,000 < 20,000
    clumped = np.zeros(m)
    clumped[:30] = 9.0                                        # all unsafe in a few episodes -> big design effect
    assert not K.power(pd.DataFrame(dict(inc=inc, uns=clumped)))["power_ok"]


def test_one_sided_upper_and_validity_semantics():
    S = K.Scorer()
    reps = np.linspace(0.05, 0.14, 1000)
    assert S.bar("x", "b", "E_ep", 0.09, reps, le=0.10, up_le=0.12) == "FAIL"    # upper95 ~ 0.1355 > 0.12
    assert S.bar("x", "b", "E_ep", 0.09, reps, le=0.10, up_le=0.14) == "PASS"
    pw = dict(n_incapable=100, deff=1.0, power_ok=False)
    assert S.bar("x", "b", "E_dec", 0.0, reps * 0, le=0.01, up_le=0.013, power_row=pw) == "UNDERPOWERED"


def test_good_mock_end_to_end(tmp_path):
    V, H, S, out = _run(tmp_path)
    assert (out / "verdicts.csv").exists() and (out / "cells.csv").exists()
    md = (out / "CONFIRMATORY_RESULTS.md").read_text()
    assert md.index("Power checks") < md.index("## Verdicts") < md.index("Holm table")
    assert V.conf_sha.nunique() == 1
    assert _v(V, "C-MAIN", "1 validity", "E_ep", plant="T1").verdict.iloc[0] == "PASS"
    assert _v(V, "C-MAIN", "P1").verdict.iloc[0] == "PASS"
    assert _v(V, "C-MAIN", "P2").verdict.iloc[0] == "PASS"
    assert (_v(V, "C-MAIN", "P3").verdict == "PASS").all()
    assert (_v(V, "C-GATE", "P5").verdict == "PASS").all()
    assert _v(V, "C-GATE", "P6").verdict.iloc[0] == "PASS"
    assert _v(V, "C-MERGE", "1 delta_comp", plant="T1").verdict.iloc[0] == "PASS"
    assert _v(V, "C-ENV", "P7").verdict.iloc[0] == "PASS"
    assert _v(V, "C-SCALE", "2 comp vs direct").verdict.iloc[0] == "PASS"
    assert H.holm_reject.sum() == 7
    # C-MAIN scored only conf_test 40-3039 minus shard 0 (eids 0-49 in the mock)
    cells = pd.read_csv(out / "cells.csv")
    n = cells[(cells.claim == "C-MAIN") & (cells.method == K.OURS) & (cells.block == "all") & (cells.plant == "T1")]
    assert n.n_ep.iloc[0] == 80 + 40 - 50


def test_bad_mock_fails_validity(tmp_path):
    V, _, _, _ = _run(tmp_path, good=False)
    assert _v(V, "C-MAIN", "1 validity", "E_ep", plant="T1").verdict.iloc[0] == "FAIL"


def test_gate_kill_when_knee_does_not_bite(tmp_path):
    V, H, _, _ = _run(tmp_path, knee_bites=False)
    assert (V[(V.claim == "C-GATE") & (V.bar == "kill")].verdict == "KILLED").all()
    assert (_v(V, "C-GATE", "P5").verdict == "FAIL").all()


def test_env_not_scored_without_prerequisite(tmp_path):
    V, H, _, _ = _run(tmp_path, ks_ok=False)
    env = V[V.claim == "C-ENV"]
    assert env[env.bar == "prerequisite"].verdict.iloc[0] == "FAIL"
    assert (env[env.bar != "prerequisite"].verdict == "NOT SCORED").all()
    assert not H.set_index("P").loc["P7"].holm_reject


def test_fixed_reference_label_is_scored_not_own(tmp_path):
    res = K.write_mock(tmp_path / "in", n_main=60, n_gate=40, n_pairs=40)
    f = res / "models/confirmatory/dec_conf_models.csv.gz"
    d = pd.read_csv(f)
    d["truth_own"] = d["truth"] + 0.5                         # own-reference labels must not change verdicts
    d.to_csv(f, index=False)
    V1, _, _ = K.run(res, tmp_path / "o1", mock=True)
    d["truth_own"] = d["truth"] - 0.5
    d.to_csv(f, index=False)
    V2, _, _ = K.run(res, tmp_path / "o2", mock=True)
    a = V1[V1.claim == "C-MAIN"].estimate.to_numpy()
    b = V2[V2.claim == "C-MAIN"].estimate.to_numpy()
    np.testing.assert_allclose(a, b)
