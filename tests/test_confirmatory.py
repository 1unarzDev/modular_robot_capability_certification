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
    S.p = {"P1": 0.001, "P2": 0.009, "P3": 0.0105, "P4": 0.2, "P5": np.nan, "P6'": 0.004, "P7": 0.03}
    H = K.holm(S).set_index("P")
    # sorted: P1 .001 <= .05/7; P6 .004 <= .05/6; P2 .009 <= .05/5=.01; P3 .0105 > .05/4=.0125? no: .0105 <= .0125
    assert H.loc["P1"].holm_reject and H.loc["P6'"].holm_reject and H.loc["P2"].holm_reject and H.loc["P3"].holm_reject
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
    assert _v(V, "C-GATE", "P6' overall").verdict.iloc[0] == "PASS"
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


# ---- CONFIRMATORY.md §9 amendments (e176756)

def test_amendment1_ep_bars_use_all_block(tmp_path):
    V, _, _, out = _run(tmp_path, good=False)
    cells = pd.read_csv(out / "cells.csv")
    for plant in ("T1", "T2"):
        bar = _v(V, "C-MAIN", "1 validity", "E_ep", plant=plant)
        assert "§9.1" in bar.note.iloc[0]
        allb = cells[(cells.claim == "C-MAIN") & (cells.method == K.OURS) & (cells.block == "all")
                     & (cells.plant == plant)].E_ep.iloc[0]
        assert bar.estimate.iloc[0] == pytest.approx(allb)
    m = _v(V, "C-MERGE", "2 validity", "E_ep per pair", plant="T1")
    assert "§9.1" in m.note.iloc[0]


def test_amendment2_ge_bars_point_judged():
    S = K.Scorer()
    reps = np.linspace(0.80, 0.90, 1000)                     # point above, one-sided lower below the threshold
    assert S.bar("C-SCALE", "1 validity", "stratum coverage", 0.86, reps, ge=0.85) == "PASS"
    assert S.rows[-1]["lower95"] < 0.85                       # the bound is still printed
    assert S.bar("C-HET", "2 degradation", "drop", 0.04, np.linspace(0.0, 0.09, 1000), le=0.05) == "PASS"


def test_amendment2_scorer_rows(tmp_path):
    V, _, _, _ = _run(tmp_path)
    for claim, bar, met in (("C-SCALE", "1 validity", "stratum coverage"), ("C-MERGE", "P4", "CF@U* comp_enum_size"),
                            ("C-HET", "2 degradation", "drop ID")):
        q = _v(V, claim, bar, met)
        assert (q.threshold.isin([">= 0.85", "<= 0.05"])).all() and q.note.str.contains("9.2").all()


def test_amendment3_dominated_contrast_when_underpowered(tmp_path):
    V, _, _, _ = _run(tmp_path)                               # small mock: dominated cells are underpowered
    dom = V[(V.claim == "C-GATE") & (V.bar == "1 dominated")]
    assert len(dom) == 2 and dom.metric.str.contains("upper95 <= 2 %").all()
    assert dom.note.str.contains("UNDERPOWERED, contrast-judged").all()
    assert (dom.threshold == "upper95 <= 0.02").all()
    # mock: raw's dominated E_dec ~2.4 % fails the contrast, par's 0 % passes; H12 overall then fails
    assert dom.set_index("method").loc["par"].verdict == "PASS"
    h = V[(V.claim == "C-GATE") & (V.bar == "1 H12 overall")]
    assert len(h) == 1 and h.verdict.iloc[0] == ("PASS" if (dom.verdict == "PASS").all() else "FAIL")


def test_amendment3_powered_keeps_original_bar(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "POWER_N", 1)
    monkeypatch.setattr(K, "POWER_UP", 1.0)                   # every cell powered
    V, _, _, _ = _run(tmp_path)
    dom = V[(V.claim == "C-GATE") & (V.bar == "1 dominated")]
    assert dom.metric.str.contains("<= 1 %").all() and (dom.threshold == "<= 0.01; upper95 <= 1.0").all()


# ---- cfBH encoding (baselines): lower = largest selected tau, -inf if never; commit-at-tau <=> lower >= tau

def _cfbh_frame():
    # one episode, three entries: selected up to 0.8 (truth 0.75: unsafe at 0.8 only), up to 0.6 (truth 0.9),
    # never selected (truth 0.5)
    return pd.DataFrame(dict(eid=[1, 1, 1], truth=[0.75, 0.9, 0.5], lower=[0.8, 0.6, -np.inf], has_fault=[1, 1, 1],
                             kind="child", cand_size=1, point=np.nan, scale=np.nan, method="cfBH"))


def test_cfbh_commit_rule_in_confirmatory():
    t = K.cluster_stats(_cfbh_frame(), _cfbh_frame()["lower"])
    # commits: entry0 at 0.6, 0.7, 0.8 (0.8 unsafe); entry1 at 0.6; entry2 never
    assert t.cert.sum() == 3 and t.uns.sum() == 1 and t.ep_unsafe.iloc[0] == 1
    assert t.cap.sum() == 2 + 3 + 0 and t.inc.sum() == 1 + 0 + 3


def test_cfbh_commit_rule_in_t1_scorer():
    import common as C
    c = C.decision_counts(_cfbh_frame(), taus=(0.6, 0.7, 0.8))
    assert (c["n_certified"], c["n_unsafe"]) == (3, 1)


def test_cfbh_has_no_matched_point_and_is_scored_on_its_rule(tmp_path):
    V, _, _, out = _run(tmp_path)
    cells = pd.read_csv(out / "cells.csv")
    cf = cells[(cells.claim == "C-MAIN") & (cells.method == "cfBH")]
    assert len(cf) and cf.CF_matched.isna().all() and cf.CF_own.notna().all()
    fdr = cf[(cf.block == "all") & (cf.plant == "T1")].FDR.iloc[0]
    assert 0 <= fdr <= 1



# ---- A-GATE / A-HET (727f995)

def test_agate_bars(tmp_path):
    V, H, _, _ = _run(tmp_path)
    g = V[V.claim == "C-GATE"]
    assert not g.bar.str.startswith("H13").any() and not (g.bar == "P6").any()      # withdrawn
    assert "P6'" in set(H.P) and "P6" not in set(H.P) and len(H) == 7
    p6 = g[g.bar == "P6'"]
    assert p6.metric.str.contains("par - robust").all() and p6.verdict.iloc[0] == "PASS"
    assert len(g[g.bar == "P6' robust undominated"]) >= 1
    assert sum(g.bar.str.startswith("robust validity s=")) >= 4                         # every severity
    assert len(g[g.bar == "price of robustness"]) == 1 and len(g[g.bar == "lawcal held-out s=3"]) == 1


def test_agate_p6_fails_when_robust_unsafe(tmp_path):
    res = K.write_mock(tmp_path / "in", n_main=60, n_gate=60, n_pairs=40)
    f = res / "stress/confirmatory/dec_conf_stress.csv.gz"
    d = pd.read_csv(f, low_memory=False)
    m = d.method == "robust"
    d.loc[m, "lower"] = d.loc[m, "truth"] + 0.2                                       # robust commits everything
    d.to_csv(f, index=False)
    V, _, _ = K.run(res, tmp_path / "out", mock=True)
    assert _v(V, "C-GATE", "P6' overall").verdict.iloc[0] == "FAIL"


def test_ahet_h1_judged_on_faulted_block(tmp_path):
    res = K.write_mock(tmp_path / "in", n_main=60, n_gate=40, n_pairs=40)
    f = res / "models/confirmatory/dec_conf_models.csv.gz"
    d = pd.read_csv(f, low_memory=False)
    m = (d.split == "conf_H1") & (d.method == K.OURS) & (d.has_fault == 0)
    d.loc[m, "lower"] = d.loc[m, "truth"] + 0.3                                       # healthy H1 candidates unsafe
    d.to_csv(f, index=False)
    V, _, _ = K.run(res, tmp_path / "out", mock=True)
    h1 = V[(V.claim == "C-HET") & (V.bar == "1 validity") & (V.method == K.OURS) & (V.cell == "conf_H1")]
    judged = h1[h1.metric == "E_ep (faulted)"]
    desc = h1[h1.metric == "E_ep (all)"]
    assert judged.verdict.iloc[0] == "PASS" and "faulted-candidate block" in judged.note.iloc[0]
    assert desc.verdict.iloc[0] == "DESCRIPTIVE" and desc.estimate.iloc[0] > 0.5
    h2 = V[(V.claim == "C-HET") & (V.bar == "1 validity") & (V.method == K.OURS) & (V.cell == "conf_H2")]
    assert (h2.metric == "E_ep (all)").all() and h2.verdict.iloc[0] == "PASS"
    assert _v(V, "C-HET", "2 degradation", "-> conf_H1 (faulted)").shape[0] == 2
