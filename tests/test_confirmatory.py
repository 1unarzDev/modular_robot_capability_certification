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
    cf = cells[(cells.claim == "C-MAIN") & (cells.method == "cfBH") & (cells.block != "merge")]   # mock C-MAIN: no merges
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


# ---- C-ENV with the env role's real output schema (scripts/env_score.py, results/env/buildings_ks.csv)

def _env_score_frame(rng, nb=40):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import importlib
    try:
        famname = importlib.import_module("env_score").FAMNAME            # the env role's family names
    except Exception:                                                     # noqa: BLE001 (cfcap.env import failure)
        famname = {"conf-ID": "ID", "conf-E-H1": "E-H1"}
    rows, mission = [], 0
    for fname in ("conf-ID", "conf-E-H1"):
        for b in range(nb):
            for _ in range(3):                                            # MISSIONS_PER_BUILDING = 3
                os_ = int(rng.random() < 0.8)
                for meth, ps, pf in (("oracle", 1.0, 0.0), ("nominal", 0.7, 0.25), ("biggest_body", 0.6, 0.1),
                                     ("greedy_feasible", 0.7, 0.2), (K.OURS, 0.92, 0.02), ("raw", 0.9, 0.12),
                                     (K.ENVCAL, 0.95, 0.04)):
                    com = int(rng.random() < 0.95)
                    fail = int(com and rng.random() < pf)
                    # exactly env_score.main's dict: claim, family, building, mission, shard, status, then score_mission
                    rows.append(dict(claim="C-ENV", family=famname.get(fname, fname), building=f"{fname}-{b}",
                                     mission=mission, shard=-1, status="confirmatory", method=meth,
                                     committed=com, failed=fail,
                                     success=int(com and not fail and os_ and rng.random() < ps),
                                     oracle_success=os_, reconfig=int(rng.random() < 0.4), time_s=30.0 * com))
                mission += 1
    return pd.DataFrame(rows)


def _env_ks(ok=True):
    # the columns of results/env/buildings_ks.csv as committed by the env role
    return pd.DataFrame(dict(stat=["route_m", "route_steps", "tortuosity", "passage_count"],
                             ks_p=[0.0497, 0.0132, 3.3e-8, 1.0], ks_d=[0.038, 0.044, 0.084 if ok else 0.12, 0.0],
                             n_hm3d=2210, n_gen=3000, family="test (= conf-ID rule)", passed=ok))


def _env_only_dir(tmp_path, ok=True):
    res = tmp_path / "in"
    (res / "env/confirmatory").mkdir(parents=True)
    (res / "MOCK.txt").write_text("mock\n")
    _env_score_frame(np.random.default_rng(1)).to_csv(res / "env/confirmatory/missions_conf_env.csv.gz", index=False)
    _env_ks(ok).to_csv(res / "env/buildings_ks.csv", index=False)
    return res


def test_env_score_schema_is_read(tmp_path):
    V, H, S = K.run(_env_only_dir(tmp_path), tmp_path / "out", mock=True)
    env = V[V.claim == "C-ENV"]
    assert env[env.bar == "prerequisite"].verdict.iloc[0] == "PASS"
    val = env[(env.bar == "1 validity") & (env.method == K.OURS)].set_index("cell")
    assert set(val.index) == {"ID", "E-H1"} and val.loc["ID"].verdict == "PASS"
    p7 = env[env.bar == "P7"]
    assert p7.verdict.iloc[0] == "PASS"
    assert any(x in p7.metric.iloc[0] for x in ("nominal", "biggest_body", "greedy_feasible"))
    assert any("shard = -1" in x for x in S.log)
    assert len(env[env.bar == "4 reconfiguration"]) == 1


def test_env_ks_prerequisite_from_env_file(tmp_path):
    V, H, _ = K.run(_env_only_dir(tmp_path, ok=False), tmp_path / "out", mock=True)
    env = V[V.claim == "C-ENV"]
    pre = env[env.bar == "prerequisite"]
    assert pre.verdict.iloc[0] == "FAIL" and "tortuosity" in pre.note.iloc[0]
    assert (env[env.bar != "prerequisite"].verdict == "NOT SCORED").all()


def test_real_buildings_ks_file_parses():
    f = pathlib.Path(__file__).resolve().parents[1] / "results/env/buildings_ks.csv"
    if not f.exists():
        pytest.skip("no env KS file")
    ks = pd.read_csv(f)
    assert all(c in ks for c in K.KS_REQ)



# ---- A-ENV (0dc1aec)

def test_aenv_exclusion_labels_and_envcal(tmp_path):
    res = _env_only_dir(tmp_path)
    f = res / "env/confirmatory/missions_conf_env.csv.gz"
    d = pd.read_csv(f)
    bad = d[d.family == "ID"].drop_duplicates("method").assign(mission=1499, building="conf-ID-999", committed=1,
                                                                failed=1, success=0, oracle_success=1)
    pd.concat([d, bad]).to_csv(f, index=False)
    V, _, S = K.run(res, tmp_path / "out", mock=True)
    # mission 1499 is excluded by A-ENV; since A-POWER (ID 0-599) the prefix rule drops it first
    assert any(("mission 1499" in x and "excluded 7 rows" in x) or "A-POWER: C-ENV ID: dropped 7" in x for x in S.log)
    val = V[(V.claim == "C-ENV") & (V.bar == "1 validity")]
    ours = val[(val.method == K.OURS) & (val.cell == "ID")].iloc[0]
    exp = d[(d.family == "ID") & (d.method == K.OURS)]
    assert ours.estimate == pytest.approx(((exp.committed == 1) & (exp.failed == 1)).sum() / exp.committed.sum())
    assert "empirical transfer bar, NOT a guarantee" in ours.note
    env = val[(val.method == K.ENVCAL)]
    assert set(env.cell) == {"ID", "E-H1"} and env.note.str.contains("exchangeability").all()
    assert env[env.cell == "ID"].threshold.iloc[0] == "<= 0.1; upper95 <= 0.13"          # same bars as the primary
    raw = V[(V.claim == "C-ENV") & (V.method == "raw")]
    assert (raw.verdict == "DESCRIPTIVE").all() and raw.note.str.contains("plug-in median").all()
    price = V[(V.claim == "C-ENV") & (V.bar == "price of the T1 constant")]
    assert len(price) == 1 and price.verdict.iloc[0] == "DESCRIPTIVE"


# ---- c2b97cd: task-library tau row (exploratory) and C-EXPLAIN (secondary)

def test_task_library_stats_hand_example(tmp_path):
    lib = tmp_path / "lib.csv"
    pd.DataFrame({"task_id": ["a", "b"], "r_surge+": [0.5, 0.9], "r_yaw+": [0.3, np.nan]}).to_csv(lib, index=False)
    tb = K.task_library_taus(lib)
    assert set(tb) == {0, 4} and list(tb[4]) == [0.3]
    d = pd.DataFrame(dict(eid=[1, 1], leg=[0, 4], truth=[0.95, 0.35], lower=[0.6, 0.2], has_fault=[0, 0]))
    t = K.task_library_stats(d, d["lower"], tb).iloc[0]
    # surge+: tau .5 -> capable, |.95-.5|>.1 -> trivial; tau .9 -> capable, in stratum, go? .6 < .9 no
    # yaw+:  tau .3 -> capable, |.35-.3|<.1 -> stratum, go? .2 < .3 no
    assert t.w == pytest.approx(2.0) and t.triv == pytest.approx(0.5) and t.cap == pytest.approx(1.5)
    assert t.cert == 0 and t.inc == 0


def test_task_library_row_in_main(tmp_path):
    V, _, _, out = _run(tmp_path)
    tl = V[V.claim == "C-MAIN (task-library tau)"]
    assert set(tl.metric) >= {"CF (stratum)", "E_dec (stratum)", "share of trivially capable decisions"}
    assert (tl.verdict == "DESCRIPTIVE").all() and tl.note.str.contains("exploratory|outside", regex=True).all()
    cells = pd.read_csv(out / "cells.csv")
    assert (cells.claim == "C-MAIN (task-library tau)").sum() > 5


def _explain_inputs(tmp_path):
    res = K.write_mock(tmp_path / "in", n_main=60, n_gate=40, n_pairs=40)
    d = pd.read_csv(res / "models/confirmatory/dec_conf_models.csv.gz", low_memory=False)
    d = d[(d.claim == "C-MAIN") & (d.method == K.OURS) & (d.test_plant == "T1") & (d.split == "conf_test")]
    eids = sorted(d.eid.unique())
    # truth: cands 1, 4, 6 hold a fault (mock has_fault); faulted wheel = wheel 4*cand of role 0
    truth = {e: {c: [(0, 4 * c + j, int(c in (1, 4, 6) and j == 0)) for j in range(4)] for c in range(8)} for e in eids}
    # posterior: the true faulted wheel blamed (p .9); plus a wrong blame on cand 2 in a few episodes
    rows = [(e, 0, 4 * c, 0.9) for e in eids for c in (1, 4, 6)] + [(e, 0, 9, 0.7) for e in eids[-5:]]
    rows.sort()
    a = np.array(rows, float)
    # models' exact format (1ffcc69): eid, asm_role, wheel, ep_wheel, p (W x 4: ok, weak, coast, brake), g_mean, classes
    pf = a[:, 3]
    P = np.stack([1 - pf, pf * 0.5, pf * 0.25, pf * 0.25], 1)
    (res / "models/confirmatory").mkdir(parents=True, exist_ok=True)
    np.savez(res / "models/confirmatory/wheels_conf_models.npz", eid=a[:, 0].astype(int), asm_role=a[:, 1].astype(int),
             wheel=a[:, 2].astype(int), ep_wheel=a[:, 2].astype(int), p=P, g_mean=np.ones(len(a)),
             classes=np.array(["ok", "weak", "coast", "brake"]))
    return res, (lambda split, index: truth[index])


def test_explain_scored_with_regenerated_truth(tmp_path):
    res, tf = _explain_inputs(tmp_path)
    V, H, _ = K.run(res, tmp_path / "out", mock=True, truth_fn=tf)
    ex = V[V.claim == "C-EXPLAIN"].set_index("bar")
    assert ex.loc["wheel precision"].estimate < 1.0 and ex.loc["wheel precision"].estimate > 0.9
    assert ex.loc["wheel precision", "threshold"] == ">= 0.95"
    assert ex.loc["false-reason rate (capable refusals)", "threshold"] == "<= 0.05"
    assert ex.loc["margin share"].verdict == "DESCRIPTIVE"
    assert "C-EXPLAIN" not in set(H.P)                                       # not in the Holm family


def test_explain_counts_hand_example():
    d = pd.DataFrame(dict(eid=[7, 7, 7], cand=[0, 0, 1], leg=0, truth=[0.5, 0.95, 0.75], has_fault=[1, 1, 0],
                          point=[0.55, 0.75, 0.72], lower=[0.4, 0.5, 0.65]))
    wheels = pd.DataFrame(dict(eid=[7, 7], role=[0, 0], wheel=[0, 5], p_fault=[0.8, 0.6]))
    tf = lambda s, i: {0: [(0, w, int(w == 0)) for w in range(4)], 1: [(0, w, 0) for w in range(4, 8)]}  # noqa
    t = K.explain_stats(d, wheels, tf).iloc[0]
    # row0 (truth .5, faulted): refused at .6/.7/.8, median .55 < tau -> 3 deficits, all correct; blamed wheel 0 ok
    # row1 (truth .95, faulted -> stratum at every tau): refused at .6/.7 (lower .5), .8: refused; median .75:
    #       deficit at .8 only (.75 < .8), wrong (capable)
    # row2 (healthy, truth .75): stratum at .7/.8 (|.75-tau| < .1); refused .7 (.65<.7) and .8; median .72 < .8 ->
    #       deficit at .8, correct (.75 < .8); blamed wheel 5 is wrong
    assert t.deficit == 5 and t.deficit_ok == 4
    assert t.blamed == 3 * 1 + 3 * 1 + 2 * 1 and t.blamed_ok == 6
    assert t.cap_ref == 3 + 1 and t.cap_wrong_reason == 1


def test_wheels_npz_models_format(tmp_path):
    (tmp_path / "models/confirmatory").mkdir(parents=True)
    P = np.array([[0.2, 0.7, 0.05, 0.05], [0.9, 0.1, 0.0, 0.0]])
    np.savez(tmp_path / "models/confirmatory/wheels_conf_models.npz", eid=np.array([40, 40]),
             asm_role=np.array([0, 1]), wheel=np.array([3, 0]), ep_wheel=np.array([3, 20]), p=P,
             g_mean=np.array([0.5, 1.0]), classes=np.array(["ok", "weak", "coast", "brake"]))
    w, _ = K.load_wheels(tmp_path)
    assert w.role.tolist() == [0, 1] and w.p_fault.round(6).tolist() == [0.8, 0.1]
    np.savez(tmp_path / "models/confirmatory/wheels_conf_models.npz", eid=np.array([40]), wheel=np.array([3]),
             p=P[:1])
    with pytest.raises(K.C.SchemaError):
        K.load_wheels(tmp_path)


# ---- A-SEED (03387d0) and scorer-computed trivial baselines (coordinator 00:35Z)

def _tagged_mock(tmp_path, add_seed1=True):
    res = K.write_mock(tmp_path / "in", n_main=60, n_gate=40, n_pairs=40)
    f = res / "models/confirmatory/dec_conf_models.csv.gz"
    d = pd.read_csv(f, low_memory=False)
    mm = d.method.isin(K.MODEL_METHODS) & d.claim.isin(K.SEED_CLAIMS)
    d["model_tag"] = np.where(mm, d.method.map(K.TAG_ALLOW), None)
    if add_seed1:
        s1 = d[mm & (d.method == K.OURS) & (d.claim == "C-MAIN")].copy()
        s1["model_tag"] = "st-n6-1"
        s1["lower"] = s1["truth"] + 0.3                           # a terrible seed: must never reach a bar
        d = pd.concat([d, s1], ignore_index=True)
    d.to_csv(f, index=False)
    return res


def test_aseed_primary_tag_only_in_bars(tmp_path):
    V0, _, _ = K.run(_tagged_mock(tmp_path / "a", add_seed1=False), tmp_path / "o0", mock=True)
    V1, _, S = K.run(_tagged_mock(tmp_path / "b", add_seed1=True), tmp_path / "o1", mock=True)
    bars = lambda V: V[(V.claim == "C-MAIN") & ~V.bar.str.contains("seed variation")].set_index(  # noqa: E731
        ["bar", "metric", "method", "plant"]).estimate
    np.testing.assert_allclose(bars(V0).sort_index().to_numpy(), bars(V1).sort_index().to_numpy())
    sv = V1[V1.bar == "seed variation (A-SEED)"]
    assert len(sv) and (sv.verdict == "DESCRIPTIVE").all() and sv.estimate.min() > 0.5
    assert any("st-n6-1" in x and "seed-variation" in x for x in S.log)


def test_aseed_mixed_tags_raise(tmp_path):
    res = _tagged_mock(tmp_path, add_seed1=False)
    f = res / "models/confirmatory/dec_conf_models.csv.gz"
    d = pd.read_csv(f, low_memory=False)
    i = d.index[(d.method == K.OURS) & (d.claim == "C-MAIN")][:5]
    d.loc[i, "model_tag"] = np.nan
    d.to_csv(f, index=False)
    with pytest.raises(K.C.SchemaError):
        K.run(res, tmp_path / "out", mock=True)


def test_trivial_baselines_computed_by_scorer(tmp_path):
    res = K.write_mock(tmp_path / "in", n_main=60, n_gate=40, n_pairs=40)
    V, _, S = K.run(res, tmp_path / "out", mock=True)
    assert any("metadata_nf" in x and "scorer computes" in x for x in S.log)      # role rows dropped
    cells = pd.read_csv(tmp_path / "out/cells.csv")
    c = cells[(cells.claim == "C-MAIN") & (cells.plant == "T1") & (cells.block == "all")].set_index("method")
    assert c.loc["always_go"].E_ep == 1.0 and c.loc["never_go"].E_ep == 0.0 and c.loc["never_go"].CF_own == 0.0
    assert np.isfinite(c.loc["metadata_nf"].CF_matched)



def test_tag_allowlist_bb_family_enters_bars(tmp_path):
    res = _tagged_mock(tmp_path / "a", add_seed1=False)
    d = pd.read_csv(res / "models/confirmatory/dec_conf_models.csv.gz", low_memory=False)
    assert set(d[(d.method == "blackbox_gnn") & (d.claim == "C-SCALE")].model_tag) == {"bb-n6-1"}
    V, _, S = K.run(res, tmp_path / "o", mock=True)
    sl = V[(V.claim == "C-SCALE") & (V.bar == "2 comp vs direct")]
    assert sl.verdict.iloc[0] in ("PASS", "FAIL") and "blackbox_gnn" in sl.note.iloc[0]
    assert not (V.bar == "seed variation (A-SEED)").any()
    # the same comparator under the st primary tag is NOT allowlisted -> seed variation, out of the bar
    f = res / "models/confirmatory/dec_conf_models.csv.gz"
    d.loc[d.method == "blackbox_gnn", "model_tag"] = "st-n6-2"
    d.to_csv(f, index=False)
    V2, _, _ = K.run(res, tmp_path / "o2", mock=True)
    assert V2[(V2.claim == "C-SCALE") & (V2.bar == "2 comp vs direct")].verdict.iloc[0] == "MISSING"
    assert (V2[V2.bar == "seed variation (A-SEED)"].method == "blackbox_gnn").any()



# ---- A-POWER (77b17b0, §11): scored prefixes

def test_apower_drops_rows_beyond_prefixes(tmp_path):
    res0 = K.write_mock(tmp_path / "a", n_main=60, n_gate=40, n_pairs=40)
    V0, _, _ = K.run(res0, tmp_path / "o0", mock=True)
    res = K.write_mock(tmp_path / "b", n_main=60, n_gate=40, n_pairs=40)
    f = res / "models/confirmatory/dec_conf_models.csv.gz"
    d = pd.read_csv(f, low_memory=False)
    cm = d[(d.claim == "C-MAIN") & (d.split == "conf_test")]
    late = cm[cm.eid == cm.eid.max()].copy()
    assert len(late)
    extra = []
    for e in (2440, 2441, 2900):                                       # beyond 40-2,439
        x = late.copy()
        x["eid"], x["shard"] = e, 40
        x["lower"] = x["truth"] + 0.3                                   # would wreck validity if scored
        extra.append(x)
    cal = d[(d.claim == "C-MAIN") & (d.split == "cal")].copy()
    cal["eid"] += 1000                                                  # cal beyond 0-999
    cal["lower"] = cal["truth"] + 0.3
    sc = d[(d.claim == "C-SCALE") & (d.split == "conf_H4")].copy()
    sc["eid"] += 600                                                    # conf_H4 beyond 0-599
    pd.concat([d] + extra + [cal, sc], ignore_index=True).to_csv(f, index=False)
    V1, _, S = K.run(res, tmp_path / "o1", mock=True)
    key = ["claim", "bar", "metric", "method", "cell", "plant"]
    a = V0.set_index(key).estimate.sort_index()
    b = V1.set_index(key).estimate.sort_index()
    np.testing.assert_allclose(a.to_numpy(), b.loc[a.index].to_numpy())
    log = "\n".join(S.log)
    assert "A-POWER: decision rows C-MAIN conf_test: dropped" in log and "beyond 40-2439" in log
    assert "A-POWER: cal rows C-MAIN cal: dropped" in log and "C-SCALE conf_H4: dropped" in log


def test_apower_env_ranges(tmp_path):
    res = _env_only_dir(tmp_path)
    f = res / "env/confirmatory/missions_conf_env.csv.gz"
    d = pd.read_csv(f)
    bad = d[d.family == "ID"].drop_duplicates("method").assign(mission=700, building="conf-ID-998", committed=1,
                                                               failed=1, success=0)
    pd.concat([d, bad]).to_csv(f, index=False)
    V, _, S = K.run(res, tmp_path / "out", mock=True)
    assert any("A-POWER: C-ENV ID: dropped 7 mission rows" in x for x in S.log)
    ours = V[(V.claim == "C-ENV") & (V.bar == "1 validity") & (V.method == K.OURS) & (V.cell == "ID")].iloc[0]
    exp = d[(d.family == "ID") & (d.method == K.OURS)]
    assert ours.estimate == pytest.approx(((exp.committed == 1) & (exp.failed == 1)).sum() / exp.committed.sum())



def test_header_note_in_results(tmp_path):
    res = K.write_mock(tmp_path / "in", n_main=40, n_gate=20, n_pairs=20)
    K.run(res, tmp_path / "out", mock=True, header_note="Pre-scoring test suite: 133 passed.")
    md = (tmp_path / "out/CONFIRMATORY_RESULTS.md").read_text()
    assert md.index("Pre-scoring test suite: 133 passed.") < md.index("## Exclusions")


# ---- critic C + A-KIND / D-CAL (5fd2720)

def test_underpowered_but_over_bar_fails():
    S = K.Scorer()
    pw = dict(n_incapable=500, deff=1.0, power_ok=False)
    assert S.bar("x", "b", "E_dec", 0.05, np.full(100, 0.05), le=0.01, up_le=0.013, power_row=pw) == \
        "FAIL (underpowered; exceeds bar)"
    assert S.bar("x", "b", "E_dec", 0.005, np.full(100, 0.005), le=0.01, up_le=0.013, power_row=pw) == "UNDERPOWERED"


def test_p3_overall_requires_every_condition():
    S = K.Scorer()
    S.bar("C-MAIN", "P3", "E_dec own (raw) > 2 %", 0.015, np.full(50, 0.015), ge=0.02, lo_gt=0.01)   # fails
    S.bar("C-MAIN", "P3", "E_dec own (ours) <= 2 %", 0.0, np.zeros(50), le=0.02, up_le=0.02)
    S.bar("C-MAIN", "P3", "E_ep(raw) - E_ep(ours), paired", 0.3, np.full(50, 0.3), lo_gt=0.0, status="HOLM")
    S.p = {"P3": 0.0001}
    K.holm(S)
    K.p_overall(S, "C-MAIN", "P3", set(), "P3 overall")
    paired = [r for r in S.rows if r["metric"].startswith("E_ep(raw)")][0]
    overall = [r for r in S.rows if r["bar"] == "P3 overall"][0]
    assert paired["verdict"] == "PASS" and overall["verdict"] == "FAIL"
    S.bar("C-MAIN", "P3", "claim", 0.015, status="DROPPED")
    K.p_overall(S, "C-MAIN", "P3", set(), "P3 overall again")
    assert S.rows[-1]["verdict"] == "DROPPED"


def test_akind_bar_1b_mondrian_calsize_and_dcal(tmp_path):
    V, _, _, out = _run(tmp_path)
    b = V[(V.claim == "C-MAIN") & (V.bar == "1b per-kind validity")]
    assert set(b.cell) >= {"lone faulted child", "child >=2 modules", "merge"}
    lone = b[b.cell == "lone faulted child"]
    assert lone.verdict.iloc[0] in ("UNDERPOWERED", "PASS", "FAIL", "FAIL (underpowered; exceeds bar)")
    if (lone.verdict == "UNDERPOWERED").any():                     # small mock: E_ep decides
        assert (lone.metric.str.startswith("E_ep own bound")).any()
    cells = pd.read_csv(out / "cells.csv")
    assert cells.block.astype(str).str.startswith("mondrian:").any()
    cs = V[V.bar == "cal-size sensitivity"]
    assert len(cs) >= 1 and (cs.verdict == "DESCRIPTIVE").all()
    md = (out / "CONFIRMATORY_RESULTS.md").read_text()
    assert "conditions on the fixed calibration split" in md
    assert len(V[V.bar == "P3 overall"]) == 1 and len(V[V.bar == "P4 overall"]) == 1


def test_shard0_exclusion_only_on_confirmatory_streams():
    d = pd.DataFrame(dict(split=["conf_het_test", "het_cal", "cal", "conf_H1", "merge_P0"], eid=[100, 100, 5, 7, 3],
                          shard=[0, 0, 0, 0, 0]))
    out = K.exclusions(d, [], "x")
    assert sorted(out.split) == ["cal", "het_cal"]            # exploratory cal rows never lose shard 0
