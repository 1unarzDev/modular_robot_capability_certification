"""Unit tests for the paper scorers (scripts/analysis/common.py) and an end-to-end run on mock data."""
import math
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts/analysis"))
import common as C  # noqa: E402
import make_all  # noqa: E402
import mock  # noqa: E402
import t3_grid  # noqa: E402


def frame(truth, lower, eid=None, point=None, scale=None):
    n = len(truth)
    return pd.DataFrame(dict(truth=truth, lower=lower, eid=eid if eid is not None else np.arange(n),
                             point=point if point is not None else np.full(n, np.nan),
                             scale=scale if scale is not None else np.full(n, np.nan)))


def test_decision_counts_hand_example():
    # tau 0.7 only. truth 0.9 / lower 0.75 -> certified; truth 0.5 / lower 0.72 -> unsafe;
    # truth 0.8 / NaN -> refuse; truth 0.6 / -inf -> refuse.
    d = frame([0.9, 0.5, 0.8, 0.6], [0.75, 0.72, np.nan, -np.inf], eid=[0, 0, 1, 1])
    c = C.decision_counts(d, taus=(0.7,))
    assert (c["n_capable"], c["n_certified"], c["n_incapable"], c["n_unsafe"]) == (2, 1, 2, 1)
    assert c["cert_frac"] == 0.5 and c["unsafe_commit"] == 0.5 and c["ep_any_unsafe"] == 0.5


def test_refusal_never_commits():
    d = frame([0.1, 0.2], [np.nan, -np.inf])
    c = C.decision_counts(d)
    assert c["n_unsafe"] == 0 and c["n_certified"] == 0


def test_nan_truth_rows_are_ignored():
    d = frame([np.nan, 0.9], [1.0, 0.95])
    c = C.decision_counts(d, taus=(0.7,))
    assert c["n_capable"] == 1 and c["n_incapable"] == 0


def test_matches_baselines_metrics():
    base = pytest.importorskip("cfcap.baselines.base")
    rng = np.random.default_rng(1)
    E, W = 30, 12
    truth = rng.uniform(0.4, 1.0, (E, W))
    lower = truth + 0.05 * rng.standard_normal((E, W)) - 0.03
    truth[rng.random((E, W)) < 0.2] = np.nan
    lower[np.isnan(truth)] = np.nan
    hf = np.zeros((E, W))
    ref = base.metrics(lower, truth, hf)
    d = pd.DataFrame(dict(truth=truth.ravel(), lower=lower.ravel(), eid=np.repeat(np.arange(E), W)))
    for r in ref:
        c = C.decision_counts(d, taus=(r["tau"],))
        assert c["n_capable"] == r["n_capable"] and c["n_incapable"] == r["n_incapable"]
        assert math.isclose(c["cert_frac"], r["cert_frac"]) and math.isclose(c["unsafe_commit"], r["unsafe_commit_rate"])
        assert math.isclose(c["ep_any_unsafe"], r["ep_any_unsafe"])


def test_rel_stratum_hand_example():
    # tau 0.7, band 0.1: rows (truth, has_fault): (0.95, 0) far+healthy -> out; (0.75, 0) near -> in;
    # (0.99, 1) faulted -> in; (0.55, 0) far below, healthy -> out; (0.65, 0) near -> in
    d = frame([0.95, 0.75, 0.99, 0.55, 0.65], [0.9, 0.72, 0.5, 0.71, 0.70])
    d["has_fault"] = [0, 0, 1, 0, 0]
    assert C.rel_mask(d, 0.7).tolist() == [False, True, True, False, True]
    r = C.decision_counts(d, taus=(0.7,), stratum="rel")
    a = C.decision_counts(d, taus=(0.7,), stratum="all")
    assert (r["n_capable"], r["n_certified"], r["n_incapable"], r["n_unsafe"]) == (2, 1, 1, 1)
    assert (a["n_capable"], a["n_certified"], a["n_incapable"], a["n_unsafe"]) == (3, 2, 2, 2)
    # the easy healthy candidate inflates 'all'
    assert r["cert_frac"] < a["cert_frac"]


def test_rel_coverage_ignores_easy_entries():
    d = frame([0.95, 0.75], [0.99, 0.7], eid=[0, 0])   # the easy entry is miscovered, the relevant one covered
    d["has_fault"] = [0, 0]
    assert C.coverage(d, stratum="all")["ep_cov"] == 0.0
    assert C.coverage(d, stratum="rel")["ep_cov"] == 1.0


def test_matched_lower_rel_stratum_controls_rel_unsafe():
    rng = np.random.default_rng(3)
    n = 4000
    truth = np.clip(rng.uniform(0.5, 1.0, n), 0, 1)
    d = pd.DataFrame(dict(truth=truth, eid=np.arange(n) // 10, has_fault=(rng.random(n) < 0.2).astype(int),
                          point=truth + 0.03 * rng.standard_normal(n), scale=np.full(n, 0.03),
                          lower=np.nan))
    lo, _ = C.matched_lower(d, d, u_star=0.01, stratum="rel")
    c = C.ep_counts(d, lo, stratum="rel")
    assert c["uns"].sum() / c["inc"].sum() <= 0.01 + 1e-12


def test_matched_margin_hits_target_and_is_tight():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(1000)
    for u in (0.0, 0.01, 0.05, 0.3):
        k = C.matched_margin(z, u)
        rate = np.mean(z >= k)
        assert rate <= u + 1e-12
        # tight: lowering k to the next order statistic would exceed u
        below = z[z < k]
        if len(below):
            assert np.mean(z >= below.max()) > u


def test_matched_margin_ties_and_edge_cases():
    assert C.matched_margin(np.array([]), 0.01) == -math.inf
    assert C.matched_margin(np.array([1.0, 1.0, 1.0]), 1.0) == -math.inf
    k = C.matched_margin(np.array([1.0, 1.0, 0.0, 0.0]), 0.25)  # one allowed, but the two 1.0 tie -> neither
    assert np.mean(np.array([1.0, 1.0, 0.0, 0.0]) >= k) <= 0.25


def test_matched_lower_applies_margin_from_fit_split():
    fit = frame([0.5] * 100, [np.nan] * 100, point=np.linspace(0.4, 0.9, 100), scale=np.full(100, 0.1))
    ev = frame([0.9], [np.nan], point=[0.95], scale=[0.1])
    lo, k = C.matched_lower(fit, ev, u_star=0.0, taus=(0.7,))
    # all fit rows incapable at 0.7; max z = (0.9-0.7)/0.1 = 2 -> k just above 2 -> lower just below 0.75
    assert 2.0 < k < 2.0 + 1e-8 and lo[0] < 0.75 and lo[0] > 0.7499


def test_coverage_episode_simultaneous():
    d = frame([0.8, 0.8, 0.8, 0.8], [0.7, 0.9, 0.7, np.nan], eid=[0, 0, 1, 1])
    c = C.coverage(d)
    assert c["entry_cov"] == 0.75 and c["ep_cov"] == 0.5


def test_boot_ratio_contains_point():
    rng = np.random.default_rng(2)
    den = rng.integers(5, 20, 200).astype(float)
    num = rng.binomial(den.astype(int), 0.3).astype(float)
    lo, hi = C.boot_ratio(num, den)
    assert lo <= num.sum() / den.sum() <= hi and hi - lo < 0.1
    assert all(np.isnan(C.boot_ratio([1.0], [2.0])))


def test_macro_names_letters_only():
    assert C.macro_name("t1.cert.amortized_sim.T1") == "cfTOneCertAmortizedSimTOne"
    assert re.fullmatch(r"[A-Za-z]+", C.macro_name("f3.legs.persec.6mod"))


def test_schema_violation_raises(tmp_path):
    p = tmp_path / "x/paper"
    p.mkdir(parents=True)
    pd.DataFrame(dict(method=["a"])).to_csv(p / "dec_x.csv", index=False)
    with pytest.raises(C.SchemaError):
        C.load("dec", tmp_path)


def test_bad_status_raises(tmp_path):
    p = tmp_path / "x/paper"
    p.mkdir(parents=True)
    pd.DataFrame(dict(source=["T1"], delta=[0.1], subset=["all"], n_dec=[1], amb_share=[0.0],
                      status=["final"])).to_csv(p / "amb_het.csv", index=False)
    with pytest.raises(C.SchemaError):
        C.load("amb", tmp_path)


def test_empty_results_gives_placeholders(tmp_path):
    nums = make_all.main(["--results", str(tmp_path / "none"), "--paper", str(tmp_path / "paper")])
    tex = (tmp_path / "paper/numbers.tex").read_text()
    assert r"\cfPH" in tex and sum(len(n.items) for n in nums) > 20
    for t in ("t1_certificate", "t2_heldout", "t3_grid", "t4_probing"):
        assert (tmp_path / f"paper/tables/{t}.tex").exists()


def test_end_to_end_mock(tmp_path):
    res = mock.write(tmp_path / "res")
    nums = make_all.main(["--results", str(res), "--paper", str(tmp_path / "paper")])
    tex = (tmp_path / "paper/numbers.tex").read_text()
    names = re.findall(r"\\newcommand\{\\(\w+)\}", tex)
    assert len(names) == len(set(names))
    got = {k: v[0] for n in nums for k, v in n.items.items()}
    for key in ("cfTOneBestCertTOne", "cfTOneCapxferCertTOne", "cfTTwoEpcovWorst", "cfTThreeVerdict",
                "cfTFourResolvedEvoi", "cfFTwoGainPts", "cfFThreeSpeedup"):
        assert "cfPH" not in got[key], key
    # matched operating point: every method with cal rows stays at or under U_STAR on T1 cal by construction,
    # and the oracle certifies everything
    t1 = pd.read_csv(tmp_path / "paper/tables/t1_certificate.csv")
    assert set(t1.stratum) == {"rel", "all"}
    assert t1[(t1.method == "oracle") & (t1.group == "all")].cf.min() == 1.0
    assert t1[(t1.method == "never_go")].cf.max() == 0.0
    # headline macros are the rel stratum, 'all' variants exist beside them
    for key in ("cfTOneBestCertTOneAll", "cfTOneCertNuisanceSmcTOne", "cfTOneCertNuisanceSmcTOneAll",
                "cfTThreeUnsafeAmortizedTTwo", "cfTThreeUnsafeAmortizedTTwoAll", "cfTFourCertEvoi",
                "cfTFourCertEvoiAll", "cfTTwoEpcovWorstAll"):
        assert "cfPH" not in got[key], key
    for t in ("t1_certificate_all", "t2_heldout_all"):
        assert (tmp_path / f"paper/tables/{t}.tex").exists()
    for f in ("f1_ambiguity", "f2_diversity", "f3_throughput"):
        assert (tmp_path / f"paper/figs/{f}.pdf").stat().st_size > 1000


def test_mock_refused_into_repo_paper(tmp_path):
    res = mock.write(tmp_path / "res")
    with pytest.raises(SystemExit):
        make_all.main(["--results", str(res), "--paper", str(C.PAPER)])


def _cell(unsafe, cert, sec):
    return dict(unsafe_commit=unsafe, cert_frac=cert, sec=sec)


def test_t3_verdict_cases():
    v = t3_grid.verdict
    assert v(_cell(0.01, 0.9, 0.01), _cell(0.02, 0.8, 4.0)) == "amortized wins on T2 robustness and latency"
    assert v(_cell(0.02, 0.8, 4.0), _cell(0.01, 0.9, 0.01)) == "SMC wins on both: drop 'amortized'"
    t = v(_cell(0.02, 0.9, 0.01), _cell(0.01, 0.8, 4.0))
    assert t.startswith("T2 tradeoff") and "amortized faster" in t
    assert "SMC wins on both" not in v(_cell(0.02, 0.9, 2.0), _cell(0.01, 0.8, 1.0))
    assert "tradeoff" in v(_cell(np.nan, 0.9, 1.0), _cell(0.01, 0.8, 2.0))


def test_duplicate_method_across_files_raises(tmp_path):
    res = mock.write(tmp_path / "res")
    d = pd.read_csv(res / "baselines/paper/dec_baselines.csv")
    d[d.method == "nominal"].to_csv(res / "baselines/paper/dec_trivial.csv", index=False)
    with pytest.raises(C.SchemaError):
        C.load("dec", res)


def test_merge_rows_stay_out_of_t1_and_feed_t5(tmp_path):
    res = mock.write(tmp_path / "res")
    nums = make_all.main(["--results", str(res), "--paper", str(tmp_path / "paper")])
    got = {k: v[0] for n in nums for k, v in n.items.items()}
    t1 = pd.read_csv(tmp_path / "paper/tables/t1_certificate.csv")
    assert "compositional_enum" not in set(t1.method)
    for key in ("cfTFiveVerdict", "cfTFiveCertComp", "cfTFiveFliphiTOne", "cfFFourVerdict", "cfFFourGateUnsafeWorst",
                "cfFFourUndomShare", "cfTOneEThreeVerdict", "cfTOneEThreeGainPi"):
        assert "cfPH" not in got[key], key
    assert (tmp_path / "paper/figs/f4_gate.pdf").stat().st_size > 1000
    for t in ("t5_merge_defect", "t5_merge_cert", "t1_by_size"):
        assert (tmp_path / f"paper/tables/{t}.tex").exists()


def test_f4_kill_rule(tmp_path):
    g = pd.DataFrame([dict(law=l, severity=s, probe="v2", certifier=c, stratum="decision", cert_frac_capable=0.9,
                           unsafe_rate=0.005, ep_any_unsafe=0.02, n=100)
                      for l, s in (("two_constant", 0), ("knee", 1), ("knee", 2)) for c in ("par", "pi", "gate")])
    (tmp_path / "theory").mkdir()
    g.to_csv(tmp_path / "theory/gate_cert.csv", index=False)
    import f4_gate
    n = f4_gate.run(tmp_path, tmp_path / "paper")
    assert n.items[C.macro_name("f4.verdict")][0].startswith("E1 killed")


def test_text_macro_underscores_are_escaped():
    n = C.Numbers("t")
    n.set("a", "pending (dec_merge.csv)", "x")
    n.set("b", r"dominance\_pi", "x")
    n.set("c", r"$\rho_l$", "x")
    v = {k: x[0] for k, x in n.items.items()}
    assert v["cfA"] == r"pending (dec\_merge.csv)" and v["cfB"] == r"dominance\_pi" and v["cfC"] == r"$\rho_l$"


def test_t1_cfbh_nominal_only(tmp_path):
    res = mock.write(tmp_path / "res")
    d = pd.read_csv(res / "baselines/paper/dec_baselines.csv")
    x = d[d.method == "nominal"].copy()
    x["method"], x["point"], x["scale"] = "cfBH", np.nan, np.nan
    x["lower"] = np.where(x.truth >= 0.81, 0.8, np.where(x.truth >= 0.71, 0.7, -np.inf))
    x.to_csv(res / "baselines/paper/dec_cfbh.csv", index=False)
    make_all.main(["--results", str(res), "--paper", str(tmp_path / "paper")])
    t1 = pd.read_csv(tmp_path / "paper/tables/t1_certificate.csv")
    assert "cfBH" not in set(t1.method)                       # no matched operating point for a selection rule
    nom = (tmp_path / "paper/tables/t1_nominal.tex").read_text()
    assert "Conformal selection (cfBH" in nom
    # scorer: commit-at-tau <=> lower >= tau; truth >= 0.71 entries selected up to 0.7 are never unsafe at 0.6/0.7
    c = C.decision_counts(x, stratum="all")
    assert c["n_unsafe"] == 0 and c["n_certified"] > 0



def _probe_rows(plant, pols, n=6, cert=0.9):
    rows = []
    for pol in pols:
        for e in range(n):
            rows.append(dict(policy=pol, split="test", status="exploratory", eid=e, shape="pair", n_probes=1,
                             probe_time_s=12.0 if pol == "fixed_v2" else 20.0, action="commit", n_dec=10,
                             n_capable=8, n_incapable=2, n_certified=int(8 * cert), n_unsafe=0, n_amb_before=2,
                             n_amb_after=1, test_plant=plant, n_capable_rel=4, n_incapable_rel=2,
                             n_certified_rel=int(4 * cert), n_unsafe_rel=0))
    return pd.DataFrame(rows)


def test_t4_headline_plant_and_appendix(tmp_path):
    import t4_probing
    p = tmp_path / "decision/paper"
    p.mkdir(parents=True)
    _probe_rows("T0", ["fixed_v2", "evoi", "eig"], cert=0.5).to_csv(p / "probe_t0_n1.csv", index=False)
    _probe_rows("T0-knee-post-ktol", ["fixed_v2", "evoi", "dominance_clamp"], cert=1.0).to_csv(p / "probe_t0n1bK.csv",
                                                                                    index=False)
    _probe_rows("T0-knee-post-K5-appendix", ["fixed_v2", "evoi"], cert=0.25).to_csv(p / "probe_t0envpost.csv",
                                                                                   index=False)
    n = t4_probing.run(tmp_path, tmp_path / "paper")
    assert n.items[C.macro_name("t4.plant")][0] == "T0-knee-post-ktol"
    assert n.items[C.macro_name("t4.cert.evoi")][0] == "100.0\\,\\%"          # 4/4 relevant: headline plant only
    assert "probe_t0n1bK.csv" in n.items[C.macro_name("t4.cert.evoi")][1]
    head = (tmp_path / "paper/tables/t4_probing.tex").read_text()
    app = (tmp_path / "paper/tables/t4_probing_appendix.tex").read_text()
    assert "Dominance + clamp" in head and "EIG" not in head
    assert "T0-knee-post-K5-appendix" in app and "T0 &" in app and "T0-knee-post-ktol &" not in app


def test_latency_table_merges_decision_and_models(tmp_path):
    (tmp_path / "decision/paper").mkdir(parents=True)
    pd.DataFrame(dict(method=["envpost_ktol"] * 3 + ["envpost_ktol"], shape=["2mod", "3mod", "4mod", "all"],
                      n_wheels=[8.0, 12.0, 16.0, np.nan], sec_per_decision=[1.49, 4.54, 12.32, 4.33],
                      hardware="cpu", status="exploratory", n=[12, 22, 14, 48])
                 ).to_csv(tmp_path / "decision/paper/latency_decision.csv", index=False)
    (tmp_path / "models").mkdir()
    pd.DataFrame(dict(method=["amortized_n2_s1"] * 2, shape=["pair", "square"], ms_per_episode=[3.4, 3.5])
                 ).to_csv(tmp_path / "models/latency.csv", index=False)
    L, srcs = C.latency_table(tmp_path)
    assert set(L.method) == {"envpost_ktol", "amortized_n2_s1"} and len(srcs) == 2
    import f3_throughput, t3_grid
    n = f3_throughput.run(tmp_path, tmp_path / "paper")
    v = {k: x[0] for k, x in n.items.items()}
    assert v["cfFThreeSecExactpostMin"] == "1.5" and v["cfFThreeSecExactpostMax"] == "12.3"
    assert v["cfFThreeMsAmortized"] == "3.5" or v["cfFThreeMsAmortized"] == "3.4"
    n3 = t3_grid.run(tmp_path, tmp_path / "paper")
    t = (tmp_path / "paper/tables/t3_latency.tex").read_text()
    assert "Exact posterior (knee-tolerant)" in t and "3.4 ms" in t and "12.32 s" in t
    assert n3.items[C.macro_name("t3.lat.envpost_ktol.max")][0] == "12.3"


def test_t4_headline_ruling_columns(tmp_path):
    import t4_probing
    p = tmp_path / "decision/paper"
    p.mkdir(parents=True)
    rows = []
    for pol, reg, served, unsafe in (("fixed_v2", 30.0, 7, 0), ("fixed_v2_clamp", 22.0, 9, 1),
                                     ("dominance_clamp", 19.0, 9, 0), ("evoi", 17.0, 9, 0)):
        for e in range(20):
            rows.append(dict(policy=pol, split="test", status="exploratory", eid=e, shape="pair", n_probes=1,
                             probe_time_s=12.0 if pol == "fixed_v2" else 18.0, action="commit", n_dec=10, n_capable=8,
                             n_incapable=2, n_certified=8, n_unsafe=0, n_amb_before=0, n_amb_after=0,
                             test_plant="T0-knee-post-ktol", regret=reg + (e % 2), n_unsafe_assigned=int(unsafe and e == 0),
                             n_served=served, oracle_served=10, n_certified_task=9, n_capable_task=10,
                             clamp_bought=int(e < 3) if pol == "fixed_v2_clamp" else np.nan))
    pd.DataFrame(rows).to_csv(p / "probe_t0n1bK.csv", index=False)
    n = t4_probing.run(tmp_path, tmp_path / "paper")
    v = {k: x[0] for k, x in n.items.items()}
    assert v["cfTFourUsefulEvoi"] == "90.0\\,\\%" and v["cfTFourUsefulFixedVTwo"] == "70.0\\,\\%"
    assert v["cfTFourDregretEvoiClamp"] == "-5.0" and v["cfTFourDregretEvoiDominance"] == "-2.0"
    assert v["cfTFourEpunsafeFixedVTwoClamp"] == "5.0\\,\\%" and v["cfTFourEpunsafeEvoi"] == "0.0\\,\\%"
    assert v["cfTFourCerttaskEvoi"] == "90.0\\,\\%" and v["cfTFourClampBought"] == "3/20"
    head = (tmp_path / "paper/tables/t4_probing.tex").read_text()
    assert "served / oracle-served" in head and "(180/200)" in head and "task-relevant" not in head
    sec = (tmp_path / "paper/tables/t4_probing_candidates.tex").read_text()
    assert "task-relevant" in sec and "alternative certified" in sec and "certifies what the task needs" not in sec
    # paired CI excludes 0 for a constant -5 s difference
    assert "-5.0 [-5.0, -5.0]" in head


def test_t4_missing_optional_columns_show_dashes(tmp_path):
    import t4_probing
    p = tmp_path / "decision/paper"
    p.mkdir(parents=True)
    d = _probe_rows("T0-knee-post-ktol", ["fixed_v2", "evoi"])
    d["regret"] = 1.0
    d.to_csv(p / "probe_t0n1bK.csv", index=False)
    n = t4_probing.run(tmp_path, tmp_path / "paper")
    head = (tmp_path / "paper/tables/t4_probing.tex").read_text()
    assert "task-relevant" not in head and "\\textemdash" in head
    assert n.items[C.macro_name("t4.clamp.bought")][2] == "placeholder"


def test_t3_selects_ours_by_posterior_amortized_compositional(tmp_path):
    import t3_grid
    res = mock.write(tmp_path / "res")
    d = pd.read_csv(res / "models/paper/dec_models.csv")
    ens = d[d.method == "amortized_sim"].copy()
    ens["method"], ens["posterior"], ens["route"] = "ensemble", "none", "simultaneous"
    pd.concat([d, ens]).to_csv(res / "models/paper/dec_models.csv", index=False)
    t3_grid.run(res, tmp_path / "paper")
    # amortized_sim (route simultaneous) and the ensemble (posterior none) are not ours; amortized_comp is
    got = pd.read_csv(res / "models/paper/dec_models.csv")
    picked = got[(got.posterior == "amortized") & (got.route == "compositional")].method.unique().tolist()
    assert picked == ["amortized_comp"]
    tex = (tmp_path / "paper/tables/t3_grid.tex").read_text()
    assert "compositional" in tex.split("\n")[1]
