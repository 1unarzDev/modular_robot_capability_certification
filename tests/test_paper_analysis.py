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
