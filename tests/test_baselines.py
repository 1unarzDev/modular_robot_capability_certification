"""Baseline interface and scorer tests (CPU; the T1 parts run on CUDA when available, else on the CPU)."""
import math

import numpy as np
import pytest
import torch

from cfcap.baselines import base
from cfcap.baselines import episode as E
from cfcap.baselines import posterior as Po
from cfcap.baselines.simple import Nominal, Oracle

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _fake_episodes(n_ep=60, C=5, seed=0, empty=False):
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(n_ep):
        cm = [] if empty else [(k,) for k in range(C)]
        rr = np.clip(1 - rng.exponential(0.1, (len(cm), base.L)), 0, 1.2)
        eps.append(E.Episode(eid=i, shape="fake", plate="plated", cmd=np.zeros((2, 3)), T=0.0, obs={},
                             cand_modules=cm, lat=None, rho_rel=rr, rho=rr, rho_dec=rr))
    return eps


def _evaluate(b, eps, alpha=0.1):
    W = max(max(e.n_cand for e in eps), 1) * base.L
    preds = [b(e) for e in eps]
    truth = base.pad([e.rho_rel for e in eps], W)
    point = np.where(np.isnan(truth), np.nan, base.pad([p.point for p in preds], W))
    scale = base.pad([p.scale for p in preds], W)
    folds = np.arange(len(eps)) % 2
    lower, _ = base.cross_fit_lower(point, scale, truth, folds, alpha)
    hf = np.ones_like(truth)
    return base.metrics(lower, truth, hf)


def test_oracle_has_zero_unsafe_and_certifies_every_capable():
    for r in _evaluate(Oracle(), _fake_episodes()):
        assert r["n_unsafe"] == 0 and r["unsafe_commit_rate"] == 0.0
        assert r["cert_frac"] == pytest.approx(1.0)
        assert r["ep_miscover"] == 0.0


def test_nominal_on_fake_faults_is_safe_at_alpha():
    # sanity: the simultaneous layer keeps the episode-level miscoverage near alpha even for a bad predictor
    rows = _evaluate(Nominal(), _fake_episodes(400, seed=3))
    assert rows[0]["ep_miscover"] <= 0.2


@pytest.mark.parametrize("b", [Nominal(), Oracle()])
def test_empty_candidate_set_refuses(b):
    ep = _fake_episodes(1, empty=True)[0]
    p = b(ep)
    assert p.point.shape == (0, base.L) and "refuse" in p.info
    for tau in base.TAUS:
        go = base.decide(p.lower, tau)
        assert go.size == 0 and not go.any()


def test_rls_empty_candidate_set_refuses():
    from cfcap.baselines.rls import RLSInterval
    ep = _fake_episodes(1, empty=True)[0]
    p = RLSInterval()(ep)
    assert p.point.shape == (0, base.L) and "refuse" in p.info


def test_capxfer_projection_empty_and_budget():
    from cfcap.baselines.capxfer_cert import child_vectors
    n = 8
    sets = [{0} for _ in range(n)]
    g_lo = np.ones(n)
    # healthy everywhere -> exactly the all-ok vector
    v = child_vectors(sets, g_lo, [0, 1, 2, 3], n)
    assert v == [((0, 0, 0, 0), (1.0, 1.0, 1.0, 1.0))]
    # three wheels outside the child exclude ok -> budget exceeded -> empty -> refuse
    s2 = [set(s) for s in sets]
    for w in (4, 5, 6):
        s2[w] = {1}
    assert child_vectors(s2, g_lo, [0, 1, 2, 3], n) == []
    # an empty wheel set inside the child -> empty
    s3 = [set(s) for s in sets]
    s3[0] = set()
    assert child_vectors(s3, g_lo, [0, 1, 2, 3], n) == []


def test_refusal_never_commits():
    lo = np.array([[np.nan, -np.inf, np.inf * 0, 0.9]])
    go = base.decide(lo, 0.8)
    assert go.tolist() == [[False, False, False, True]]


def test_hypothesis_count_and_prior():
    HM, HG, nf = Po.hypotheses(8)
    S = len(Po.fault_states())
    assert len(HM) == 1 + 8 * S + math.comb(8, 2) * S * S
    lp = Po.log_prior_full(HM, nf, 8)
    assert np.exp(lp).sum() == pytest.approx(1.0, rel=1e-9)


def test_nominal_on_healthy_is_exact():
    """A healthy nominal body (identified theta as a point domain, no spread): rho_dec = 1 on every candidate and leg,
    which is exactly the nominal baseline's prediction; zero unsafe commits."""
    from cfcap.t1 import params as P
    from cfcap.t1 import plant
    par, asm, cl = E.context("pair")
    lat = plant.sample_latents(asm, 1, np.random.default_rng(0), plant.HET_OFF,
                               domain=P.point_domain(P.load_theta("plated"), 1, device=DEV), device=DEV)
    lat["yaw_err"] = torch.zeros_like(lat["yaw_err"])
    ep = E.Episode(eid=0, shape="pair", plate="plated", cmd=np.zeros((2, 3)), T=0.0, obs={},
                   cand_modules=[tuple(c.modules) for c in cl], lat=E.lat_to(lat, "cpu"))
    rr, rho, rd = E.candidate_rho(par, ep.cands, lat, DEV)
    ep.rho_rel, ep.rho, ep.rho_dec = rr[0], rho[0], rd[0]
    p = Nominal()(ep)
    assert np.allclose(p.point, ep.truth, atol=1e-5)
    assert np.allclose(rr[0], 1.0, atol=1e-6)
    rows = _evaluate(Nominal(), [ep] * 20)
    assert all(r["n_unsafe"] == 0 for r in rows)


def test_smc_task_units_cover_split():
    import argparse
    from cfcap.baselines import smc
    from cfcap.runner import parse_chunk
    p = argparse.ArgumentParser()
    smc._args(p)
    a = p.parse_args(["--episodes", "37", "--per-unit", "4"])
    eps = [j for k in range(5) for u in smc._units(a, k, 5) for j in parse_chunk(u)]
    assert eps == list(range(37))


def test_decision_stratum_counts_only_faulted_near_tau():
    truth = np.array([[0.75, 0.95, 0.65, 0.78]])
    hf = np.array([[1, 1, 0, 1]], float)
    lower = truth - 0.01
    r = {x["tau"]: x for x in base.metrics(lower, truth, hf, taus=(0.7,), band=0.1)}[0.7]
    assert r["n_dec"] == 2            # 0.75 and 0.78 (faulted, within 0.1 of tau); 0.95 too far, 0.65 not faulted
    assert r["cert_frac"] == 1.0 and r["n_unsafe"] == 0


def test_compile_failure_falls_back_to_graph(monkeypatch):
    from cfcap.t1 import plant
    calls = []

    def fake(jobs, par, T, device=None, engine=None):
        calls.append(engine)
        if engine == "compile":
            raise RuntimeError("Failed to find C compiler")
        return ["ok"]
    monkeypatch.setattr(plant, "simulate_bucket", fake)
    monkeypatch.setattr(Po, "_COMPILE_BROKEN", False)
    assert Po._simulate(None, None, None, None, 1.0, "cpu", "compile") == "ok"
    assert Po._simulate(None, None, None, None, 1.0, "cpu", "compile") == "ok"
    assert calls == ["compile", "graph", "graph"]


def test_conformal_selection_fdr_and_refusal():
    rng = np.random.default_rng(5)
    E_, W = 400, 20
    truth = np.clip(1 - rng.exponential(0.15, (E_, W)), 0, 1.2)
    point = truth + rng.normal(0, 0.05, (E_, W))
    scale = np.full_like(point, 0.05)
    folds = np.arange(E_) % 2
    go = base.conformal_selection(point, scale, truth, folds, 0.8, 0.1)
    inc = truth < 0.8
    assert go.sum() > 0 and (go & inc).sum() / go.sum() <= 0.15
    point[:5] = np.nan
    go2 = base.conformal_selection(point, scale, truth, folds, 0.8, 0.1)
    assert not go2[:5].any()


def test_fault_shift_weights():
    w = base.fault_shift_weights([0, 1, 2, 2])
    assert np.isclose(w.mean(), 1) and np.isclose(w[2] / w[0], 4)


def test_from_runs_end_to_end(tmp_path, monkeypatch):
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("bp", pathlib.Path(__file__).parents[1] / "scripts/baselines_pilot.py")
    bp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bp)
    monkeypatch.setattr(bp, "RES", tmp_path / "res")
    rng = np.random.default_rng(1)
    units = tmp_path / "runs" / "units"
    units.mkdir(parents=True)
    cols = {k: [] for k in bp.DEC_COLS + ("ep_n_faults", "cand_fault_class", "sims", "peak_vram_gb")}
    for split in ("cal", "test"):
        for e in range(60):
            C = 3
            tr = np.clip(1 - rng.exponential(0.15, (C, base.L)), 0, 1.2)
            for meth in ("nominal", "oracle", "nuisance_smc", "capxfer"):
                pt = {"nominal": np.ones_like(tr), "oracle": tr, "nuisance_smc": tr + rng.normal(0, .03, tr.shape),
                      "capxfer": np.full_like(tr, np.nan)}[meth]
                lo = tr - 0.1 if meth == "capxfer" else pt - 0.05
                n = tr.size
                vals = dict(method=[meth] * n, route=["raw"] * n, posterior=["none"] * n, train=["T1"] * n,
                            test_plant=["T1"] * n, split=[split] * n, status=["exploratory"] * n, eid=[e] * n,
                            shape=["pair" if e % 2 else "square"] * n, cand=np.repeat(np.arange(C), base.L),
                            kind=["child"] * n, cand_size=[1] * n, has_fault=np.repeat([1, 0, 1], base.L),
                            leg=np.tile(np.arange(base.L), C), truth=tr.ravel(), point=pt.ravel(),
                            scale=np.full(n, 0.03), lower=lo.ravel(), alpha=[0.1] * n, seconds=[1.0] * n,
                            ep_n_faults=[e % 3] * n, cand_fault_class=np.repeat([1, 0, 2], base.L), sims=[0] * n,
                            peak_vram_gb=[0.0] * n)
                for k in cols:
                    cols[k] += list(vals[k])
    np.savez(units / "u.npz", **{k: np.asarray(v) for k, v in cols.items()})
    import argparse
    a = argparse.Namespace(runs=str(tmp_path / "runs"), split="test", cal_split="cal", tag="t_", hardware="cpu")
    bp.cmd_from_runs(a)
    import pandas as pd
    met = pd.read_csv(tmp_path / "res" / "t_metrics.csv")
    o = met[(met.baseline == "oracle") & (met.certifier == "sim_a10") & (met.stratum == "all") & (met["shape"] == "all")]
    assert (o.n_unsafe == 0).all() and np.allclose(o.cert_frac, 1.0)
    assert {"sim_a05", "sim_a10", "mondrian_a10", "raw", "cselect_q10"} <= set(met.certifier)
    assert {"faults2x"} <= set(met["shift"]) and {"class0", "class2", "decision"} <= set(met.stratum)
    dec = pd.read_csv(tmp_path / "res" / "paper" / "dec_baselines.csv.gz")
    assert set(dec.split) == {"test"} and {"capxfer_raw", "nuisance_smc_raw"} <= set(dec.method)
