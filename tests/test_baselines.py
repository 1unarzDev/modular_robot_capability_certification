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
