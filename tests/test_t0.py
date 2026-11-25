"""T0 unit tests: QS wrapper vs capxfer, candidate enumeration, consistent set contains the truth, scorer."""
import numpy as np
import pytest

from capxfer import assembly as A
from capxfer.ideas.i2 import theory as Th
from capxfer.sim import OK, WEAK, COAST, BRAKE

from cfcap.t0 import candidates as C
from cfcap.t0 import observe as O
from cfcap.t0 import probesets as PS
from cfcap.t0 import qs, stagea


@pytest.fixture(scope="module")
def P():
    return qs.load_params("plated")


@pytest.fixture(scope="module")
def theta():
    return qs.load_theta("plated")


def test_single_healthy_rho_matches_capxfer(P, theta):
    asm = A.physical("single", P)
    pl = qs.Plant(asm, P, theta)
    rho, r = qs.rho(pl, *qs.healthy(asm.n))
    xis = qs.decision_legs(asm)
    xr, _, _ = Th.qs_solve(asm, P, *qs.healthy(asm.n), xis, dom=theta)
    rho_c = Th.rho_of(xr, xis, asm.char_length())
    assert r["resid"].max() < 1e-6
    np.testing.assert_allclose(rho, rho_c, atol=1e-6)
    assert np.all(rho > 0.8) and np.all(rho < 1.05)
    # the decision legs are capxfer's P1..P5 legs (both signs)
    np.testing.assert_allclose(xis[[0, 1, 2, 3, 8, 9]], Th.legs(asm)[[0, 1, 2, 3, 8, 9]])


@pytest.mark.parametrize("name", ["pair", "L3", "square4"])
def test_faulted_matches_capxfer_where_both_converge(P, theta, name):
    asm = A.physical(name, P)
    rng = np.random.default_rng(3)
    B = 12
    modes, gs = qs.healthy(asm.n, B)
    for b in range(1, B):
        i = rng.integers(asm.n)
        modes[b, i] = rng.integers(1, 4)
        gs[b, i] = rng.uniform(0.3, 0.9)
    xis = qs.decision_legs(asm)
    xr, _, du = Th.qs_solve(asm, P, modes, gs, xis, dom=theta)
    rc = Th.qs_solve.last_resid
    pl = qs.Plant(asm, P, theta)
    r = qs.solve(pl, modes, gs, pl.ik(xis))
    ok = (rc < 1e-6) & (r["resid"] < 1e-6)
    assert ok.mean() > 0.7
    np.testing.assert_allclose(r["xi"][ok], xr[ok], atol=2e-4)


def test_per_row_heterogeneity_equals_per_plant(P, theta):
    asm = A.physical("pair", P)
    rng = np.random.default_rng(0)
    draws = [O.sample_het(rng, asm, True) for _ in range(3)]
    pl0 = qs.Plant(asm, P, theta)
    xis = qs.decision_legs(asm)
    m, g = qs.healthy(asm.n, 3)
    m[1, 2], g[1, 2] = WEAK, 0.5
    r_rows = qs.solve(pl0, m, g, pl0.ik(xis), het_rows=[np.stack([d[i] for d in draws]) for i in range(3)])
    for b, (s, l, mu) in enumerate(draws):
        pl = qs.Plant(asm, P, theta, stall=s, noload=l, mu_f=mu)
        r = qs.solve(pl, m[b:b + 1], g[b:b + 1], pl.ik(xis))
        np.testing.assert_allclose(r_rows["xi"][b], r["xi"][0], atol=1e-8)


def test_per_wheel_commands(P, theta):
    asm = A.physical("square4", P)
    pl = qs.Plant(asm, P, theta)
    # IK references reproduce the twist command path
    xis = qs.decision_legs(asm)[:2]
    a = qs.solve(pl, *qs.healthy(asm.n), pl.ik(xis))
    np.testing.assert_allclose(a["om"][0], pl.ik(xis), atol=1e-6)
    # tug-of-war: body stays near rest, commanded wheels saturate the duty clamp or slip
    legs = PS.probe_legs(pl, "tug")
    om, dc = stagea.legs_arrays(legs)
    r = qs.solve(pl, *qs.healthy(asm.n), om, dc)
    assert r["resid"].max() < 1e-6
    assert np.abs(r["xi"][0, :, :2]).max() < 0.05
    active = np.abs(om) > 0
    assert np.abs(r["duty"][0][active]).min() > 0.5
    # open-loop duty command is honoured
    dcmd = np.full((1, asm.n), np.nan)
    dcmd[0, 0] = 0.4
    r = qs.solve(pl, *qs.healthy(asm.n), np.zeros((1, asm.n)), dcmd)
    assert abs(r["duty"][0, 0, 0] - 0.4) < 1e-12


def test_children_of_square(P):
    sq = A.physical("square4", P)
    ch = C.children(sq, P)
    sizes = sorted(c.size for c in ch)
    assert sizes == [1] * 4 + [2] * 4 + [3] * 4                     # diagonal pairs are not connected
    pairs = {c.modules for c in ch if c.size == 2}
    assert pairs == {(0, 1), (0, 2), (1, 3), (2, 3)}               # square4 cells (0,0),(1,0),(0,1),(1,1)
    for c in ch:
        assert [sq.modules[k] for k in c.modules] == list(c.asm.modules)          # module identity carried
        np.testing.assert_array_equal(sq.wheel_module[c.src] , np.repeat(c.modules, 4))
        assert [sq.wheel_corner[i] for i in c.src] == c.asm.wheel_corner
    assert len(C.children(sq, P, include_full=True)) == 13


def test_merge_and_carry(P):
    pair = A.physical("pair", P)
    single = A.physical("single", P)
    pl = C.merge_placements(C.cells_of(pair), C.cells_of(single))
    assert len(pl) == 6                                             # 2x1 bar: 6 face-adjacent free cells
    c = C.merge(pair, (0, 1), single, pl[0], P)
    assert c.size == 3 and c.n_other == 1
    modes_p = np.arange(8)[None]
    modes_o = np.array([[100, 101, 102, 103]])
    np.testing.assert_array_equal(C.carry(modes_p, modes_o, c.src), [[0, 1, 2, 3, 4, 5, 6, 7, 100, 101, 102, 103]])


def test_child_normal_loads_recomputed(P, theta):
    sq = A.physical("square4", P)
    ch = [c for c in C.children(sq, P) if c.size == 1][0]
    pl = qs.Plant(sq, P, theta)
    sub = qs.sub_plant(pl, ch.asm, ch.src)
    assert abs(sub.N.sum() - ch.asm.mass * 9.81) < 1e-6
    assert abs(sub.N.sum() - pl.N.sum() / 4) < 1e-6


def test_hypothesis_space_size():
    m, g = stagea.hypotheses(4)
    S = len(stagea.FAULT_STATES)
    assert len(m) == 1 + 4 * S + 6 * S * S
    assert len(np.unique(np.concatenate([m, np.round(g * 100)], 1), axis=0)) == len(m)


@pytest.mark.parametrize("het", [False, True])
def test_consistent_set_contains_truth(P, theta, het):
    rng = np.random.default_rng(7 + het)
    asm = A.physical("L3", P)
    st, nl, mu = O.sample_het(rng, asm, het)
    pl = qs.Plant(asm, P, theta, stall=st, noload=nl, mu_f=mu)
    HM, HG = stagea.hypotheses(asm.n, 1)
    legs = PS.probe_legs(pl, "v2")
    om, dc = stagea.legs_arrays(legs)
    sig_h = O.signals(qs.solve(pl, HM, HG, om, dc))
    nuis = O.nuisance_sd(pl, legs, rng, 4) if het else None
    for _ in range(5):
        ti = int(rng.integers(len(HM)))
        ts = sig_h[ti]
        for p in O.PRESETS:
            ok = O.consistent(sig_h, ts, O.sigma(ts, p, het, nuis))
            assert ok[ti]
            assert ok.sum() < len(HM)


@pytest.mark.parametrize("het", [0, 1])
def test_episode_truth_in_set_and_oracle_safe(P, theta, het):
    rec, rows = stagea.run_episode(11, P, theta, sizes=(2, 3), n_merge=2, het=het, nuis_M=3, prior_M=2)
    assert rec["n_union"] >= 1
    known = [r for r in rows if r["rho_model"] == "known"]
    assert sum(r["n_unsafe"] for r in known) == 0                  # truth in the set -> oracle never unsafe
    for r in rows:
        assert r["n_amb"] + r["n_go"] <= r["n_dec"]
        assert r["n_go_cap"] <= min(r["n_go"], r["n_cap"])
    sets = {r["probe_set"] for r in rows}
    assert {"v2", "v2+tug", "diag_split"} <= sets


def test_decision_counts_scorer():
    rho_h = np.array([[0.9, 0.5, 0.75], [0.8, 0.65, 0.95]])
    rt = np.array([0.8, 0.65, 0.95])
    m = np.array([True, True])
    # tau 0.7: leg0 go (both >=), leg1 no-go (both <), leg2 ambiguous (0.75 / 0.95 -> min 0.75 >= 0.7 -> go)
    np.testing.assert_array_equal(stagea.decision_counts(rho_h, m, rt, 0.7), [3, 0, 2, 2, 0, 2])
    # tau 0.8: leg0 go (0.9, 0.8), leg1 no-go, leg2 ambiguous (0.75 < 0.8 <= 0.95)
    np.testing.assert_array_equal(stagea.decision_counts(rho_h, m, rt, 0.8), [3, 1, 1, 2, 0, 1])
    # boundary: rho == tau counts as capable / go
    np.testing.assert_array_equal(stagea.decision_counts(np.array([[0.7]]), np.array([True]), np.array([0.7]), 0.7),
                                  [1, 0, 1, 1, 0, 1])
    # a set that excludes the truth can be unsafe
    np.testing.assert_array_equal(stagea.decision_counts(rho_h, np.array([True, False]), rt, 0.85),
                                  [3, 0, 1, 1, 1, 0])
    # empty set refuses: no go, everything ambiguous
    np.testing.assert_array_equal(stagea.decision_counts(rho_h, np.array([False, False]), rt, 0.7), [3, 3, 0, 2, 0, 0])
    # nuisance draws axis (M, H, L)
    r3 = np.stack([rho_h, rho_h - 0.2])
    np.testing.assert_array_equal(stagea.decision_counts(r3, m, rt, 0.7), [3, 2, 0, 2, 0, 0])
