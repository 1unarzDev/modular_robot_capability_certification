"""T2 stress plant tests: zero added effects reproduce crane's kernel; a healthy single has rho_rel = 1."""
import numpy as np
import pytest

from cfcap.t2 import build, kernel, capability as cap2
from cfcap.t1 import params as P
from cfcap.t1 import plant
from engine.kernels import mecanum as K
from engine.kernels import mecanum_run as KR


@pytest.fixture(scope="module")
def setup():
    par = P.base_params("plated")
    th = P.load_theta("plated")
    return par, th, P.theta_dr(th)


def _latents(asm, B, seed, dr, het=plant.HET_FULL, faults=True):
    rng = np.random.default_rng(seed)
    f = plant.random_faults(B, asm.n, rng) if faults else None
    return plant.sample_latents(asm, B, rng, het, dr=dr, faults=f, device="cpu")


def _crane_spec(asm, lat, par):
    S = build.episode_specs(asm, lat, par, build.Effects(firmware="crane"), seed=1)
    for k in ("phimax",):                                    # crane's own firmware clip active
        wp = S["p"][:, K.G:].reshape(S["p"].shape[0], -1, K.Q)
        wp[..., K.IW[k]] = 12.0
    return S


def test_step_cmd_is_mecanum_step(setup):
    par, th, dr = setup
    asm = P.named_assembly("pair", par)
    lat = _latents(asm, 5, 0, dr)
    S = _crane_spec(asm, lat, par)
    p = S["p"]
    rng = np.random.default_rng(0)
    x = K.init(p)
    x[:, 3:6] = rng.normal(0, 0.2, (p.shape[0], 3))
    u = rng.normal(0, 0.3, (p.shape[0], 3))
    ref = K.step(x, u, p, 0.005)
    got, _ = kernel.step_cmd(x, K.firmware(K._wheels(p, np), u), p, 0.005)
    assert np.array_equal(ref, got)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("name", ["single", "pair", "square"])
def test_zero_effects_reproduce_crane_replay(setup, backend, name):
    """With no added effects and crane's drive, T2 is mecanum_run.replay of the same rows (same backend)."""
    par, th, dr = setup
    asm = P.named_assembly(name, par)
    lat = _latents(asm, 6, 2, dr)
    S = _crane_spec(asm, lat, par)
    tabs, _ = __import__("cfcap.t1.capability", fromlist=["x"]).leg_tables(asm, par)
    tables = tabs.astype(np.float64)
    which = np.arange(6) % tables.shape[0]
    T = 60
    got = kernel.run(S, tables, which, T, backend)
    ref = KR.replay(S["p"], tables, which=which, delay=S["delay"], backend=backend, ticks=T)
    for c in ("twist", "pose", "duty", "speed"):
        assert np.allclose(got[c], ref[c], rtol=0, atol=1e-12), c
    assert np.allclose(got["gz"], ref["gz"], atol=1e-12)


def test_backends_agree_with_all_effects(setup):
    par, th, dr = setup
    asm = P.named_assembly("L3", par)
    lat = _latents(asm, 3, 4, dr)
    a = cap2.capability([(asm, lat)], par, build.ALL, relative=True, backend="numpy")[0]
    b = cap2.capability([(asm, lat)], par, build.ALL, relative=True, backend="numba")[0]
    assert np.allclose(a["rho"], b["rho"], atol=1e-10)
    assert np.allclose(a["rho_rel"], b["rho_rel"], atol=1e-10)


def test_healthy_single_capability(setup):
    par, th, dr = setup
    s = P.named_assembly("single", par)
    lat = plant.sample_latents(s, 4, np.random.default_rng(0), plant.HET_OFF, domain=P.point_domain(th, 4, "cpu"),
                               device="cpu")
    for eff in (build.NONE, build.ALL):
        o = cap2.capability([(s, lat)], par, eff, relative=True)[0]
        assert np.allclose(o["rho_rel"], 1.0, atol=1e-9)
        ss = [cap2.LEGS.index(k) for k in ("surge+", "surge-", "sway+", "sway-")]
        assert np.all((o["rho"][:, ss] > 0.85) & (o["rho"][:, ss] < 1.05))
    o10 = cap2.capability([(s, lat)], par, T=10.0)[0]["rho"]
    assert np.all(np.abs(o10[:, ss] - 1.0) < 0.06)


def test_single_has_no_compliance(setup):
    """A single module has no latched face: the compliance effect is a no-op."""
    par, th, dr = setup
    s = P.named_assembly("single", par)
    lat = _latents(s, 3, 5, dr)
    a = cap2.capability([(s, lat)], par, build.NONE)[0]["rho"]
    b = cap2.capability([(s, lat)], par, build.Effects(compliance=True))[0]["rho"]
    assert np.array_equal(a, b)


def test_weak_wheel_lowers_relative_capability(setup):
    par, th, dr = setup
    s = P.named_assembly("single", par)
    lat = plant.sample_latents(s, 1, np.random.default_rng(0), plant.HET_OFF, domain=P.point_domain(th, 1, "cpu"),
                               device="cpu", faults=(np.array([[1, 0, 0, 0]]), np.array([[0.3, 1, 1, 1]], np.float32)))
    o = cap2.capability([(s, lat)], par, build.ALL, relative=True)[0]
    assert o["rho_rel"].min() < 0.9


def test_observe_shapes(setup):
    par, th, dr = setup
    s = P.named_assembly("pair", par)
    lat = _latents(s, 2, 6, dr)
    tab, T = plant.v2_probe(s, par, 2, np.random.default_rng(0))
    R = cap2.simulate(s, lat, par, tab, T, build.ALL)
    obs = cap2.observe(R, lat)
    assert obs["pre_duty"].shape == (2, R["t"].shape[0], 8) and np.isfinite(obs["gz"]).all()


def test_eval_simulate_episode():
    """t2_eval simulate on one generator episode: labels for every candidate, unfaulted candidates have rho_rel = 1."""
    from cfcap.gen import episode as E
    from cfcap.t2 import eval as ev
    ep = E.make_episode("plated", "test", 0)
    lab, faulted, ob = ev.simulate_episode(ep, ("all",), obs=True)
    rr = lab["all"]["rho_rel"]
    assert rr.shape == (len(ep.cands), 10) and np.isfinite(rr).all()
    assert np.all(rr[~faulted] == 1.0)
    assert ob["duty"].shape[1] == ep.parent.asm.n and ob["imu"].shape[1] == 3


def test_adv_space_and_score(setup):
    """t2_adv: decoding z gives valid latents (0-2 faults, payload) and the score is finite."""
    par, th, dr = setup
    from cfcap.t2 import adv
    asm = P.named_assembly("pair", par)
    sp = adv.Space(asm, dr)
    z = np.random.default_rng(0).uniform(size=(6, sp.d))
    refs = adv._refs(asm, par, "plated", "numba")
    sc, leg, tau, r1, r2, lat = adv.score_members(sp, z, par, refs, 1, "numba")
    assert np.isfinite(sc).all() and r1.shape == r2.shape == (6, 10)
    assert ((lat["mode"].numpy() > 0).sum(1) <= 2).all()


def test_motor_mode_backends_agree_and_is_off_by_default(setup):
    par, th, dr = setup
    asm = P.named_assembly("pair", par)
    lat = _latents(asm, 3, 7, dr)
    assert not build.ALL.motor and not build.NONE.motor
    a = cap2.capability([(asm, lat)], par, build.ABLATIONS["all+motor"], relative=True, backend="numpy")[0]
    b = cap2.capability([(asm, lat)], par, build.ABLATIONS["all+motor"], relative=True, backend="numba")[0]
    c = cap2.capability([(asm, lat)], par, build.ALL, relative=True, backend="numba")[0]
    assert np.allclose(a["rho"], b["rho"], atol=1e-10)
    assert np.abs(b["rho"] - c["rho"]).max() > 1e-3          # the motor law changes the plant


@pytest.mark.parametrize("sev", [1, 3])
def test_knee_law_backends_agree(setup, sev):
    """motor_kind='knee' (theory's cfcap/t0/envelope.py law): numpy (calls envelope.knee_factors) == numba."""
    par, th, dr = setup
    asm = P.named_assembly("square", par)
    lat = _latents(asm, 2, 8, dr)
    e = build.ABLATIONS[f"knee{sev}"]
    assert e.motor_on and not e.motor
    a = cap2.capability([(asm, lat)], par, e, backend="numpy")[0]["rho"]
    b = cap2.capability([(asm, lat)], par, e, backend="numba")[0]["rho"]
    c = cap2.capability([(asm, lat)], par, build.NONE, backend="numba")[0]["rho"]
    assert np.allclose(a, b, atol=1e-10) and np.abs(b - c).max() > 1e-4
