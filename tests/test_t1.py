"""T1 plant and capability-label tests (GPU)."""
import numpy as np
import pytest
import torch

from cfcap.t1 import capability as cap
from cfcap.t1 import params as P
from cfcap.t1 import plant
from capxfer import probes, sim

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def setup():
    par = P.base_params("plated")
    th = P.load_theta("plated")
    return par, th, P.theta_dr(th)


def _clone(d):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}


@pytest.mark.parametrize("name", ["single", "pair", "square"])
def test_het_off_reproduces_capxfer(setup, name):
    par, th, dr = setup
    asm = P.named_assembly(name, par)
    rng = np.random.default_rng(1)
    B = 12
    dom = sim.sample_domain(B, rng, dr, device="cuda")
    mode, g = plant.random_faults(B, asm.n, rng)
    sched = probes.Schedule(asm.char_length(), pids=("P1", "P2", "P5"))
    R0 = sim.simulate(asm, par, _clone(dom), torch.tensor(mode), torch.tensor(g), schedule=sched, record_hz=50,
                      seed=5)
    lat = plant.sample_latents(asm, B, rng, plant.HET_OFF, domain=_clone(dom), faults=(mode, g))
    lat["yaw_err"] = None                                  # use capxfer's own mounting-yaw draw (same seed)
    tab = plant.schedule_ticks(sched, par)
    R1 = plant.simulate_bucket([dict(asm=asm, lat=lat, cmd=tab)], par, sim.probe_duration(sched), seed=5)[0]
    for k in ("duty", "twist", "wheel_angle", "torque"):
        assert torch.allclose(R0[k], R1[k], atol=1e-5, rtol=0), k
    # the per-wheel step with empty (NaN) tables is the same physics
    R2 = plant.simulate_bucket([dict(asm=asm, lat=lat, cmd=tab, pw_ref=np.full((tab.shape[0], asm.n), np.nan,
                                                                              np.float32))],
                               par, sim.probe_duration(sched), seed=5)[0]
    assert torch.allclose(R0["twist"], R2["twist"], atol=1e-5, rtol=0)


def test_mixed_buckets_match_single_runs(setup):
    par, th, dr = setup
    rng = np.random.default_rng(2)
    targets = []
    for name in ("single", "pair", "single"):
        asm = P.named_assembly(name, par)
        targets.append((asm, plant.sample_latents(asm, 3, rng, plant.HET_FULL, dr=dr)))
    mixed = cap.capability(targets, par)
    for (asm, lat), o in zip(targets, mixed):
        alone = cap.capability([(asm, lat)], par)[0]
        assert torch.allclose(o["rho"], alone["rho"], atol=1e-5)


def test_healthy_single_capability(setup):
    par, th, dr = setup
    s = P.named_assembly("single", par)
    lat = plant.sample_latents(s, 4, np.random.default_rng(0), plant.HET_OFF, domain=P.point_domain(th, 4, "cuda"))
    o = cap.capability([(s, lat)], par, relative=True)[0]
    rho = o["rho"].cpu().numpy()
    surge_sway = [cap.LEGS.index(k) for k in ("surge+", "surge-", "sway+", "sway-")]
    # 3 s legs: the measured firmware's slow integral leaves a healthy module at ~0.91 (see capability docstring)
    assert np.all((rho[:, surge_sway] > 0.85) & (rho[:, surge_sway] < 1.05))
    assert np.allclose(o["rho_rel"].cpu().numpy(), 1.0, atol=1e-6)
    # long legs converge to ~1
    o10 = cap.capability([(s, lat)], par, T=10.0)[0]["rho"].cpu().numpy()
    assert np.all(np.abs(o10[:, surge_sway] - 1.0) < 0.05)


def test_weak_wheel_lowers_relative_capability(setup):
    par, th, dr = setup
    s = P.named_assembly("single", par)
    mode = np.zeros((1, 4), np.int64)
    g = np.ones((1, 4), np.float32)
    mode[0, 0], g[0, 0] = plant.WEAK, 0.4
    lat = plant.sample_latents(s, 1, np.random.default_rng(0), plant.HET_OFF, domain=P.point_domain(th, 1, "cuda"),
                               faults=(mode, g))
    rel = cap.capability([(s, lat)], par, relative=True)[0]["rho_rel"].cpu().numpy()[0]
    assert rel.min() < 0.9
    # fl ("/" axis) needs no spin on the (+,+) diagonal: that leg is unaffected (capxfer A.md physics note)
    assert abs(rel[cap.LEGS.index("diag++")] - 1.0) < 0.03


def test_per_wheel_speed_leg(setup):
    """Per-wheel reference = the IK of a surge twist reproduces the twist command; a tug-of-war (front wheels forward,
    rear wheels backward) produces ~no net surge."""
    par, th, dr = setup
    s = P.named_assembly("single", par)
    lat = plant.sample_latents(s, 2, np.random.default_rng(0), plant.HET_OFF, domain=P.point_domain(th, 2, "cuda"))
    tabs, tw = cap.leg_tables(s, par, legs=("surge+",))
    ref = s.ik(tabs[0], 0.0501).astype(np.float32)                    # (K+1, n) rad/s
    zero = np.zeros_like(tabs[0])
    R_cmd = plant.simulate_bucket([dict(asm=s, lat=lat, cmd=tabs[0])], par, cap.T_LEG)[0]
    R_pw = plant.simulate_bucket([dict(asm=s, lat=lat, cmd=zero, pw_ref=ref)], par, cap.T_LEG)[0]
    assert torch.allclose(R_cmd["twist"], R_pw["twist"], atol=1e-4)
    tug = ref.copy()
    tug[:, 2:] *= -1                                                   # rear wheels (rl, rr) reversed
    R_tug = plant.simulate_bucket([dict(asm=s, lat=lat, cmd=zero, pw_ref=tug)], par, cap.T_LEG)[0]
    assert R_tug["twist"][:, -50:, 0].abs().mean() < 0.05
    # duty leg: open-loop duty 0.5 on all wheels drives forward, and is what the firmware reports
    duty = np.where(np.arange(ref.shape[0])[:, None] > 10, 0.5 * np.sign(ref[-2]), np.nan).astype(np.float32)
    duty = np.broadcast_to(duty, ref.shape).copy()
    R_d = plant.simulate_bucket([dict(asm=s, lat=lat, cmd=zero, pw_duty=duty)], par, cap.T_LEG)[0]
    assert torch.allclose(R_d["duty"][:, -10:], torch.tensor(duty[-12], device="cuda").expand(2, 10, 4), atol=1e-6)
    assert R_d["twist"][:, -50:, 0].mean() > 0.1


def test_heterogeneity_and_observe(setup):
    par, th, dr = setup
    asm = P.named_assembly("pair", par)
    rng = np.random.default_rng(3)
    lat = plant.sample_latents(asm, 64, rng, plant.HET_FULL, dr=dr)
    sw = lat["stall_w"].cpu().numpy()
    assert sw.min() >= 0.88 and sw.max() <= 1.12 and sw.std() > 0.02
    tab, T = plant.v2_probe(asm, par, 64, rng)
    R = plant.simulate_bucket([dict(asm=asm, lat=lat, cmd=tab)], par, T)[0]
    obs = plant.observe(R, lat, seed=1)
    assert obs["pre_duty"].shape == R["duty"].shape and torch.isfinite(obs["speed"]).all()
    # subset -> child keeps the module's wheels
    child = plant.subset(lat, [1])
    assert torch.equal(child["stall_w"], lat["stall_w"][:, 4:8])
