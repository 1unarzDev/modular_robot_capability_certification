"""Probe misfit of T2 against T1 with the latents known (oracle misfit; an upper bound on what a T1-model check can
detect). IMU from the 50 Hz twist the same way for both plants (gz = w, a = dv/dt + w x v); duty with a 0.01 floor.
D = mean over samples and channels of ((T2 - T1) / sigma)^2, sigma = capxfer IMU noise (0.002, 0.02, 0.02).
Same statistic as scripts/stress_A5_clamp.py."""
from __future__ import annotations

import numpy as np

SIG = np.array([0.002, 0.02, 0.02])
SIG_DUTY = 0.01


def imu_from_twist(tw):
    dv = np.zeros_like(tw)
    dv[:, 1:] = np.diff(tw, axis=1) / 0.02
    ax = dv[..., 0] - tw[..., 2] * tw[..., 1]
    ay = dv[..., 1] + tw[..., 2] * tw[..., 0]
    return np.stack([tw[..., 2], ax, ay], -1)


def misfit(tw1, tw2, d1, d2):
    T = min(tw1.shape[1], tw2.shape[1])
    di = ((imu_from_twist(tw1[:, :T]) - imu_from_twist(tw2[:, :T])) / SIG) ** 2
    dd = ((d1[:, :T] - d2[:, :T]) / SIG_DUTY) ** 2
    return di.reshape(len(tw1), -1).mean(1), dd.reshape(len(tw1), -1).mean(1)


YAWC_T = 0.5 + 0.3 + 1.5 + 0.6 + 1.5 + 0.3 + 1.0


def yaw_rate_table(asm, par, rng):
    """Child-rate yaw probe `yawc`: yaw at the child's decision-leg rate (t1.capability.leg_twists: the farthest
    wheel at WHEEL_SPEED_CAP): lead U(0, 0.5) s, ramp 0.3 s to +w, hold 1.5 s, ramp 0.6 s to -w, hold 1.5 s, ramp 0.3 s
    to 0, 1 s tail. Returns (K+1, 3) at 50 Hz, the length of YAWC_T."""
    from cfcap.t1 import capability as cap1, plant
    hz = float(par["firmware"]["cmd_rate_hz"])
    K = plant.n_ticks(YAWC_T, par)
    w = float(cap1.leg_twists(asm)[cap1.LEGS.index("yaw+")][2])
    t = np.arange(K + 1) / hz - float(rng.uniform(0.0, 0.5))
    knots = np.array([0.0, 0.3, 1.8, 2.4, 3.9, 4.2])
    vals = np.array([0.0, w, w, -w, -w, 0.0])
    wz = np.interp(t, knots, vals, left=0.0, right=0.0)
    tab = np.zeros((K + 1, 3), np.float32)
    tab[:, 2] = wz
    tab[K:] = 0.0
    return tab


WHEELRATE_T = 0.5 + 4 * (0.4 + 1.0 + 0.3) + 0.5


def wheelrate_table(asm, par, rng):
    """Feasible parent probe `wheelrate` (N5): body-twist command 0; every module runs its twist-free wheel mode (the
    null vector n of its IK map, J_m^T n = 0: the wheels oppose each other in antagonistic pairs, so no net force or
    moment; for the standard X layout n = (+1, -1, -1, +1) on fl, fr, rl, rr), scaled so each module's fastest wheel
    reaches the largest |omega| its wheels need on the child's yaw leg (IK of t1.capability.leg_twists yaw: the
    Prop. 2b closed form). Lead U(0, 0.2) s, 0.5 s settle, legs +, -, +, - of 0.4 s ramp + 1.0 s hold + 0.3 s ramp
    down, 0.5 s tail. Returns (cmd (K+1, 3) zeros, pw_ref (K+1, n) rad/s, 0 outside the legs), at 50 Hz."""
    from cfcap.t1 import capability as cap1, plant
    hz = float(par["firmware"]["cmd_rate_hz"])
    K = plant.n_ticks(WHEELRATE_T, par)
    tw = cap1.leg_twists(asm)[cap1.LEGS.index("yaw+")]
    pos, w_hat, r_hat = np.asarray(asm.wheel_pos), np.asarray(asm.w_hat), np.asarray(asm.r_hat)
    r = float(par["wheel"]["radius_m"])
    den = r * (w_hat * r_hat).sum(-1)
    vc = np.stack([tw[0] - tw[2] * pos[:, 1], tw[1] + tw[2] * pos[:, 0]], -1)
    om = np.abs((r_hat * vc).sum(-1) / den)                                       # |omega_j| on the yaw leg
    wm = np.asarray(asm.wheel_module)
    mode = np.zeros(asm.n)
    for m in np.unique(wm):
        j = np.nonzero(wm == m)[0]
        J = np.stack([r_hat[j, 0], r_hat[j, 1], pos[j, 0] * r_hat[j, 1] - pos[j, 1] * r_hat[j, 0]], 1) / den[j, None]
        nv = np.linalg.svd(J.T)[2][-1]                                            # J^T n = 0
        nv = nv / np.abs(nv).max()
        mode[j] = nv * om[j].max()
    lead = float(rng.uniform(0.0, 0.2))
    t = np.arange(K + 1) / hz - lead
    ref = np.zeros((K + 1, asm.n))
    t0 = 0.5
    for sg in (+1.0, -1.0, +1.0, -1.0):
        prof = np.interp(t, [t0, t0 + 0.4, t0 + 1.4, t0 + 1.7], [0.0, 1.0, 1.0, 0.0], left=0.0, right=0.0)
        ref += (sg * prof)[:, None] * mode[None]
        t0 += 1.7
    ref[K:] = 0.0
    return np.zeros((K + 1, 3), np.float32), ref.astype(np.float32)


def _tables(kind, asm, par, rng):
    """(twist table, T, per-wheel table or None)."""
    from cfcap.gen import simrun as SR
    if kind == "yawc":
        return yaw_rate_table(asm, par, rng), YAWC_T, None
    if kind == "wheelrate":
        c, ref = wheelrate_table(asm, par, rng)
        return c, WHEELRATE_T, ref
    return SR.probe_tables(kind, asm, par, rng)[0], SR.probe_T(kind), None


def probe_misfit(asm, lat, par, effects, seed, kinds=("v2", "clamp", "yawc"), rng=None, backend="numba"):
    """{kind: (D_imu, D_duty)} of one or more episodes (rows of lat) on `asm` (T1 torch CPU vs T2)."""
    from cfcap.gen import simrun as SR
    from cfcap.t1 import plant
    from cfcap.t2 import capability as cap2
    rng = rng or np.random.default_rng(seed)
    B = lat["mode"].shape[0]
    out = {}
    lt = {k: v for k, v in lat.items() if not k.startswith("t2_")}
    for kind in kinds:
        tt = [_tables(kind, asm, par, rng) for _ in range(B)]
        tabs, T = np.stack([x[0] for x in tt]), tt[0][1]
        pw = np.stack([x[2] for x in tt]) if tt[0][2] is not None else None
        R1 = plant.simulate_bucket([dict(asm=asm, lat=lt, cmd=tabs, pw_ref=pw)], par, T, device="cpu")[0]
        R2 = cap2.simulate(asm, lat, par, tabs, T, effects, backend=backend, seed=seed, pw_ref=pw)
        out[kind] = misfit(R1["twist"].numpy().astype(float), R2["twist"], R1["duty"].numpy().astype(float),
                           R2["duty"])
        if kind == "yawc":
            # tracking statistic (post-hoc, A5y): |yaw ratio measured in T2 - T1's prediction| over the two holds
            Tn = min(R1["twist"].shape[1], R2["twist"].shape[1])
            cmdz = tabs[:, 1:Tn + 1, 2]
            hold = (np.abs(cmdz) > 0.999 * np.abs(cmdz).max(1, keepdims=True)) & (np.arange(Tn)[None] % 1 == 0)
            g1 = R1["twist"].numpy()[:, :Tn, 2].astype(float)
            g2 = R2["twist"][:, :Tn, 2]
            ratio = lambda g: (np.where(hold, g * np.sign(cmdz), 0).sum(1) /                         # noqa: E731
                               np.maximum(np.where(hold, np.abs(cmdz), 0).sum(1), 1e-9))
            out["yawc_track"] = np.abs(ratio(g2) - ratio(g1))
    return out
