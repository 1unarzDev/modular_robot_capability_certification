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


def _tables(kind, asm, par, rng):
    from cfcap.gen import simrun as SR
    if kind == "yawc":
        return yaw_rate_table(asm, par, rng), YAWC_T
    return SR.probe_tables(kind, asm, par, rng)[0], SR.probe_T(kind)


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
        R1 = plant.simulate_bucket([dict(asm=asm, lat=lt, cmd=tabs)], par, T, device="cpu")[0]
        R2 = cap2.simulate(asm, lat, par, tabs, T, effects, backend=backend, seed=seed)
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
