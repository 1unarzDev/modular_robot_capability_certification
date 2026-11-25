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


def probe_misfit(asm, lat, par, effects, seed, kinds=("v2", "clamp"), rng=None, backend="numba"):
    """{kind: (D_imu, D_duty)} of one or more episodes (rows of lat) on `asm` (T1 torch CPU vs T2)."""
    from cfcap.gen import simrun as SR
    from cfcap.t1 import plant
    from cfcap.t2 import capability as cap2
    rng = rng or np.random.default_rng(seed)
    B = lat["mode"].shape[0]
    out = {}
    lt = {k: v for k, v in lat.items() if not k.startswith("t2_")}
    for kind in kinds:
        tabs = np.stack([SR.probe_tables(kind, asm, par, rng)[0] for _ in range(B)])
        T = SR.probe_T(kind)
        R1 = plant.simulate_bucket([dict(asm=asm, lat=lt, cmd=tabs)], par, T, device="cpu")[0]
        R2 = cap2.simulate(asm, lat, par, tabs, T, effects, backend=backend, seed=seed)
        out[kind] = misfit(R1["twist"].numpy().astype(float), R2["twist"], R1["duty"].numpy().astype(float),
                           R2["duty"])
    return out
