"""T2 capability labels and probe observations: the cfcap.t1.capability interface on the T2 stress plant.

EVALUATION / ADVERSARIAL SEARCH ONLY: T2 labels must never enter training or calibration.

    out = capability(targets, params, effects=build.ALL, relative=True, backend="numba")
    # targets: list of (capxfer Assembly, T1 latents) as for cfcap.t1.capability.capability
    # out[i]: rho (B_i, L), drift (B_i, L) [+ rho_healthy, rho_rel] as numpy arrays

Same decision legs, leg tables (cfcap.t1.capability.leg_tables), steady window and rho definition (capxfer
targets_from_twist: rho = <xi, xi_cmd>_W / |xi_cmd|^2_W over the window, W = diag(1, 1, Lc^2)), records at the end of
every 50 Hz tick labelled t = j / 50 s as capxfer's integrator records. Targets of equal wheel count run as one batch.

`simulate` + `observe` give the robot-visible channels of arbitrary command tables (the probe-observation
interface, as cfcap.t1.plant.simulate_bucket + observe).
"""
from __future__ import annotations

import math

import numpy as np

from cfcap.t2 import build, kernel
from cfcap.t1 import capability as cap1

LEGS, T_LEG = cap1.LEGS, cap1.T_LEG
DT_OBS = 0.02


def rho_from_twist(twist, t, tw, T, Lc):
    """(B, L) rho and drift per leg; twist (B, L, Tn, 3), tw (L, 3)."""
    a, b = cap1.window(T)
    m = (t >= a) & (t <= b)
    xr = twist[:, :, m].mean(2)
    w = np.array([1.0, 1.0, Lc * Lc])
    nc = (w * tw * tw).sum(-1)
    r = (w * xr * tw).sum(-1) / nc
    res = xr - r[..., None] * tw
    return r, np.sqrt((w * res * res).sum(-1) / nc)


def _n_ticks(T, params):
    return int(math.ceil(T * float(params["firmware"]["cmd_rate_hz"]) - 1e-6))


def capability(targets, params, effects=build.NONE, legs=LEGS, T=T_LEG, relative=False, backend="numba",
               seed=0, dtype="float32"):
    """See module docstring. seed: effect-draw seed (target i uses seed + 1000003 i)."""
    L = len(legs)
    K = _n_ticks(T, params)
    jobs = []                                            # (target idx, healthy?, spec, tables)
    for i, (asm, lat) in enumerate(targets):
        tabs, tw = cap1.leg_tables(asm, params, legs, T)
        for healthy in ((False, True) if relative else (False,)):
            S = build.episode_specs(asm, lat, params, effects, seed + 1000003 * i, clear_faults=healthy)
            jobs.append((i, healthy, build.repeat(S, L), tabs, tw))
    by_w = {}
    for q, j in enumerate(jobs):
        by_w.setdefault(j[2]["px0"].shape[1], []).append(q)
    tw_all = {}
    res = {}
    for W, qs in by_w.items():
        S = build.concat([jobs[q][2] for q in qs])
        tables = np.concatenate([jobs[q][3] for q in qs], 0).astype(np.float64)
        which = np.concatenate([np.tile(np.arange(L), jobs[q][2]["px0"].shape[0] // L) + L * k
                                for k, q in enumerate(qs)])
        out = kernel.run(S, tables, which, K, backend, dtype)
        o = 0
        for q in qs:
            Bq = jobs[q][2]["px0"].shape[0]
            res[q] = out["twist"][o:o + Bq]
            o += Bq
    t = np.arange(K) * DT_OBS
    outs = []
    for i, (asm, lat) in enumerate(targets):
        qs = [q for q, j in enumerate(jobs) if j[0] == i]
        tw = jobs[qs[0]][4]
        Lc = asm.char_length()
        tws = res[qs[0]].reshape(-1, L, K, 3)
        rho, drift = rho_from_twist(tws, t, tw, T, Lc)
        o = dict(rho=rho, drift=drift)
        if relative:
            rh, _ = rho_from_twist(res[qs[1]].reshape(-1, L, K, 3), t, tw, T, Lc)
            o.update(rho_healthy=rh, rho_rel=rho / rh)
        outs.append(o)
    return outs


def simulate(asm, lat, params, cmd, T, effects=build.NONE, backend="numba", seed=0, dtype="float32"):
    """Command tables cmd (B, K+1, 3) or (K+1, 3) at 50 Hz for duration T -> channels (B, K, .) at the end of every
    tick: twist, pose, duty (pre-fault firmware duty), speed (true wheel speed), angle, gz, ax, ay, plus t."""
    S = build.episode_specs(asm, lat, params, effects, seed)
    B = S["px0"].shape[0]
    cmd = np.asarray(cmd, np.float64)
    if cmd.ndim == 2:
        tables, which = cmd[None], np.zeros(B, np.int64)
    else:
        tables, which = cmd, np.arange(B)
    K = _n_ticks(T, params)
    out = kernel.run(S, tables, which, K, backend, dtype)
    out["t"] = np.arange(K) * DT_OBS
    return out


def observe(R, lat, seed=0, cpr=None):
    """Robot-visible channels on the 50 Hz grid, as cfcap.t1.plant.observe: pre_duty, speed from quantised encoder
    count differences, gz/ax/ay with the IMU noise and the latents' biases, wheel-sample dropout (hold)."""
    from cfcap.t1 import params as P
    from cfcap.t1.plant import IMU_NOISE
    cpr = P.CPR if cpr is None else cpr
    rng = np.random.default_rng(seed)
    cnt = np.floor(R["angle"] / (2 * math.pi) * cpr)
    spd = np.zeros_like(cnt)
    spd[:, 1:] = np.diff(cnt, axis=1) / cpr * 2 * math.pi / DT_OBS
    duty = R["duty"].copy()
    B, T, n = duty.shape
    dp = float(lat.get("drop_p", 0.0) or 0.0)
    if dp > 0:
        drop = rng.uniform(size=(B, T, n)) < dp
        drop[:, 0] = False
        idx = np.where(drop, 0, np.arange(T)[None, :, None])
        idx = np.maximum.accumulate(idx, axis=1)
        duty, spd = np.take_along_axis(duty, idx, 1), np.take_along_axis(spd, idx, 1)
    gb = build._np(lat["gyro_bias"])[:, None]
    ab = build._np(lat["acc_bias"])
    return dict(pre_duty=duty, speed=spd, gz=R["gz"] + IMU_NOISE[0] * rng.normal(size=(B, T)) + gb,
                ax=R["ax"] + IMU_NOISE[1] * rng.normal(size=(B, T)) + ab[:, 0:1],
                ay=R["ay"] + IMU_NOISE[2] * rng.normal(size=(B, T)) + ab[:, 1:2])
