"""Noise-thresholded observation model for T0 probes.

Robot-visible QS signals per probe leg (hold-window means):
  gz      IMU yaw rate (true body yaw rate; the only body-motion signal the robot has without lidar);
  om_i    wheel speeds from the encoders (the encoder-odometry twist is a linear function of these, so it adds nothing);
  u_i     per-wheel published duty (PI output before weak scaling; capxfer sim.py convention).
Lidar odometry is not used (only M1 carries a lidar; capxfer marks it unused by the study).

Noise (sigma of a hold-window mean, window 0.6 s), "sensor" preset, from capxfer `sensors.py` + `params/module_measured.yaml`:
  gz : per-acquisition gyro noise 0.004 rad/s x the sensors.py DR max multiplier 2.0, 400 Hz -> 0.008/sqrt(240);
       gyro bias (sensors.py gyro_bias_std 0.01) removed with the 0.5 s settle rest (200 acquisitions) -> residual
       0.008/sqrt(200). Total 0.00077 rad/s.
  om : encoder quantisation, cpr 2048 (params counts_per_output_rev): 2 pi/2048/0.6 s step, two uniform ends ->
       0.0021 rad/s.
  u  : duty jitter from the 50 Hz loop's count-based speed (kp 0.0501 duty per rad/s x 2 pi/2048/0.02 s step,
       two uniform ends), averaged over 30 ticks -> 0.0006.
"floor" preset (ESTIMATE, no corpus hold-window ripple number available): adds 0.02 duty, 0.05 rad/s wheel speed and
0.005 rad/s gz in quadrature, standing in for floor ripple and QS-vs-dynamic residuals.

Nuisance (heterogeneity on): the robot does not know each wheel's motor spread or each module's mu factor. The
consistency test absorbs this by inflating sigma (an inflated-noise approximation to marginalising the nuisance):
  * empirical SD of every signal over healthy-assembly draws of the heterogeneity prior (`nuisance_sd`);
  * a fault-scaled motor term: c |u_i| on duties and c |om_i| on saturated/passive wheel speeds, with
    c = HET_SPREAD / sqrt(3) (uniform spread).
Both hypothesis and truth signals are evaluated on the same (true) continuous plant, so the truth is always consistent
(the 'idealised oracle': uncertainty only over the discrete fault hypothesis, tolerance for the nuisance).
A hypothesis h' is consistent with the true h iff every signal on every leg satisfies |s(h') - s(h)| <= k sigma.
"""
from __future__ import annotations

import math

import numpy as np

HET_SPREAD = 0.08          # per-wheel stall / no-load scale ~ U(1 - 0.08, 1 + 0.08) (PLAN §4 T1: +-8 %)
MU_SPREAD = 0.10           # per-module mu factor ~ U(0.9, 1.1)
WINDOW_S = 0.6
SENSOR = dict(gz=math.hypot(0.008 / math.sqrt(400 * WINDOW_S), 0.008 / math.sqrt(200)),
              om=2 * math.pi / 2048 / WINDOW_S * math.sqrt(2 / 12),
              u=0.0501 * (2 * math.pi / 2048 / 0.02) * math.sqrt(2 / 12) / math.sqrt(WINDOW_S * 50))
FLOOR = dict(gz=0.005, om=0.05, u=0.02)
PRESETS = ("sensor", "floor")


def signals(res):
    """QS solve result -> (B, K, 1 + 2n) signal array [gz, om_1..n, u_1..n]."""
    return np.concatenate([res["xi"][..., 2:3], res["om"], res["duty"]], -1)


def sigma(truth_sig, preset="sensor", het=False, nuis_sd=None):
    """truth_sig (K, 1+2n) -> sigma (K, 1+2n)."""
    K, D = truth_sig.shape
    n = (D - 1) // 2
    base = np.concatenate([[SENSOR["gz"]], np.full(n, SENSOR["om"]), np.full(n, SENSOR["u"])])
    var = np.broadcast_to(base ** 2, (K, D)).copy()
    if preset == "floor":
        fl = np.concatenate([[FLOOR["gz"]], np.full(n, FLOOR["om"]), np.full(n, FLOOR["u"])])
        var += fl ** 2
    if het:
        c = HET_SPREAD / math.sqrt(3)
        u = truth_sig[:, 1 + n:]
        om = truth_sig[:, 1:1 + n]
        var[:, 1 + n:] += (c * u) ** 2
        free = np.abs(u) > 0.999                                        # saturated: speed ceiling scales with spread
        var[:, 1:1 + n] += np.where(free, (c * om) ** 2, 0.0)
        if nuis_sd is not None:
            var += nuis_sd ** 2
    return np.sqrt(var)


def zmax(h_sig, truth_sig, sig):
    """Max over signals and legs of |s(h') - s(h)| / sigma -> (B,)."""
    return (np.abs(h_sig - truth_sig[None]) / sig[None]).reshape(len(h_sig), -1).max(1)


def consistent(h_sig, truth_sig, sig, k=3.0):
    return zmax(h_sig, truth_sig, sig) <= k


def sample_het(rng, asm, on=True):
    """Per-wheel stall / no-load scale and per-wheel mu factor (per-module draw broadcast to its wheels)."""
    n, m = asm.n, len(asm.modules)
    if not on:
        return np.ones(n), np.ones(n), np.ones(n)
    st = rng.uniform(1 - HET_SPREAD, 1 + HET_SPREAD, n)
    nl = rng.uniform(1 - HET_SPREAD, 1 + HET_SPREAD, n)
    mu = rng.uniform(1 - MU_SPREAD, 1 + MU_SPREAD, m)[asm.wheel_module]
    return st, nl, mu


def nuisance_sd(plant, legs, rng, M=8):
    """Empirical SD of the healthy assembly's signals over M heterogeneity draws (batched).
    plant: Plant of the assembly (its own heterogeneity is ignored); legs: list of (name, om_ref, duty_cmd)."""
    from . import qs
    asm = plant.asm
    om_ref = np.stack([l[1] for l in legs])
    dc = np.stack([l[2] for l in legs])
    draws = [sample_het(rng, asm, True) for _ in range(M)]
    hr = [np.stack([d[i] for d in draws]) for i in range(3)]
    m, g = qs.healthy(asm.n, M)
    r = qs.solve(plant, m, g, om_ref, dc, het_rows=hr)
    return np.std(signals(r), 0, ddof=1)
