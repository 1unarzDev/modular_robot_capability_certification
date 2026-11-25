"""Misspecified motor laws ("wrong law") shared by theory E1 (T0) and the stress T2 motor-law arm.

T0/T1 motors follow the two-constant law (THEORY_cfcap.md Lemma 1): published duty u = a alpha + b beta with constant
per-wheel stall / no-load scales. A *knee* law makes those scales depend on the operating point:

    nl_eff(om)  = nl  * max(FLOOR, 1 - c_w * (|om|  / om_k  - 1)_+)       (speed knee: no-load scale drops above om_k)
    st_eff(tau) = st  * max(FLOOR, 1 - c_t * (|tau| / tau_k - 1)_+)       (torque knee: stall scale drops above tau_k)

with om_k = f_w * (healthy no-load speed of that wheel) and tau_k = f_t * (stall torque of that wheel). Knees are drawn
per wheel, independently of any probe. Severity 0 = the two-constant law (no knee).

API (pure numpy, no torch; import it from any plant):
    env = motor_envelope("knee", severity, rng, n)          # per-wheel relative knee parameters (dict of (n,) arrays)
    st_mult, nl_mult = knee_factors(env, om, tau, om_noload, tau_stall)
        om, tau: the wheel's current speed (rad/s) and load torque (N m), any shape broadcasting with (n,);
        om_noload, tau_stall: that wheel's healthy no-load speed (rad/s) and stall torque (N m) under the base law.
        Multiply the base stall / no-load scales by the returned factors (in a dynamic plant: every step, or with a
        first-order lag; in T0: as a fixed point, `solve_knee`).
    solve_knee(plant, modes, gs, om_ref, env, iters)       # T0 QS fixed point, one problem per leg
"""
from __future__ import annotations

import numpy as np

FLOOR = 0.4
# severity -> (f_w range, c_w range, f_t range, c_t range); 0 = no knee
SEVERITY = {
    0: None,
    1: ((0.75, 0.95), (0.3, 0.7), (0.30, 0.50), (0.3, 0.7)),
    2: ((0.60, 0.90), (0.6, 1.2), (0.20, 0.40), (0.6, 1.2)),
    3: ((0.50, 0.80), (1.0, 2.0), (0.10, 0.30), (1.0, 2.0)),
}


def motor_envelope(kind, severity, rng, n):
    """Per-wheel knee parameters. kind 'none' or 'knee'; severity in SEVERITY (0 = none)."""
    if kind == "none" or severity == 0 or SEVERITY.get(severity) is None:
        return dict(kind="none", severity=0, f_w=np.full(n, np.inf), c_w=np.zeros(n), f_t=np.full(n, np.inf),
                    c_t=np.zeros(n))
    if kind != "knee":
        raise ValueError(kind)
    (fw, cw, ft, ct) = SEVERITY[severity]
    return dict(kind="knee", severity=int(severity), f_w=rng.uniform(*fw, n), c_w=rng.uniform(*cw, n),
                f_t=rng.uniform(*ft, n), c_t=rng.uniform(*ct, n))


def knee_factors(env, om, tau, om_noload, tau_stall):
    """(st_mult, nl_mult) multipliers of the base stall / no-load scales at operating point (om, tau)."""
    om_k = env["f_w"] * np.asarray(om_noload, float)
    tau_k = env["f_t"] * np.asarray(tau_stall, float)
    nl = np.maximum(FLOOR, 1 - env["c_w"] * np.clip(np.abs(om) / om_k - 1, 0, None))
    st = np.maximum(FLOOR, 1 - env["c_t"] * np.clip(np.abs(tau) / tau_k - 1, 0, None))
    return st, nl


def solve_knee(plant, modes, gs, om_ref, env, iters=30, damp=0.3, tol=1e-4):
    """T0 QS equilibrium under the knee law: fixed point of qs.solve_flat with operating-point-dependent per-wheel
    scales (damping `damp`, stops when the scales move < tol; r['conv'] (L,) flags converged legs). modes, gs (n,); om_ref (L, n), one problem per leg.
    Returns dict xi (L,3), om (L,n), duty (L,n), tau (L,n) load torque, st (L,n), nl (L,n) effective scales."""
    from cfcap.t0 import qs
    from capxfer.sim import OK, WEAK
    L, n = om_ref.shape
    M, G = np.tile(modes, (L, 1)), np.tile(gs, (L, 1))
    st0, nl0, mu = (np.tile(np.asarray(x, float), (L, 1)) for x in (plant.stall, plant.noload, plant.mu_f))
    d, nom = plant.d, plant.nom
    vb = d["v_batt"] / nom["v0"]
    om_nl = vb * nom["omega0"] * d["noload_scale"] * nl0                # healthy no-load speed per wheel
    tau_st = nom["tau_s"] * d["stall_scale"] * st0
    driven = (M == OK) | (M == WEAK)
    gw = np.where(M == WEAK, G, 1.0)
    nan = np.full((L, n), np.nan)
    st, nl = st0.copy(), nl0.copy()
    knee = env.get("kind", "none") != "none"

    def torque(r, st, nl):
        TS = nom["tau_s"] * d["stall_scale"] * st
        OM0 = nom["omega0"] * d["noload_scale"] * nl
        return (r["duty"] * gw * vb - r["om"] / OM0) * TS

    r = qs.solve_flat(plant, M, G, om_ref, nan, het_rows=(st, nl, mu))
    conv = np.ones(L, bool)
    for _ in range(iters if knee else 0):
        tau = torque(r, st, nl)
        fs, fn = knee_factors(env, r["om"], tau, om_nl, tau_st)
        st_t = np.where(driven, st0 * fs, st0)
        nl_t = np.where(driven, nl0 * fn, nl0)
        ch = np.maximum(np.abs(st_t - st).max(1), np.abs(nl_t - nl).max(1))
        conv = ch < tol
        if conv.all():
            break
        st = st + damp * (st_t - st)
        nl = nl + damp * (nl_t - nl)
        r = qs.solve_flat(plant, M, G, om_ref, nan, het_rows=(st, nl, mu))
    r["tau"] = torque(r, st, nl)
    r["st"], r["nl"], r["conv"] = st, nl, conv
    return r
