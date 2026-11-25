"""T0 quasi-static (QS) plant: capability vector rho over K decision legs, per-wheel heterogeneity, per-wheel commands.

Physics and equations are those of capxfer's QS solver (`capxfer/ideas/i2/theory.py::qs_solve`, docs/THEORY.md Def. 1,
Lemma 0; QS vs dynamic sim R^2 0.98), re-implemented here because cfcap needs three things qs_solve does not allow:
  * per-wheel motor heterogeneity (stall / no-load scale per wheel) and per-wheel mu factor;
  * arbitrary per-wheel speed references omega_ref (not only the IK of a body twist): module-differential twists and
    per-wheel tug-of-war probes;
  * per-wheel open-loop duty commands.
It also replaces the finite-difference Newton Jacobian by the analytic block Jacobian (body 3x3 + diagonal wheel block,
Schur complement), the same structure as capxfer `sim.py`'s implicit step, which makes it O(n) per iteration.
`tests/test_t0.py` checks it against capxfer qs_solve on shared cases.

Unknowns per problem: body twist xi (3) and wheel speeds om (n). Per wheel i:
  slips    s1 = a1.xi - c1 om,  s2 = a2.xi - c2 om   (traction channel, free/roller channel)
  forces   f1 = -mu_i N_i tanh(s1/s0);  f2 = -c_r s2  (or friction for a roller-locked wheel)
  body     sum_i f1 a1 + f2 a2 = 0
  load     tau_c = f1 c1 + f2 c2 + crr N r tanh(om/0.5)
  wheel    tracking: om = om_ref (PI integral) | fixed duty U (saturated or commanded): tau_s(U g vb - om/om0) = tau_c
           passive (coast/brake): -tau_bd tanh(om/0.3) - brake (tau_s/om0) om = tau_c
Published duty (what the firmware logs, before weak scaling): tracking -> required duty; saturated -> +-1; commanded
duty -> U; passive -> clamp(kff om_ref + kp (om_ref - om)) (the PI integral is frozen for passive wheels, as in sim.py).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from capxfer import assembly as A
from capxfer import probes
from capxfer.ideas.i2.theory import normal_loads, rho_of  # noqa: F401  (re-exported)
from capxfer.sim import OK, WEAK, COAST, BRAKE, ROLLER, nominal as sim_nominal

CAPXFER_ROOT = os.environ.get("CAPXFER_ROOT", os.path.expanduser("~/worktrees/capxfer-cluster/capxfer"))
THETA_DIR = os.environ.get("CAPXFER_THETA_DIR", os.path.expanduser("~/.cache/modular-runs/2026-11/capxfer/sysid"))
NOM = dict(mu=0.8, s0=0.03, c_roller=2.25, crr=0.0175, tau_bd=0.75, v_batt=11.85, stall_scale=1.0, noload_scale=1.0,
           radius_scale=1.0)                                   # capxfer theory.NOM
MODULE_MASS = {"plated": 12.9, "plateless": 5.2}               # corpus per-module mass by plate state (as cfcap/t1)

# decision legs: surge+-, sway+-, yaw+-, 4 diagonals (= capxfer P1..P5, both signs; yaw scaled per assembly)
LEG_NAMES = ("surge+", "surge-", "sway+", "sway-", "diag++", "diag--", "diag+-", "diag-+", "yaw+", "yaw-")


def load_params(plate="plated"):
    P = A.load_params(os.path.join(CAPXFER_ROOT, "params", "module_measured.yaml"))
    for k in list(P["module"]["mass_kg"]):
        P["module"]["mass_kg"][k] = MODULE_MASS[plate]
    return P


def load_theta(plate="plated"):
    with open(os.path.join(THETA_DIR, f"theta_{plate}.json")) as fh:
        th = json.load(fh)["theta"]
    return {k: float(v) for k, v in th.items() if k in NOM}


def yaw_scale(asm):
    return min(1.0, probes.WHEEL_SPEED_CAP / max(asm.char_length() * probes.W_NOM, 1e-9))


def decision_legs(asm):
    """(10, 3) commanded twists in LEG_NAMES order."""
    v, d, w = probes.V_NOM, probes.V_DIAG, probes.W_NOM * yaw_scale(asm)
    return np.array([[v, 0, 0], [-v, 0, 0], [0, v, 0], [0, -v, 0], [d, d, 0], [-d, -d, 0], [d, -d, 0], [-d, d, 0],
                     [0, 0, w], [0, 0, -w]], float)


def ik(asm, xi, radius):
    """Nominal-geometry IK wheel speeds for twists xi (..., 3) -> (..., n)."""
    return asm.ik(np.asarray(xi, float), radius)


@dataclass
class Plant:
    """Continuous plant of one assembly: theta (shared) + per-wheel heterogeneity. Arrays are per wheel (n,)."""
    asm: object
    params: dict
    theta: dict
    stall: np.ndarray = None       # per-wheel stall-torque scale (x theta stall_scale)
    noload: np.ndarray = None      # per-wheel no-load-speed scale (x theta noload_scale)
    mu_f: np.ndarray = None        # per-wheel mu factor (per-module factor broadcast to its wheels)
    N: np.ndarray = None           # normal loads (recomputed for every assembly)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = self.asm.n
        one = np.ones(n)
        self.stall = one.copy() if self.stall is None else np.asarray(self.stall, float)
        self.noload = one.copy() if self.noload is None else np.asarray(self.noload, float)
        self.mu_f = one.copy() if self.mu_f is None else np.asarray(self.mu_f, float)
        self.N = normal_loads(self.asm) if self.N is None else np.asarray(self.N, float)
        self.d = {**NOM, **self.theta}
        self.nom = sim_nominal(self.params)

    @property
    def radius(self):
        return self.nom["radius"] * self.d["radius_scale"]

    def ik(self, xi):
        return ik(self.asm, xi, self.nom["radius"])


def sub_plant(plant, asm_child, wheel_idx):
    """Plant of a child/merged assembly that carries over the per-wheel latents of `wheel_idx` (indices into the
    source plant, or arrays given directly); normal loads are recomputed for the new geometry."""
    return Plant(asm_child, plant.params, plant.theta, stall=plant.stall[wheel_idx], noload=plant.noload[wheel_idx],
                 mu_f=plant.mu_f[wheel_idx])


# ----------------------------------------------------------------------------- solver
def solve(plant, modes, gs, om_ref, duty_cmd=None, iters=40, tol=1e-9, max_outer=8, chunk=60000, het_rows=None):
    """Batched QS equilibrium.

    modes, gs : (B, n) fault modes and weak gains (hypotheses or truth).
    om_ref    : (K, n) per-wheel speed references (rad/s); use plant.ik(xis) for body-twist commands.
    duty_cmd  : optional (K, n) open-loop duty per wheel (NaN = speed-tracking wheel).
    het_rows  : optional (stall, noload, mu_f), each (B, n): per-row heterogeneity overriding the plant's (batched
                nuisance draws on one geometry).
    Returns dict xi (B,K,3), om (B,K,n), duty (B,K,n) published duty, resid (B,K) max-abs residual (N, Nm, rad/s).
    QS equilibria need not exist (e.g. a lone module with a coast wheel in full slip spins up); such problems return
    the least-residual point found (resid > 1e-6), as capxfer qs_solve does.
    """
    modes = np.asarray(modes)
    gs = np.asarray(gs, float)
    om_ref = np.atleast_2d(np.asarray(om_ref, float))
    B, n = modes.shape
    K = om_ref.shape[0]
    if duty_cmd is None:
        duty_cmd = np.full((K, n), np.nan)
    duty_cmd = np.atleast_2d(np.asarray(duty_cmd, float))
    M = np.repeat(modes, K, 0)                                           # flatten (B, K) -> problems
    Gs = np.repeat(gs, K, 0)
    OMR = np.tile(om_ref, (B, 1))
    DC = np.tile(duty_cmd, (B, 1))
    HR = None if het_rows is None else [np.repeat(np.asarray(h, float), K, 0) for h in het_rows]
    out = {k: [] for k in ("xi", "om", "duty", "resid")}
    for p0 in range(0, B * K, chunk):
        sl = slice(p0, p0 + chunk)
        r = solve_flat(plant, M[sl], Gs[sl], OMR[sl], DC[sl], iters, tol, max_outer,
                       None if HR is None else [h[sl] for h in HR])
        for k in out:
            out[k].append(r[k])
    res = {k: np.concatenate(v, 0) for k, v in out.items()}
    return dict(xi=res["xi"].reshape(B, K, 3), om=res["om"].reshape(B, K, n), duty=res["duty"].reshape(B, K, n),
                resid=res["resid"].reshape(B, K))


def solve_flat(plant, modes, gs, om_ref, duty_cmd, iters=40, tol=1e-9, max_outer=8, het_rows=None):
    """One QS problem per row: modes, gs, om_ref, duty_cmd all (P, n). Returns a numpy dict (P, ...)."""
    f64 = dict(dtype=torch.float64)
    T = lambda a: torch.as_tensor(np.asarray(a), **f64)
    asm, d, nom = plant.asm, plant.d, plant.nom
    P, n = modes.shape
    pos, w, r = T(asm.wheel_pos), T(asm.w_hat), T(asm.r_hat)
    f_hat = torch.stack([-r[:, 1], r[:, 0]], -1)
    n_hat = torch.stack([-w[:, 1], w[:, 0]], -1)
    modes_t = torch.as_tensor(modes)
    roller = modes_t == ROLLER
    rl = roller[..., None]
    d1 = torch.where(rl, w, r)
    d2 = torch.where(rl, n_hat, f_hat)
    rad = nom["radius"] * d["radius_scale"]
    lever = lambda dd: torch.stack([dd[..., 0], dd[..., 1], pos[:, 0] * dd[..., 1] - pos[:, 1] * dd[..., 0]], -1)
    G = dict(A1=lever(d1).expand(P, n, 3), A2=lever(d2).expand(P, n, 3),
             C1=(rad * (w * d1).sum(-1)).expand(P, n), C2=(rad * (w * d2).sum(-1)).expand(P, n),
             FR2=roller, PAS=(modes_t == COAST) | (modes_t == BRAKE), BRK=(modes_t == BRAKE).double(),
             GW=torch.where(modes_t == WEAK, T(gs), torch.ones(P, n, **f64)), OMR=T(om_ref))
    DC = T(duty_cmd)
    CMD = ~torch.isnan(DC) & ~G["PAS"]                                   # open-loop duty wheels (driven only)
    stall, noload, mu_f = (plant.stall, plant.noload, plant.mu_f) if het_rows is None else het_rows
    tau_s = T(nom["tau_s"] * d["stall_scale"] * stall)                 # (n,) or (P, n)
    om0 = T(nom["omega0"] * d["noload_scale"] * noload)
    vb = d["v_batt"] / nom["v0"]
    s0, c_r, tau_bd = d["s0"], d["c_roller"], d["tau_bd"]
    muN = T(d["mu"] * mu_f * plant.N)
    G.update(TS=tau_s.expand(P, n), OM0=om0.expand(P, n), MUN=muN.expand(P, n))
    Nr = T(plant.N) * d["crr"] * rad
    XI_SCALE = torch.tensor([1.0, 1.0, 1.0 / max(asm.char_length(), 0.1)], **f64)
    MAX_DOM, MAX_DXI = 3.0, 0.1

    def evaluate(g, xi, om, fixed, U, S0, jac=True):
        A1, A2, C1, C2 = g["A1"], g["A2"], g["C1"], g["C2"]
        tau_s, om0, muN = g["TS"], g["OM0"], g["MUN"]
        s1 = (A1 * xi[:, None]).sum(-1) - C1 * om
        s2 = (A2 * xi[:, None]).sum(-1) - C2 * om
        t1, t2 = torch.tanh(s1 / S0), torch.tanh(s2 / S0)
        f1 = -muN * t1
        f2 = torch.where(g["FR2"], -muN * t2, -c_r * s2)
        tr = torch.tanh(om / 0.5)
        tau_c = f1 * C1 + f2 * C2 + Nr * tr
        Rb = (f1[..., None] * A1 + f2[..., None] * A2).sum(1)
        tp = torch.tanh(om / 0.3)
        BRK, PAS = g["BRK"], g["PAS"]
        tau_pas = -tau_bd * tp - BRK * (tau_s / om0) * om
        tau_mot = tau_s * (U * g["GW"] * vb - om / om0)
        Rw = torch.where(PAS, tau_pas - tau_c, torch.where(fixed, tau_mot - tau_c, om - g["OMR"]))
        if not jac:
            return Rb, Rw, tau_c
        k1 = -muN * (1 - t1 * t1) / S0
        k2 = torch.where(g["FR2"], -muN * (1 - t2 * t2) / S0, torch.full_like(t2, -c_r))
        Jbb = torch.einsum("bi,bij,bik->bjk", k1, A1, A1) + torch.einsum("bi,bij,bik->bjk", k2, A2, A2)
        q = (k1 * C1)[..., None] * A1 + (k2 * C2)[..., None] * A2          # d tau_c / d xi
        dtc = -(k1 * C1 * C1 + k2 * C2 * C2) + Nr * (1 - tr * tr) / 0.5    # d tau_c / d om
        dpas = -tau_bd * (1 - tp * tp) / 0.3 - BRK * (tau_s / om0) - dtc
        dmot = -tau_s / om0 - dtc
        Dw = torch.where(PAS, dpas, torch.where(fixed, dmot, torch.ones_like(om)))
        Cw = torch.where((PAS | fixed)[..., None], -q, torch.zeros_like(q))   # d Rw / d xi
        return Rb, Rw, tau_c, Jbb, -q, Cw, Dw                               # -q: d Rb / d om_i (row i)

    def merit(Rb, Rw):
        return (Rb * Rb).sum(1) + (Rw * Rw).sum(1)

    def newton(xi, om, fixed, U, S0, n_it):
        """Damped Newton with a trust region; iterates only on rows that are neither converged nor stalled."""
        act = torch.arange(P)
        for _ in range(n_it):
            g = {k: v[act] for k, v in G.items()}
            x, o, fx, uu = xi[act], om[act], fixed[act], U[act]
            Rb, Rw, _, Jbb, Bb, Cw, Dw = evaluate(g, x, o, fx, uu, S0)
            f0 = merit(Rb, Rw)
            keep = f0 > tol * tol
            if not bool(keep.any()):
                break
            if not bool(keep.all()):
                act = act[keep]
                g = {k: v[keep] for k, v in g.items()}
                x, o, fx, uu, Rb, Rw, Jbb, Bb, Cw, Dw, f0 = (t[keep] for t in (x, o, fx, uu, Rb, Rw, Jbb, Bb, Cw, Dw,
                                                                              f0))
            Dinv = 1.0 / Dw
            S = Jbb - torch.einsum("bik,bi,bil->bkl", Bb, Dinv, Cw)
            rhs = -Rb + torch.einsum("bik,bi->bk", Bb, Dinv * Rw)
            dxi = torch.linalg.solve(S + 1e-12 * torch.eye(3, **f64), rhs[..., None])[..., 0]
            dom = (-Rw - (Cw * dxi[:, None]).sum(-1)) * Dinv
            dxi, dom = torch.nan_to_num(dxi), torch.nan_to_num(dom)
            sc = torch.minimum(MAX_DOM / dom.abs().amax(1).clamp(min=1e-12),
                               MAX_DXI / (dxi.abs() / XI_SCALE).amax(1).clamp(min=1e-12)).clamp(max=1.0)
            dxi, dom = dxi * sc[:, None], dom * sc[:, None]
            bx, bo, bf = x, o, f0
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.03):
                xt, ot = x + alpha * dxi, o + alpha * dom
                ft = merit(*evaluate(g, xt, ot, fx, uu, S0, jac=False)[:2])
                better = ft < bf
                bx = torch.where(better[:, None], xt, bx)
                bo = torch.where(better[:, None], ot, bo)
                bf = torch.where(better, ft, bf)
                if bool(better.all()):
                    break
            xi[act], om[act] = bx, bo
            stalled = bf >= f0
            if bool(stalled.all()):
                break
            if bool(stalled.any()):
                act = act[~stalled]
        return xi, om

    # initial guess: least-squares body twist of the references; every wheel at its reference
    Gk = torch.stack([r[:, 0], r[:, 1], pos[:, 0] * r[:, 1] - pos[:, 1] * r[:, 0]], -1) / (rad * (w * r).sum(-1))[:, None]
    xi = torch.linalg.lstsq(Gk, G["OMR"].T).solution.T.contiguous()      # (P,3)
    om = G["OMR"].clone()
    fixed = CMD.clone()
    U = torch.where(CMD, torch.nan_to_num(DC), torch.zeros_like(DC))
    for s0e in (30 * s0, 10 * s0, 3 * s0):                               # homotopy in the slip scale
        xi, om = newton(xi, om, fixed, U, s0e, 10)
    for outer in range(max_outer):                                       # duty-clamp active set
        xi, om = newton(xi, om, fixed, U, s0, iters)
        _, _, tau_c = evaluate(G, xi, om, fixed, U, s0, jac=False)
        u_req = (tau_c / G["TS"] + om / G["OM0"]) / (G["GW"] * vb)
        sat_now = fixed & ~CMD
        new_sat = ~G["PAS"] & ~CMD & ~fixed & (u_req.abs() > 1.0)
        release = sat_now & (torch.sign(U) * (om - G["OMR"]) > 0)        # passed its reference: tracks again
        if not bool(new_sat.any()) and not bool(release.any()):
            break
        U = torch.where(new_sat, torch.sign(u_req), torch.where(release, torch.zeros_like(U), U))
        fixed = (fixed | new_sat) & ~release
    Rb, Rw, tau_c = evaluate(G, xi, om, fixed, U, s0, jac=False)
    resid = torch.cat([Rb, Rw], 1).abs().amax(1)
    u_req = (tau_c / G["TS"] + om / G["OM0"]) / (G["GW"] * vb)
    kff, kp = nom["ff_duty_per_radps"], nom["kp"]
    u_pas = torch.clamp(kff * G["OMR"] + kp * (G["OMR"] - om), -1, 1)
    duty = torch.where(G["PAS"], u_pas, torch.where(fixed, U, torch.clamp(u_req, -1, 1)))
    return dict(xi=xi.numpy(), om=om.numpy(), duty=duty.numpy(), resid=resid.numpy())


# ----------------------------------------------------------------------------- capability
def rho(plant, modes, gs, xis=None, **kw):
    """Capability vector rho (B, K) on the decision legs (default: the 10 LEG_NAMES legs) for body-twist commands."""
    xis = decision_legs(plant.asm) if xis is None else np.asarray(xis, float)
    r = solve(plant, modes, gs, plant.ik(xis), **kw)
    return rho_of(r["xi"], xis, plant.asm.char_length()), r


def healthy(n, B=1):
    return np.zeros((B, n), np.int64), np.ones((B, n))
