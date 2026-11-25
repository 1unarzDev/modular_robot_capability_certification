"""Closed-loop capability labels: ρ per decision leg from T1 rollouts (never from a formula).

Each episode runs every decision leg as its own ~3 s trial of the target morphology with the episode's latents
(leg trials of one episode share every latent, including mounting-yaw errors). ρ per leg is capxfer
`sim.targets_from_twist` over the leg's steady window: ρ = <ξ, ξ_cmd>_W / |ξ_cmd|²_W, W = diag(1, 1, Lc²).

Under the measured firmware (feed-forward 0.65 m/s, ki 0.8 duty/m) a healthy module converges slowly: at theta_plated
a healthy single reaches rho ~0.91 in a 3 s leg and ~0.99 after 10 s. Hence both the absolute rho and capxfer's
relative capability (D17: divided by the same latents' fault-free rollout) are returned.

Leg timing: 0.1 s rest, 0.3 s ramp, hold to the end (T = 3.0 s); window = the last 2.15 s minus 0.05 s ([0.8 s, 2.95 s]).
Speeds follow the capxfer probe library: surge/sway 0.30 m/s, diagonals 0.21/0.21 m/s, yaw W_NOM scaled so the
farthest wheel stays below WHEEL_SPEED_CAP.
"""
from __future__ import annotations

import numpy as np
import torch

from cfcap.t1 import plant
from capxfer import probes, sim

LEGS = ("surge+", "surge-", "sway+", "sway-", "yaw+", "yaw-", "diag++", "diag+-", "diag-+", "diag--")
T_LEG = 3.0
T0, RAMP = 0.1, 0.3
WIN_LEN = 2.15                       # steady window = the last 2.15 s of the leg minus 0.05 s ([0.8, 2.95] at 3 s)


def window(T):
    return (max(T0 + RAMP + 0.4, T - 0.05 - WIN_LEN), T - 0.05)


def leg_twists(asm):
    wz = probes.W_NOM * min(1.0, probes.WHEEL_SPEED_CAP / max(asm.char_length() * probes.W_NOM, 1e-9))
    v, d = probes.V_NOM, probes.V_DIAG
    return np.array([(v, 0, 0), (-v, 0, 0), (0, v, 0), (0, -v, 0), (0, 0, wz), (0, 0, -wz),
                     (d, d, 0), (d, -d, 0), (-d, d, 0), (-d, -d, 0)], float)


class _LegWindow:
    """Duck-typed schedule for targets_from_twist: one window with one commanded twist."""

    def __init__(self, name, tw, T):
        a, b = window(T)
        self._w = [(name, 1, a, b, tw)]

    def windows(self):
        return self._w


def leg_tables(asm, params, legs=LEGS, T=T_LEG):
    """(L, K+1, 3) tick tables of the decision legs."""
    hz = float(params["firmware"]["cmd_rate_hz"])
    K = plant.n_ticks(T, params)
    t = np.arange(K + 1) / hz
    ramp = np.clip((t - T0) / RAMP, 0, 1)
    ramp[K:] = 0.0
    tw = leg_twists(asm)[[LEGS.index(l) for l in legs]]
    return (ramp[None, :, None] * tw[:, None, :]).astype(np.float32), tw


def job(asm, params, lat, legs=LEGS, T=T_LEG):
    """A plant job running every leg for every episode (rows episode-major: b*L + l)."""
    tabs, tw = leg_tables(asm, params, legs, T)
    B, L = lat["mode"].shape[0], len(legs)
    cmd = np.tile(tabs, (B, 1, 1))
    return dict(asm=asm, lat=plant.repeat(lat, L), cmd=cmd), tw


def rho_from(R, asm, tw, legs=LEGS, T=T_LEG):
    """R of a `job` -> rho (B, L), drift (B, L) via capxfer targets_from_twist, one leg group at a time."""
    L = len(legs)
    Bt = R["twist"].shape[0]
    rho = torch.empty(Bt // L, L, device=R["twist"].device)
    drift = torch.empty_like(rho)
    for li, name in enumerate(legs):
        Rl = dict(twist=R["twist"][li::L], t=R["t"])
        r, d = sim.targets_from_twist(Rl, _LegWindow(name, tw[li], T), asm.char_length())
        rho[:, li], drift[:, li] = r[:, 0], d[:, 0]
    return rho, drift


def capability(targets, params, legs=LEGS, T=T_LEG, relative=False, **kw):
    """targets: list of (asm, lat) target morphologies with their latents (B_i episodes each, any sizes).
    Returns a list of dicts rho (B_i, L), drift (B_i, L) from closed-loop rollouts bucketed by wheel count; with
    relative=True also rho_healthy (the same latents with every fault cleared, a paired rollout) and
    rho_rel = rho / rho_healthy (capxfer D17 relative capability)."""
    jobs, tws = [], []
    for asm, lat in targets:
        j, tw = job(asm, params, lat, legs, T)
        jobs.append(j)
        tws.append(tw)
        if relative:
            lh = dict(lat, mode=torch.zeros_like(lat["mode"]), g=torch.ones_like(lat["g"]))
            jobs.append(job(asm, params, lh, legs, T)[0])
    Rs = plant.simulate_mixed(jobs, params, T, **kw)
    out = []
    step = 2 if relative else 1
    for i, ((asm, _), tw) in enumerate(zip(targets, tws)):
        rho, drift = rho_from(Rs[step * i], asm, tw, legs, T)
        o = dict(rho=rho, drift=drift)
        if relative:
            rh, _ = rho_from(Rs[step * i + 1], asm, tw, legs, T)
            o.update(rho_healthy=rh, rho_rel=rho / rh)
        out.append(o)
    return out
