"""T0 probe library: per-wheel speed references (and optional open-loop duties) per probe leg, plus time cost.

A probe set is a list of legs; each leg is (name, om_ref (n,), duty_cmd (n,) or NaN). Times follow capxfer
`probes.leg_duration` (ramp 0.3 s + hold + ramp 0.3 s + rest 0.4 s); observations are hold-window means
(hold window = hold - 0.4 s, as capxfer `Schedule.windows`).

  v2       capxfer probe v2: +-0.30 vx, +-0.30 vy, +-1.0 wz (yaw scaled to the 0.45 m/s wheel-speed cap). 6 legs, 12 s.
  clamp    clamp ramp: +-0.45 m/s surge and sway (capxfer P6x/P6y), near the duty clamp of a healthy wheel. 4 legs.
  moddiff  module-differential twists: every module spins about its own centre (+-), and checkerboard modules
           (cell parity) are commanded opposite sway / surge 0.30 m/s. 4 legs.
  tug      per-wheel tug-of-war at body rest: in every module the diagonal pair (fl, rr) [or (fr, rl)] is commanded
           opposite wheel speeds +-0.40 m/s surface while all other wheels hold zero speed. 4 legs.
Diagnostic split (observe the child directly with its own v2) is handled in the Stage A engine (cost t_split + 12 s).
"""
from __future__ import annotations

import numpy as np

from capxfer import probes

from . import qs

HOLD = {"v2": 1.0, "clamp": 1.2, "moddiff": 1.0, "tug": 1.0}
T_SPLIT_S = 30.0          # ESTIMATE: one split or merge manoeuvre (B3 split/combine timing not measured yet)
V_CLAMP, V_TUG = 0.45, 0.40


def leg_time(hold):
    return probes.RAMP + hold + probes.RAMP + probes.REST


def window_s(hold):
    return hold - 0.4


def v2_twists(asm):
    L = qs.decision_legs(asm)
    return L[[0, 1, 2, 3, 8, 9]]


V2_LEG_IDX = (0, 1, 2, 3, 8, 9)       # v2 legs inside qs.decision_legs


def _legs_from_twists(plant, names, xis):
    om = plant.ik(np.asarray(xis, float))
    return [(nm, om[k], np.full(plant.asm.n, np.nan)) for k, nm in enumerate(names)]


def module_twist_refs(plant, twists):
    """twists: (m, 3) one body twist per module, about that module's centre -> per-wheel om_ref (n,)."""
    asm = plant.asm
    pos = asm.wheel_pos
    c = asm.module_pos[asm.wheel_module]
    tw = np.asarray(twists, float)[asm.wheel_module]
    rel = pos - c
    vc = np.stack([tw[:, 0] - tw[:, 2] * rel[:, 1], tw[:, 1] + tw[:, 2] * rel[:, 0]], -1)
    return (vc * asm.r_hat).sum(-1) / (plant.nom["radius"] * (asm.w_hat * asm.r_hat).sum(-1))


def probe_legs(plant, kind):
    asm = plant.asm
    n = asm.n
    nan = np.full(n, np.nan)
    if kind == "v2":
        return _legs_from_twists(plant, ("surge+", "surge-", "sway+", "sway-", "yaw+", "yaw-"), v2_twists(asm))
    if kind == "clamp":
        v = V_CLAMP
        return _legs_from_twists(plant, ("clamp_x+", "clamp_x-", "clamp_y+", "clamp_y-"),
                                 [[v, 0, 0], [-v, 0, 0], [0, v, 0], [0, -v, 0]])
    if kind == "moddiff":
        m = len(asm.modules)
        parity = np.array([(i + j) % 2 for _, i, j, _ in asm.modules]) * 2 - 1.0      # +-1 checkerboard
        w = probes.W_NOM * min(1.0, probes.WHEEL_SPEED_CAP / float(np.hypot(*asm.wheel_pos[0] - asm.module_pos[0])))
        legs = []
        for nm, tw in (("spin+", np.tile([0, 0, w], (m, 1))), ("spin-", np.tile([0, 0, -w], (m, 1))),
                       ("check_y", np.stack([0 * parity, probes.V_NOM * parity, 0 * parity], 1)),
                       ("check_x", np.stack([probes.V_NOM * parity, 0 * parity, 0 * parity], 1))):
            legs.append((nm, module_twist_refs(plant, tw), nan.copy()))
        return legs
    if kind == "tug":
        corner = np.array([c.split(":")[1] for c in asm.wheel_corner])
        wv = V_TUG / plant.nom["radius"]
        legs = []
        for pair, (a, b) in (("A", ("fl", "rr")), ("B", ("fr", "rl"))):
            for s in (+1, -1):
                om = np.zeros(n)
                om[corner == a] = s * wv
                om[corner == b] = -s * wv
                legs.append((f"tug{pair}{'+' if s > 0 else '-'}", om, nan.copy()))
        return legs
    raise ValueError(kind)


def probe_time(kind):
    return len(probe_legs_names(kind)) * leg_time(HOLD[kind])


def probe_legs_names(kind):
    return dict(v2=6, clamp=4, moddiff=4, tug=4)[kind] * [None]


PROBE_SETS = {                       # name -> extra probe kinds on top of v2
    "v2": (),
    "v2+clamp": ("clamp",),
    "v2+moddiff": ("moddiff",),
    "v2+tug": ("tug",),
    "v2+clamp+moddiff+tug": ("clamp", "moddiff", "tug"),
}


def set_time(name):
    return probe_time("v2") + sum(probe_time(k) for k in PROBE_SETS[name])
