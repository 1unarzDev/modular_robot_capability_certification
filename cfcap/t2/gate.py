"""C-GATE in T2 (docs/CONFIRMATORY.md §3 + amendment A-GATE): theory's E1 certifier constructions with T2 as the truth.

E1 (cfcap/theory_task.py gate_episode) runs everything in T0. Here the *truth* and the *probe observations* come from
T2 (`all` + theory's knee law, cfcap/t0/envelope.py via Effects(motor_kind='knee', knee_severity=s); s = 0 is T2 `all`),
and the robot's model is E1's: the two-constant T0 quasi-static plant at the identified theta, fitted per wheel from the
parent's probe windows (v2, and v2+clamp), with the episode's faults known (as E1). Per child and decision leg:
  truth          T2 rho_abs / rho_ref^T1 (fixed T1 reference, cfcap.t1.reference)
  rho_par_<p>    T0 two-constant prediction from the fit at probe set p                     (E1 `rho_par`)
  rho_pi_<p>     identified-set low end over E1's class corners                             (E1 `rho_pi`)
  rho_nom        T0 nominal healthy prediction                                              (E1 `rho_nom`)
  dom_<p>, undom_speed_<p>, undom_torque_<p>   E1's closed-form dominance flags (Prop. 2b)
  misfit_<p>     E1's probe misfit of the child's wheels
Certifier lower bounds and margins are applied by the exporter (scripts/stress_conf_gate.py), with margins fitted on
exploratory T2 knee episodes only. Episodes are the generator's (plated, split, index); children = the episode's
`self` and `child` candidates; knee and T2 effect draws are seeded by `root` + [index] and shared over severities
(paired design; only the knee ranges change with s).
"""
from __future__ import annotations

import zlib

import numpy as np

PROBES = {"v2": ("v2",), "v2clamp": ("v2", "clamp")}
PI_ST, PI_MU = 0.34, 0.9
DOM_SPEED_TOL, DOM_TAU_FRAC = 1.01, 0.02


def t2_seed(root, index):
    return int(zlib.crc32(("gate|" + ",".join(map(str, list(root) + [int(index)]))).encode()))


def _effects(sev):
    from cfcap.t2 import build
    return build.ALL if sev == 0 else build.ABLATIONS[f"all+knee{int(sev)}"]


def probe_windows(kind, asm, par, rng):
    """Generator probe command (cfcap.gen.simrun.probe_tables) + its hold windows [(t0, t1, twist)] in trial time."""
    from capxfer import probes
    from cfcap.gen import simrun as SR
    cmd, _, info = SR.probe_tables(kind, asm, par, rng)
    pids = ("P1", "P2", "P5") if kind == "v2" else ("P6x", "P6y")
    sched = probes.Schedule(asm.char_length(), pids=pids)
    wins = [(ta + info["lead"], tb + info["lead"], np.asarray(tw, float) * info["amp"], info["amp"])
            for _, _, ta, tb, tw in sched.windows()]
    return cmd, SR.probe_T(kind), wins


def gate_episode(index, sevs=(0, 1, 2, 3), split="conf_test", root=(92599, 22), plate="plated", backend="auto",
                 iters_unused=None):
    """Rows (child x leg x severity) for one generator episode."""
    from capxfer.sim import OK, WEAK, COAST, BRAKE
    from cfcap.gen import episode as E
    from cfcap.t0 import qs
    from cfcap.t1 import capability as cap1
    from cfcap.t1.reference import rho_ref_many
    from cfcap.t2 import capability as cap2, eval as ev
    backend = ev.resolve_backend(backend)
    ep = E.make_episode(plate, split, int(index))
    par = E.params(plate)
    P, th = qs.load_params(plate), qs.load_theta(plate)
    pa = ep.parent
    parent = pa.asm
    n = parent.n
    tm, tg = np.asarray(pa.mode), np.asarray(pa.g, float)
    driven = (tm == OK) | (tm == WEAK)
    gw = np.where(tm == WEAK, tg, 1.0)
    kids = [(ci, c) for ci, c in enumerate(ep.cands) if c.kind in (E.KIND["self"], E.KIND["child"])]
    seed = t2_seed(root, index)
    lat_p = E.plant_latents(ep, [(0, k) for k in range(pa.m)], pa.load_z, E.parent_payload(ep),
                            gyro=(pa.gyro_bias, pa.acc_bias))
    lat_c = [E.plant_latents(ep, c.modules, c.load_z, c.payload, B=1) for _, c in kids]
    ref = rho_ref_many([c.asm for _, c in kids], plate, "cpu")                      # (C, L) fixed T1 reference
    perm = [qs.LEG_NAMES.index(l) for l in cap1.LEGS]                               # T0 leg order -> T1 order
    pm = qs.Plant(parent, P, th)
    d, nom = pm.d, pm.nom
    vb = d["v_batt"] / nom["v0"]
    om0, ts0 = nom["omega0"] * d["noload_scale"], nom["tau_s"] * d["stall_scale"]
    # probe commands + windows (generator streams), shared over severities
    prb = {}
    for kind in ("v2", "clamp"):
        rng = E.episode_rng(plate, split, int(index), stream=1 + 16 * pa.role + ("v2", "clamp", "tug").index(kind))
        prb[kind] = probe_windows(kind, parent, par, rng)
    rows = []
    for sev in sevs:
        eff = _effects(sev)
        # truth on every child
        o = cap2.capability([(c.asm, l) for (_, c), l in zip(kids, lat_c)], par, eff, backend=backend, seed=seed)
        truth = np.stack([oo["rho"][0] for oo in o]) / ref
        # parent probes in T2: window means of |omega| and |duty| per wheel, and the commanded IK references
        om_list, u_list, ref_list, amp_list = [], [], [], []
        for kind in ("v2", "clamp"):
            cmd, T, wins = prb[kind]
            R = cap2.simulate(parent, lat_p, par, cmd, T, eff, backend=backend, seed=seed)
            t = R["t"]
            for ta, tb, tw, amp in wins:
                amp_list.append(amp)
                m = (t >= ta) & (t <= tb)
                om_list.append(np.abs(R["speed"][0, m].mean(0)))
                u_list.append(np.abs(R["duty"][0, m].mean(0)))
                ref_list.append(pm.ik(tw[None])[0])
        om_all, u_all, om_ref = np.stack(om_list), np.stack(u_list), np.stack(ref_list)
        amps = np.asarray(amp_list)
        fit = {}
        for key, kinds in PROBES.items():
            sl = slice(0, 6) if key == "v2" else slice(0, 10)
            om_o, u_o = om_all[sl], u_all[sl]
            track = (u_o < 0.999) & (om_o > 0.5)
            nl_k = om_o / (om0 * gw * vb * np.maximum(u_o, 1e-6))
            nl_hat, mis = np.ones(n), np.zeros(n)
            for i in np.flatnonzero(driven):
                tt = track[:, i]
                nl_hat[i] = np.median(nl_k[tt, i]) if tt.any() else om_o[:, i].max() / (om0 * gw[i] * vb)
                m_tr = np.abs(nl_k[tt, i] / nl_hat[i] - 1).max() if tt.any() else 0.0
                sat = ~tt & (om_o[:, i] > 0.5)
                m_sat = np.clip(1 - om_o[sat, i] / (om0 * gw[i] * vb * nl_hat[i]), 0, None).max() if sat.any() else 0.0
                mis[i] = max(m_tr, m_sat)
            # dominance (closed form, Prop. 2b) against the NOMINAL routine: the generator's amplitude jitter is
            # divided out, so a leg is dominated iff the unjittered v2 (/ clamp) visits its wheel speeds and torques
            # the visited speed is the commanded IK reference (the robot's own closed form), not the T2-measured one
            vmax = (np.abs(om_ref[sl]) / amps[sl, None]).max(0)
            pf = qs.Plant(parent, P, th, stall=nl_hat, noload=nl_hat, mu_f=np.ones(n))
            rp = qs.solve(pf, tm[None], tg[None], om_ref[sl])
            tau_p = np.abs((rp["duty"][0] * gw * vb - rp["om"][0] / (om0 * nl_hat)) * ts0 * nl_hat).max(0)
            fit[key] = dict(nl=nl_hat, mis=mis, vmax=vmax, tau_p=tau_p,
                            nl_pi=np.where(driven, np.minimum(nl_hat, vmax / (om0 * gw * vb)), 1.0))
        for k, ((ci, c), lc) in enumerate(zip(kids, lat_c)):
            src = np.concatenate([np.arange(4 * mk, 4 * mk + 4) for _, mk in c.modules])
            nc = c.asm.n
            xis = qs.decision_legs(c.asm)
            pn = qs.Plant(c.asm, P, th)
            omr = pn.ik(xis)
            rn = qs.solve(pn, np.zeros((1, nc), np.int64), np.ones((1, nc)), omr)
            rho_n = qs.rho_of(rn["xi"][0], xis, c.asm.char_length())[perm]
            drv = driven[src]
            row = dict(index=int(index), severity=int(sev), cand=int(ci), cand_size=len(c.modules),
                       has_fault=int((tm[src] != OK).any()), kind=int(c.kind), truth=truth[k], rho_nom=rho_n)
            for key, f in fit.items():
                ph = qs.Plant(c.asm, P, th, stall=f["nl"][src], noload=f["nl"][src], mu_f=np.ones(nc))
                r = qs.solve(ph, tm[src][None], tg[src][None], omr)
                rho_p = qs.rho_of(r["xi"][0], xis, c.asm.char_length())
                tau_c = np.abs((r["duty"][0] * gw[src] * vb - r["om"][0] / (om0 * f["nl"][src])) * ts0 * f["nl"][src])
                us = (np.abs(r["om"][0]) > DOM_SPEED_TOL * f["vmax"][src][None]) & drv[None]
                ut = (tau_c > f["tau_p"][src][None] + DOM_TAU_FRAC * ts0) & drv[None]
                pas = ~drv
                cs = [(sf, nf, mf, pb) for sf in (PI_ST, 0.67, 1.0) for nf in ("pi", "fit", "box") for mf in (PI_MU, 1.1)
                      for pb in (0, 1)]
                cs += [(5.0, "pi", mf, pb) for mf in (PI_MU, 1.1) for pb in (0, 1)]
                hr = (np.stack([np.where(pas, (1.08 if pb else 0.92), 1.08 if sf == 1.0 else (5.0 if sf == 5.0 else
                                                                                                sf * f["nl"][src]))
                                for sf, _, _, pb in cs]),
                      np.stack([np.where(pas, (0.92 if pb else 1.08), f["nl_pi"][src] if nf == "pi" else
                                         (f["nl"][src] if nf == "fit" else np.full(nc, 1.08)))
                                for _, nf, _, pb in cs]),
                      np.stack([np.full(nc, mf) for _, _, mf, _ in cs]))
                rpi = qs.solve(ph, np.tile(tm[src], (len(cs), 1)), np.tile(tg[src], (len(cs), 1)), omr, het_rows=hr)
                rho_pi = qs.rho_of(rpi["xi"], xis, c.asm.char_length()).min(0)
                row[f"rho_par_{key}"] = rho_p[perm]
                row[f"rho_pi_{key}"] = np.minimum(rho_pi, rho_p)[perm]
                row[f"undom_speed_{key}"] = us.any(1)[perm]
                row[f"undom_torque_{key}"] = ut.any(1)[perm]
                row[f"dom_{key}"] = (~(us | ut).any(1))[perm]
                row[f"misfit_{key}"] = np.full(10, f["mis"][src][drv].max() if drv.any() else 0.0)
            rows.append(row)
    return rows
