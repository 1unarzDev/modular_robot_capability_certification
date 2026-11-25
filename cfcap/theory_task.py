"""cfcap.runner Tasks for role theory (docs/board/theory.md, docs/theory/THEORY_cfcap.md).

    python -m cfcap.runner theory_phase --task-spec cfcap.theory_task:PHASE --shard K --n-shards N \
        --seeds 0:5000 --per-unit 10 --sizes 4,5,6,7,8 --faults mix --np-sets v2,v2clamp,v2obl,v2yawhi,v2clampyawhi
    python -m cfcap.runner theory_envelope --task-spec cfcap.theory_task:ENVELOPE --shard K --n-shards N \
        --seeds 0:3000 --per-unit 20

PHASE: the Prop. 2 / Prop. 3 phase diagram. One episode = scripts/theory_nuisance.py run_episode (parametric box image
of the child capability under the motor-spread / mu box, and the speed-only nonparametric class per probe set) on a
random 4-8-module polyomino parent; rows = (episode, candidate, decision leg).

ENVELOPE: the envelope-misspecification check (board prediction S1). The truth motor of every wheel has a knee: above
a speed omega_k its no-load scale drops and above a load torque tau_k its stall scale drops,
    nl_eff = nl * max(0.4, 1 - c_w (|om|/om_k - 1)_+),   st_eff = st * max(0.4, 1 - c_t (|tau|/tau_k - 1)_+),
with om_k = U(0.55, 0.95) vb omega0, tau_k = U(0.1, 0.4) tau_s, c ~ U(0.5, 1.5), per wheel, independent of the probes.
It is solved as a fixed point of the T0 solver with operating-point-dependent per-wheel constants (no change to
cfcap/t0/qs.py). The two-constant oracle plugs in each wheel's effective constants at the parent's highest-speed
probed leg (v2, or v2+clamp) and predicts the child; rows give rho_hat (oracle), rho_true (knee truth), and the
dominance ratios of the child's operating point (speed, torque) against the parent's probed ones.
Outputs are the runner's columnar units (npz); no parquet.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from cfcap.runner import Stop, Task

REPO = Path(__file__).resolve().parents[1]
_TN = None


def _tn():
    """scripts/theory_nuisance.py as a module (the episode code that produced results/theory)."""
    global _TN
    if _TN is None:
        spec = importlib.util.spec_from_file_location("theory_nuisance", REPO / "scripts" / "theory_nuisance.py")
        _TN = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_TN)
    return _TN


def _seed_units(a, shard, n_shards, tag):
    lo, hi = map(int, a.seeds.split(":"))
    items = [f"{tag}-{s:06d}-{min(hi, s + a.per_unit):06d}" for s in range(lo, hi, a.per_unit)]
    return items[shard::n_shards]


def _columns(rows):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(rows)
    out = {}
    for c in df.columns:
        v = df[c].to_numpy()
        out[c] = v.astype(str) if v.dtype.kind == "O" else v
    return out


# ----------------------------------------------------------------------------- PHASE
def _phase_args(p):
    p.add_argument("--seeds", default="0:5000")
    p.add_argument("--per-unit", type=int, default=10)
    p.add_argument("--sizes", default="4,5,6,7,8")
    p.add_argument("--faults", default="mix", choices=("mix", "weak1"))
    p.add_argument("--np-sets", default="v2,v2clamp,v2obl,v2yawhi,v2clampyawhi")
    p.add_argument("--deltas", default="0.04,0.08,0.16")
    p.add_argument("--n-merge", type=int, default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--plate", default="plated")


def _phase_run(unit, ctx):
    import time
    import torch
    from cfcap.t0 import qs
    a = ctx.args
    torch.set_num_threads(a.threads)
    P, th = qs.load_params(a.plate), qs.load_theta(a.plate)
    _, lo, hi = unit.split("-")
    rows, eps = [], []
    for k, seed in enumerate(range(int(lo), int(hi))):
        if ctx.should_stop():
            raise Stop("between episodes")
        t0 = time.time()
        r, _ = _tn().run_episode(seed, P, th, sizes=tuple(int(x) for x in a.sizes.split(",")), faults=a.faults,
                                 np_sets=tuple(a.np_sets.split(",")), n_merge=a.n_merge,
                                 deltas=tuple(float(x) for x in a.deltas.split(",")))
        rows += r
        eps.append(dict(seed=seed, rows=len(r), s=round(time.time() - t0, 2)))
        ctx.progress((k + 1) / (int(hi) - int(lo)))
    ctx.meta["episodes"] = eps
    return _columns(rows)


def _summary(recs, a):
    eps = [e for r in recs for e in r.get("meta", {}).get("episodes", [])]
    return {"theory.units": len(recs), "theory.episodes": len(eps),
            "theory.s_per_ep_mean": sum(float(e["s"]) for e in eps) / max(len(eps), 1)}


PHASE = Task("theory_phase", lambda a, s, n: _seed_units(a, s, n, "ph"), _phase_run, _phase_args, _summary,
             ("numpy", "torch", "pandas"), "theory phase diagram: parametric box image + speed-only class per probe set")


# ----------------------------------------------------------------------------- ENVELOPE
def _env_args(p):
    p.add_argument("--seeds", default="0:3000")
    p.add_argument("--per-unit", type=int, default=20)
    p.add_argument("--sizes", default="4,5,6")
    p.add_argument("--fp-iters", type=int, default=5)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--plate", default="plated")


def knee_solve(plant, modes, gs, om_ref, knee, iters=5):
    """Fixed point of the T0 QS solve with operating-point-dependent per-wheel constants (knee truth).
    modes, gs (n,), om_ref (L, n) one problem per leg. Returns (xi (L,3), om (L,n), tau (L,n), st_eff, nl_eff)."""
    import numpy as np
    from cfcap.t0 import qs
    L, n = om_ref.shape
    M, G = np.tile(modes, (L, 1)), np.tile(gs, (L, 1))
    st0, nl0, mu = np.tile(plant.stall, (L, 1)), np.tile(plant.noload, (L, 1)), np.tile(plant.mu_f, (L, 1))
    st, nl = st0.copy(), nl0.copy()
    d, nom = plant.d, plant.nom
    vb = d["v_batt"] / nom["v0"]
    driven = (M == 0) | (M == 1)
    for _ in range(iters):
        r = qs.solve_flat(plant, M, G, om_ref, np.full((L, n), np.nan), het_rows=(st, nl, mu))
        om, u = r["om"], r["duty"]
        TS = nom["tau_s"] * d["stall_scale"] * st
        OM0 = nom["omega0"] * d["noload_scale"] * nl
        tau = (u * np.where(M == 1, G, 1.0) * vb - om / OM0) * TS
        nl_new = nl0 * np.maximum(0.4, 1 - knee["cw"] * np.clip(np.abs(om) / knee["omk"] - 1, 0, None))
        st_new = st0 * np.maximum(0.4, 1 - knee["ct"] * np.clip(np.abs(tau) / knee["tauk"] - 1, 0, None))
        nl = np.where(driven, 0.5 * nl + 0.5 * nl_new, nl0)
        st = np.where(driven, 0.5 * st + 0.5 * st_new, st0)
    r = qs.solve_flat(plant, M, G, om_ref, np.full((L, n), np.nan), het_rows=(st, nl, mu))
    TS = nom["tau_s"] * d["stall_scale"] * st
    OM0 = nom["omega0"] * d["noload_scale"] * nl
    tau = (r["duty"] * np.where(M == 1, G, 1.0) * vb - r["om"] / OM0) * TS
    return r["xi"], r["om"], tau, st, nl


def env_episode(seed, P, th, sizes=(4, 5, 6), iters=5):
    import numpy as np
    from capxfer.sim import WEAK, COAST, BRAKE
    from cfcap.t0 import qs, stagea, candidates as C, probesets as PS, observe as O
    rng = np.random.default_rng(20_000 + seed)
    parent = stagea.random_parent(rng, P, tuple(sizes))
    n = parent.n
    st, nl, mu = O.sample_het(rng, parent, True)
    tm, tg = np.zeros(n, np.int64), np.ones(n)
    if seed % 5 != 0:
        for w in rng.choice(n, int(rng.integers(1, 3)), replace=False):
            md = int(rng.choice([WEAK, WEAK, COAST, BRAKE]))
            tm[w], tg[w] = md, (float(rng.uniform(0.3, 1.0)) if md == WEAK else 1.0)
    plant = qs.Plant(parent, P, th, stall=st, noload=nl, mu_f=mu)
    vb = plant.d["v_batt"] / plant.nom["v0"]
    om0 = plant.nom["omega0"] * plant.d["noload_scale"]
    knee = dict(omk=rng.uniform(0.55, 0.95, n) * vb * om0 * nl, cw=rng.uniform(0.5, 1.5, n),
                tauk=rng.uniform(0.1, 0.4, n) * plant.nom["tau_s"] * plant.d["stall_scale"] * st,
                ct=rng.uniform(0.5, 1.5, n))
    # parent probes under the knee truth; oracle constants = effective constants at the highest-speed probed leg
    fits = {}
    probed = {}
    for key, kinds in (("v2", ("v2",)), ("v2clamp", ("v2", "clamp"))):
        om_ref = np.concatenate([np.stack([l[1] for l in PS.probe_legs(plant, k)]) for k in kinds])
        _, om, tau, ste, nle = knee_solve(plant, tm, tg, om_ref, knee, iters)
        j = np.abs(om).argmax(0)
        fits[key] = (ste[j, np.arange(n)], nle[j, np.arange(n)])
        probed[key] = (np.abs(om).max(0), np.abs(tau).max(0))
    rows = []
    for cand in C.children(parent, P):
        src = cand.src
        cpl = qs.Plant(cand.asm, P, th, stall=st[src], noload=nl[src], mu_f=mu[src])
        xis = qs.decision_legs(cand.asm)
        omr = cpl.ik(xis)
        ck = {k: v[src] for k, v in knee.items()}
        xi_t, om_t, tau_t, _, _ = knee_solve(cpl, tm[src], tg[src], omr, ck, iters)
        rho_t = qs.rho_of(xi_t, xis, cand.asm.char_length())
        out = dict(seed=np.full(10, seed), cand=np.array([cand.name] * 10), cand_size=np.full(10, cand.size),
                   has_fault=np.full(10, int((tm[src] != 0).any())), leg=np.arange(10), rho_true=rho_t,
                   n_faults=np.full(10, int((tm != 0).sum())), parent_size=np.full(10, len(parent.modules)))
        drv = (tm[src] == 0) | (tm[src] == WEAK)
        for key in fits:
            sth, nlh = fits[key]
            ph = qs.Plant(cand.asm, P, th, stall=sth[src], noload=nlh[src], mu_f=mu[src])
            r = qs.solve(ph, tm[src][None], tg[src][None], omr)
            out[f"rho_hat_{key}"] = qs.rho_of(r["xi"][0], xis, cand.asm.char_length())
            vmax, tmax = probed[key]
            out[f"speed_ratio_{key}"] = np.where(drv[None], np.abs(om_t) / np.maximum(vmax[src][None], 1e-6), 0).max(1)
            out[f"torque_ratio_{key}"] = np.where(drv[None], np.abs(tau_t) / np.maximum(tmax[src][None], 1e-3),
                                                  0).max(1)
        rows.append(out)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def _env_run(unit, ctx):
    import time
    import numpy as np
    import torch
    from cfcap.t0 import qs
    a = ctx.args
    torch.set_num_threads(a.threads)
    P, th = qs.load_params(a.plate), qs.load_theta(a.plate)
    _, lo, hi = unit.split("-")
    parts, eps = [], []
    for k, seed in enumerate(range(int(lo), int(hi))):
        if ctx.should_stop():
            raise Stop("between episodes")
        t0 = time.time()
        r = env_episode(seed, P, th, tuple(int(x) for x in a.sizes.split(",")), a.fp_iters)
        parts.append(r)
        eps.append(dict(seed=seed, rows=len(r["leg"]), s=round(time.time() - t0, 2)))
        ctx.progress((k + 1) / (int(hi) - int(lo)))
    ctx.meta["episodes"] = eps
    return {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}


ENVELOPE = Task("theory_envelope", lambda a, s, n: _seed_units(a, s, n, "env"), _env_run, _env_args, _summary,
                ("numpy", "torch", "pandas"), "theory envelope-misspecification check (knee truth vs two-constant oracle)")


# ----------------------------------------------------------------------------- GATE (E1: trust-gated certificate)
GATE_SIZES = (2, 3, 4, 5, 6)
PROBE_IDX = {"v2": slice(0, 6), "v2clamp": slice(0, 10)}        # parent legs: 6 v2 legs then 4 clamp legs
PI_ST, PI_MU = 0.34, 0.9          # identified-set floor: stall >= 0.4 x (1-d)/(1+d) of the fit, mu >= 0.9 x prior centre
DOM_SPEED_TOL, DOM_TAU_FRAC = 1.01, 0.02


def gate_law(seed):
    """E1 design: seed % 4 = knee severity (0 = two-constant truth), 300 parents per law for seeds 0..1199."""
    return seed % 4


def gate_episode(seed, P, th, sizes=GATE_SIZES, iters=40):
    """One E1 parent: knee-law truth (severity gate_law(seed)), robot's two-constant fit at v2 / v2+clamp, and for
    every connected child and decision leg: rho_true, rho_nom, and per probe set rho_par (fit), rho_pi (identified-set
    low end), dominance flags and the probe misfit of the child's wheels. Returns a dict of (rows,) arrays."""
    import numpy as np
    from capxfer.sim import OK, WEAK, COAST, BRAKE
    from cfcap.t0 import qs, stagea, candidates as C, probesets as PS, observe as O
    from cfcap.t0 import envelope as EV
    sev = gate_law(seed)
    rng = np.random.default_rng(30_000 + seed)
    parent = stagea.random_parent(rng, P, tuple(sizes))
    n = parent.n
    st, nl, mu = O.sample_het(rng, parent, True)
    tm, tg = np.zeros(n, np.int64), np.ones(n)
    for w in rng.choice(n, int(rng.integers(1, 3)), replace=False):
        md = int(rng.choice([WEAK, WEAK, COAST, BRAKE]))
        tm[w], tg[w] = md, (float(rng.uniform(0.3, 1.0)) if md == WEAK else 1.0)
    env = EV.motor_envelope("knee" if sev else "none", sev, np.random.default_rng((40_000 + seed, sev)), n)
    plant = qs.Plant(parent, P, th, stall=st, noload=nl, mu_f=mu)
    d, nom = plant.d, plant.nom
    vb = d["v_batt"] / nom["v0"]
    om0 = nom["omega0"] * d["noload_scale"]
    ts0 = nom["tau_s"] * d["stall_scale"]
    driven = (tm == OK) | (tm == WEAK)
    gw = np.where(tm == WEAK, tg, 1.0)
    om_ref = np.concatenate([np.stack([l[1] for l in PS.probe_legs(plant, k)]) for k in ("v2", "clamp")])
    obs = EV.solve_knee(plant, tm, tg, om_ref, env, iters)
    fit = {}
    for key, sl in PROBE_IDX.items():
        om_o, u_o = np.abs(obs["om"][sl]), np.abs(obs["duty"][sl])
        track = (u_o < 0.999) & (om_o > 0.5)
        nl_k = om_o / (om0 * gw * vb * np.maximum(u_o, 1e-6))
        nl_hat, mis = np.ones(n), np.zeros(n)
        for i in np.flatnonzero(driven):
            t = track[:, i]
            nl_hat[i] = np.median(nl_k[t, i]) if t.any() else om_o[:, i].max() / (om0 * gw[i] * vb)
            m_tr = np.abs(nl_k[t, i] / nl_hat[i] - 1).max() if t.any() else 0.0
            sat = ~t & (om_o[:, i] > 0.5)
            m_sat = np.clip(1 - om_o[sat, i] / (om0 * gw[i] * vb * nl_hat[i]), 0, None).max() if sat.any() else 0.0
            mis[i] = max(m_tr, m_sat)
        vmax = om_o.max(0)
        # the robot's own model of the parent's probe torques (two-constant fit)
        pf = qs.Plant(parent, P, th, stall=nl_hat, noload=nl_hat, mu_f=np.ones(n))
        rp = qs.solve(pf, tm[None], tg[None], om_ref[sl])
        tau_p = np.abs((rp["duty"][0] * gw * vb - rp["om"][0] / (om0 * nl_hat)) * ts0 * nl_hat).max(0)
        fit[key] = dict(nl=nl_hat, mis=mis, vmax=vmax, tau_p=tau_p,
                        nl_pi=np.where(driven, np.minimum(nl_hat, vmax / (om0 * gw * vb)), 1.0))
    out = {k: [] for k in ("seed", "law", "cand", "cand_size", "has_fault", "leg", "rho_true", "rho_nom")}
    for key in PROBE_IDX:
        for c in ("rho_par", "rho_pi", "dom", "undom_speed", "undom_torque", "misfit"):
            out[f"{c}_{key}"] = []
    for cand in C.children(parent, P):
        src = cand.src
        nc = cand.asm.n
        xis = qs.decision_legs(cand.asm)
        cpl = qs.Plant(cand.asm, P, th, stall=st[src], noload=nl[src], mu_f=mu[src])
        omr = cpl.ik(xis)
        cenv = {k: (v[src] if isinstance(v, np.ndarray) else v) for k, v in env.items()}
        rt = EV.solve_knee(cpl, tm[src], tg[src], omr, cenv, iters)
        rho_t = qs.rho_of(rt["xi"], xis, cand.asm.char_length())
        out.setdefault("conv", []).append(rt["conv"] & bool(obs["conv"].all()))
        pn = qs.Plant(cand.asm, P, th)
        rn = qs.solve(pn, np.zeros((1, nc), np.int64), np.ones((1, nc)), omr)
        rho_n = qs.rho_of(rn["xi"][0], xis, cand.asm.char_length())
        out["seed"].append(np.full(10, seed)); out["law"].append(np.full(10, sev))
        out["cand"].append(np.array([cand.name] * 10)); out["cand_size"].append(np.full(10, cand.size))
        out["has_fault"].append(np.full(10, int((tm[src] != OK).any()))); out["leg"].append(np.arange(10))
        out["rho_true"].append(rho_t); out["rho_nom"].append(rho_n)
        drv = driven[src]
        for key, f in fit.items():
            ph = qs.Plant(cand.asm, P, th, stall=f["nl"][src], noload=f["nl"][src], mu_f=np.ones(nc))
            r = qs.solve(ph, tm[src][None], tg[src][None], omr)
            rho_p = qs.rho_of(r["xi"][0], xis, cand.asm.char_length())
            tau_c = np.abs((r["duty"][0] * gw[src] * vb - r["om"][0] / (om0 * f["nl"][src])) * ts0 * f["nl"][src])
            us = (np.abs(r["om"][0]) > DOM_SPEED_TOL * f["vmax"][src][None]) & drv[None]
            ut = (tau_c > f["tau_p"][src][None] + DOM_TAU_FRAC * ts0) & drv[None]
            # identified-set low end: min over the class corners (stall floor / fit, ceiling / fit, mu box), since
            # rho need not be monotone in every wheel's constants (yaw coupling)
            # passive wheels: brake strength ~ st/nl is only known up to the motor-spread box
            pas = ~drv
            cs = [(sf, nf, mf, pb) for sf in (PI_ST, 0.67, 1.0) for nf in ("pi", "fit", "box") for mf in (PI_MU, 1.1)
                  for pb in (0, 1)]
            # hard speed clamp at the probed maximum (step envelope: an overdriven wheel may resist without bound)
            cs += [(5.0, "pi", mf, pb) for mf in (PI_MU, 1.1) for pb in (0, 1)]
            # sf = 1.0 means the box top (1.08, absolute): a saturated, overdriven wheel brakes in proportion to its
            # stall scale, so the low end of rho can sit at the strongest motor, not the weakest
            hr = (np.stack([np.where(pas, (1.08 if pb else 0.92), 1.08 if sf == 1.0 else (5.0 if sf == 5.0 else sf * f["nl"][src]))
                            for sf, _, _, pb in cs]),
                  np.stack([np.where(pas, (0.92 if pb else 1.08), f["nl_pi"][src] if nf == "pi" else
                                     (f["nl"][src] if nf == "fit" else np.full(nc, 1.08)))
                            for _, nf, _, pb in cs]),
                  np.stack([np.full(nc, mf) for _, _, mf, _ in cs]))
            rpi = qs.solve(ph, np.tile(tm[src], (len(cs), 1)), np.tile(tg[src], (len(cs), 1)), omr, het_rows=hr)
            rho_pi = qs.rho_of(rpi["xi"], xis, cand.asm.char_length()).min(0)
            out[f"rho_par_{key}"].append(rho_p)
            out[f"rho_pi_{key}"].append(np.minimum(rho_pi, rho_p))
            out[f"undom_speed_{key}"].append(us.any(1)); out[f"undom_torque_{key}"].append(ut.any(1))
            out[f"dom_{key}"].append(~(us | ut).any(1))
            out[f"misfit_{key}"].append(np.full(10, f["mis"][src][drv].max() if drv.any() else 0.0))
    return {k: np.concatenate(v) for k, v in out.items()}


def _gate_args(p):
    p.add_argument("--seeds", default="0:1200")
    p.add_argument("--per-unit", type=int, default=20)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--plate", default="plated")


def _gate_run(unit, ctx):
    import time
    import numpy as np
    import torch
    from cfcap.t0 import qs
    a = ctx.args
    torch.set_num_threads(a.threads)
    P, th = qs.load_params(a.plate), qs.load_theta(a.plate)
    _, lo, hi = unit.split("-")
    parts, eps = [], []
    for k, seed in enumerate(range(int(lo), int(hi))):
        if ctx.should_stop():
            raise Stop("between episodes")
        t0 = time.time()
        parts.append(gate_episode(seed, P, th))
        eps.append(dict(seed=seed, rows=len(parts[-1]["leg"]), s=round(time.time() - t0, 2)))
        ctx.progress((k + 1) / (int(hi) - int(lo)))
    ctx.meta["episodes"] = eps
    return {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}


GATE = Task("theory_gate", lambda a, s, n: _seed_units(a, s, n, "gate"), _gate_run, _gate_args, _summary,
            ("numpy", "torch", "pandas"), "E1 trust-gated certificate rows (knee-law truth vs two-constant fit)")
