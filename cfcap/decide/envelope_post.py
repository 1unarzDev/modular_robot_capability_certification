"""Envelope-class episodes with fault modes from a posterior (H3 follow-up 1): motor-law AND fault uncertainty.

Truth and observations are those of theory E1 (`cfcap/theory_task.py:gate_episode`, same RNG order, knee law of
severity seed % 4, 1-2 faults with continuous weak g, heterogeneity on). Differences:
  * the parent's probe signals [gz, om, duty] get floor-preset noise (t0.observe.sigma) before anything uses them;
  * the robot does not know the fault modes. A posterior over <= 2-fault hypotheses (weak g on stagea.G_GRID, coast,
    brake) is computed from the noisy v2 legs only (the two-constant likelihood is not trusted at clamp speeds,
    where a knee may bind; a first version that also used the clamp legs emptied the screen on knee episodes): sequential screen at K_SCREEN
    (nominal two-constant plant, floor + heterogeneity sd x SD_INFLATE, as cfcap.decide.t0data), generator prior,
    Gaussian likelihood with the same sd; kept hypotheses = top mass up to 0.99, at most H_MAX (the rest is recorded
    as dropped mass and absorbed by end-to-end calibration);
  * for every kept hypothesis h the E1 bounds are recomputed with h's modes: the two-constant fit at the observed
    probes (rho_par_h) and the identified-set low end over the speed-ceiling class corners (rho_pi_h), exactly the
    gate_episode construction;
  * the episode's bounds: rho_pi = min over kept h (credible-set envelope), rho_par = posterior-weighted mean over kept
    h; undom_speed = the MAP hypothesis's flag; plus rho_par_hi = max over kept h (the dominance-trigger baseline's
    upper end).
Columns match gate_episode, so cfcap.decide.envelope.episodes_from_gate / run consume them unchanged.
"""
from __future__ import annotations

import math

import numpy as np

H_MAX = 6
COVER = 0.99
K_SCREEN = 5.0
SD_INFLATE = 1.5
SIZES = (2, 3, 4)


def _bounds(parent, cands, P, th, st, nl, mu, hm, hg, obs, sl, plant):
    """gate_episode's per-probe-set fit and per-child bounds for one fault hypothesis (hm, hg)."""
    from capxfer.sim import OK, WEAK
    from cfcap.t0 import qs
    from cfcap import theory_task as TT
    n = parent.n
    d, nom = plant.d, plant.nom
    vb = d["v_batt"] / nom["v0"]
    om0 = nom["omega0"] * d["noload_scale"]
    ts0 = nom["tau_s"] * d["stall_scale"]
    driven = (hm == OK) | (hm == WEAK)
    gw = np.where(hm == WEAK, hg, 1.0)
    om_o, u_o = np.abs(obs["om"][sl]), np.abs(obs["duty"][sl])
    track = (u_o < 0.999) & (om_o > 0.5)
    nl_k = om_o / (om0 * gw * vb * np.maximum(u_o, 1e-6))
    nl_hat = np.ones(n)
    for i in np.flatnonzero(driven):
        t = track[:, i]
        nl_hat[i] = np.median(nl_k[t, i]) if t.any() else om_o[:, i].max() / (om0 * gw[i] * vb)
    nl_hat = np.clip(np.nan_to_num(nl_hat, nan=1.0, posinf=1.0), 0.05, 5.0)
    vmax = om_o.max(0)
    nl_pi = np.where(driven, np.minimum(nl_hat, vmax / (om0 * gw * vb)), 1.0)
    out = []
    for cand in cands:
        src = cand.src
        nc = cand.asm.n
        xis = qs.decision_legs(cand.asm)
        cpl = qs.Plant(cand.asm, P, th, stall=st[src], noload=nl[src], mu_f=mu[src])
        omr = cpl.ik(xis)
        ph = qs.Plant(cand.asm, P, th, stall=nl_hat[src], noload=nl_hat[src], mu_f=np.ones(nc))
        r = qs.solve(ph, hm[src][None], hg[src][None], omr)
        rho_p = qs.rho_of(r["xi"][0], xis, cand.asm.char_length())
        drv = driven[src]
        us = (np.abs(r["om"][0]) > TT.DOM_SPEED_TOL * vmax[src][None]) & drv[None]
        pas = ~drv
        cs = [(sf, nf, mf, pb) for sf in (TT.PI_ST, 0.67, 1.0) for nf in ("pi", "fit", "box") for mf in (TT.PI_MU, 1.1)
              for pb in (0, 1)]
        cs += [(5.0, "pi", mf, pb) for mf in (TT.PI_MU, 1.1) for pb in (0, 1)]
        hr = (np.stack([np.where(pas, (1.08 if pb else 0.92),
                                 1.08 if sf == 1.0 else (5.0 if sf == 5.0 else sf * nl_hat[src])) for sf, _, _, pb in cs]),
              np.stack([np.where(pas, (0.92 if pb else 1.08), nl_pi[src] if nf == "pi" else
                                 (nl_hat[src] if nf == "fit" else np.full(nc, 1.08))) for _, nf, _, pb in cs]),
              np.stack([np.full(nc, mf) for _, _, mf, _ in cs]))
        rpi = qs.solve(ph, np.tile(hm[src], (len(cs), 1)), np.tile(hg[src], (len(cs), 1)), omr, het_rows=hr)
        rho_pi = qs.rho_of(rpi["xi"], xis, cand.asm.char_length()).min(0)
        out.append((rho_p, np.minimum(rho_pi, rho_p), us.any(1)))
    return out


def episode(seed, P, th, iters=40, sizes=SIZES):
    from capxfer.sim import OK, WEAK, COAST, BRAKE
    from cfcap import theory_task as TT
    from cfcap.decide import t0data as TD
    from cfcap.t0 import qs, stagea, candidates as C, probesets as PS, observe as O
    from cfcap.t0 import envelope as EV
    sev = TT.gate_law(seed)
    rng = np.random.default_rng(30_000 + seed)                 # E1 truth stream (identical draws for sizes = E1's)
    parent = stagea.random_parent(rng, P, tuple(sizes))
    n = parent.n
    st, nl, mu = O.sample_het(rng, parent, True)
    tm, tg = np.zeros(n, np.int64), np.ones(n)
    for w in rng.choice(n, int(rng.integers(1, 3)), replace=False):
        md = int(rng.choice([WEAK, WEAK, COAST, BRAKE]))
        tm[w], tg[w] = md, (float(rng.uniform(0.3, 1.0)) if md == WEAK else 1.0)
    env = EV.motor_envelope("knee" if sev else "none", sev, np.random.default_rng((40_000 + seed, sev)), n)
    plant = qs.Plant(parent, P, th, stall=st, noload=nl, mu_f=mu)
    nplant = qs.Plant(parent, P, th)
    legs = PS.probe_legs(plant, "v2") + PS.probe_legs(plant, "clamp")
    om_ref = np.stack([l[1] for l in legs])
    obs = EV.solve_knee(plant, tm, tg, om_ref, env, iters)
    # robot-visible noisy signals; the bounds use the noisy om / duty too
    nrng = np.random.default_rng((50_000 + seed,))
    sig = O.signals(dict(xi=obs["xi"][None], om=obs["om"][None], duty=obs["duty"][None]))[0]   # (10, 1 + 2n)
    y = sig + O.sigma(sig, "floor", False) * nrng.standard_normal(sig.shape)
    obs_n = dict(om=y[:, 1:1 + n], duty=np.clip(y[:, 1 + n:], -1, 1))
    HM, HG = stagea.hypotheses(n)
    lp = TD.log_prior(HM, n)
    cands = C.children(parent, P)
    res = {}
    meta = {}
    # fault posterior from the v2 legs only: the two-constant likelihood is not trusted at clamp speeds (the knee
    # may bind there), so the clamp legs enter only the model-free fit / identified set, never the fault likelihood
    sl2 = TT.PROBE_IDX["v2"]
    yk = y[sl2]
    sd = SD_INFLATE * O.sigma(yk, "floor", True, O.nuisance_sd(nplant, legs[:6], np.random.default_rng((60_000 + seed,)), 8))
    alive = np.arange(len(HM))
    ll = np.zeros(len(HM))
    for li in range(yk.shape[0]):
        r = qs.solve(nplant, HM[alive], HG[alive], om_ref[sl2][li:li + 1])
        z = (O.signals(r)[:, 0] - yk[li][None]) / sd[li][None]
        ll[alive] += -0.5 * (z * z).sum(1)
        alive = alive[np.abs(z).max(1) <= K_SCREEN]
        if len(alive) == 0:
            break
    meta["n_screen"] = len(alive)
    if len(alive) == 0:
        meta["empty_screen"] = 1
        res = {key: None for key in TT.PROBE_IDX}
    else:
        lw = lp[alive] + ll[alive]
        w = np.exp(lw - lw.max()); w /= w.sum()
        o = np.argsort(-w)
        k = int(min(H_MAX, np.searchsorted(np.cumsum(w[o]), COVER) + 1))
        kept, wk = alive[o[:k]], w[o[:k]]
        meta["mass_dropped"] = float(1 - wk.sum())
        meta["truth_kept"] = int(any(((HM[h] == tm) & ((tm != WEAK) | (np.abs(HG[h] - tg) <= 0.051))).all()
                                     for h in kept))
        wk = wk / wk.sum()
        for key, sl in TT.PROBE_IDX.items():
            res[key] = ([_bounds(parent, cands, P, th, st, nl, mu, HM[h], HG[h], obs_n, sl, plant) for h in kept], wk)
    out = {k: [] for k in ("seed", "law", "cand", "cand_size", "has_fault", "leg", "rho_true", "conv")}
    for key in TT.PROBE_IDX:
        for c in ("rho_par", "rho_pi", "rho_par_hi", "undom_speed"):
            out[f"{c}_{key}"] = []
    for ci, cand in enumerate(cands):
        src = cand.src
        xis = qs.decision_legs(cand.asm)
        cpl = qs.Plant(cand.asm, P, th, stall=st[src], noload=nl[src], mu_f=mu[src])
        cenv = {kk: (v[src] if isinstance(v, np.ndarray) else v) for kk, v in env.items()}
        rt = EV.solve_knee(cpl, tm[src], tg[src], cpl.ik(xis), cenv, iters)
        out["rho_true"].append(qs.rho_of(rt["xi"], xis, cand.asm.char_length()))
        out["conv"].append(rt["conv"] & bool(obs["conv"].all()))
        out["seed"].append(np.full(10, seed)); out["law"].append(np.full(10, sev))
        out["cand"].append(np.array([cand.name] * 10)); out["cand_size"].append(np.full(10, cand.size))
        out["has_fault"].append(np.full(10, int((tm[src] != OK).any()))); out["leg"].append(np.arange(10))
        for key in TT.PROBE_IDX:
            if res[key] is None:                               # empty screen: support unknown -> no bound
                nanv = np.full(10, np.nan)
                out[f"rho_par_{key}"].append(nanv); out[f"rho_pi_{key}"].append(nanv)
                out[f"rho_par_hi_{key}"].append(nanv); out[f"undom_speed_{key}"].append(np.ones(10, bool))
                continue
            b, wk = res[key]
            rp = np.stack([bb[ci][0] for bb in b]); rpi = np.stack([bb[ci][1] for bb in b])
            out[f"rho_par_{key}"].append(wk @ rp)
            out[f"rho_pi_{key}"].append(rpi.min(0))
            out[f"rho_par_hi_{key}"].append(rp.max(0))
            out[f"undom_speed_{key}"].append(b[0][ci][2])
    cols = {k: np.concatenate(v) for k, v in out.items()}
    for k, v in meta.items():
        cols[k] = np.full(len(cols["leg"]), v)
    return cols
