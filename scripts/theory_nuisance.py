"""Theory (a)+(b): what the parent's v2 identifies per wheel, and the nuisance-induced ambiguity band (T0, exploratory).

Per driven wheel the T0 motor law is linear in two inverse constants (THEORY_cfcap.md Lemma 1):
    published duty u = a * alpha + b * beta,   a = tau_c / tau_s_nom,  b = omega / omega0_nom,
    alpha = 1 / (g vb st),  beta = 1 / (g vb nl);   the wheel saturates on the line a alpha + b beta = 1.
v2 runs every tracking wheel at a ~ 0 (no load torque), so it identifies beta (to the duty noise) but not alpha; mu is
not excited either. The child's identified interval is then the image of the box
    x_i = log(alpha_i / alpha_i_true) in [-w, w],  w = log((1+delta)/(1-delta))    (alpha/beta = nl/st, unknown ratio)
    y_k = log(mu_k / mu_k_true)       in [-w_mu, w_mu]
    z_i = log(beta_i / beta_i_true)   in [-eps_i, eps_i],  eps_i = k sigma_u / |u_i|       (duty noise only)
(variant 'ident'), against the A2 'prior'-like variant where st, nl, mu are all free in their prior box ('none').
The box image is computed (1) linearly from finite-difference gradients and (2) exactly at the two gradient-sign
corners per leg. Also the speed-only (nonparametric ceiling) class: every child wheel's no-load ceiling is only known
to exceed the parent's max probed speed (v2, or v2+clamp) unless it saturated there.

  run      --seeds 0:40   -> $CFCAP_OUT/theory/nuis_<s0>_<s1>.pkl
  analyze  -> results/theory/nuisance_curve.csv, nuisance_wheel_ident.csv, coverage_speed_only.csv
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pathlib
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "theory"
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", os.path.expanduser("~/.cache/modular-runs/2026-11/cfcap"))) / "theory"
DELTAS = (0.02, 0.04, 0.08, 0.12, 0.16)
TAUS = (0.6, 0.7, 0.8)
K_SIG = 3.0
H = 1e-3


NP_SETS = {"v2": ("v2",), "v2clamp": ("v2", "clamp"), "v2obl": ("v2", "obl"), "v2yawhi": ("v2", "yawhi"),
           "v2clampyawhi": ("v2", "clamp", "yawhi")}


def extra_legs(plant, kind):
    """Probe kinds beyond cfcap.t0.probesets: obl = +-22.5 deg translations at 0.30 m/s (capxfer Prop. 4 heading);
    yawhi = +-2x the v2 yaw rate (reaches the yaw operating points of smaller children; saturating is allowed)."""
    from cfcap.t0 import probesets as PS
    if kind in ("v2", "clamp", "moddiff", "tug"):
        return PS.probe_legs(plant, kind)
    if kind == "obl":
        c, s_ = 0.3 * math.cos(math.pi / 8), 0.3 * math.sin(math.pi / 8)
        xis = [[c, s_, 0], [-c, -s_, 0], [c, -s_, 0], [-c, s_, 0]]
    elif kind == "yawhi":
        w = 2 * PS.v2_twists(plant.asm)[4, 2]
        xis = [[0, 0, w], [0, 0, -w]]
    else:
        raise ValueError(kind)
    om = plant.ik(np.asarray(xis, float))
    return [(f"{kind}{k}", om[k], np.full(plant.asm.n, np.nan)) for k in range(len(om))]


def run_episode(seed, P, th, sizes=(4, 5, 6), faults="weak1", np_sets=("v2", "v2clamp"), n_merge=4, deltas=None):
    """faults: 'weak1' (healthy if seed % 4 == 0, else one weak wheel g ~ U(0.3, 1)) or 'mix' (healthy if seed % 5 == 0,
    else 1-2 faults, weak g ~ U(0.3, 1) / coast / brake w.p. 0.5 / 0.25 / 0.25). Defaults reproduce results/theory."""
    from cfcap.t0 import qs, stagea, candidates as C, probesets as PS, observe as O
    from capxfer.sim import WEAK, COAST, BRAKE
    DELTAS_ = DELTAS if deltas is None else tuple(deltas)
    rng = np.random.default_rng(10_000 + seed)
    parent = stagea.random_parent(rng, P, tuple(sizes))
    n = parent.n
    st, nl, mu = O.sample_het(rng, parent, True)
    tm, tg = np.zeros(n, np.int64), np.ones(n)
    if faults == "weak1":
        if seed % 4 != 0:
            w = int(rng.integers(n))
            tm[w], tg[w] = WEAK, float(rng.uniform(0.3, 1.0))
    elif seed % 5 != 0:
        for w in rng.choice(n, int(rng.integers(1, 3)), replace=False):
            md = int(rng.choice([WEAK, WEAK, COAST, BRAKE]))
            tm[w], tg[w] = md, (float(rng.uniform(0.3, 1.0)) if md == WEAK else 1.0)
    other = stagea.random_parent(rng, P, (1, 2), name="other")
    ost, onl, omu = O.sample_het(rng, other, True)
    plant = qs.Plant(parent, P, th, stall=st, noload=nl, mu_f=mu)
    vb = plant.d["v_batt"] / plant.nom["v0"]
    om0 = plant.nom["omega0"] * plant.d["noload_scale"]
    # ---- parent observations: v2 and clamp
    obs = {}
    kinds = sorted({k for s_ in np_sets for k in NP_SETS[s_]} | {"v2", "clamp"})
    for kind in kinds:
        L = extra_legs(plant, kind)
        r = qs.solve(plant, tm[None], tg[None], np.stack([l[1] for l in L]), np.stack([l[2] for l in L]))
        obs[kind] = r
    u2, om2 = obs["v2"]["duty"][0], obs["v2"]["om"][0]                    # (6, n)
    speed_part = om2 / (om0 * nl * tg * vb)
    torque_share = np.where(np.abs(u2) > 1e-6, 1 - speed_part / u2, np.nan)
    sat = {k: (np.abs(obs[k]["duty"][0]) > 0.999).any(0) for k in obs}
    vmax = {k: np.abs(obs[k]["om"][0]).max(0) for k in obs}
    for s_ in set(np_sets) | {"v2clamp"}:
        vmax[s_] = np.max([vmax[k] for k in NP_SETS[s_]], 0)
        sat[s_] = np.any([sat[k] for k in NP_SETS[s_]], 0)
    umax = np.abs(u2).max(0)
    wheel_rows = [dict(seed=seed, wheel=i, fault=int(tm[i]), g=tg[i], umax_v2=umax[i],
                       torque_share_med=float(np.nanmedian(np.abs(torque_share[:, i]))),
                       torque_share_max=float(np.nanmax(np.abs(torque_share[:, i]))), sat_v2=bool(sat["v2"][i]),
                       sat_v2clamp=bool(sat["v2clamp"][i])) for i in range(n)]
    sig_u = {"sensor": O.SENSOR["u"], "floor": math.hypot(O.SENSOR["u"], O.FLOOR["u"])}
    # per parent wheel: duty-noise box on log beta
    eps = {p: np.minimum(K_SIG * s / np.maximum(umax, 1e-3), 0.5) for p, s in sig_u.items()}
    omu_wheel = omu
    cands = C.children(parent, P) + C.sample_merges(parent, other, P, rng, n_merge)
    rows = []
    for cand in cands:
        src = cand.src
        nc, mods = cand.asm.n, len(cand.asm.modules)
        cm = C.carry(tm[None], np.zeros((1, other.n), np.int64), src)[0]
        cg = C.carry(tg[None], np.ones((1, other.n)), src)[0]
        cst, cnl, cmu = C.carry(st, ost, src), C.carry(nl, onl, src), C.carry(mu, omu_wheel, src)
        cpl = qs.Plant(cand.asm, P, th, stall=cst, noload=cnl, mu_f=cmu)
        xis = qs.decision_legs(cand.asm)
        omr = cpl.ik(xis)
        wm = cand.asm.wheel_module
        # truth-ratio box centre: alpha/beta = nl/st; the prior ratio box is [(1-d)/(1+d), (1+d)/(1-d)]
        lr = np.log(cnl / cst)
        # coordinates: x_i (alpha), z_i (beta), y_k (mu per module)
        D = 2 * nc + mods

        def het_of(c):
            c = np.atleast_2d(c)
            x, z, y = c[:, :nc], c[:, nc:2 * nc], c[:, 2 * nc:]
            stx = cst[None] * np.exp(-x)                                   # alpha ~ 1/st
            nlx = cnl[None] * np.exp(-z)                                   # beta ~ 1/nl
            mux = cmu[None] * np.exp(y[:, wm])
            return stx, nlx, mux

        def rho_rows(c, legs=None):
            c = np.atleast_2d(c)
            hr = het_of(c)
            B = len(c)
            o = omr if legs is None else omr[legs]
            xx = xis if legs is None else xis[legs]
            r = qs.solve(cpl, np.tile(cm, (B, 1)), np.tile(cg, (B, 1)), o, het_rows=hr)
            return qs.rho_of(r["xi"], xx, cand.asm.char_length())

        base = np.zeros(D)
        pert = np.vstack([base, base + H * np.eye(D)])
        R = rho_rows(pert)                                                 # (1+D, 10)
        rho_t = R[0]
        grad = (R[1:] - R[0]) / H                                          # (D, 10)
        # box half-widths per variant and delta
        src_par = np.where(src >= 0, src, 0)
        is_par = src >= 0
        variants = []
        for d in DELTAS_:
            w = math.log((1 + d) / (1 - d))
            wmu = math.log((1 + 1.25 * d) / (1 - 1.25 * d))
            for p in ("sensor", "floor"):
                e = np.where(is_par, eps[p][src_par], eps[p].mean())
                variants.append((f"ident_{p}", d, np.concatenate([np.full(nc, w), e, np.full(mods, wmu)])))
            wn = math.log((1 + d) / (1 - d)) / 2                           # st and nl each +-d (prior-like)
            variants.append(("none", d, np.concatenate([np.full(nc, wn), np.full(nc, wn), np.full(mods, wmu)])))
        lo_c = np.zeros((len(variants), 10)); hi_c = np.zeros((len(variants), 10))
        lo_l = np.zeros((len(variants), 10)); hi_l = np.zeros((len(variants), 10))
        for li in range(10):
            sg = np.sign(grad[:, li])
            C_ = []
            for vi, (_, _, wv) in enumerate(variants):
                lo_l[vi, li] = rho_t[li] - np.abs(grad[:, li]) @ wv
                hi_l[vi, li] = rho_t[li] + np.abs(grad[:, li]) @ wv
                C_ += [-sg * wv, sg * wv]
            rr = rho_rows(np.array(C_), legs=[li])[:, 0]
            lo_c[:, li] = np.minimum(rr[0::2], rr[1::2])
            hi_c[:, li] = np.maximum(rr[0::2], rr[1::2])
        # speed-only (nonparametric ceiling) class
        np_rows = []
        for key in np_sets:
            gp = np.ones(nc)
            for j in range(nc):
                if src[j] >= 0 and cm[j] in (0, WEAK):
                    s = src[j]
                    gp[j] = tg[s] if sat[key][s] else min(1.0, vmax[key][s] / (om0 * nl[s] * vb))
            modes = np.where(gp < 1, WEAK, cm)
            np_rows.append((modes, gp))
        Mlo = np.stack([m for m, _ in np_rows]); Glo = np.stack([g for _, g in np_rows])
        nk = len(np_sets)
        hr = (np.tile(cst, (nk, 1)), np.tile(cnl, (nk, 1)), np.tile(cmu, (nk, 1)))
        rn = qs.solve(cpl, Mlo, Glo, omr, het_rows=hr)
        rho_np_lo = qs.rho_of(rn["xi"], xis, cand.asm.char_length())       # (2, 10)
        passive = (cm == COAST) | (cm == BRAKE)                            # passive modes are identified (A2)
        rh = qs.solve(cpl, np.where(passive, cm, 0)[None], np.ones((1, nc)), omr,
                      het_rows=(cst[None], cnl[None], cmu[None]))
        rho_np_hi = np.maximum(qs.rho_of(rh["xi"], xis, cand.asm.char_length())[0], rho_t)
        has_fault = bool((cm != 0).any())
        for li in range(10):
            row = dict(seed=seed, cand=cand.name, kind=cand.kind, cand_size=cand.size, has_fault=has_fault, leg=li,
                       rho_t=float(rho_t[li]), S_alpha=float(np.abs(grad[:nc, li]).sum()),
                       S_beta=float(np.abs(grad[nc:2 * nc, li]).sum()), S_mu=float(np.abs(grad[2 * nc:, li]).sum()),
                       np_hi=float(rho_np_hi[li]), n_faults=int((tm != 0).sum()),
                       parent_size=len(parent.modules))
            for ki, key in enumerate(np_sets):
                row[f"np_lo_{key}"] = float(rho_np_lo[ki, li])
            for vi, (vn, d, _) in enumerate(variants):
                row[f"lo_{vn}_{d}"] = lo_c[vi, li]; row[f"hi_{vn}_{d}"] = hi_c[vi, li]
                row[f"lolin_{vn}_{d}"] = lo_l[vi, li]; row[f"hilin_{vn}_{d}"] = hi_l[vi, li]
            rows.append(row)
    return rows, wheel_rows


def cmd_run(a):
    import torch
    torch.set_num_threads(4)
    from cfcap.t0 import qs
    P, th = qs.load_params(), qs.load_theta()
    s0, s1 = map(int, a.seeds.split(":"))
    rows, wrows = [], []
    t0 = time.time()
    for seed in range(s0, s1):
        r, w = run_episode(seed, P, th)
        rows += r; wrows += w
        print(f"seed {seed} rows {len(r)} {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_pickle(OUT / f"nuis_{s0:05d}_{s1:05d}.pkl")
    pd.DataFrame(wrows).to_pickle(OUT / f"nuiswheel_{s0:05d}_{s1:05d}.pkl")


def amb(lo, hi, tau):
    return (lo < tau) & (hi >= tau)


def cmd_analyze(a):
    D = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(str(OUT / "nuis_*.pkl")))])
    W = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(str(OUT / "nuiswheel_*.pkl")))])
    RES.mkdir(parents=True, exist_ok=True)
    out = []
    for vn in ("ident_sensor", "ident_floor", "none"):
        for d in DELTAS:
            for sub, mask in (("all", np.ones(len(D), bool)), ("faulted_cand", D.has_fault.values)):
                X = D[mask]
                a_ex = np.mean([amb(X[f"lo_{vn}_{d}"], X[f"hi_{vn}_{d}"], t).mean() for t in TAUS])
                a_li = np.mean([amb(X[f"lolin_{vn}_{d}"], X[f"hilin_{vn}_{d}"], t).mean() for t in TAUS])
                # smooth first-order prediction: share = mean over tau of P(|rho - tau| <= half-width)
                hw = (X[f"hilin_{vn}_{d}"] - X[f"lolin_{vn}_{d}"]) / 2
                a_sm = np.mean([(np.abs(X.rho_t - t) <= hw).mean() for t in TAUS])
                cover = np.mean([((X[f"lo_{vn}_{d}"] <= X.rho_t + 1e-9) & (X[f"hi_{vn}_{d}"] >= X.rho_t - 1e-9)).mean()])
                out.append(dict(variant=vn, delta=d, subset=sub, n_dec=len(X) * len(TAUS), amb_exact_corner=a_ex,
                                amb_linear=a_li, amb_smooth=a_sm,
                                median_halfwidth=float(hw.median()), p99_halfwidth=float(hw.quantile(0.99))))
    S = pd.DataFrame(out)
    S.to_csv(RES / "nuisance_curve.csv", index=False)
    print(S.to_string())
    # slope of the band: half-width / delta for the alpha and mu parts
    G = pd.DataFrame(dict(S_alpha_med=[D.S_alpha.median()], S_alpha_p99=[D.S_alpha.quantile(.99)],
                          S_mu_med=[D.S_mu.median()], S_mu_p99=[D.S_mu.quantile(.99)],
                          S_beta_med=[D.S_beta.median()], S_beta_p99=[D.S_beta.quantile(.99)],
                          share_S_alpha_zero=[(D.S_alpha < 1e-4).mean()], share_S_mu_zero=[(D.S_mu < 1e-4).mean()],
                          n_parents=[D.seed.nunique()], n_cand_legs=[len(D)]))
    G.to_csv(RES / "nuisance_sensitivity.csv", index=False)
    print(G.to_string())
    # speed-only class
    sp = []
    for key in ("v2", "v2clamp"):
        for sub, mask in (("all", np.ones(len(D), bool)), ("faulted_cand", D.has_fault.values)):
            X = D[mask]
            for t in TAUS:
                sp.append(dict(probe=key, subset=sub, tau=t, n=len(X),
                               amb_share=amb(X[f"np_lo_{key}"], X.np_hi, t).mean(),
                               cert_frac_capable=((X[f"np_lo_{key}"] >= t) & (X.rho_t >= t)).sum() / max(1, (X.rho_t >= t).sum())))
    SP = pd.DataFrame(sp)
    SP.to_csv(RES / "coverage_speed_only.csv", index=False)
    print(SP.to_string())
    Wd = W.assign(faulted=W.fault != 0).groupby("faulted").agg(
        n=("wheel", "size"), torque_share_med=("torque_share_med", "median"),
        torque_share_max_p90=("torque_share_max", lambda s: s.quantile(.9)), sat_v2=("sat_v2", "mean"),
        sat_v2clamp=("sat_v2clamp", "mean"))
    Wd.reset_index().to_csv(RES / "nuisance_wheel_ident.csv", index=False)
    print(Wd.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--seeds", default="0:40")
    sp.add_parser("analyze")
    a = ap.parse_args()
    dict(run=cmd_run, analyze=cmd_analyze)[a.cmd](a)
