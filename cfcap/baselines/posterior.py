"""Simulator-likelihood posteriors over parent latents, propagated to every candidate (shared code).

Two posteriors over the parent's v2 observation, both with T1 (`cfcap.t1.plant`) as the likelihood:
  * `enumerate_known`: exact enumeration over every <= 2-fault wheel-mode vector (weak g on GRID, coast, brake) with
    every nuisance at its true value (the A6 oracle-nuisance posterior; heterogeneity on);
  * `nuisance_smc`: faults x sampled nuisances. Unknown: per-wheel/per-module motor spread, per-module battery offset
    and mu factor, the assembly's normal-load split (HET prior) and the floor mu (DR prior); the rest of the domain
    (s0, latency, v_batt, ...) and the mounting-yaw errors stay known, as in A6. Gyro bias is estimated from the
    0.5 s rest at the start of v2. Algorithm (resample-move SMC with one adaptive tempering step):
      1. screen every fault hypothesis at the prior-mean nuisance with a tempered likelihood (sd x SCREEN_TEMPER);
      2. keep the top K hypotheses; joint particles (hypothesis, nuisance draw) = K x M with shared draws;
      3. adaptive tempering ladder (at most max_stages): the next beta keeps the ESS of the incremental weights
         >= ess_frac x particles; resample N particles; mh_steps random-walk Metropolis moves on the nuisance;
      4. stop at beta = 1 or after max_stages (then the result is a tempered, i.e. wider, posterior: beta reported).
Likelihood: independent Gaussian on pre_duty, quantised encoder speed and gz with sd duty 0.02, speed 0.10 rad/s,
gz 0.005 rad/s (A6; tempered above the actual noise). Prior over faults: the generator's (p_none 0.2, 1 or 2 faults
equally likely, modes uniform, g uniform on GRID).

Propagation (`propagate`): each posterior particle's latents are carried to every candidate through `plant.subset`
and the candidate's rho_rel is simulated with T1 (closed-loop decision legs, divided by the same particle's
fault-free rollout). Candidate summaries: weighted mean, sd, 5 % quantile per (candidate, leg).

Role nuisance: reuse `sim_signals`, `loglik`, `sample_nuisance`, `nuisance_smc` (additive changes only, please).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from cfcap.t1 import capability as cap
from cfcap.t1 import plant

GRID = np.round(np.arange(0.30, 0.9001, 0.10), 2)
SD = dict(duty=0.02, speed=0.10, gz=0.005)
CHUNK_WHEELS = 1 << 16
NUIS_KEYS = ("stall_w", "noload_w", "vbatt_off", "mu_mod", "load_z")


# ----------------------------------------------------------------------------- hypotheses and prior
def fault_states(grid=GRID):
    return [(plant.WEAK, float(g)) for g in grid] + [(plant.COAST, 1.0), (plant.BRAKE, 1.0)]


def hypotheses(n, grid=GRID, max_faults=2):
    """Every <= max_faults wheel-mode vector -> mode (H, n) int64, g (H, n) float32, n_faults (H,)."""
    st = fault_states(grid)
    rows_m, rows_g, nf = [np.zeros(n, np.int64)], [np.ones(n, np.float32)], [0]
    for k in range(1, max_faults + 1):
        for ws in itertools.combinations(range(n), k):
            for ss in itertools.product(st, repeat=k):
                m, g = np.zeros(n, np.int64), np.ones(n, np.float32)
                for w, (mm, gg) in zip(ws, ss):
                    m[w], g[w] = mm, gg
                rows_m.append(m)
                rows_g.append(g)
                nf.append(k)
    return np.array(rows_m), np.array(rows_g), np.array(nf)


def log_prior(nf, n, p_none=0.2, k_max=2):
    """Generator prior over the fault count and wheel set: P(0) = p_none, P(k) = (1 - p_none)/k_max for k = 1..k_max,
    the k wheels uniform without replacement."""
    nf = np.asarray(nf)
    lp = np.full(nf.shape, -np.inf)
    lp[nf == 0] = math.log(p_none)
    for k in range(1, k_max + 1):
        lp[nf == k] = math.log((1 - p_none) / k_max) - math.log(math.comb(n, k))
    return lp


def log_prior_full(HM, nf, n, grid=GRID, p_none=0.2, k_max=2):
    """+ each faulted wheel's state: weak(g) 1/(3 |grid|) (g uniform on the grid), coast 1/3, brake 1/3."""
    lp = log_prior(nf, n, p_none, k_max)
    n_weak = (HM == plant.WEAK).sum(1)
    n_pas = ((HM == plant.COAST) | (HM == plant.BRAKE)).sum(1)
    return lp + n_weak * (math.log(1 / 3) - math.log(len(grid))) + n_pas * math.log(1 / 3)


# ----------------------------------------------------------------------------- likelihood
def _signals(R, cpr):
    th = R["wheel_angle"].double()
    cnt = torch.floor(th / (2 * np.pi) * cpr)
    spd = torch.zeros_like(th)
    spd[:, 1:] = torch.diff(cnt, dim=1) / cpr * 2 * np.pi / plant.DT_OBS
    return R["duty"], spd.float(), R["twist"][:, :, 2]


def rows_from(lat1, B):
    """Latents of one episode (B = 1, on device) repeated B times."""
    return plant._index(lat1, torch.zeros(B, dtype=torch.long, device=lat1["mode"].device))


ENGINE = "compile"


def loglik(asm, par, cmd, T, lat_rows, y, gyro_bias=0.0, sd=SD, chunk=None, engine=None):
    """Log-likelihood of observation y (dict pre_duty (T,n), speed (T,n), gz (T,)) for every latent row.
    lat_rows: latents dict with B rows (any faults / nuisances). Returns (B,) float64 numpy (temper 1).
    With engine "compile" every chunk is padded to the same row count (one compiled specialisation per wheel count
    and chunk size, as A6)."""
    from cfcap.t1 import params as P
    engine = engine or (ENGINE if lat_rows["mode"].device.type == "cuda" else "eager")
    B, n = lat_rows["mode"].shape
    chunk = chunk or max(1, CHUNK_WHEELS // n)
    if engine == "compile":
        chunk = min(chunk, 1 << max(7, int(np.ceil(np.log2(max(B, 1))))))   # power-of-two shapes only
    dev = lat_rows["mode"].device
    yd, ys, yg = (y[k].to(dev) for k in ("pre_duty", "speed", "gz"))
    out = np.empty(B)
    for b0 in range(0, B, chunk):
        nb = min(B, b0 + chunk) - b0
        size = chunk if engine == "compile" else nb
        idx = torch.arange(b0, b0 + size, device=dev).clamp(max=B - 1)
        lr = plant._index(lat_rows, idx)
        R = plant.simulate_bucket([dict(asm=asm, lat=lr, cmd=cmd)], par, T, device=dev, engine=engine)[0]
        d, s, g = _signals(R, P.CPR)
        Tm = min(d.shape[1], yd.shape[0])
        l = -0.5 * (((d[:, :Tm] - yd[:Tm]) / sd["duty"]) ** 2).sum((1, 2)) \
            - 0.5 * (((s[:, :Tm] - ys[:Tm]) / sd["speed"]) ** 2).sum((1, 2)) \
            - 0.5 * (((g[:, :Tm] + gyro_bias - yg[:Tm]) / sd["gz"]) ** 2).sum(1)
        out[b0:b0 + nb] = l[:nb].double().cpu().numpy()
        del R, d, s, g
    return out


def gyro_bias_rest(obs, cmd):
    rest = np.abs(np.asarray(cmd)[:obs["gz"].shape[0]]).sum(1) == 0
    rest[int(1.0 / plant.DT_OBS):] = False
    return float(obs["gz"][torch.as_tensor(rest)].mean()) if rest.any() else 0.0


def set_faults(lat, HM, HG):
    lat = dict(lat)
    dev = lat["mode"].device
    lat["mode"] = torch.as_tensor(HM, device=dev).long()
    lat["g"] = torch.as_tensor(HG, device=dev).float()
    return lat


# ----------------------------------------------------------------------------- nuisance prior
def sample_nuisance(asm, lat_true, M, rng, het=plant.HET_FULL, mu_range=(0.35, 1.0)):
    """M nuisance particles: the episode's latents with every NUIS_KEYS field redrawn from the HET prior and the
    floor mu from the DR prior; the rest of the domain and the mounting-yaw errors are kept (known)."""
    dev = lat_true["mode"].device
    base = rows_from(lat_true, M)
    fresh = plant.sample_latents(asm, M, rng, het, domain=base["dom"], device=dev)
    out = dict(base)
    for k in NUIS_KEYS:
        out[k] = fresh[k]
    dom = dict(base["dom"])
    dom["mu"] = torch.as_tensor(rng.uniform(*mu_range, M), dtype=torch.float32, device=dev)
    out["dom"] = dom
    return out


def nuisance_mean(asm, lat_true, mu_range=(0.35, 1.0)):
    out = rows_from(lat_true, 1)
    out = dict(out)
    for k in NUIS_KEYS:
        out[k] = torch.ones_like(out[k]) if k in ("stall_w", "noload_w", "mu_mod") else torch.zeros_like(out[k])
    dom = dict(out["dom"])
    dom["mu"] = torch.full_like(dom["mu"], float(np.mean(mu_range)))
    out["dom"] = dom
    return out


def _nuis_vec(lat):
    """Flatten the nuisance of B rows -> (B, D) tensor (and a function to write it back)."""
    parts = [lat[k] for k in NUIS_KEYS] + [lat["dom"]["mu"][:, None]]
    sizes = [p.shape[1] for p in parts]
    return torch.cat(parts, 1), sizes


def _set_nuis(lat, vec, sizes):
    out = dict(lat)
    o = 0
    for k, s in zip(NUIS_KEYS + ("mu",), sizes):
        v = vec[:, o:o + s]
        o += s
        if k == "mu":
            dom = dict(lat["dom"])
            dom["mu"] = v[:, 0].contiguous()
            out["dom"] = dom
        else:
            out[k] = v.contiguous()
    return out


def _nuis_bounds(lat, het=plant.HET_FULL, mu_range=(0.35, 1.0)):
    """(lo, hi, step) per nuisance coordinate of the HET prior (uniform ones) and mu; load_z is Gaussian."""
    vec, sizes = _nuis_vec(lat)
    D = vec.shape[1]
    lo, hi, step = (torch.full((D,), -1e9, device=vec.device) for _ in range(3))
    o = 0
    spread = dict(stall_w=het.wheel_motor + het.module_motor, noload_w=het.wheel_motor + het.module_motor,
                  vbatt_off=het.vbatt_module, mu_mod=het.mu_module, load_z=None)
    for k, s in zip(NUIS_KEYS + ("mu",), sizes):
        if k == "mu":
            lo[o:o + s], hi[o:o + s], step[o:o + s] = mu_range[0], mu_range[1], 0.05
        elif k == "load_z":
            lo[o:o + s], hi[o:o + s], step[o:o + s] = -1e9, 1e9, 0.25 * max(het.load_split, 1e-6)
        else:
            h = spread[k]
            c = 0.0 if k == "vbatt_off" else 1.0
            lo[o:o + s], hi[o:o + s], step[o:o + s] = c - h, c + h, 0.25 * max(h, 1e-6)
        o += s
    return lo, hi, step, sizes


def _ess(lw):
    w = np.exp(lw - lw.max())
    return w.sum() ** 2 / (w * w).sum()


def _temper(ll, lp, frac):
    """Largest beta in (0, 1] with ESS(lp + beta ll) >= frac * len."""
    target = frac * len(ll)
    if _ess(lp + ll) >= target:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _ess(lp + mid * ll) >= target:
            lo = mid
        else:
            hi = mid
    return max(lo, 1e-8)


# ----------------------------------------------------------------------------- posteriors
def enumerate_known(asm, par, cmd, T, lat_true, y, grid=GRID):
    """Exact posterior over fault hypotheses with every nuisance known. Returns dict HM, HG, w (normalised),
    lat (the hypotheses' latent rows are lat_true with faults replaced; not materialised)."""
    n = asm.n
    HM, HG, nf = hypotheses(n, grid)
    rows = set_faults(rows_from(lat_true, len(HM)), HM, HG)
    gb = float(lat_true["gyro_bias"][0])
    ll = loglik(asm, par, cmd, T, rows, y, gyro_bias=gb)
    lw = ll + log_prior_full(HM, nf, n, grid)
    w = np.exp(lw - lw.max())
    return dict(HM=HM, HG=HG, w=w / w.sum(), n_sims=len(HM))


def nuisance_smc(asm, par, cmd, T, lat_true, y, rng, grid=GRID, K=96, M=24, N=128, mh_steps=2, screen_temper=3.0,
                 ess_frac=0.5, max_stages=12, het=plant.HET_FULL):
    """Faults x nuisance posterior (see module docstring). lat_true supplies only the known part of the domain and
    the mounting-yaw errors; its NUIS_KEYS / mu / faults / gyro bias are never read. Returns dict of N equally weighted
    particles: lat (N rows), HM, HG, beta, n_sims."""
    n = asm.n
    HM, HG, nf = hypotheses(n, grid)
    lpr = log_prior_full(HM, nf, n, grid)
    gb = gyro_bias_rest(y, cmd)
    sd_s = {k: v * screen_temper for k, v in SD.items()}
    # 1. screen at the prior-mean nuisance
    rows = set_faults(rows_from(nuisance_mean(asm, lat_true), len(HM)), HM, HG)
    ll0 = loglik(asm, par, cmd, T, rows, y, gyro_bias=gb, sd=sd_s)
    top = np.argsort(-(ll0 + lpr))[:K]
    n_sims = len(HM)
    # 2. joint particles: K hypotheses x M shared nuisance draws
    nu = sample_nuisance(asm, lat_true, M, rng, het)
    rep = lambda lat, r: plant._index(lat, torch.arange(lat["mode"].shape[0], device=lat["mode"].device).repeat(r))
    J = rep(nu, len(top))                                             # row = k*M + j  (hypothesis-major)
    hm, hg = np.repeat(HM[top], M, 0), np.repeat(HG[top], M, 0)
    J = set_faults(J, hm, hg)
    ll = loglik(asm, par, cmd, T, J, y, gyro_bias=gb)
    lp = np.repeat(lpr[top], M)
    n_sims += len(ll)
    # 3-4. adaptive tempering ladder: next beta keeps the ESS >= ess_frac * particles; resample; MH nuisance moves
    dev = J["mode"].device
    gen = torch.Generator(device=dev).manual_seed(int(rng.integers(1 << 31)))
    P_, cur_ll, hm_, hg_ = J, ll, hm, hg
    beta, n_stage = 0.0, 0
    while beta < 1.0 and n_stage < max_stages:
        n_stage += 1
        base_lw = lp if n_stage == 1 else np.zeros(len(cur_ll))     # stage 1 carries the prior over hypotheses
        if _ess(base_lw + (1.0 - beta) * cur_ll) >= ess_frac * len(cur_ll):
            nb = 1.0
        else:
            lo_b, hi_b = beta, 1.0
            for _ in range(40):
                mid = 0.5 * (lo_b + hi_b)
                if _ess(base_lw + (mid - beta) * cur_ll) >= ess_frac * len(cur_ll):
                    lo_b = mid
                else:
                    hi_b = mid
            nb = max(lo_b, beta + 1e-9)
        lw = base_lw + (nb - beta) * cur_ll
        w = np.exp(lw - lw.max())
        w /= w.sum()
        beta = nb
        idx = rng.choice(len(w), size=N, p=w)
        P_ = plant._index(P_, torch.as_tensor(idx, device=dev))
        cur_ll, hm_, hg_ = cur_ll[idx], hm_[idx], hg_[idx]
        lo, hi, step, sizes = _nuis_bounds(P_, het)
        for _ in range(mh_steps):
            vec, _ = _nuis_vec(P_)
            prop = vec + step * torch.randn(vec.shape, generator=gen, device=dev)
            inb = ((prop >= lo) & (prop <= hi)).all(1).cpu().numpy()
            Q = _set_nuis(P_, prop, sizes)
            llq = loglik(asm, par, cmd, T, Q, y, gyro_bias=gb)
            n_sims += N
            # Gaussian prior on load_z (sd het.load_split); uniform elsewhere (in-bounds check)
            o = sum(sizes[:NUIS_KEYS.index("load_z")])
            s = sizes[NUIS_KEYS.index("load_z")]
            lz0, lz1 = vec[:, o:o + s], prop[:, o:o + s]
            lpz = (-0.5 * ((lz1 ** 2 - lz0 ** 2).sum(1)) / max(het.load_split, 1e-6) ** 2).cpu().numpy()
            acc = inb & (np.log(rng.uniform(size=N)) < beta * (llq - cur_ll) + lpz)
            acc_t = torch.as_tensor(acc, device=dev)
            P_ = _set_nuis(P_, torch.where(acc_t[:, None], prop, vec), sizes)
            cur_ll = np.where(acc, llq, cur_ll)
    hm, hg, idx = hm_, hg_, np.arange(N)
    return dict(lat=P_, HM=hm[idx], HG=hg[idx], beta=beta, n_sims=n_sims, n_stages=n_stage, gyro_bias=gb)


# ----------------------------------------------------------------------------- propagation to candidates
def wquantile(x, w, q):
    """Weighted quantile along axis 0: x (P, ...), w (P,)."""
    o = np.argsort(x, 0)
    xs = np.take_along_axis(x, o, 0)
    cw = np.cumsum(w[o], 0)
    k = (cw < q - 1e-12).sum(0)
    k = np.minimum(k, x.shape[0] - 1)
    return np.take_along_axis(xs, k[None], 0)[0]


def summarise(rho, w):
    """rho (P, C, L), w (P,) -> mean, sd, q05 (C, L)."""
    w = w / w.sum()
    m = np.tensordot(w, rho, 1)
    sd = np.sqrt(np.maximum(np.tensordot(w, (rho - m) ** 2, 1), 0))
    return m, sd, wquantile(rho, w, 0.05)


def propagate(par, cands, lat_rows, w, device="cuda", plate="plated"):
    """rho_dec (P, C, L) of every particle row (lat_rows, P rows on device) on every candidate."""
    from .episode import candidate_rho
    return candidate_rho(par, cands, lat_rows, device, relative=False, plate=plate)[2]


def top_mass(w, cover=0.999, cap_=32):
    o = np.argsort(-w)
    c = np.cumsum(w[o])
    k = int(min(cap_, np.searchsorted(c, cover) + 1))
    return o[:k]
