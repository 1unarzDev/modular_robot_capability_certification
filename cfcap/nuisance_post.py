"""Nuisance pilot (exploratory): decision ambiguity of the T1 posterior when the robot does NOT know its nuisances.

Pre-registration and thresholds: docs/board/nuisance.md (written before the first run).

Per episode: a parent (pair / square / 2x3, plated theta, module yaws 0), 0-2 true faults (weak g emphasised in
0.6-0.9, coast, brake) and true nuisances drawn from the prior below; a probe routine (v2, v2+clamp, v2+tug,
v2+moddiff, v2+clamp+moddiff+tug) is simulated in T1 and observed (pre_duty + N(0, 0.005), quantised encoder speed,
gz + IMU noise). The posterior over the joint (mode vector, g, nuisances) is sampled by tempered SMC; 64 posterior
draws are pushed through t1.capability (relative=True) for every connected child and 10 random merges with a healthy
1-2-module assembly. A decision (episode, candidate, leg, tau) is ambiguous iff q05 < tau <= q95 of the draws.

Latent space (standard-normal reparametrisation z, prior N(0, I)):
  global DR (params.theta_dr): 13 uniform keys + CoM offset (2 normal); per wheel: g (mixture inverse CDF, used only if
  the wheel is weak), stall and no-load multipliers U(0.92, 1.08), mounting-yaw error N(0, 0.02), load-split N(0, 0.15);
  per module: V_batt offset U(-0.3, 0.3) V, mu factor U(0.9, 1.1).
Discrete prior: P(k faults) = 0.2 / 0.4 / 0.4 (k = 0 / 1 / 2), uniform wheels, mode weak/coast/brake 0.5/0.25/0.25.

SMC: geometric bridge pi_b ∝ q(H)^(1-b) [p(H) L(H, z)]^b N(z), q = single-fault screen at the prior-median nuisance
mixed with the prior; adaptive tempering (ESS 0.5); per stage 3 sweeps of {adaptive-covariance random walk on z,
one discrete move (independence from q / mode swap / wheel swap within a module), exact ridge Gibbs}. Ridge Gibbs:
weak(g) multiplies the drive voltage (capxfer sim: volt = u g vb), so weak g with multipliers (s, n) is the same
plant as healthy with (g s, g n); the move resamples (ok|weak, g, s, n) given the effective (g s, g n) from the exact
conditional, with no simulation.

Entry points: `run_batch` (one shape x probe set x episode list; used by cfcap/nuisance_task.py:TASK and
scripts/nuisance_pilot.py). Episode truth, merges and observation noise derive from numpy seeds of (seed, shape,
episode) only, so every probe set sees the same episodes on every host.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import time

import numpy as np
import pandas as pd
import torch

from capxfer import probes, sim
from cfcap.t0 import candidates as Cd
from cfcap.t1 import capability as cap
from cfcap.t1 import params as P
from cfcap.t1 import plant

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results/nuisance"
DEV = "cuda"
OK, WEAK, COAST, BRAKE = plant.OK, plant.WEAK, plant.COAST, plant.BRAKE
FMODES = (WEAK, COAST, BRAKE)
PM = {WEAK: 0.5, COAST: 0.25, BRAKE: 0.25}
P_K = (0.2, 0.4, 0.4)
G_PIECES = ((0.3, 0.6, 0.3), (0.6, 0.9, 0.7))          # (lo, hi, mass)
SPREAD = (0.92, 1.08)                                  # default (delta 0.08); Space(delta=) overrides
VB_H, MU_H, LZ_SD = 0.3, 0.1, 0.15
SD = dict(duty=0.004, speed=0.02, gz=0.001)             # per 0.5-s bin mean
BIN = 25
DUTY_NOISE = 0.005
TAUS = (0.7, 0.8)
UNI_KEYS = ("mu", "s0", "c_roller", "crr", "tau_bd", "latency", "stall_scale", "noload_scale", "mass_scale",
            "izz_scale", "radius_scale", "v_batt", "jeff_scale")
G_SCREEN = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
PROBE_SETS = {"v2": (), "v2+clamp": ("clamp",), "v2+tug": ("tug",), "v2+moddiff": ("moddiff",),
              "v2+clamp+moddiff+tug": ("clamp", "moddiff", "tug")}
ndtr, ndtri = torch.special.ndtr, torch.special.ndtri


# ----------------------------------------------------------------------------- latent space
class Space:
    def __init__(self, asm, dr, delta=0.08):
        self.asm, self.n, self.m = asm, asm.n, len(asm.modules)
        self.spread = (1.0 - delta, 1.0 + delta)
        n, m = self.n, self.m
        f = dict(device=DEV, dtype=torch.float32)
        self.lo = torch.tensor([dr[k][0] for k in UNI_KEYS], **f)
        self.hi = torch.tensor([dr[k][1] for k in UNI_KEYS], **f)
        self.cg, self.yaw = float(dr["cg_off"]), float(dr["wheel_yaw_err"])
        o = 15
        self.sl = {}
        for k, w in (("g", n), ("ws", n), ("wn", n), ("yaw", n), ("lz", n), ("vb", m), ("mu", m)):
            self.sl[k] = slice(o, o + w)
            o += w
        self.d = o
        self.wm = torch.as_tensor(np.asarray(asm.wheel_module), device=DEV)

    def to_lat(self, modes, z):
        """modes (B, n) long, z (B, d) -> plant latents dict."""
        B = z.shape[0]
        vals = self.lo + (self.hi - self.lo) * ndtr(z[:, :13])
        dom = {k: vals[:, i].contiguous() for i, k in enumerate(UNI_KEYS)}
        one = torch.ones(B, device=DEV)
        dom.update(kp_scale=one, ki_scale=one, cg_dx=z[:, 13] * self.cg, cg_dy=z[:, 14] * self.cg,
                   wheel_yaw_err_std=0.0)
        g = torch.where(modes == WEAK, gmap(z[:, self.sl["g"]]), torch.ones_like(z[:, self.sl["g"]]))
        u = lambda k: ndtr(z[:, self.sl[k]])
        return dict(dom=dom, mode=modes, g=g.contiguous(),
                    stall_w=self.spread[0] + (self.spread[1] - self.spread[0]) * u("ws"),
                    noload_w=self.spread[0] + (self.spread[1] - self.spread[0]) * u("wn"),
                    vbatt_off=-VB_H + 2 * VB_H * u("vb"), mu_mod=1 - MU_H + 2 * MU_H * u("mu"),
                    yaw_err=z[:, self.sl["yaw"]] * self.yaw, load_z=z[:, self.sl["lz"]] * LZ_SD,
                    gyro_bias=torch.zeros(B, device=DEV), acc_bias=torch.zeros(B, 2, device=DEV), drop_p=0.0,
                    payload=None)


def gmap(zg):
    u = ndtr(zg)
    return torch.where(u < 0.3, 0.3 + u, 0.6 + (u - 0.3) / 0.7 * 0.3)


def ginv(g):
    u = torch.where(g < 0.6, g - 0.3, 0.3 + (g - 0.6) / 0.3 * 0.7)
    return ndtri(u.clamp(1e-6, 1 - 1e-6))


def spread_inv(w, spread=(0.92, 1.08)):
    return ndtri(((w - spread[0]) / (spread[1] - spread[0])).clamp(1e-6, 1 - 1e-6))


def log_prior_modes(modes):
    """log p(H) of mode vectors (..., n)."""
    n = modes.shape[-1]
    k = (modes != OK).sum(-1)
    lp = torch.zeros(modes.shape[:-1], device=modes.device)
    for mo, p in PM.items():
        lp = lp + (modes == mo).sum(-1) * math.log(p)
    comb = torch.tensor([0.0, math.log(n), math.log(n * (n - 1) / 2)], device=modes.device)
    pk = torch.log(torch.tensor(P_K, device=modes.device))
    kk = k.clamp(max=2)
    return torch.where(k <= 2, lp + pk[kk] - comb[kk], torch.full_like(lp, -1e9))


def sample_truth(space, rng):
    """One episode's true (modes (n,), z (d,)) from a numpy Generator (host-independent)."""
    n = space.n
    modes = np.zeros(n, np.int64)
    k = rng.choice(3, p=P_K)
    for w in rng.choice(n, size=k, replace=False):
        modes[w] = int(rng.choice(FMODES, p=[PM[m] for m in FMODES]))
    z = rng.standard_normal(space.d).astype(np.float32)
    return torch.as_tensor(modes, device=DEV), torch.as_tensor(z, device=DEV)


# ----------------------------------------------------------------------------- probes
class Probe:
    def __init__(self, asm, par, kinds):
        sched = probes.Schedule(asm.char_length(), pids=("P1", "P2", "P5"))
        t = sim.probe_duration(sched)
        legs = []
        n = asm.n
        radius = float(par["wheel"]["radius_m"])
        for kind in kinds:
            if kind == "clamp":
                v = 0.45
                for tw in ([v, 0, 0], [-v, 0, 0], [0, v, 0], [0, -v, 0]):
                    legs.append((t, 1.2, np.array(tw, float), None))
                    t += 0.6 + 1.2 + 0.4
            elif kind == "tug":
                corner = np.array([c.split(":")[1] for c in asm.wheel_corner])
                wv = 0.40 / radius
                for a, b in (("fl", "rr"), ("fr", "rl")):
                    for s in (+1, -1):
                        om = np.zeros(n)
                        om[corner == a] = s * wv
                        om[corner == b] = -s * wv
                        legs.append((t, 1.0, None, om))
                        t += 2.0
            elif kind == "moddiff":
                for om in moddiff_refs(asm, radius):
                    legs.append((t, 1.0, None, om))
                    t += 2.0
            else:
                raise ValueError(kind)
        self.T = t
        hz = float(par["firmware"]["cmd_rate_hz"])
        K = plant.n_ticks(t, par)
        tt = np.arange(K + 1) / hz
        tw = sched.twist(tt)
        pw = np.full((K + 1, n), np.nan) if any(l[3] is not None for l in legs) else None
        for t0, hold, twv, om in legs:
            r = np.minimum(np.clip((tt - t0) / 0.3, 0, 1), np.clip((t0 + 0.6 + hold - tt) / 0.3, 0, 1))
            msk = (tt >= t0) & (tt < t0 + 0.6 + hold)
            if twv is not None:
                tw[msk] += r[msk, None] * twv
            else:
                pw[msk] = r[msk, None] * om[None]
        tw[K:] = 0.0
        self.tw = tw.astype(np.float32)
        self.pw = None if pw is None else pw.astype(np.float32)


def moddiff_refs(asm, radius):
    """T0 probesets 'moddiff' legs as per-wheel speed references (rad/s)."""
    pos, c = asm.wheel_pos, asm.module_pos[asm.wheel_module]
    rel = pos - c
    m = len(asm.modules)
    parity = np.array([(i + j) % 2 for _, i, j, _ in asm.modules]) * 2 - 1.0
    w = probes.W_NOM * min(1.0, probes.WHEEL_SPEED_CAP / float(np.hypot(*rel[0])))
    out = []
    for tw in (np.tile([0, 0, w], (m, 1)), np.tile([0, 0, -w], (m, 1)),
               np.stack([0 * parity, probes.V_NOM * parity, 0 * parity], 1),
               np.stack([probes.V_NOM * parity, 0 * parity, 0 * parity], 1)):
        t = tw[asm.wheel_module]
        vc = np.stack([t[:, 0] - t[:, 2] * rel[:, 1], t[:, 1] + t[:, 2] * rel[:, 0]], -1)
        out.append((vc * asm.r_hat).sum(-1) / (radius * (asm.w_hat * asm.r_hat).sum(-1)))
    return out


# ----------------------------------------------------------------------------- simulation -> features
def feats_from(duty, angle, gz):
    th = angle.double()
    cnt = torch.floor(th / (2 * math.pi) * P.CPR)
    spd = torch.zeros_like(th)
    spd[:, 1:] = torch.diff(cnt, dim=1) / P.CPR * 2 * math.pi / plant.DT_OBS
    T = (duty.shape[1] // BIN) * BIN
    b = lambda x: x[:, :T].reshape(x.shape[0], T // BIN, BIN, *x.shape[2:]).mean(2)
    return dict(duty=b(duty.float()), speed=b(spd.float()), gz=b(gz.float()))


class Sim:
    def __init__(self, asm, par, probe, engine, chunk):
        self.asm, self.par, self.probe, self.engine, self.chunk = asm, par, probe, engine, chunk
        self.n_sims = 0

    def run(self, lat, raw=False):
        B = lat["mode"].shape[0]
        outs = []
        for r0 in range(0, B, self.chunk):
            idx = torch.arange(r0, min(B, r0 + self.chunk), device=DEV)
            li = plant._index(lat, idx)
            job = dict(asm=self.asm, lat=li, cmd=self.probe.tw)
            if self.probe.pw is not None:
                job["pw_ref"] = self.probe.pw
            R = plant.simulate_bucket([job], self.par, self.probe.T, engine=self.engine)[0]
            self.n_sims += len(idx)
            if raw:
                return R
            outs.append(feats_from(R["duty"], R["wheel_angle"], R["twist"][:, :, 2]))
            del R
        return {k: torch.cat([o[k] for o in outs], 0) for k in outs[0]}


def observe_feats(R, lat, seed):
    obs = plant.observe(R, lat, seed=seed)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    duty = obs["pre_duty"] + DUTY_NOISE * torch.randn(obs["pre_duty"].shape, generator=g, device=DEV)
    T = (duty.shape[1] // BIN) * BIN
    b = lambda x: x[:, :T].reshape(x.shape[0], T // BIN, BIN, *x.shape[2:]).mean(2)
    return dict(duty=b(duty), speed=b(obs["speed"]), gz=b(obs["gz"]))


def loglik(f, y, rows_per_ep):
    """f: feats (E*P, ...), y: feats (E, ...) -> (E, P)."""
    E = y["duty"].shape[0]
    ll = 0
    for k in ("duty", "speed", "gz"):
        a = f[k].reshape(E, rows_per_ep, *f[k].shape[1:])
        r = ((a - y[k][:, None]) / SD[k]) ** 2
        ll = ll - 0.5 * r.reshape(E, rows_per_ep, -1).sum(-1)
    return ll


# ----------------------------------------------------------------------------- screening proposal q(H)
def configs(n):
    cw, cm = [(-1, -1)], [(0, 0)]
    for w in range(n):
        for mo in FMODES:
            cw.append((w, -1)); cm.append((mo, 0))
    for w1, w2 in itertools.combinations(range(n), 2):
        for m1, m2 in itertools.product(FMODES, FMODES):
            cw.append((w1, w2)); cm.append((m1, m2))
    return torch.tensor(cw, device=DEV), torch.tensor(cm, device=DEV)


def cfg_modes(cw, cm, n):
    M = torch.zeros(len(cw), n, dtype=torch.long, device=DEV)
    for s in range(2):
        ok = cw[:, s] >= 0
        M[ok, cw[ok, s]] = cm[ok, s]
    return M


def screen(space, simr, y, E):
    """Single-fault screen at the prior-median nuisance -> S (E, n, 4) log-evidence gain per (wheel, mode)."""
    n = space.n
    hm, hg = [np.zeros(n, np.int64)], [np.ones(n, np.float32)]
    tags = [(-1, OK)]
    for w in range(n):
        for g in G_SCREEN:
            m = np.zeros(n, np.int64); gg = np.ones(n, np.float32); m[w] = WEAK; gg[w] = g
            hm.append(m); hg.append(gg); tags.append((w, WEAK))
        for mo in (COAST, BRAKE):
            m = np.zeros(n, np.int64); m[w] = mo
            hm.append(m); hg.append(np.ones(n, np.float32)); tags.append((w, mo))
    H = len(hm)
    modes = torch.as_tensor(np.stack(hm), device=DEV).repeat(E, 1)
    z = torch.zeros(E * H, space.d, device=DEV)
    lat = space.to_lat(modes, z)
    lat["g"] = torch.as_tensor(np.stack(hg), device=DEV).repeat(E, 1)
    ll = loglik(simr.run(lat), y, H)                                      # (E, H)
    S = torch.zeros(E, n, 4, device=DEV)
    for w in range(n):
        for mo in FMODES:
            j = [i for i, t in enumerate(tags) if t == (w, mo)]
            S[:, w, mo] = torch.logsumexp(ll[:, j], 1) - math.log(len(j)) - ll[:, 0]
    return S


class Q:
    """q(H) = 0.3 p(H) + 0.7 p(H) exp(kappa sum S) / Z, evaluable from mode vectors and sampleable."""

    def __init__(self, S, n):
        self.S, self.n = S, n
        E = S.shape[0]
        self.kappa = 8.0 / S.abs().amax((1, 2)).clamp(min=1.0)              # (E,)
        self.cw, self.cm = configs(n)
        self.M = cfg_modes(self.cw, self.cm, n)                            # (Ncfg, n)
        lp = log_prior_modes(self.M)                                       # (Ncfg,)
        sc = self._score(self.M[None].expand(E, -1, -1))                   # (E, Ncfg)
        self.logZ = torch.logsumexp(lp[None] + self.kappa[:, None] * sc, 1)
        self.logq_cfg = self.eval(self.M[None].expand(E, -1, -1))

    def _score(self, modes):                                               # modes (E, P, n)
        Sx = self.S[:, None].expand(-1, modes.shape[1], -1, -1)
        return torch.gather(Sx, 3, modes[..., None])[..., 0].sum(-1)

    def eval(self, modes):
        lp = log_prior_modes(modes)
        a = lp + math.log(0.3)
        b = lp + math.log(0.7) + self.kappa[:, None] * self._score(modes) - self.logZ[:, None]
        return torch.logaddexp(a, b)

    def sample(self, Pn, gen):
        idx = torch.multinomial(torch.exp(self.logq_cfg), Pn, replacement=True, generator=gen)
        return self.M[idx]                                                 # (E, P, n)


# ----------------------------------------------------------------------------- SMC
def systematic(logw, gen):
    E, Pn = logw.shape
    w = torch.softmax(logw, 1)
    c = torch.cumsum(w, 1)
    c[:, -1] = 1.0
    u = (torch.rand(E, 1, device=DEV, generator=gen) + torch.arange(Pn, device=DEV)[None]) / Pn
    return torch.searchsorted(c, u).clamp(max=Pn - 1)


def ess(logw):
    w = torch.softmax(logw, 1)
    return 1.0 / (w ** 2).sum(1)


def next_dbeta(a, beta, target):
    lo = torch.zeros_like(beta)
    hi = 1.0 - beta
    ok = ess(hi[:, None] * a) >= target
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        good = ess(mid[:, None] * a) >= target
        lo = torch.where(good, mid, lo)
        hi = torch.where(good, hi, mid)
    return torch.where(ok, 1.0 - beta, torch.where(beta >= 1, torch.zeros_like(beta), lo.clamp(min=1e-7)))


def g_integral(a, b):
    """int_a^b f_g(g) g^-2 dg (piecewise-constant mixture density), elementwise; 0 where a >= b."""
    tot = torch.zeros_like(a)
    for lo, hi, mass in G_PIECES:
        aa, bb = torch.maximum(a, torch.full_like(a, lo)), torch.minimum(b, torch.full_like(b, hi))
        tot = tot + torch.where(bb > aa, mass / (hi - lo) * (1 / aa - 1 / bb), torch.zeros_like(a))
    return tot


def g_sample(a, b, gen):
    """g ~ f_g(g) g^-2 on [a, b] (requires a < b)."""
    masses = []
    for lo, hi, mass in G_PIECES:
        aa, bb = torch.maximum(a, torch.full_like(a, lo)), torch.minimum(b, torch.full_like(b, hi))
        masses.append(torch.where(bb > aa, mass / (hi - lo) * (1 / aa - 1 / bb), torch.zeros_like(a)))
    m = torch.stack(masses, -1)
    u0 = torch.rand(a.shape, device=DEV, generator=gen) * m.sum(-1)
    first = u0 < m[..., 0]
    lo_ = torch.where(first, torch.maximum(a, torch.full_like(a, 0.3)), torch.maximum(a, torch.full_like(a, 0.6)))
    hi_ = torch.where(first, torch.minimum(b, torch.full_like(b, 0.6)), torch.minimum(b, torch.full_like(b, 0.9)))
    u = torch.rand(a.shape, device=DEV, generator=gen)
    inv = 1 / hi_ + u * (1 / lo_ - 1 / hi_)                                   # 1/g uniform  <=> density ∝ g^-2
    return 1 / inv


def ridge_gibbs(space, modes, z, beta, q, gen):
    """Exact Gibbs over (ok|weak, g, s, n) per driven wheel given the effective (g s, g n). In place."""
    E, Pn, n = modes.shape
    sg, ss, sn = space.sl["g"], space.sl["ws"], space.sl["wn"]
    lo, hi = space.spread
    dens = 1.0 / (hi - lo) ** 2
    for i in range(n):
        mi = modes[..., i]
        drv = (mi == OK) | (mi == WEAK)
        g_cur = torch.where(mi == WEAK, gmap(z[..., sg.start + i]), torch.ones_like(z[..., 0]))
        ws = lo + (hi - lo) * ndtr(z[..., ss.start + i])
        wn = lo + (hi - lo) * ndtr(z[..., sn.start + i])
        es, en = g_cur * ws, g_cur * wn
        m_ok = modes.clone(); m_ok[..., i] = OK
        m_wk = modes.clone(); m_wk[..., i] = WEAK
        lpd = lambda M: (1 - beta)[:, None] * q.eval(M) + beta[:, None] * log_prior_modes(M)
        l_ok = lpd(m_ok) + math.log(dens) + torch.where((es >= lo) & (es <= hi) & (en >= lo) & (en <= hi),
                                                        torch.zeros_like(es), torch.full_like(es, -1e30))
        a = torch.maximum(es, en) / hi
        b = torch.minimum(es, en) / lo
        I = g_integral(a, b)
        l_wk = lpd(m_wk) + math.log(dens) + torch.log(I.clamp(min=1e-38)) + torch.where(
            I > 0, torch.zeros_like(I), torch.full_like(I, -1e30))
        p_wk = torch.sigmoid(l_wk - l_ok)
        choose_wk = (torch.rand(p_wk.shape, device=DEV, generator=gen) < p_wk) & drv & (I > 0)
        choose_ok = drv & ~choose_wk
        g_new = g_sample(torch.where(I > 0, a, torch.full_like(a, 0.5)),
                         torch.where(I > 0, b, torch.full_like(b, 0.6)), gen)
        g_eff = torch.where(choose_wk, g_new, torch.ones_like(g_new))
        z[..., ss.start + i] = torch.where(drv, spread_inv(es / g_eff, space.spread), z[..., ss.start + i])
        z[..., sn.start + i] = torch.where(drv, spread_inv(en / g_eff, space.spread), z[..., sn.start + i])
        zg_ok = torch.randn(z[..., 0].shape, device=DEV, generator=gen)
        z[..., sg.start + i] = torch.where(choose_wk, ginv(g_new), torch.where(choose_ok & (mi == WEAK), zg_ok,
                                                                               z[..., sg.start + i]))
        modes[..., i] = torch.where(choose_wk, torch.full_like(mi, WEAK), torch.where(choose_ok, torch.full_like(mi, OK), mi))


def discrete_proposal(space, modes, z, q, gen):
    """One discrete proposal per particle. Returns (modes', z', log correction c) with
    log alpha = beta (ll' - ll) + extra(beta), extra computed by the caller from the kind."""
    E, Pn, n = modes.shape
    kind = torch.randint(0, 3, (E, Pn), device=DEV, generator=gen)          # 0 indep, 1 mode swap, 2 wheel swap
    m_ind = q.sample(Pn, gen)
    m2, z2 = modes.clone(), z.clone()
    nf = (modes != OK).sum(-1)
    # pick a faulted wheel uniformly (if any)
    r = torch.rand(E, Pn, n, device=DEV, generator=gen) * (modes != OK)
    fw = r.argmax(-1)
    fmode = torch.gather(modes, 2, fw[..., None])[..., 0]
    # mode swap: to one of the two other fault modes
    alt = torch.tensor([[0, 0], [COAST, BRAKE], [WEAK, BRAKE], [WEAK, COAST]], device=DEV)
    pick = torch.randint(0, 2, (E, Pn), device=DEV, generator=gen)
    newmode = alt[fmode, pick]
    # wheel swap within module
    wm = space.wm
    same = (wm[None, None, :] == wm[fw][..., None]) & (modes == OK)
    r2 = torch.rand(E, Pn, n, device=DEV, generator=gen) * same
    nw = r2.argmax(-1)
    has_nw = same.any(-1)
    ar = torch.arange(E, device=DEV)[:, None], torch.arange(Pn, device=DEV)[None]
    # apply
    k0 = kind == 0
    m2[k0] = m_ind[k0]
    k1 = (kind == 1) & (nf > 0)
    mm = m2.clone(); mm[ar[0], ar[1], fw] = newmode
    m2 = torch.where(k1[..., None], mm, m2)
    k2 = (kind == 2) & (nf > 0) & has_nw
    mm = m2.clone(); mm[ar[0], ar[1], nw] = fmode; mm[ar[0], ar[1], fw] = OK
    m2 = torch.where(k2[..., None], mm, m2)
    sg = space.sl["g"].start
    zz = z2.clone()
    gfw = zz[ar[0], ar[1], sg + fw].clone(); gnw = zz[ar[0], ar[1], sg + nw].clone()
    zz[ar[0], ar[1], sg + nw] = gfw; zz[ar[0], ar[1], sg + fw] = gnw
    z2 = torch.where(k2[..., None], zz, z2)
    valid = k0 | k1 | k2
    return m2, z2, kind, valid


def smc(space, simr, y, q, E, Pn, gen, max_stages=150, sweeps=3, final_sweeps=4):
    d, n = space.d, space.n
    modes = q.sample(Pn, gen)
    z = torch.randn(E, Pn, d, device=DEV, generator=gen)
    flat = lambda t: t.reshape(E * Pn, *t.shape[2:])
    ev = lambda M, Z: loglik(simr.run(space.to_lat(flat(M), flat(Z))), y, Pn)
    ll = ev(modes, z)
    beta = torch.zeros(E, device=DEV)
    scale = torch.full((E,), 0.5 * 2.38 / math.sqrt(d), device=DEV)
    stages, acc_c, acc_d = 0, [], []
    ar = torch.arange(E, device=DEV)[:, None]
    done_final = False
    while True:
        lp, lq = log_prior_modes(modes), q.eval(modes)
        a = ll + lp - lq
        if stages >= max_stages:
            db = 1.0 - beta
        else:
            db = next_dbeta(a, beta, 0.5 * Pn)
        logw = db[:, None] * a
        idx = systematic(logw, gen)
        modes, z, ll = modes[ar, idx], z[ar, idx], ll[ar, idx]
        beta = (beta + db).clamp(max=1.0)
        stages += 1
        last = bool((beta >= 1 - 1e-9).all())
        # proposal covariance
        zc = z - z.mean(1, keepdim=True)
        cov = zc.transpose(1, 2) @ zc / (Pn - 1)
        dg = torch.diagonal(cov, dim1=1, dim2=2)
        cov = 0.8 * cov + 0.2 * torch.diag_embed(dg) + 1e-6 * torch.eye(d, device=DEV)
        L = torch.linalg.cholesky(cov)
        for _ in range(final_sweeps if last else sweeps):
            # continuous RW
            zp = z + scale[:, None, None] * (L[:, None] @ torch.randn(E, Pn, d, 1, device=DEV, generator=gen))[..., 0]
            llp = ev(modes, zp)
            la = -0.5 * ((zp ** 2).sum(-1) - (z ** 2).sum(-1)) + beta[:, None] * (llp - ll)
            acc = torch.log(torch.rand(E, Pn, device=DEV, generator=gen)) < la
            z = torch.where(acc[..., None], zp, z)
            ll = torch.where(acc, llp, ll)
            ar_c = acc.float().mean(1)
            scale = scale * torch.exp(ar_c - 0.234)
            acc_c.append(float(ar_c.mean()))
            # discrete
            m2, z2, kind, valid = discrete_proposal(space, modes, z, q, gen)
            llp = ev(m2, z2)
            lp1, lq1, lp2, lq2 = log_prior_modes(modes), q.eval(modes), log_prior_modes(m2), q.eval(m2)
            b = beta[:, None]
            la = b * (llp - ll) + torch.where(kind == 0, b * ((lp2 - lq2) - (lp1 - lq1)),
                                              b * (lp2 - lp1) + (1 - b) * (lq2 - lq1))
            acc = (torch.log(torch.rand(E, Pn, device=DEV, generator=gen)) < la) & valid
            modes = torch.where(acc[..., None], m2, modes)
            z = torch.where(acc[..., None], z2, z)
            ll = torch.where(acc, llp, ll)
            acc_d.append(float(acc.float().mean()))
            # ridge Gibbs (likelihood-invariant)
            ridge_gibbs(space, modes, z, beta, q, gen)
        if last:
            break
    return modes, z, ll, dict(stages=stages, acc_cont=float(np.mean(acc_c)), acc_disc=float(np.mean(acc_d)),
                              forced=int(stages >= max_stages))


# ----------------------------------------------------------------------------- labels
def concat(la, lb):
    out = dict(la)
    for k in ("mode", "g", "stall_w", "noload_w", "yaw_err", "load_z", "vbatt_off", "mu_mod"):
        out[k] = torch.cat([la[k], lb[k]], 1)
    return out


def fresh_split(lat, gen):
    lat = dict(lat)
    lat["load_z"] = LZ_SD * torch.randn(lat["mode"].shape, device=DEV, generator=gen)
    return lat


def labels(space, par, cands, other_lat, modes_rows, z_rows, gen):
    """rows: truth first then posterior draws. Returns list of rho_rel (rows, L) per candidate."""
    lat = space.to_lat(modes_rows, z_rows)
    B = modes_rows.shape[0]
    targets = []
    for c in cands:
        if c.kind == "child":
            lc = fresh_split(plant.subset(lat, c.modules), gen)
        else:
            lo_ = plant._index(other_lat, torch.zeros(B, dtype=torch.long, device=DEV))
            lc = fresh_split(concat(plant.subset(lat, c.modules), lo_), gen)
        targets.append((c.asm, lc))
    out = cap.capability(targets, par, relative=True)
    return [o["rho_rel"] for o in out]




# ----------------------------------------------------------------------------- one batch of episodes
_CACHE = {}
HET_OTHER = plant.Het(wheel_motor=0.08, vbatt_module=0.3, mu_module=0.1)


def setup(shape, probe_set, engine="graph", chunk=8192, delta=0.08):
    key = (shape, probe_set, engine, chunk, delta)
    if key not in _CACHE:
        par = P.base_params("plated")
        dr = P.theta_dr(P.load_theta("plated"))
        asm = P.named_assembly(shape, par)
        _CACHE[key] = dict(par=par, dr=dr, asm=asm, space=Space(asm, dr, delta), kids=Cd.children(asm, par),
                           sim=Sim(asm, par, Probe(asm, par, PROBE_SETS[probe_set]), engine, chunk))
    return _CACHE[key]


def episode_rng(seed, shape, episode):
    return np.random.default_rng([int(seed), sum(map(ord, shape)) * 7 + len(shape), int(episode)])


def run_batch(shape, probe_set, episodes, seed=909, particles=256, draws=64, max_stages=150, engine="graph",
              chunk=8192, delta=0.08):
    """SMC posterior + labels for the given episode indices. Returns (decision columns dict, episode dicts)."""
    S_ = setup(shape, probe_set, engine, chunk, delta)
    par, dr, asm, space, kids, simr = S_["par"], S_["dr"], S_["asm"], S_["space"], S_["kids"], S_["sim"]
    n, E = asm.n, len(episodes)
    truths, others, merges, lab_seeds = [], [], [], []
    for ep in episodes:
        rng = episode_rng(seed, shape, ep)
        truths.append(sample_truth(space, rng))
        other = P.named_assembly(str(rng.choice(["single", "pair"])), par)
        torch.manual_seed(int(rng.integers(2**31)))
        others.append(plant.sample_latents(other, 1, rng, HET_OTHER, dr=dr))
        merges.append(Cd.sample_merges(asm, other, par, rng, n_merge=10))
        lab_seeds.append(int(rng.integers(2**31)))
    modes_t = torch.stack([t[0] for t in truths])
    z_t = torch.stack([t[1] for t in truths])
    gen = torch.Generator(device=DEV).manual_seed(int(episode_rng(seed + 1, shape + probe_set, episodes[0])
                                                      .integers(2**62)))
    t0 = time.time()
    simr.n_sims = 0
    lat_t = space.to_lat(modes_t, z_t)
    R = simr.run(lat_t, raw=True)
    y = observe_feats(R, lat_t, seed=lab_seeds[0] % (2**31 - 10))       # obs noise: from the first episode's seed
    del R
    ll_true = loglik(simr.run(lat_t), y, 1)[:, 0]
    q = Q(screen(space, simr, y, E), n)
    modes, z, ll, info = smc(space, simr, y, q, E, particles, gen, max_stages=max_stages)
    torch.cuda.synchronize()
    t_smc, sims_smc = time.time() - t0, simr.n_sims
    t1 = time.time()
    dec = {k: [] for k in ("delta", "shape", "probe", "episode", "cand", "kind", "size", "has_fault", "leg", "truth", "q05",
                           "q50", "q95")}
    eps = []
    for e, ep in enumerate(episodes):
        pick = torch.randperm(particles, device=DEV, generator=gen)[:draws]
        Mr = torch.cat([modes_t[e:e + 1], modes[e, pick]], 0)
        Zr = torch.cat([z_t[e:e + 1], z[e, pick]], 0)
        lg = torch.Generator(device=DEV).manual_seed(lab_seeds[e])
        cands = kids + merges[e]
        rr = labels(space, par, cands, others[e], Mr, Zr, lg)
        true_set = set(np.flatnonzero(modes_t[e].cpu().numpy()).tolist())
        gt = space.to_lat(modes_t[e:e + 1], z_t[e:e + 1])["g"][0]
        for c, r in zip(cands, rr):
            r = r.float().cpu().numpy()
            qs_ = np.quantile(r[1:], [0.05, 0.5, 0.95], axis=0)
            cw = set(c.src[c.src >= 0].tolist())
            for li, leg in enumerate(cap.LEGS):
                for k, v in (("delta", float(delta)), ("shape", shape), ("probe", probe_set), ("episode", int(ep)), ("cand", c.name),
                             ("kind", c.kind), ("size", int(c.size)), ("has_fault", int(bool(cw & true_set))),
                             ("leg", leg), ("truth", float(r[0, li])), ("q05", float(qs_[0, li])),
                             ("q50", float(qs_[1, li])), ("q95", float(qs_[2, li]))):
                    dec[k].append(v)
        eps.append(dict(shape=shape, probe=probe_set, delta=delta, episode=int(ep), n_faults=len(true_set),
                        faults=";".join(f"{w}:{int(modes_t[e, w])}:{float(gt[w]):.2f}" for w in sorted(true_set)),
                        ll_true=float(ll_true[e]), ll_best=float(ll[e].max()), ll_med=float(ll[e].median()),
                        post_mass_true_modes=float((modes[e] == modes_t[e]).all(-1).float().mean()),
                        stages=info["stages"], forced=info["forced"], acc_cont=info["acc_cont"],
                        acc_disc=info["acc_disc"], smc_sims_per_ep=sims_smc / E, smc_s_per_ep=t_smc / E,
                        n_cands=len(cands)))
    torch.cuda.synchronize()
    t_lab = time.time() - t1
    for r in eps:
        r["label_s_per_ep"] = t_lab / E
        r["total_s_per_ep"] = r["smc_s_per_ep"] + t_lab / E
    return dec, eps
