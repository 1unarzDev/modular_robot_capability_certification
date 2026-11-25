"""T0 decision episodes: a parent of 2-4 modules with 0-2 hidden faults and unknown per-wheel / per-module nuisances,
observed with v2; a weighted particle posterior pushed to every candidate; per-particle predictions of every probe in
the library (clamp, moddiff, tug, and a diagnostic split per proper child).

Episode (seed):
  * parent = stagea.random_parent (sizes 2-4, random yaws), plated theta; heterogeneity ON: per-wheel stall / no-load
    scale U(0.92, 1.08), per-module mu factor U(0.9, 1.1) (t0.observe.sample_het);
  * faults from the generator prior: none w.p. 0.2, else 1 or 2 wheels, mode uniform weak / coast / brake, weak g
    uniform on stagea.G_GRID (0.3..0.9);
  * other assembly: one healthy module with known heterogeneity (merge partner); up to N_MERGE random merges;
  * observation of a probe: QS hold-window signals [gz, om, duty] + Gaussian noise with the "floor" preset sd
    (t0.observe.sigma, sensor noise + ripple / QS-residual stand-in).
Posterior (`DecisionPosterior`): hypotheses = every <= 2-fault mode vector; screened with the stagea sequential v2
filter (nominal nuisance, sd inflated for heterogeneity, |z| <= K_SCREEN vs the noisy y; truth retention recorded);
particles = screened hypotheses x M prior nuisance draws, weight = prior x v2 likelihood of the hypothesis with the
nuisance marginalised by sd inflation (the draws keep weight 1/M; an 8-draw importance sampler on the joint collapses
to one particle); extra probes and splits reweight with the particle's own signal and floor + heterogeneity sd; the top-mass particles (cover 0.9999, at most P_MAX) are kept. Every particle carries its
nuisance draw to every candidate through `candidates.carry` (the other assembly's nuisance is known).
Predictors on the same episode: `rho_true` (oracle), `rho_nom` (every wheel healthy, nominal nuisance).
"""
from __future__ import annotations

import math
import time

import numpy as np

from capxfer.sim import WEAK, COAST, BRAKE

from cfcap.t0 import candidates as Cn
from cfcap.t0 import observe as O
from cfcap.t0 import probesets as PS
from cfcap.t0 import qs
from cfcap.t0 import stagea as SA

from . import posterior as DP
from . import problem as Pb

SIZES = (2, 3, 4)
M_NUIS = 8
P_MAX = 192
K_SCREEN = 5.0
SD_INFLATE = 1.5         # extra factor on the screen / v2-likelihood sd (nuisance marginalisation is approximate)
N_MERGE = 4
EXTRA = ("clamp", "moddiff", "tug")


def _ctx():
    if not hasattr(_ctx, "v"):
        _ctx.v = (qs.load_params(), qs.load_theta())
    return _ctx.v


def log_prior(HM, n, p_none=0.2):
    nf = (HM != 0).sum(1)
    lp = np.where(nf == 0, math.log(p_none), 0.0)
    for k in (1, 2):
        lp = np.where(nf == k, math.log((1 - p_none) / 2) - math.log(math.comb(n, k)), lp)
    n_weak = (HM == WEAK).sum(1)
    n_pas = ((HM == COAST) | (HM == BRAKE)).sum(1)
    return lp + n_weak * (math.log(1 / 3) - math.log(len(SA.G_GRID))) + n_pas * math.log(1 / 3)


def _sd(sig, het=False):
    """sd (..., K, D) of the floor-preset noise at signal sig (..., K, D); het=True adds the heterogeneity inflation
    (likelihood of a particle whose own nuisance draw is only a prior sample)."""
    sh = sig.shape
    s2 = sig.reshape(-1, sh[-2], sh[-1])
    return np.stack([O.sigma(s, "floor", het) for s in s2]).reshape(sh)


def episode(seed, sizes=SIZES, M=M_NUIS, p_max=P_MAX, n_merge=N_MERGE, faulted=None):
    """faulted: None (prior), True (>= 1 fault), False (healthy)."""
    t0 = time.time()
    params, theta = _ctx()
    rng = np.random.default_rng(seed)
    parent = SA.random_parent(rng, params, sizes)
    n, m = parent.n, len(parent.modules)
    st, nl, mu = O.sample_het(rng, parent, True)
    tplant = qs.Plant(parent, params, theta, stall=st, noload=nl, mu_f=mu)
    nplant = qs.Plant(parent, params, theta)
    u = rng.random()
    nf = (0 if u < 0.2 else int(rng.integers(1, 3))) if faulted is None else (int(rng.integers(1, 3)) if faulted else 0)
    tm, tg = np.zeros(n, np.int64), np.ones(n)
    for w in rng.choice(n, nf, replace=False):
        mm = int(rng.choice([WEAK, COAST, BRAKE]))
        tm[w], tg[w] = mm, (float(rng.choice(SA.G_GRID)) if mm == WEAK else 1.0)
    other = SA.random_parent(rng, params, (1,), name="other")
    ost, onl, omu = O.sample_het(rng, other, True)

    legs = {k: PS.probe_legs(tplant, k) for k in ("v2",) + EXTRA}
    arr = {k: SA.legs_arrays(v) for k, v in legs.items()}
    ytrue = {}
    for k, (om, dc) in arr.items():
        s = O.signals(qs.solve(tplant, tm[None], tg[None], om, dc))[0]
        ytrue[k] = s + _sd(s) * rng.standard_normal(s.shape)

    # screen (nominal nuisance, inflated sd) with the sequential v2 filter
    HM, HG = SA.hypotheses(n)
    nuis = O.nuisance_sd(nplant, legs["v2"], rng, 8)
    sig_scr = SD_INFLATE * O.sigma(ytrue["v2"], "floor", True, nuis)
    alive = np.arange(len(HM))
    om2, dc2 = arr["v2"]
    for li in SA.V2_ORDER:
        r = qs.solve(nplant, HM[alive], HG[alive], om2[li:li + 1], dc2[li:li + 1])
        z = (np.abs(O.signals(r)[:, 0] - ytrue["v2"][li][None]) / sig_scr[li][None]).max(1)
        alive = alive[z <= K_SCREEN]
        if len(alive) == 0:
            break
    ti = SA.find_row(HM, HG, tm, tg)
    truth_in = bool(np.isin(ti, alive))
    if len(alive) == 0:                                           # keep the prior's best guess: healthy
        alive = np.array([0])
    # particles = screened hypotheses x nuisance draws
    draws = [O.sample_het(rng, parent, True) for _ in range(M)]
    U = len(alive)
    pm, pg = np.repeat(HM[alive], M, 0), np.repeat(HG[alive], M, 0)
    pj = np.tile(np.arange(M), U)
    ph = np.repeat(np.arange(U), M)
    het = [np.stack([draws[j][i] for j in pj]) for i in range(3)]
    # v2 likelihood of each hypothesis with the nuisance marginalised by sd inflation (nominal-nuisance signals,
    # sd = floor + heterogeneity + healthy-draw spread at y); the nuisance draws stay at their prior (weight 1/M)
    s_h = O.signals(qs.solve(nplant, HM[alive], HG[alive], om2, dc2))
    ll_h = -0.5 * (((ytrue["v2"][None] - s_h) / sig_scr[None]) ** 2).reshape(U, -1).sum(1)
    ll = ll_h[ph]
    lw = log_prior(pm, n) + ll
    w = np.exp(lw - lw.max())
    w /= w.sum()
    o = np.argsort(-w)
    k = int(min(p_max, np.searchsorted(np.cumsum(w[o]), 0.9999) + 1))
    keep = o[:k]
    pm, pg, pj, ph, w = pm[keep], pg[keep], pj[keep], ph[keep], w[keep] / w[keep].sum()
    het = [h[keep] for h in het]
    ess = float(1.0 / (w ** 2).sum())
    P = len(w)

    probes = {}
    for kd in EXTRA:
        om, dc = arr[kd]
        s = O.signals(qs.solve(tplant, pm, pg, om, dc, het_rows=het))
        probes[kd] = DP.Probe(s.reshape(P, -1), _sd(s, True).reshape(P, -1), PS.probe_time(kd), ytrue[kd].ravel())

    # candidates: whole parent, connected children, merges
    kids = Cn.children(parent, params)
    merges = Cn.sample_merges(parent, other, params, rng, n_merge)
    full = Cn.Candidate("parent", "child", parent, np.arange(n), tuple(range(m)))
    cands = [full] + kids + merges
    inst = Pb.Instance.from_candidates(Cn.cells_of(parent), kids + merges)
    C, L = len(cands), 10
    rho = np.full((P, C, L), np.nan)
    rho_true = np.full((C, L), np.nan)
    rho_nom = np.full((C, L), np.nan)
    has_fault = np.zeros(C, bool)
    for ci, cd in enumerate(cands):
        src = cd.src
        no = other.n
        cm = Cn.carry(pm, np.zeros((1, no), np.int64), src)
        cg = Cn.carry(pg, np.ones((1, no)), src)
        hrows = [Cn.carry(het[i], np.broadcast_to(x, (1, no)), src) for i, x in enumerate((ost, onl, omu))]
        key = np.concatenate([cm, np.round(cg * 100), pj[:, None]], 1)
        uq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        cpl_t = qs.Plant(cd.asm, params, theta, stall=Cn.carry(st, ost, src), noload=Cn.carry(nl, onl, src),
                         mu_f=Cn.carry(mu, omu, src))
        xis = qs.decision_legs(cd.asm)
        r = qs.solve(cpl_t, cm[first], cg[first], cpl_t.ik(xis), het_rows=[h[first] for h in hrows])
        rho[:, ci] = qs.rho_of(r["xi"], xis, cd.asm.char_length())[inv]
        tmc = Cn.carry(tm[None], np.zeros((1, no), np.int64), src)
        tgc = Cn.carry(tg[None], np.ones((1, no)), src)
        rt = qs.solve(cpl_t, tmc, tgc, cpl_t.ik(xis))
        rho_true[ci] = qs.rho_of(rt["xi"], xis, cd.asm.char_length())[0]
        cpl_n = qs.Plant(cd.asm, params, theta, mu_f=Cn.carry(np.ones(n), omu, src),
                         stall=Cn.carry(np.ones(n), ost, src), noload=Cn.carry(np.ones(n), onl, src))
        hm, hg = qs.healthy(cd.asm.n)
        rn = qs.solve(cpl_n, hm, hg, cpl_n.ik(xis))
        rho_nom[ci] = qs.rho_of(rn["xi"], xis, cd.asm.char_length())[0]
        has_fault[ci] = bool((tmc != 0).any())
        if 0 < ci <= len(kids):                                   # diagnostic split: the child's own v2
            vi = list(PS.V2_LEG_IDX)
            sv = O.signals(dict(xi=r["xi"][:, vi], om=r["om"][:, vi], duty=r["duty"][:, vi]))[inv]
            stv = O.signals(dict(xi=rt["xi"][:, vi], om=rt["om"][:, vi], duty=rt["duty"][:, vi]))[0]
            yv = stv + _sd(stv) * rng.standard_normal(stv.shape)
            probes[f"split:{ci}"] = DP.Probe(sv.reshape(P, -1), _sd(sv, True).reshape(P, -1),
                                             PS.T_SPLIT_S + PS.set_time("v2"), yv.ravel())
    post = DP.DecisionPosterior(w, rho, ph, probes)
    meta = dict(seed=seed, parent_size=m, n_wheels=n, n_faults=nf, n_hyp=len(HM), n_screen=U, truth_in=int(truth_in),
                P=P, ess=ess, C=C, n_children=len(kids), n_merges=len(merges), gen_s=time.time() - t0)
    return dict(inst=inst, post=post, rho_true=rho_true, rho_nom=rho_nom, has_fault=has_fault, meta=meta,
                legs=qs.LEG_NAMES)
