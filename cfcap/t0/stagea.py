"""Stage A2/A3 engine: decision ambiguity of the idealised-oracle consistent set over children and merges.

One episode = one random parent (polyomino of 4-8 modules, random module yaws, plated theta), 1-2 true faults on
random wheels (weak g on the hypothesis grid, coast, brake), heterogeneity on/off, and one healthy 1-2-module 'other'
assembly for merges. Hypothesis space: every <= 2-fault mode vector on the parent's wheels (weak g in G_GRID, coast,
brake). The consistent set is computed sequentially over probe legs (exact: consistency is a conjunction over legs),
then for every candidate (child / merge) the capability rho of every consistent hypothesis is solved on the 10 decision
legs under the candidate's own (true) continuous plant.

Decision (candidate, leg, tau): ambiguous iff min_S rho < tau <= max_S rho; oracle go iff min_S rho >= tau;
capable iff rho_true >= tau; unsafe go iff go and not capable.
rho_model 'known': every hypothesis' rho on the candidate's true continuous plant (uncertainty only over the discrete
fault hypothesis; truth always in the set, so oracle unsafe-go = 0 by construction). rho_model 'prior' (heterogeneity
on only): min/max also over prior_M draws of the candidate's motor spread and mu factors, an upper bound on the
ambiguity that the unknown nuisance at the candidate's operating points adds (the parent's evidence on that nuisance
is ignored, so this is conservative).
"""
from __future__ import annotations

import itertools
import time
from collections import defaultdict

import numpy as np

from capxfer import assembly as A
from capxfer.sim import OK, WEAK, COAST, BRAKE

from . import candidates as C
from . import observe as O
from . import probesets as PS
from . import qs

G_GRID = np.round(np.arange(0.3, 0.91, 0.1), 2)
FAULT_STATES = [(WEAK, g) for g in G_GRID] + [(COAST, 1.0), (BRAKE, 1.0)]
TAUS = (0.6, 0.7, 0.8)
K_SIGMA = 3.0
MODE_NAME = {WEAK: "weak", COAST: "coast", BRAKE: "brake"}
_SHAPES = {}
V2_ORDER = (2, 4, 0, 3, 5, 1)        # sequential filter order over the v2 legs (sway+ first: excites every wheel)


def shapes(k):
    if k not in _SHAPES:
        _SHAPES[k] = A._polyominoes(k)
    return _SHAPES[k]


def random_parent(rng, params, sizes, name="parent"):
    """As capxfer assembly.random_assembly (uniform size, uniform fixed polyomino, random yaws), shapes cached."""
    k = int(rng.choice(sizes))
    sh = shapes(k)
    cells = sh[int(rng.integers(len(sh)))]
    mods = [(str(rng.choice(("M1", "M2", "M3", "M4"))), i, j, int(rng.integers(4))) for i, j in cells]
    a = A.build(f"{name}{k}", mods, params)
    a.meta["family"] = A.shape_family(cells)
    return a


def hypotheses(n, max_faults=2):
    """All <= max_faults fault mode vectors -> modes (H, n) int, gs (H, n)."""
    rows_m, rows_g = [np.zeros(n, np.int64)], [np.ones(n)]
    for k in range(1, max_faults + 1):
        for wheels in itertools.combinations(range(n), k):
            for states in itertools.product(FAULT_STATES, repeat=k):
                m = np.zeros(n, np.int64)
                g = np.ones(n)
                for w, (mm, gg) in zip(wheels, states):
                    m[w], g[w] = mm, gg
                rows_m.append(m)
                rows_g.append(g)
    return np.array(rows_m), np.array(rows_g)


def find_row(M, Gs, m, g):
    hit = np.flatnonzero((M == m).all(1) & np.isclose(Gs, g).all(1))
    assert len(hit) == 1, hit
    return int(hit[0])


def legs_arrays(legs):
    return np.stack([l[1] for l in legs]), np.stack([l[2] for l in legs])


def sigmas(truth_sig, het, nuis):
    return {p: O.sigma(truth_sig, p, het, nuis) for p in O.PRESETS}


def decision_counts(rho_h, mask, rho_t, tau):
    """Scorer. rho_h: (H, L) or (M, H, L) rho of every hypothesis (x nuisance draw) on L decision legs; mask (H,)
    consistent set; rho_t (L,) truth. Returns [n_dec, n_ambiguous, n_go, n_capable, n_unsafe_go, n_go_and_capable].
    Ambiguous: min < tau <= max over the set; go: min >= tau; capable: rho_t >= tau. Empty set -> refuse (no go,
    counted ambiguous)."""
    rho_h = np.asarray(rho_h, float)
    mask = np.asarray(mask, bool)
    L = rho_h.shape[-1]
    cap = np.asarray(rho_t) >= tau
    if not mask.any():
        return np.array([L, L, 0, cap.sum(), 0, 0])
    sel = rho_h[mask] if rho_h.ndim == 2 else rho_h[:, mask].reshape(-1, L)
    lo, hi = sel.min(0), sel.max(0)
    amb = (lo < tau) & (hi >= tau)
    go = lo >= tau
    return np.array([L, amb.sum(), go.sum(), cap.sum(), (go & ~cap).sum(), (go & cap).sum()])


def run_episode(seed, params, theta, sizes=(4, 5, 6, 7, 8), n_merge=10, het=None, n_faults=None, kinds_extra=(
        "clamp", "moddiff", "tug"), diag_split=True, nuis_M=8, prior_M=6):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    parent = random_parent(rng, params, sizes)
    n = parent.n
    het = bool(rng.integers(2)) if het is None else bool(het)
    st, nl, mu = O.sample_het(rng, parent, het)
    plant = qs.Plant(parent, params, theta, stall=st, noload=nl, mu_f=mu)
    # true faults
    nf = int(rng.integers(1, 3)) if n_faults is None else int(n_faults)
    wheels = rng.choice(n, nf, replace=False)
    tm, tg = np.zeros(n, np.int64), np.ones(n)
    for w in wheels:
        mm = int(rng.choice([WEAK, COAST, BRAKE]))
        tm[w] = mm
        tg[w] = float(rng.choice(G_GRID)) if mm == WEAK else 1.0
    ftype = "+".join(sorted(MODE_NAME[int(tm[w])] for w in wheels))
    # other assembly (healthy, known) for merges
    other = random_parent(rng, params, (1, 2), name="other")
    ost, onl, omu = O.sample_het(rng, other, het)
    # hypothesis space
    HM, HG = hypotheses(n)
    ti = find_row(HM, HG, tm, tg)
    # ---------------- parent observation: v2 sequential filter
    legs = {k: PS.probe_legs(plant, k) for k in ("v2",) + tuple(kinds_extra)}
    rec = dict(seed=seed, parent_size=len(parent.modules), n_wheels=n, family=parent.meta["family"], het=int(het),
               n_faults=nf, fault_type=ftype, n_hyp=len(HM))
    tsig, sig = {}, {}
    for k, L in legs.items():
        om, dc = legs_arrays(L)
        r = qs.solve(plant, tm[None], tg[None], om, dc)
        tsig[k] = O.signals(r)[0]
        nuis = O.nuisance_sd(plant, L, rng, nuis_M) if het else None
        sig[k] = sigmas(tsig[k], het, nuis)
    om_v2, dc_v2 = legs_arrays(legs["v2"])
    alive = np.arange(len(HM))
    z = {p: np.zeros(len(HM)) for p in O.PRESETS}
    n_solved = 0
    for li in V2_ORDER:
        r = qs.solve(plant, HM[alive], HG[alive], om_v2[li:li + 1], dc_v2[li:li + 1])
        n_solved += len(alive)
        s = O.signals(r)[:, 0]
        for p in O.PRESETS:
            zz = (np.abs(s - tsig["v2"][li][None]) / sig["v2"][p][li][None]).max(1)
            z[p][alive] = np.maximum(z[p][alive], zz)
        keep = z["floor"][alive] <= K_SIGMA                              # floor sigma >= sensor sigma: superset
        alive = alive[keep]
    U = alive                                                            # union consistent set (v2, floor)
    assert ti in U, "truth must be consistent"
    zU = {("v2", p): z[p][U] for p in O.PRESETS}
    for k in kinds_extra:
        om, dc = legs_arrays(legs[k])
        r = qs.solve(plant, HM[U], HG[U], om, dc)
        n_solved += len(U) * len(om)
        s = O.signals(r)
        for p in O.PRESETS:
            zU[(k, p)] = (np.abs(s - tsig[k][None]) / sig[k][p][None]).reshape(len(U), -1).max(1)
    masks = {}
    for set_name, kinds in PS.PROBE_SETS.items():
        if any(k not in kinds_extra for k in kinds):
            continue
        for p in O.PRESETS:
            zz = zU[("v2", p)].copy()
            for k in kinds:
                zz = np.maximum(zz, zU[(k, p)])
            masks[(set_name, p)] = zz <= K_SIGMA
    for (sn, p), m in masks.items():
        rec[f"setsize_{sn}_{p}"] = int(m.sum())
    t_parent = time.time() - t0
    # ---------------- candidates
    cands = C.children(parent, params) + C.sample_merges(parent, other, params, rng, n_merge)
    agg = defaultdict(lambda: np.zeros(6, np.int64))   # key -> [n_dec, n_amb, n_go, n_cap, n_unsafe, n_go_cap]
    n_child_solved, nonconv = 0, []
    HMU, HGU = HM[U], HG[U]
    tU = int(np.flatnonzero(U == ti)[0])
    for cand in cands:
        src = cand.src
        cm = C.carry(HMU, np.zeros((1, other.n), np.int64), src)
        cg = C.carry(HGU, np.ones((1, other.n)), src)
        key = np.concatenate([cm, np.round(cg * 100).astype(np.int64)], 1)
        uq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        cpl = qs.Plant(cand.asm, params, theta, stall=C.carry(st, ost, src), noload=C.carry(nl, onl, src),
                       mu_f=C.carry(mu, omu, src))
        xis = qs.decision_legs(cand.asm)
        r = qs.solve(cpl, cm[first], cg[first], cpl.ik(xis))
        n_child_solved += len(first) * len(xis)
        nonconv.append(float((r["resid"] > 1e-6).mean()))
        rho_u = qs.rho_of(r["xi"], xis, cand.asm.char_length())          # (Uq, 10)
        rho_h = rho_u[inv]                                               # (|U|, 10)
        rho_t = rho_h[tU]
        rho_models = {"known": rho_h}
        if het and prior_M:
            # nuisance-prior bound: the consistent hypotheses' rho under prior draws of the candidate's motor spread
            # and mu factors (the truth's own draw is not among them)
            nq = len(first)
            draws = [O.sample_het(rng, cand.asm, True) for _ in range(prior_M)]
            hr = [np.repeat(np.stack([d_[i] for d_ in draws]), nq, 0) for i in range(3)]
            rr = qs.solve(cpl, np.tile(cm[first], (prior_M, 1)), np.tile(cg[first], (prior_M, 1)), cpl.ik(xis),
                          het_rows=hr)
            n_child_solved += prior_M * nq * len(xis)
            R = qs.rho_of(rr["xi"], xis, cand.asm.char_length()).reshape(prior_M, nq, -1)
            rho_models["prior"] = R[:, inv]                              # (M, |U|, 10)
        cmasks = dict(masks)
        if diag_split:
            # observe the candidate itself with its own v2 (the v2 legs are decision legs 0,1,2,3,8,9)
            sv = O.signals(dict(xi=r["xi"][:, PS.V2_LEG_IDX], om=r["om"][:, PS.V2_LEG_IDX],
                                duty=r["duty"][:, PS.V2_LEG_IDX]))
            tsv = sv[inv[tU]]
            nuis = None
            if het:
                nuis = O.nuisance_sd(cpl, PS.probe_legs(cpl, "v2"), rng, nuis_M)
                n_child_solved += nuis_M * 6
            for p in O.PRESETS:
                sgm = O.sigma(tsv, p, het, nuis)
                okc = ((np.abs(sv - tsv[None]) / sgm[None]).reshape(len(sv), -1).max(1) <= K_SIGMA)[inv]
                cmasks[("diag_split", p)] = masks[("v2", p)] & okc
        has_fault = int(bool(np.any(tm[src[src >= 0]] != OK)))
        for (sn, p), m in cmasks.items():
            for rm, rh in rho_models.items():
                for tau in TAUS:
                    agg[(cand.kind, cand.size, has_fault, sn, p, rm, tau)] += decision_counts(rh, m, rho_t, tau)
    rec.update(n_union=len(U), n_cands=len(cands), n_children=sum(c.kind == "child" for c in cands),
               qs_parent=n_solved, qs_child=n_child_solved, t_parent_s=round(t_parent, 2),
               t_total_s=round(time.time() - t0, 2), nonconv_child=float(np.mean(nonconv)))
    rows = []
    for (kind, size, hf, sn, p, rm, tau), v in agg.items():
        rows.append(dict(seed=seed, kind=kind, cand_size=size, has_fault=hf, probe_set=sn, noise=p, rho_model=rm,
                         tau=tau,
                         n_dec=int(v[0]), n_amb=int(v[1]), n_go=int(v[2]), n_cap=int(v[3]), n_unsafe=int(v[4]),
                         n_go_cap=int(v[5])))
    return rec, rows
