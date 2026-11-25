"""Theory (d): compositional certificate for merges of modules that were never assembled together (T0, exploratory).

Two assemblies G1, G2 (1-2 modules each, random yaws, <= 1 fault each: weak 0.4/0.6/0.8 or coast) are probed
separately. Stage 1 per wheel: a noisy 2-D feature of the wheel's class (stand-in for the self-model output:
duty ratio 1/g and a passive indicator, Gaussian noise), split-conformal class thresholds calibrated on fresh wheels,
levels alpha1/(2 n_G) for OK and alpha1/(2 K) for faulted classes (capxfer Thm 5 recipe) in each assembly
separately. Stage 2 is exact (T0 with the true continuous plant): lb(B) = min over hypotheses in S1|B1 x S2|B2 of
rho_B. Every merge layout of G1 with G2 (and both pieces) is a candidate; all are certified from the same sets.
Reported: miscoverage P(rho_true < lb) per (merge, leg), per merge, and simultaneously over all merges and legs of
a pair, against alpha1 + alpha1' and 1 - (1 - alpha1)(1 - alpha1').
-> results/theory/merge_coverage.csv
"""
from __future__ import annotations

import itertools
import os
import pathlib
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "theory"
CLASSES = [("ok", 0, 1.0), ("w0.4", 1, 0.4), ("w0.6", 1, 0.6), ("w0.8", 1, 0.8), ("coast", 2, 1.0)]
FEAT = np.array([[1.0, 0.0], [2.5, 0.0], [1 / 0.6, 0.0], [1.25, 0.0], [1.0, 1.0]])
NOISE = 0.18
K = 1


def thresholds(rng, alpha1, n_G, n_cal=4000):
    t = []
    for ci in range(len(CLASSES)):
        a = alpha1 / (2 * n_G) if ci == 0 else alpha1 / (2 * K)
        s = np.linalg.norm(rng.normal(0, NOISE, (n_cal, 2)), axis=1)
        k = int(np.ceil((n_cal + 1) * (1 - a)))
        t.append(np.sort(s)[min(k, n_cal) - 1])
    return np.array(t)


def wheel_sets(rng, cls, thr):
    x = FEAT[cls] + rng.normal(0, NOISE, (len(cls), 2))
    d = np.linalg.norm(x[:, None] - FEAT[None], axis=2)
    return d <= thr[None]


def assembly_hyps(sets):
    """<= K faults, each wheel's class in its set -> list of class vectors."""
    n = len(sets)
    out = []
    if sets[:, 0].all():
        out.append(np.zeros(n, int))
    for j in range(n):
        others_ok = np.delete(sets[:, 0], j).all()
        if not others_ok:
            continue
        for c in np.flatnonzero(sets[j, 1:]) + 1:
            v = np.zeros(n, int); v[j] = c
            out.append(v)
    return out


def main(n_pairs=400, alpha1=0.1, seed=0):
    import torch
    torch.set_num_threads(4)
    from cfcap.t0 import qs, stagea, candidates as C, observe as O
    from capxfer.sim import WEAK, COAST
    P, th = qs.load_params(), qs.load_theta()
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    modes_of = np.array([0, WEAK, WEAK, WEAK, COAST])
    g_of = np.array([c[2] for c in CLASSES])
    for pi in range(n_pairs):
        G1 = stagea.random_parent(rng, P, (1, 2), name="g1")
        G2 = stagea.random_parent(rng, P, (1, 2), name="g2")
        truth, sets, hyps, het = [], [], [], []
        for G in (G1, G2):
            cls = np.zeros(G.n, int)
            if rng.random() < 0.7:
                cls[rng.integers(G.n)] = rng.integers(1, len(CLASSES))
            thr = thresholds(rng, alpha1, G.n)
            S = wheel_sets(rng, cls, thr)
            truth.append(cls); sets.append(S); hyps.append(assembly_hyps(S))
            het.append(O.sample_het(rng, G, True))
        covered = [all(S[j, c] for j, c in enumerate(cl)) and (cl != 0).sum() <= K for S, cl in zip(sets, truth)]
        # candidates: every merge layout of whole G1 with whole G2
        placements = C.merge_placements(C.cells_of(G1), C.cells_of(G2))
        ep_viol = False
        for t in placements:
            cand = C.merge(G1, tuple(range(len(G1.modules))), G2, t, P)
            src = cand.src
            st = C.carry(het[0][0], het[1][0], src); nl = C.carry(het[0][1], het[1][1], src)
            mu = C.carry(het[0][2], het[1][2], src)
            cpl = qs.Plant(cand.asm, P, th, stall=st, noload=nl, mu_f=mu)
            xis = qs.decision_legs(cand.asm)
            H = [np.concatenate([h1, h2]) for h1 in hyps[0] for h2 in hyps[1]]
            tv = C.carry(truth[0], truth[1], src)
            allv = [tv] + [C.carry(h[:G1.n], h[G1.n:], src) for h in H]
            Mv = np.stack([modes_of[v] for v in allv]); Gv = np.stack([g_of[v] for v in allv])
            r = qs.solve(cpl, Mv, Gv, cpl.ik(xis))
            rho = qs.rho_of(r["xi"], xis, cand.asm.char_length())
            rho_t = rho[0]
            lb = rho[1:].min(0) if len(H) else np.full(10, -np.inf)
            viol = rho_t < lb - 1e-9
            ep_viol |= bool(viol.any())
            rows.append(dict(pair=pi, merge=str(t), n_wheels=cand.asm.n, n_hyp=len(H), cov1=covered[0],
                             cov2=covered[1], viol_any_leg=bool(viol.any()), viol_legs=int(viol.sum()),
                             certified_06=int((lb >= 0.6).sum()), capable_06=int((rho_t >= 0.6).sum())))
        for rr in rows:
            if rr["pair"] == pi:
                rr["viol_pair_simult"] = ep_viol
        if pi % 50 == 0:
            print(pi, f"{time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    pairs = D.groupby("pair").agg(sim=("viol_pair_simult", "first"), cov1=("cov1", "first"), cov2=("cov2", "first"))
    S = pd.DataFrame([dict(n_pairs=n_pairs, n_merges=len(D), alpha1_each=alpha1,
                           bound_union=2 * alpha1, bound_product=1 - (1 - alpha1) ** 2,
                           miscov_per_merge_any_leg=D.viol_any_leg.mean(),
                           miscov_per_merge_leg=D.viol_legs.sum() / (10 * len(D)),
                           miscov_simultaneous_per_pair=pairs.sim.mean(),
                           stage1_uncovered_G1=1 - pairs.cov1.mean(), stage1_uncovered_G2=1 - pairs.cov2.mean(),
                           stage1_uncovered_either=1 - (pairs.cov1 & pairs.cov2).mean(),
                           viol_given_covered=D[D.cov1 & D.cov2].viol_any_leg.mean(),
                           cert_frac_capable_06=D.certified_06.sum() / max(1, D.capable_06.sum()),
                           mean_merges_per_pair=len(D) / n_pairs)])
    RES.mkdir(parents=True, exist_ok=True)
    S.to_csv(RES / "merge_coverage.csv", index=False)
    print(S.T.to_string())


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
