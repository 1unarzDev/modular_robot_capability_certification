"""capxfer's certificate (study 05) adapted to multi-module children: the per-wheel compositional route of capxfer
THEORY.md Theorem 5(a), with capxfer's own self-model, thresholds and lookup-table construction.

What is capxfer's, unchanged (imported read-only):
  * the v2 TCN self-model ensemble (rung_v2, 3 seeds, plated) run on the parent's v2 trial: per-wheel class
    probabilities p (ok/weak/coast/brake) and weak-gain estimate g_hat (trial = mean log-prob over motion windows, as
    capxfer `trial_probs`; the batched feature path is the one of scripts/stageA1_hidden.py);
  * Mondrian per-class conformal thresholds on 1 - p_c (`corpus.scoring.mondrian_thresholds`, local copy `SC`) and the g interval
    g_lo = g_hat - dg (`evaluate.g_delta`), both fitted on capxfer's own simulated calibration split
    (rung_v2/sim_probs_plated.csv, part = calib);
  * the empty-set convention: an empty projected set gives lower = -inf (refuse).
How it is adapted to a multi-module child B of a parent G (Theorem 5(a) construction, per leg):
  1. levels: t_OK at 1 - alpha1/(2 n_G) (n_G = parent wheels), fault classes at 1 - alpha1/(2K), K = 2, dg at
     1 - alpha1/(2K); alpha1 = alpha2 = alpha/2;
  2. projection S|_B: every child mode vector drawn from the child wheels' sets with at most K - (forced faults
     outside B) faults (a wheel is forced faulted when its set excludes ok); weak wheels at their gain corner g_lo,
     floored to capxfer's 0.05 table grid (all of 0.30..0.90 allowed below the grid, i.e. 0.30);
     more than MAX_VECTORS vectors -> refuse (capxfer's composition cost cap);
  3. per-mode band b_m(leg): the floor((M2+1) alpha2)-th smallest rho_dec (= rho / nominal healthy reference, the
     study's primary label; capxfer's own z is relative to the same draw's healthy rollout) of the child at mode vector m over M2 fresh
     plant draws (v2 self-model DR x HET_FULL heterogeneity prior), simulated with T1 on the child's own decision legs
     (capxfer used a lone-module lookup table with z = min over directions; here the table is per child shape and per
     leg, cached across episodes since the draws are independent of the episode);
  4. lower(B, leg) = min over m in S|_B of b_m(leg).
Output: point = lower (the raw capxfer certificate), scale = 1, so SimultaneousLower re-calibrates it with an additive
margin; the raw certificate is also reported on its own ("capxfer_raw").
"""
from __future__ import annotations

import itertools
import math
import pathlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cfcap.t1 import capability as cap
from cfcap.t1 import params as P
from cfcap.t1 import plant

from .base import L, Baseline, Prediction

class SC:
    """capxfer corpus.scoring's three conformal helpers, copied verbatim in logic (that module imports scipy at load
    time, which the rented torch image lacks): split-conformal quantile, Mondrian per-class thresholds, wheel sets."""

    @staticmethod
    def conformal_quantile(scores, alpha):
        s = np.sort(np.asarray(scores, float))
        n = len(s)
        k = math.ceil((n + 1) * (1 - alpha))
        if n == 0 or k > n:
            return np.inf
        return float(s[k - 1])

    @staticmethod
    def mondrian_thresholds(probs, labels, alpha, n_classes=4):
        probs, labels = np.asarray(probs, float), np.asarray(labels)
        return np.array([SC.conformal_quantile(1 - probs[labels == c, c], alpha) for c in range(n_classes)])

    @staticmethod
    def wheel_sets(probs, q):
        return (1 - np.asarray(probs, float)) <= np.asarray(q, float)[None, :]


BASE = pathlib.Path.home() / ".cache/modular-runs/2026-11/capxfer/rung_v2"
P_COLS = ["p_ok", "p_weak", "p_coast", "p_brake"]
TABLE_GRID = tuple(round(0.30 + 0.05 * k, 2) for k in range(13))   # capxfer corpus.capability.G_GRID
K_FAULTS = 2
MAX_VECTORS = 400


class SelfModelV2:
    """capxfer v2 self-model ensemble, batched over episodes (from scripts/stageA1_hidden.py, unchanged logic)."""

    def __init__(self, plate="plated", device="cuda"):
        from capxfer.corpus import selfmodel as S
        self.S, self.device, self.models = S, device, []
        for s in (1, 2, 3):
            m = S.SelfModel().to(device)
            m.load_state_dict(torch.load(BASE / "models" / f"selfmodel_{plate}_s{s}.pt", map_location=device))
            m.eval()
            self.models.append(m)

    @torch.no_grad()
    def __call__(self, asm, cmd, obs, bs=256):
        """cmd (T, 3) numpy; obs dict of (T, n)/(T,) tensors (one episode) -> p (n, 4), g_hat (n,)."""
        S, dev = self.S, self.device
        T, n = obs["pre_duty"].shape
        cmd = np.asarray(cmd[:T], np.float32)
        ref = torch.tensor(S.wheel_ref(asm, cmd), device=dev)
        cmd_t = torch.tensor(cmd, device=dev)
        wf = torch.stack([ref / 5.0, obs["pre_duty"].to(dev), obs["speed"].to(dev) / 5.0], -1).half().float()
        af = torch.stack([cmd_t[:, 0] / 0.3, cmd_t[:, 1] / 0.3, cmd_t[:, 2], obs["gz"].to(dev), obs["ax"].to(dev),
                          obs["ay"].to(dev)], -1)
        geom = torch.tensor(S.geometry(asm), device=dev)
        starts = S.motion_windows(cmd, np.ones(T, bool))
        if not starts:
            return None, None
        ar = torch.arange(S.W, device=dev)
        lp = torch.zeros(n, 4, device=dev)
        gs = torch.zeros(n, device=dev)
        for i in range(0, len(starts), bs):
            ss = torch.tensor(starts[i:i + bs], device=dev)
            tt = ss[:, None] + ar[None, :]
            w, a = wf[tt], af[tt]
            g = geom.expand(len(ss), n, 5)
            mask = torch.ones(len(ss), n, dtype=torch.bool, device=dev)
            for m in self.models:
                logits, gh = m(w, a, g, mask)
                lp += F.log_softmax(logits, -1).sum(0) / len(self.models)
                gs += gh.sum(0) / len(self.models)
        return torch.softmax(lp / len(starts), -1).cpu().numpy(), (gs / len(starts)).cpu().numpy()


def thresholds(n_parent, alpha, plate="plated"):
    """capxfer Mondrian thresholds and dg from capxfer's simulated calibration split at the Theorem 5 levels."""
    sim = pd.read_csv(BASE / f"sim_probs_{plate}.csv")
    cal = sim[sim.part == "calib"]
    a1 = alpha / 2
    q = SC.mondrian_thresholds(cal[P_COLS].values, cal["mode"].values, a1 / (2 * K_FAULTS))
    q[0] = SC.conformal_quantile(1 - cal[cal["mode"] == 0].p_ok.values, a1 / (2 * n_parent))
    w = cal[cal["mode"] == 1]
    dg = SC.conformal_quantile(np.abs(w.g_hat - w.g_true), a1 / (2 * K_FAULTS))
    return q, float(dg)


def corner(g_lo):
    return max([g for g in TABLE_GRID if g <= g_lo + 1e-9], default=TABLE_GRID[0])


def child_vectors(sets, g_lo, child_wheels, n_parent, k=K_FAULTS, cap_=MAX_VECTORS, fixed=None):
    """S|_B as a list of (mode tuple, g tuple) over the child's wheels; [] = empty (refuse); None = over the cap.
    child_wheels: parent wheel index per candidate wheel, or -1 for a merge partner's wheel, whose (mode, g) is
    taken from `fixed` (known partner; it does not use the parent's fault budget)."""
    if any(len(sets[w]) == 0 for w in range(n_parent)):
        return []                    # globally empty support (some parent wheel has no class): refuse
    forced = [w for w in range(n_parent) if 0 not in sets[w]]
    outside = sum(1 for w in forced if w not in set(child_wheels))
    budget = k - outside
    if budget < 0:
        return []
    opts = []
    for j, w in enumerate(child_wheels):
        if w < 0:
            opts.append([(("P", int(fixed[0][j])), float(fixed[1][j]))])
            continue
        o = []
        for c in sorted(sets[w]):
            o.append((c, corner(g_lo[w]) if c == plant.WEAK else 1.0))
        if not o:
            return []
        opts.append(o)
    out = []
    for combo in itertools.product(*opts):
        if sum(1 for c, _ in combo if not isinstance(c, tuple) and c != 0) <= budget:
            out.append((tuple(c[1] if isinstance(c, tuple) else c for c, _ in combo), tuple(g for _, g in combo)))
            if len(out) > cap_:
                return None
    return out


class CapxferCert(Baseline):
    name = "capxfer"

    def __init__(self, alpha=0.1, M2=39, plate="plated", device="cuda", seed=505):
        self.alpha, self.M2, self.plate, self.device, self.seed = alpha, M2, plate, device, seed
        self.sm = SelfModelV2(plate, device)
        self.k_band = math.floor((M2 + 1) * alpha / 2)
        assert self.k_band >= 1, "M2 too small for alpha2"
        self._thr = {}
        self._bands = {}
        self._draws = {}
        self.dr = P.theta_dr(P.load_theta(plate))

    def _draw(self, cand):
        """Fresh plant draws per child shape (common across mode vectors and episodes)."""
        key = self._shape_key(cand)
        if key not in self._draws:
            rng = np.random.default_rng(self.seed + hash(key) % 100003)
            self._draws[key] = plant.sample_latents(cand.asm, self.M2, rng, plant.HET_FULL, dr=self.dr,
                                                    device=self.device)
        return self._draws[key]

    @staticmethod
    def _shape_key(cand):
        cells = [(int(i), int(j), int(y)) for _, i, j, y in cand.asm.modules]
        x0, y0 = min(c[0] for c in cells), min(c[1] for c in cells)
        return tuple((i - x0, j - y0, y) for i, j, y in cells)

    def bands(self, par, items):
        """items: list of (cand, mode tuple, g tuple) -> (len, L) bands; computes the uncached ones in one batch."""
        need, keys = [], []
        for cand, m, g in items:
            k = (self._shape_key(cand), m, g)
            keys.append(k)
            if k not in self._bands and k not in [x[0] for x in need]:
                need.append((k, cand, m, g))
        if need:
            targets = []
            for k, cand, m, g in need:
                lat = dict(self._draw(cand))
                lat["mode"] = torch.tensor([m] * self.M2, device=self.device).long()
                lat["g"] = torch.tensor([g] * self.M2, device=self.device).float()
                targets.append((cand.asm, lat))
            from cfcap.t1.reference import rho_ref_many
            from . import posterior as Po
            outs = Po.with_fallback(lambda e: cap.capability(targets, par, device=self.device, engine=e,
                                                             max_wheels=Po.CAP_MAX_WHEELS), "graph")
            ref = rho_ref_many([x[1].asm for x in need], self.plate, self.device)
            for (k, *_), o, rf in zip(need, outs, ref):
                r = np.sort(o["rho"].cpu().numpy() / rf[None], 0)
                self._bands[k] = r[self.k_band - 1]
        return np.stack([self._bands[k] for k in keys]) if keys else np.zeros((0, L))

    def predict(self, ep):
        par, asm, cands = ep.params, ep.parent_asm, ep.cands
        truth_c = ep.cand_true_modes()  # used only for merge-partner wheels (known)
        n = asm.n
        if n not in self._thr:
            self._thr[n] = thresholds(n, self.alpha, self.plate)
        q, dg = self._thr[n]
        t0 = time.perf_counter()
        p, gh = self.sm(asm, ep.cmd, ep.obs)
        if p is None:
            lo = np.full((len(cands), L), -np.inf)
            return Prediction(lo, np.ones_like(lo), lo, dict(refuse="no motion windows"))
        sets = [set(np.flatnonzero(s)) for s in SC.wheel_sets(p, q)]
        g_lo = np.clip(gh - dg, 0.0, 1.0)
        lower = np.full((len(cands), L), -np.inf)
        items, owner, n_vec = [], [], []
        for ci, c in enumerate(cands):
            vecs = child_vectors(sets, g_lo, list(c.src), n, fixed=truth_c[ci])
            n_vec.append(-1 if vecs is None else len(vecs))
            if not vecs:
                continue                       # empty or over the cap -> refuse (-inf)
            for m, g in vecs:
                items.append((c, m, g))
                owner.append(ci)
        b = self.bands(par, items)
        owner = np.asarray(owner)
        for ci in np.unique(owner):
            lower[ci] = b[owner == ci].min(0)
        n_empty = sum(1 for v in n_vec if v == 0)
        return Prediction(lower.copy(), np.ones_like(lower), lower,
                          dict(t_selfmodel=time.perf_counter() - t0, n_vectors=int(np.sum(np.maximum(n_vec, 0))),
                               n_refuse_empty=n_empty, n_refuse_cap=sum(1 for v in n_vec if v < 0),
                               set_sizes=[len(s) for s in sets], bands_cached=len(self._bands)))
