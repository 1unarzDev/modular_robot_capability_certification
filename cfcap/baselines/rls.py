"""RLS + intervals: per-wheel effective motor gain from duty and encoder speed, propagated through the T0 QS plant.

Per wheel i, over the moving samples of the parent's v2 trial, least squares (= RLS with forgetting factor 1, run in
batch form) of the steady motor law
    omega_i(t) = k_i u_i(t) + b_i sign(omega_ref,i(t))          (u = published pre_duty, omega = encoder speed)
gives k_i and its standard error. The effective gain is relative to the assembly: g_hat_i = k_i / median_j k_j
(removes battery voltage and floor), with interval g_hat_i +- (Z_SE se_i + H_SPREAD). H_SPREAD = 0.20 is the 1-99 %
band of g_hat on healthy wheels with heterogeneity on (exploratory check on 64 square episodes: 0.77-1.28).
A passive wheel's speed does not follow its duty (k ~ 0): g_hat_i < PASSIVE_G flags it as possibly coast/brake.

Propagation (T0, `cfcap.t0.qs`, nominal theta, no heterogeneity): for every candidate, the QS analogue of rho_dec,
rho_QS(h) /
rho_QS(healthy) on the 10 decision legs, where
  point: weak(clip(g_hat, 0.3, 1)) on unflagged wheels (ok if >= 0.97), flagged wheels at the worse of coast / brake;
  lower: weak(g_lo) on unflagged wheels (rho is monotone in a wheel's gain), flagged wheels at the worst combination
         of {coast, brake, weak(0.3)} (enumerated; more than MAX_FLAGGED flagged wheels in a candidate -> refuse).
Scale = point - lower + SCALE_FLOOR.
"""
from __future__ import annotations

import itertools

import numpy as np

from cfcap.t0 import qs
from cfcap.t1 import capability as cap
from cfcap.t1 import plant

from .base import L, Baseline, Prediction

Z_SE = 2.0
H_SPREAD = 0.20
PASSIVE_G = 0.5
MAX_FLAGGED = 3
SCALE_FLOOR = 0.02
G_MIN = 0.3
# T0 leg order -> T1 order
_T0_IDX = [qs.LEG_NAMES.index(l) for l in cap.LEGS]


def rls_gains(asm, cmd, obs, moving=0.05):
    """-> g_hat (n,), se (n,), k (n,)."""
    u = obs["pre_duty"].numpy().astype(float)
    w = obs["speed"].numpy().astype(float)
    T = u.shape[0]
    c = np.asarray(cmd[:T], float)
    ref = asm.ik(c, 0.0501)
    mov = np.abs(c).sum(1) > moving
    n = asm.n
    k, se = np.zeros(n), np.zeros(n)
    for i in range(n):
        X = np.stack([u[mov, i], np.sign(ref[mov, i])], 1)
        y = w[mov, i]
        XtX = X.T @ X
        try:
            beta = np.linalg.solve(XtX, X.T @ y)
            r = y - X @ beta
            s2 = (r @ r) / max(len(y) - 2, 1)
            k[i] = beta[0]
            se[i] = np.sqrt(s2 * np.linalg.inv(XtX)[0, 0])
        except np.linalg.LinAlgError:
            k[i], se[i] = 0.0, np.inf
    med = np.median(k)
    med = med if abs(med) > 1e-9 else 1.0
    return k / med, se / abs(med), k


class RLSInterval(Baseline):
    name = "rls_interval"

    def __init__(self, plate="plated"):
        self.plate = plate
        self.theta = qs.load_theta(plate)
        self._plants = {}

    def _plant(self, cand, par):
        key = (cand.name, id(cand.asm))
        if key not in self._plants:
            if len(self._plants) > 4096:
                self._plants.clear()
            self._plants[key] = qs.Plant(cand.asm, par, self.theta)
        return self._plants[key]

    def _rho_rel(self, pl, modes, gs):
        modes = np.asarray(modes)
        allm = np.concatenate([np.zeros((1, modes.shape[1]), np.int64), modes])
        allg = np.concatenate([np.ones((1, modes.shape[1])), np.asarray(gs, float)])
        r, _ = qs.rho(pl, allm, allg)
        r = r[:, _T0_IDX]
        return r[1:] / np.where(np.abs(r[0:1]) > 1e-9, r[0:1], 1e-9)

    def predict(self, ep):
        par, asm, cands = ep.params, ep.parent_asm, ep.cands
        truth_c = ep.cand_true_modes()  # used only for merge-partner wheels (known)
        gh, se, _ = rls_gains(asm, ep.cmd, ep.obs)
        flag = gh < PASSIVE_G
        g_pt = np.clip(gh, G_MIN, 1.0)
        g_lo = np.clip(gh - (Z_SE * se + H_SPREAD), G_MIN, 1.0)
        C = len(cands)
        point, lower = np.zeros((C, L)), np.full((C, L), -np.inf)
        for ci, c in enumerate(cands):
            src = np.asarray(c.src)
            own = src >= 0
            sp = np.where(own, src, 0)
            pl = self._plant(c, par)
            fl = [j for j, w in enumerate(src) if w >= 0 and flag[w]]
            tm, tg = truth_c[ci]                     # merge-partner wheels: known (already certified partner)
            gp, gl = np.where(own, g_pt[sp], tg), np.where(own, g_lo[sp], tg)
            base_m_pt = np.where(own, np.where(gp < 0.97, plant.WEAK, plant.OK), tm)
            base_m_lo = np.where(own, np.where(gl < 1.0, plant.WEAK, plant.OK), tm)
            if len(fl) > MAX_FLAGGED:
                point[ci] = 0.0
                continue
            rows_m, rows_g, tag = [], [], []
            for combo in itertools.product((plant.COAST, plant.BRAKE), repeat=len(fl)):
                m, g = base_m_pt.copy(), gp.copy()
                for j, mm in zip(fl, combo):
                    m[j], g[j] = mm, 1.0
                rows_m.append(m), rows_g.append(g), tag.append(0)
            for combo in itertools.product(((plant.COAST, 1.0), (plant.BRAKE, 1.0), (plant.WEAK, G_MIN)),
                                           repeat=len(fl)):
                m, g = base_m_lo.copy(), gl.copy()
                for j, (mm, gg) in zip(fl, combo):
                    m[j], g[j] = mm, gg
                rows_m.append(m), rows_g.append(g), tag.append(1)
            rr = self._rho_rel(pl, np.array(rows_m), np.array(rows_g))
            tag = np.array(tag)
            point[ci] = rr[tag == 0].min(0)
            lower[ci] = np.minimum(rr[tag == 1].min(0), point[ci])
        scale = np.where(np.isfinite(lower), point - lower, 1.0) + SCALE_FLOOR
        return Prediction(point, scale, lower, dict(n_flagged=int(flag.sum()), g_hat=np.round(gh, 3).tolist()))
