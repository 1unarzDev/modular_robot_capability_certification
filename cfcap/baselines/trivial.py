"""Trivial / heuristic baselines (research lead's E3; review round 0): what a certificate must beat.

All are robot-visible (they read only the parent's v2 observation and the candidate morphology) and CPU-cheap. Each
outputs (point, scale) for SimultaneousLower like every other baseline.
  * detector_gated: a wheel detector (RLS effective gain g_hat < DET_G, or the passive flag, cfcap.baselines.rls) gates
    the nominal prediction: a candidate holding a flagged wheel gets point 0 (never certified), else 1; scale 1.
  * physics_gains: T0 QS physics on the point estimate of the wheel gains (RLS g_hat, flagged wheels at the worse of
    coast / brake), constant scale (no interval: the point-physics ablation of rls_interval).
  * copy_parent: each candidate inherits the parent's own measured capability on the matching v2 direction
    (encoder-odometry body twist, IMU yaw rate, hold windows found from the command): surge/sway/yaw +-; a diagonal
    takes the min of its surge and sway components; scale 1.
  * weakest_member: a candidate is as capable as its weakest module: per leg, the min over its parent modules of the
    lone-module QS rho with that module's g_hat (merge-partner modules: their known modes); scale 1.
  * model_free (offline, scripts/baselines_pilot.py): per (candidate size, leg) the calibration median of rho_dec,
    scale = median - 5 % quantile + 0.01, no observation at all.
The uncalibrated posterior (enum_known_raw / nuisance_smc_raw) and conformal selection are offline certifiers over
the posterior rows (scripts/baselines_pilot.py from-runs).
"""
from __future__ import annotations

import itertools

import numpy as np

from cfcap.t1 import params as _P  # noqa: F401  (yaml stub before capxfer.assembly)
from cfcap.t0 import qs
from cfcap.t1 import capability as cap
from cfcap.t1 import plant

from .base import L, Baseline, Prediction
from .rls import G_MIN, PASSIVE_G, RLSInterval, rls_gains

DET_G = 0.85
LEG_DIR = {"surge+": (0, 1), "surge-": (0, -1), "sway+": (1, 1), "sway-": (1, -1), "yaw+": (2, 1), "yaw-": (2, -1)}


class DetectorGated(Baseline):
    name = "detector_gated"

    def predict(self, ep):
        gh, _, _ = rls_gains(ep.parent_asm, ep.cmd, ep.obs)
        flag = (gh < DET_G) | (gh < PASSIVE_G)
        pt = np.ones((ep.n_cand, L))
        for ci, c in enumerate(ep.cands):
            src = c.src[c.src >= 0]
            bad = bool(flag[src].any())
            if c.other is not None:
                bad = bad or bool((c.other["mode"] != 0).any())
            if bad:
                pt[ci] = 0.0
        return Prediction(pt, np.ones_like(pt), pt.copy(), dict(n_flagged=int(flag.sum())))


class PhysicsGains(RLSInterval):
    """rls_interval's point only, constant scale."""
    name = "physics_gains"

    def predict(self, ep):
        p = super().predict(ep)
        return Prediction(p.point, np.full_like(p.point, 0.05), p.point.copy(), dict(p.info))


def parent_rho_v2(asm, cmd, obs, radius=0.0501):
    """Parent's measured rho per v2 direction (6,), order surge+, surge-, sway+, sway-, yaw+, yaw- (NaN if absent)."""
    om = obs["speed"].numpy().astype(float)
    T = om.shape[0]
    c = np.asarray(cmd[:T], float)
    J = asm.ik(np.eye(3), radius)                                    # (3, n): wheel speeds per unit twist
    xi = om @ np.linalg.pinv(J)                                      # (T, 3) encoder-odometry twist
    xi[:, 2] = obs["gz"].numpy().astype(float)                       # yaw from the IMU
    Lc = asm.char_length()
    w = np.array([1.0, 1.0, Lc * Lc])
    out = np.full(6, np.nan)
    steady = np.r_[False, np.all(np.abs(np.diff(c, axis=0)) < 1e-6, 1)]
    for k, (name, (ax, sgn)) in enumerate(LEG_DIR.items()):
        v = c[:, ax] * sgn
        if v.max() <= 0:
            continue
        m = steady & (v > 0.95 * v.max()) & (np.abs(np.delete(c, ax, 1)).sum(1) < 1e-6)
        idx = np.flatnonzero(m)
        if idx.size < 5:
            continue
        idx = idx[int(0.2 / plant.DT_OBS):] if idx.size > int(0.4 / plant.DT_OBS) else idx
        xc = c[idx]
        out[k] = float(((xi[idx] * xc * w).sum(1)).sum() / (xc * xc * w).sum(1).sum())
    return out


class CopyParent(Baseline):
    name = "copy_parent"

    def predict(self, ep):
        r6 = parent_rho_v2(ep.parent_asm, ep.cmd, ep.obs)
        d = dict(zip(LEG_DIR, r6))
        legs = []
        for l in cap.LEGS:
            if l in d:
                legs.append(d[l])
            else:                                                     # diag(sx, sy): min of its two components
                sx = "surge+" if l[4] == "+" else "surge-"
                sy = "sway+" if l[5] == "+" else "sway-"
                legs.append(np.nanmin([d[sx], d[sy]]))
        row = np.nan_to_num(np.asarray(legs, float), nan=0.0)
        pt = np.tile(row, (ep.n_cand, 1))
        return Prediction(pt, np.ones_like(pt), pt.copy(), dict(parent_rho=np.round(r6, 3).tolist()))


class WeakestMember(Baseline):
    name = "weakest_member"

    def __init__(self, plate="plated"):
        self.theta = qs.load_theta(plate)
        self._lone = {}

    def _lone_rho(self, par, asm_single, modes, gs):
        key = (tuple(modes), tuple(np.round(gs, 3)), asm_single.modules[0][3])
        if key not in self._lone:
            pl = qs.Plant(asm_single, par, self.theta)
            m = np.array([np.zeros(4, np.int64), np.asarray(modes)])
            g = np.array([np.ones(4), np.asarray(gs, float)])
            r, _ = qs.rho(pl, m, g)
            r = r[:, [qs.LEG_NAMES.index(l) for l in cap.LEGS]]
            self._lone[key] = r[1] / np.where(np.abs(r[0]) > 1e-9, r[0], 1e-9)
        return self._lone[key]

    def predict(self, ep):
        from capxfer import assembly as A
        par = ep.params
        gh, _, _ = rls_gains(ep.parent_asm, ep.cmd, ep.obs)
        flag = gh < PASSIVE_G
        g_pt = np.clip(gh, G_MIN, 1.0)
        pt = np.zeros((ep.n_cand, L))
        single = {}
        for ci, c in enumerate(ep.cands):
            per = []
            for j, k in enumerate(c.pmods):
                mid, i0, j0, yaw = ep.parent_asm.modules[k]
                if yaw not in single:
                    single[yaw] = A.build(f"lone{yaw}", [(mid, 0, 0, yaw)], par)
                w = np.arange(4 * k, 4 * k + 4)
                opts = [((plant.COAST, plant.BRAKE) if flag[x] else
                         ((plant.WEAK if g_pt[x] < 0.97 else plant.OK),)) for x in w]
                r = None
                for combo in itertools.product(*opts):
                    gs = [1.0 if mm != plant.WEAK else g_pt[x] for mm, x in zip(combo, w)]
                    rr = self._lone_rho(par, single[yaw], list(combo), gs)
                    r = rr if r is None else np.minimum(r, rr)
                per.append(r)
            if c.other is not None:                                  # known partner modules
                om, og = c.other["mode"][0].numpy(), c.other["g"][0].numpy()
                for k in range(len(om) // 4):
                    yaw = int(c.asm.modules[len(c.pmods) + k][3])
                    if yaw not in single:
                        single[yaw] = A.build(f"lone{yaw}", [("M1", 0, 0, yaw)], par)
                    per.append(self._lone_rho(par, single[yaw], list(om[4 * k:4 * k + 4]), list(og[4 * k:4 * k + 4])))
            pt[ci] = np.min(per, 0)
        return Prediction(pt, np.ones_like(pt), pt.copy(), {})
