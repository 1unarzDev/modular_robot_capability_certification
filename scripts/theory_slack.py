"""Theory (c): certificate slack decomposition, checked on the T0 decision margins of the A2 episodes (exploratory).

Input: the A2 re-run decision table of scripts/theory_coverage.py (rho_true per (episode, candidate, leg, tau); probe
v2, sensor noise, rho model known). A synthetic estimator rho_hat = rho + s * r with r = sqrt(c) * e_episode +
sqrt(1 - c) * e_entry (Gaussian, or Student-t3 for a heavy tail), scale s = sigma. Certificates on the normalised
residual r (commit iff rho_hat - q s >= tau):
  marginal      q = conformal (1-alpha) quantile of the entry scores           (per-decision guarantee)
  simultaneous  q = conformal (1-alpha) quantile of the per-episode max score  (cfcap.conformal.SimultaneousLower)
Decomposition (THEORY_cfcap.md Prop. 4): a capable decision with margin m = rho - tau >= 0 is lost iff
m < s (q - r); with G(t) = P(0 <= m < t | m >= 0) and density bound f on [0, t_max],
  1 - CF <= f * s * ( E(q_pop_marg - r)_+  +  (q_pop_sim - q_pop_marg)  +  (q_hat - q_pop_sim)_+ ) + oracle ambiguity,
and the first-order value G(s (q - r)) averaged over r. Output results/theory/slack_decomposition.csv,
slack_margin_density.csv.
"""
from __future__ import annotations

import glob
import os
import pathlib

import numpy as np
import pandas as pd

from cfcap.conformal.core import SimultaneousLower, split_quantile

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "theory"
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", os.path.expanduser("~/.cache/modular-runs/2026-11/cfcap"))) / "theory"
ALPHA = 0.05
N_CAL = 50                      # calibration episodes per split
REPS = 40


def load():
    D = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(str(OUT / "decisions_v2_*.pkl")))])
    D = D[(D.noise == "sensor") & (D.rho_model == "known")]
    D = D.drop_duplicates(["seed", "cand", "leg", "tau"])
    return D


def episodes_matrix(D):
    """Entries per episode as ragged arrays padded with NaN: rho, tau, lo (oracle), lone-faulted flag."""
    eps = []
    for s, d in D.groupby("seed"):
        eps.append((d.rho_t.values, d.tau.values, d.lo.values,
                    ((d.cand_size == 1) & (d.fault_wheels.fillna("") != "")).values))
    L = max(len(e[0]) for e in eps)
    pad = lambda k: np.stack([np.pad(e[k].astype(float), (0, L - len(e[k])), constant_values=np.nan) for e in eps])
    return pad(0), pad(1), pad(2), pad(3) > 0.5


def draw_r(rng, shape, c, tail):
    if tail == "t3":
        z = lambda sh: rng.standard_t(3, sh) / np.sqrt(3.0)
    else:
        z = lambda sh: rng.standard_normal(sh)
    return np.sqrt(c) * z((shape[0], 1)) + np.sqrt(1 - c) * z(shape)


def main():
    D = load()
    rho, tau, lo, lone = episodes_matrix(D)
    present = ~np.isnan(rho)
    m = rho - tau
    cap = present & (m >= 0)
    n_ep = len(rho)
    RES.mkdir(parents=True, exist_ok=True)
    # margin density of capable decisions near 0
    mm = m[cap]
    dens = []
    for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        dens.append(dict(t=t, G_all=float((mm < t).mean()), f_all=float((mm < t).mean() / t),
                         G_lone=float((m[cap & lone] < t).mean()), f_lone=float((m[cap & lone] < t).mean() / t)))
    pd.DataFrame(dens).to_csv(RES / "slack_margin_density.csv", index=False)
    print(pd.DataFrame(dens).to_string())
    oracle_cf = float(((lo >= tau) & cap).sum() / cap.sum())
    rng = np.random.default_rng(0)
    rows = []
    M_ent = present.sum(1).mean()
    for tail in ("gauss", "t3"):
        for c in (0.0, 0.5, 0.9):
            for sig in (0.01, 0.02, 0.05, 0.1):
                acc = {k: [] for k in ("cf_marg", "cf_sim", "cf_lone_marg", "cf_lone_sim", "ug_marg_dec", "ug_sim_ep",
                                       "ug_marg_ep", "q_marg", "q_sim", "pred_marg", "pred_sim")}
                # population quantiles from a large draw
                big = draw_r(rng, (4000, present.shape[1]), c, tail)
                bigp = np.where(present[rng.integers(n_ep, size=4000)], big, np.nan)
                qpm = float(np.nanquantile(bigp, 1 - ALPHA))
                qps = float(np.nanquantile(np.nanmax(bigp, 1), 1 - ALPHA))
                for rep in range(REPS):
                    r = np.where(present, draw_r(rng, rho.shape, c, tail), np.nan)
                    perm = rng.permutation(n_ep)
                    cal, te = perm[:N_CAL], perm[N_CAL:]
                    q_m = split_quantile(r[cal][present[cal]], ALPHA)
                    sl = SimultaneousLower(ALPHA).fit(np.nan_to_num(r[cal]) * sig + np.nan_to_num(rho[cal]),
                                                      np.full(r[cal].shape, sig),
                                                      np.where(present[cal], rho[cal], np.nan))
                    q_s = sl.q
                    rh = rho[te] + sig * r[te]
                    for name, q in (("marg", q_m), ("sim", q_s)):
                        go = present[te] & (rh - q * sig >= tau[te])
                        capt = cap[te]
                        acc[f"cf_{name}"].append((go & capt).sum() / capt.sum())
                        ln = capt & lone[te]
                        acc[f"cf_lone_{name}"].append((go & ln).sum() / max(1, ln.sum()))
                        uns = go & present[te] & (rho[te] < tau[te])
                        if name == "marg":
                            acc["ug_marg_dec"].append(uns.sum() / present[te].sum())
                            acc["ug_marg_ep"].append(uns.any(1).mean())
                        else:
                            acc["ug_sim_ep"].append(uns.any(1).mean())
                        # attribution: loss with the population marginal / simultaneous quantile on the same draws
                        qq = qpm if name == "marg" else qps
                        need = sig * (qq - r[te])
                        pred = ((m[te] < need) & capt).sum() / capt.sum()
                        acc[f"pred_{name}"].append(1 - pred)
                    acc["q_marg"].append(q_m); acc["q_sim"].append(q_s)
                A = {k: float(np.mean(v)) for k, v in acc.items()}
                # computable bound (Prop. 4) with f_bar from the empirical capable-margin cdf on (0, t_max]
                tmax = sig * (A["q_sim"] + 4.0)
                fbar = float(max((mm < t).mean() / t for t in np.linspace(tmax / 20, tmax, 20)))
                er = float(np.nanmean(np.clip(qpm - bigp, 0, None)))
                bound_sim = min(1.0, fbar * sig * (er + (qps - qpm) + max(0.0, A["q_sim"] - qps)))
                from scipy.stats import norm
                q_bonf = float(norm.ppf(1 - ALPHA / M_ent)) if tail == "gauss" else np.nan
                rows.append(dict(tail=tail, corr=c, sigma=sig, M_entries=M_ent, n_cal=N_CAL, alpha=ALPHA,
                                 oracle_cf=oracle_cf, q_pop_marg=qpm, q_pop_sim=qps, q_hat_marg=A["q_marg"],
                                 q_hat_sim=A["q_sim"], bonferroni_q=q_bonf,
                                 cf_marg=A["cf_marg"], cf_sim=A["cf_sim"], cf_popq_marg=A["pred_marg"],
                                 cf_popq_sim=A["pred_sim"], cf_lone_marg=A["cf_lone_marg"],
                                 cf_lone_sim=A["cf_lone_sim"],
                                 loss_bound_sim=bound_sim, loss_sim=1 - A["cf_sim"],
                                 loss_est_part=fbar * sig * er, loss_simprice_part=fbar * sig * (qps - qpm),
                                 loss_finite_n_part=fbar * sig * max(0.0, A["q_sim"] - qps),
                                 unsafe_marg_per_dec=A["ug_marg_dec"], unsafe_marg_per_ep=A["ug_marg_ep"],
                                 unsafe_sim_per_ep=A["ug_sim_ep"]))
                print(rows[-1]["tail"], c, sig, round(A["cf_marg"], 4), round(A["cf_sim"], 4),
                      round(A["pred_sim"], 4), round(1 - bound_sim, 4), round(A["ug_sim_ep"], 4), flush=True)
    pd.DataFrame(rows).to_csv(RES / "slack_decomposition.csv", index=False)


if __name__ == "__main__":
    main()
