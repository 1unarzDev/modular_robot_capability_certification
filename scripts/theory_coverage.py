"""Theory (a): operating-point coverage on T0 (exploratory).

  closed   closed-form wheel speeds of every decision leg / probe leg vs T0 IK, and the speed-dominance shares of
           children's decision legs by the parent's v2 (and v2+clamp) legs, over random 4-8-module parents.
           -> results/theory/coverage_closedform.csv, coverage_dominance.csv
  rerun    re-runs the A2 engine (cfcap/t0/stagea.run_episode, unchanged) with a read-only hook on the scorer that
           records every (candidate, leg, tau) decision of probe set v2 (sensor and floor noise, rho models known and
           prior): rho_true, set min/max, and for the faulted wheels the child's leg operating point vs the parent's
           v2 operating points.  -> $CFCAP_OUT/theory/decisions_v2_<s0>_<s1>.pkl
  analyze  attribution of the ambiguous decisions -> results/theory/coverage_ambiguous.csv, coverage_attrib.csv

Run heavy parts via scripts/heavy.sh.
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "theory"
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", os.path.expanduser("~/.cache/modular-runs/2026-11/cfcap"))) / "theory"
LEG_KIND = {0: "surge", 1: "surge", 2: "sway", 3: "sway", 4: "diag", 5: "diag", 6: "diag", 7: "diag", 8: "yaw", 9: "yaw"}


def closed_form_speeds(asm, xis, R):
    """X-pattern closed form: omega_i = sqrt2 (v_i . r_i) / R with v_i the contact velocity (w.r = 1/sqrt2)."""
    p = asm.wheel_pos
    vc = np.stack([xis[:, None, 0] - xis[:, None, 2] * p[None, :, 1], xis[:, None, 1] + xis[:, None, 2] * p[None, :, 0]],
                  -1)
    wr = (asm.w_hat * asm.r_hat).sum(-1)
    return (vc * asm.r_hat[None]).sum(-1) / (R * wr[None]), np.abs(wr)


def cmd_closed(a):
    from cfcap.t0 import qs, stagea, candidates as C, probesets as PS
    P, th = qs.load_params(), qs.load_theta()
    rng = np.random.default_rng(a.seed)
    R = qs.sim_nominal(P)["radius"]
    rows, dom = [], []
    max_err, wr_all = 0.0, []
    for ep in range(a.n):
        parent = stagea.random_parent(rng, P, (4, 5, 6, 7, 8))
        pl = qs.Plant(parent, P, th)
        v2 = pl.ik(PS.v2_twists(parent))                                  # (6, n)
        clamp = np.stack([l[1] for l in PS.probe_legs(pl, "clamp")])
        cf, wr = closed_form_speeds(parent, PS.v2_twists(parent), R)
        max_err = max(max_err, float(np.abs(cf - v2).max()))
        wr_all.append(wr)
        vmax = np.abs(v2).max(0)
        vmax_c = np.maximum(vmax, np.abs(clamp).max(0))
        for cand in C.children(parent, P):
            xis = qs.decision_legs(cand.asm)
            om = qs.ik(cand.asm, xis, R)                                  # (10, n_c)
            cfc, _ = closed_form_speeds(cand.asm, xis, R)
            max_err = max(max_err, float(np.abs(cfc - om).max()))
            src = cand.src
            for li in range(10):
                w = np.abs(om[li])
                act = w > 1e-9
                und = w > vmax[src] * (1 + 1e-9)
                und_c = w > vmax_c[src] * (1 + 1e-9)
                dom.append(dict(ep=ep, parent_size=len(parent.modules), cand_size=cand.size, leg=li,
                                leg_kind=LEG_KIND[li], n_wheels=len(w), n_active=int(act.sum()),
                                n_undom_v2=int(und.sum()), n_undom_v2clamp=int(und_c.sum()),
                                max_ratio_v2=float((w / vmax[src]).max()),
                                yaw_rate=float(xis[li, 2]), Lc=cand.asm.char_length()))
    D = pd.DataFrame(dom)
    RES.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "coverage_dominance_raw.csv.gz", index=False)
    g = D.groupby("leg_kind")
    S = pd.DataFrame(dict(
        n_child_legs=g.size(),
        share_legs_with_undominated_wheel_v2=g.apply(lambda d: (d.n_undom_v2 > 0).mean()),
        share_wheels_undominated_v2=g.n_undom_v2.sum() / g.n_wheels.sum(),
        share_legs_with_undominated_wheel_v2clamp=g.apply(lambda d: (d.n_undom_v2clamp > 0).mean()),
        share_wheels_undominated_v2clamp=g.n_undom_v2clamp.sum() / g.n_wheels.sum(),
        max_speed_ratio_v2_median=g.max_ratio_v2.median(), max_speed_ratio_v2_max=g.max_ratio_v2.max()))
    S.reset_index().to_csv(RES / "coverage_dominance.csv", index=False)
    Y = D[D.leg_kind == "yaw"].groupby("cand_size").apply(
        lambda d: pd.Series(dict(n=len(d), share_undom_v2=(d.n_undom_v2 > 0).mean(),
                                 share_undom_v2clamp=(d.n_undom_v2clamp > 0).mean(),
                                 ratio_median=d.max_ratio_v2.median())))
    Y.reset_index().to_csv(RES / "coverage_dominance_yaw_by_size.csv", index=False)
    pd.DataFrame([dict(n_parents=a.n, max_abs_err_closed_form_radps=max_err,
                       w_dot_r_abs_min=float(np.concatenate(wr_all).min()),
                       w_dot_r_abs_max=float(np.concatenate(wr_all).max()),
                       surge_speed_radps=0.3 / R, diag_speed_radps=np.sqrt(2) * 0.21 * np.sqrt(2) / R,
                       clamp_speed_radps=0.45 / R)]).to_csv(RES / "coverage_closedform.csv", index=False)
    print(S.to_string())
    print(Y.to_string())
    print("closed-form max err", max_err)


# ----------------------------------------------------------------------------- A2 re-run with a scorer hook
def truth_of(seed, params, sizes=(4, 5, 6, 7, 8)):
    """Replays the first rng draws of stagea.run_episode (parent, het, nuisances, true faults)."""
    from cfcap.t0 import stagea, observe as O
    from capxfer.sim import WEAK, COAST, BRAKE
    rng = np.random.default_rng(seed)
    parent = stagea.random_parent(rng, params, sizes)
    n = parent.n
    het = bool(rng.integers(2))
    st, nl, mu = O.sample_het(rng, parent, het)
    nf = int(rng.integers(1, 3))
    wheels = rng.choice(n, nf, replace=False)
    tm, tg = np.zeros(n, np.int64), np.ones(n)
    for w in wheels:
        mm = int(rng.choice([WEAK, COAST, BRAKE]))
        tm[w] = mm
        tg[w] = float(rng.choice(stagea.G_GRID)) if mm == WEAK else 1.0
    return parent, het, (st, nl, mu), tm, tg


def cmd_rerun(a):
    import torch
    torch.set_num_threads(4)
    from cfcap.t0 import qs, stagea, candidates as C, probesets as PS
    P, th = qs.load_params(), qs.load_theta()
    R = qs.sim_nominal(P)["radius"]
    s0, s1 = map(int, a.seeds.split(":"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    state = {}
    orig_dl, orig_dc = qs.decision_legs, stagea.decision_counts

    def dl_hook(asm):
        state["cand"] = asm.name
        state["k"] = 0
        return orig_dl(asm)

    def dc_hook(rho_h, mask, rho_t, tau):
        # call order inside run_episode per candidate: for (set, noise) in cmasks: for rho_model: for tau
        k = state["k"]
        state["k"] += 1
        key = state["keys"][state["cand"]]
        nm, nr = key["n_masks"], key["n_rm"]
        mi, rest = divmod(k, nr * len(stagea.TAUS))
        ri = rest // len(stagea.TAUS)
        (sn, p) = key["mask_names"][mi]
        rm = key["rm_names"][ri]
        if sn == "v2":
            m = np.asarray(mask, bool)
            rh = np.asarray(rho_h, float)
            sel = rh[m] if rh.ndim == 2 else rh[:, m].reshape(-1, rh.shape[-1])
            lo, hi = sel.min(0), sel.max(0)
            for li in range(len(rho_t)):
                rows.append(dict(seed=state["seed"], cand=state["cand"], noise=p, rho_model=rm, tau=tau, leg=li,
                                 rho_t=float(rho_t[li]), lo=float(lo[li]), hi=float(hi[li]),
                                 n_set=int(m.sum())))
        return orig_dc(rho_h, mask, rho_t, tau)

    qs.decision_legs = dl_hook
    stagea.decision_counts = dc_hook
    t0 = time.time()
    try:
        for seed in range(s0, s1):
            parent, het, (st, nl, mu), tm, tg = truth_of(seed, P)
            # candidate bookkeeping: mask order = PROBE_SETS x PRESETS (+ diag_split x PRESETS), rho models
            mask_names = [(sn, p) for sn in PS.PROBE_SETS for p in ("sensor", "floor")] + \
                         [("diag_split", p) for p in ("sensor", "floor")]
            rm_names = ["known", "prior"] if het else ["known"]
            state.update(seed=seed, keys=KeyDict(dict(n_masks=len(mask_names), n_rm=len(rm_names),
                                                      mask_names=mask_names, rm_names=rm_names)))
            rec, _ = stagea.run_episode(seed, P, th)
            # parent v2 operating points per wheel (true faulted parent)
            pl = qs.Plant(parent, P, th, stall=st, noload=nl, mu_f=mu)
            L = PS.probe_legs(pl, "v2")
            r = qs.solve(pl, tm[None], tg[None], np.stack([l[1] for l in L]), np.stack([l[2] for l in L]))
            om_p = np.abs(r["om"][0]).max(0)
            ref_p = np.abs(np.stack([l[1] for l in L])).max(0)
            childs = {c.name: c for c in C.children(parent, P)}
            for row in rows:
                if row["seed"] != seed or "fault_wheels" in row:
                    continue
                c = childs.get(row["cand"])
                if c is None:                                              # merge
                    row.update(fault_wheels="", child_ref_max=np.nan, parent_ref_max=np.nan, kind="merge",
                               cand_size=np.nan)
                    continue
                fw = [j for j, s in enumerate(c.src) if tm[s] != 0]
                xis = qs.decision_legs(c.asm)
                om_c = np.abs(qs.ik(c.asm, xis[row["leg"]:row["leg"] + 1], R))[0]
                row.update(kind="child", cand_size=c.size,
                           fault_wheels=";".join(f"{int(tm[c.src[j]])}:{tg[c.src[j]]:.1f}" for j in fw),
                           child_ref_max=float(max([om_c[j] for j in fw], default=0.0)),
                           parent_ref_max=float(max([ref_p[c.src[j]] for j in fw], default=0.0)),
                           parent_om_v2_max=float(max([om_p[c.src[j]] for j in fw], default=0.0)),
                           het=int(het), parent_size=len(parent.modules))
            print(f"seed {seed} het {int(het)} {rec['fault_type']} |U| {rec['n_union']} {time.time() - t0:.0f}s",
                  flush=True)
    finally:
        qs.decision_legs, stagea.decision_counts = orig_dl, orig_dc
    pd.DataFrame(rows).to_pickle(OUT / f"decisions_v2_{s0:05d}_{s1:05d}.pkl")
    print("done", time.time() - t0)


class KeyDict(dict):
    """Same bookkeeping for every candidate of an episode."""

    def __init__(self, v):
        super().__init__()
        self.v = v

    def __getitem__(self, k):
        return self.v


def cmd_analyze(a):
    D = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(str(OUT / "decisions_v2_*.pkl")))])
    D["amb"] = (D.lo < D.tau) & (D.hi >= D.tau)
    D["leg_kind"] = D.leg.map(LEG_KIND)
    D["undominated_v2"] = D.child_ref_max > D.parent_ref_max * (1 + 1e-9)
    D["above_clamp"] = D.child_ref_max > 0.45 / 0.0501 * (1 + 1e-9)
    RES.mkdir(parents=True, exist_ok=True)
    A = D[D.amb & (D.noise == "sensor")]
    A.to_csv(RES / "coverage_ambiguous.csv", index=False)
    out = []
    for (nz, rm), d in D.groupby(["noise", "rho_model"]):
        amb = d[d.amb]
        out.append(dict(noise=nz, rho_model=rm, n_dec=len(d), n_amb=len(amb), amb_share=d.amb.mean(),
                        amb_share_faulted_child_undominated=d[d.undominated_v2 == True].amb.mean(),
                        amb_share_faulted_child_dominated=d[(d.undominated_v2 == False) &
                                                            (d.fault_wheels.fillna("") != "")].amb.mean(),
                        share_amb_undominated=(amb.undominated_v2 == True).mean() if len(amb) else np.nan,
                        share_amb_diag=(amb.leg_kind == "diag").mean() if len(amb) else np.nan,
                        share_amb_yaw=(amb.leg_kind == "yaw").mean() if len(amb) else np.nan,
                        share_amb_lone=(amb.cand_size == 1).mean() if len(amb) else np.nan,
                        share_dec_undominated=(d.undominated_v2 == True).mean()))
    S = pd.DataFrame(out)
    S.to_csv(RES / "coverage_attrib.csv", index=False)
    print(S.to_string())
    print(A[A.rho_model == "known"][["seed", "cand", "leg_kind", "tau", "rho_t", "lo", "hi", "fault_wheels",
                                     "child_ref_max", "parent_ref_max", "undominated_v2"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("closed"); c.add_argument("--n", type=int, default=300); c.add_argument("--seed", type=int, default=7)
    r = sp.add_parser("rerun"); r.add_argument("--seeds", default="0:100")
    sp.add_parser("analyze")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dict(closed=cmd_closed, rerun=cmd_rerun, analyze=cmd_analyze)[a.cmd](a)
