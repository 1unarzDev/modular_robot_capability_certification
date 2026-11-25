"""E1 (exploratory): trust-gated certificate on T0 under a misspecified (knee) motor law.

Pre-registration: docs/board/theory.md "E1 pre-registration" (copied from docs/board/research.md §4 E1 before the run).
Episode code: cfcap/theory_task.py gate_episode (runner task GATE); wrong law: cfcap/t0/envelope.py (shared with stress).

  run      --seeds 0:400 --procs 4      -> $CFCAP_OUT/theory/gate/gate_<s0>_<s1>.pkl   (run through scripts/heavy.sh)
  analyze                               -> results/theory/gate_cert.csv, gate_extrapolated_share.csv

Certifiers (lower bound l per (candidate, leg); commit iff l >= tau; alpha = 0.1 simultaneous per episode;
2-fold cross-fit by seed parity, cal fold -> test fold, both directions pooled):
  raw            rho_par (two-constant fit at the probes), q = 0                                   (uncalibrated)
  par            rho_par - q, q = conformal max-over-episode residual on two-constant-truth cal episodes (trusts physics)
  par_pooled     rho_par - q, q pooled over all laws in the cal fold
  pi             rho_pi (identified-set low end, class corners), no calibration                    (model-free)
  pi_cal         rho_pi - q, q pooled
  gate           l0 = rho_par if (dominated or child misfit <= q_gate) else rho_pi; l = l0 - q;
                 q_gate chosen on cal half A (max certified share), q on cal half B (end-to-end, pooled over laws)
  nominal        healthy-body prediction - q (pooled);  detector_gated: nominal on fault-free children only
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pathlib
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "theory"
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", os.path.expanduser("~/.cache/modular-runs/2026-11/cfcap"))) / "theory" / "gate"
ALPHA = 0.1
TAUS = (0.6, 0.7, 0.8)
REL_BAND = 0.10
PROBES = ("v2", "v2clamp")
LEG_KIND = {0: "surge", 1: "surge", 2: "sway", 3: "sway", 4: "diag", 5: "diag", 6: "diag", 7: "diag", 8: "yaw", 9: "yaw"}


def _work(seeds):
    import torch
    torch.set_num_threads(1)
    from cfcap.t0 import qs
    from cfcap.theory_task import gate_episode
    P, th = qs.load_params(), qs.load_theta()
    out = []
    for s in seeds:
        t0 = time.time()
        r = gate_episode(s, P, th)
        r["wall_s"] = np.full(len(r["leg"]), time.time() - t0)
        out.append(pd.DataFrame(r))
    return pd.concat(out)


def cmd_run(a):
    from multiprocessing import get_context
    s0, s1 = map(int, a.seeds.split(":"))
    seeds = list(range(s0, s1))
    chunks = [seeds[i::a.procs * 4] for i in range(a.procs * 4)]
    t0 = time.time()
    with get_context("spawn").Pool(a.procs) as pool:
        parts = []
        for k, df in enumerate(pool.imap_unordered(_work, chunks)):
            parts.append(df)
            print(f"chunk {k + 1}/{len(chunks)} {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_pickle(OUT / f"gate_{s0:05d}_{s1:05d}.pkl")
    print("done", f"{time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- analysis
def q_sim(D, lo_col, ep_mask=None):
    """Conformal (1-alpha) quantile of the per-episode max of (l0 - rho_true) (finite-sample corrected)."""
    from cfcap.conformal.core import split_quantile
    X = D if ep_mask is None else D[ep_mask]
    s = (X[lo_col] - X.rho_true).groupby(X.seed).max()
    return split_quantile(s.values, ALPHA)


def lower_bounds(cal, test, probe):
    """Dict certifier -> lower bound array on `test` (cal: calibration rows)."""
    par, pi = f"rho_par_{probe}", f"rho_pi_{probe}"
    out = {"raw": test[par].values, "pi": test[pi].values}
    out["par"] = test[par].values - q_sim(cal, par, cal.law == 0)
    out["par_pooled"] = test[par].values - q_sim(cal, par)
    out["pi_cal"] = test[pi].values - max(0.0, q_sim(cal, pi))
    cal = cal.assign(nom_l0=cal.rho_nom, det_l0=np.where(cal.has_fault == 1, -np.inf, cal.rho_nom))
    out["nominal"] = test.rho_nom.values - q_sim(cal, "nom_l0")
    qd = q_sim(cal[cal.has_fault == 0], "det_l0")
    out["detector_gated"] = np.where(test.has_fault.values == 1, -np.inf, test.rho_nom.values - qd)
    # gate: q_gate on half A, margin on half B
    A = cal[cal.seed % 4 < 2]
    B = cal[cal.seed % 4 >= 2]
    mis = f"misfit_{probe}"
    grid = np.unique(np.concatenate([[-1.0], np.quantile(A[mis], np.linspace(0, 1, 21)), [np.inf]]))
    best, best_cf = -1.0, -1.0
    for qg in grid:
        l0 = np.where(A[f"dom_{probe}"] | (A[mis] <= qg), A[par], A[pi])
        X = A.assign(g_l0=l0)
        q = q_sim(X, "g_l0")
        cf = np.mean([((l0 - q >= t) & (A.rho_true >= t)).sum() / max(1, (A.rho_true >= t).sum()) for t in TAUS])
        if cf > best_cf:
            best, best_cf = qg, cf
    l0B = np.where(B[f"dom_{probe}"] | (B[mis] <= best), B[par], B[pi])
    qB = q_sim(B.assign(g_l0=l0B), "g_l0")
    l0T = np.where(test[f"dom_{probe}"] | (test[mis] <= best), test[par], test[pi])
    out["gate"] = l0T - qB
    return out, dict(q_gate=best, q_gate_margin=qB)


def cmd_analyze(a):
    parts = [pd.read_pickle(f) for f in sorted(glob.glob(str(OUT / "gate_*.pkl")))]
    for d in a.from_runs or []:                                          # runner outputs (campaign): units/*.npz
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(d), "**", "units", "*.npz"), recursive=True)):
            z = np.load(f, allow_pickle=False)
            parts.append(pd.DataFrame({k: z[k] for k in z.files}))
    D = pd.concat(parts).reset_index(drop=True)
    if "wall_s" not in D:
        D["wall_s"] = np.nan
    D = D.drop_duplicates(["seed", "cand", "leg"])
    print("episodes", D.seed.nunique(), "rows", len(D), "per law", D.groupby("law").seed.nunique().to_dict(),
          "conv", round(D.conv.mean(), 4), "s/ep", round(D.groupby("seed").wall_s.first().mean(), 1))
    rows, qrows, ext = [], [], []
    for probe in PROBES:
        L = {}
        for fold in (0, 1):
            cal, test = D[D.seed % 2 != fold], D[D.seed % 2 == fold]
            lb, info = lower_bounds(cal, test, probe)
            qrows.append(dict(probe=probe, fold=fold, **info))
            for k, v in lb.items():
                L.setdefault(k, []).append(pd.Series(v, index=test.index))
        L = {k: pd.concat(v).reindex(D.index) for k, v in L.items()}
        undom = ~D[f"dom_{probe}"]
        for law in sorted(D.law.unique()):
            lm = D.law == law
            for cert, lo in L.items():
                for t in TAUS:
                    pass
                for stratum in ("all", "decision", "undominated"):
                    acc = dict(n=0, cap=0, inc=0, go_cap=0, unsafe=0)
                    ep_uns = {}
                    for t in TAUS:
                        m = lm.copy()
                        if stratum == "decision":
                            m &= (D.has_fault == 1) | ((D.rho_true - t).abs() < REL_BAND)
                        elif stratum == "undominated":
                            m &= undom
                        go = (lo >= t) & m
                        cap = (D.rho_true >= t) & m
                        acc["n"] += int(m.sum()); acc["cap"] += int(cap.sum()); acc["inc"] += int((m & ~cap).sum())
                        acc["go_cap"] += int((go & cap).sum()); acc["unsafe"] += int((go & ~cap).sum())
                        u = (go & ~cap).groupby(D.seed).any()
                        for s_, v in u[D[lm].seed.unique()].items():
                            ep_uns[s_] = ep_uns.get(s_, False) or bool(v)
                    rows.append(dict(law="two-constant" if law == 0 else "knee", severity=int(law), probe=probe,
                                     certifier=cert, stratum=stratum,
                                     cert_frac_capable=acc["go_cap"] / max(1, acc["cap"]),
                                     unsafe_rate=acc["unsafe"] / max(1, acc["inc"]),
                                     ep_any_unsafe=float(np.mean(list(ep_uns.values()))) if ep_uns else np.nan,
                                     n=acc["n"], n_capable=acc["cap"], n_incapable=acc["inc"],
                                     n_episodes=int(D[lm].seed.nunique())))
            # share of certified decisions that are extrapolated (undominated), by leg kind and child size
            if law == 0:
                for cert in ("par", "gate"):
                    for t in TAUS:
                        go = (L[cert] >= t) & lm
                        G = D[go].assign(undom=undom[go], leg_kind=D.leg[go].map(LEG_KIND),
                                         size_bin=np.where(D.cand_size[go] >= 6, "6+", D.cand_size[go].astype(str)))
                        for (lk, sb), g in G.groupby(["leg_kind", "size_bin"]):
                            ext.append(dict(probe=probe, certifier=cert, tau=t, leg_kind=lk, child_size=sb,
                                            n_certified=len(g), share_undominated=g.undom.mean(),
                                            share_undom_speed=g[f"undom_speed_{probe}"].mean(),
                                            share_undom_torque=g[f"undom_torque_{probe}"].mean()))
                        ext.append(dict(probe=probe, certifier=cert, tau=t, leg_kind="all", child_size="all",
                                        n_certified=int(go.sum()), share_undominated=float(undom[go].mean()),
                                        share_undom_speed=float(D[f"undom_speed_{probe}"][go].mean()),
                                        share_undom_torque=float(D[f"undom_torque_{probe}"][go].mean())))
    R = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    R.to_csv(RES / "gate_cert.csv", index=False)
    pd.DataFrame(ext).to_csv(RES / "gate_extrapolated_share.csv", index=False)
    pd.DataFrame(qrows).to_csv(RES / "gate_thresholds.csv", index=False)
    pd.set_option("display.width", 250)
    for st in ("decision", "undominated"):
        X = R[(R.stratum == st) & R.certifier.isin(["raw", "par", "par_pooled", "pi", "pi_cal", "gate", "nominal",
                                                     "detector_gated"])]
        print(f"--- stratum {st}: CF")
        print(X.pivot_table(index=["probe", "certifier"], columns="severity", values="cert_frac_capable").round(3))
        print(f"--- stratum {st}: unsafe_rate")
        print(X.pivot_table(index=["probe", "certifier"], columns="severity", values="unsafe_rate").round(4))
    print(pd.DataFrame(qrows))
    verdict(R)


def verdict(R):
    d = R[R.stratum == "decision"]
    u = R[R.stratum == "undominated"]
    g = lambda X, probe, cert, sev, col: float(X[(X.probe == probe) & (X.certifier == cert) & (X.severity == sev)][col].iloc[0])
    kill = all(g(u, p, "par", s, "unsafe_rate") <= 0.01 for p in PROBES for s in (1, 2, 3))
    s1 = g(d, "v2", "gate", 0, "cert_frac_capable") >= g(d, "v2", "par", 0, "cert_frac_capable") - 0.02
    sev_bad = [s for s in (1, 2, 3) if g(d, "v2", "par", s, "unsafe_rate") > 0.02]
    worst = max(sev_bad) if sev_bad else None
    s2 = (worst is not None and g(d, "v2", "gate", worst, "unsafe_rate") <= 0.01 and
          g(d, "v2", "gate", worst, "cert_frac_capable") >= g(d, "v2", "pi", worst, "cert_frac_capable") + 0.05)
    lines = [f"KILL (par unsafe on undominated <= 1 % at every knee severity, both probes): {kill}",
             f"SUCCESS-1 (two-constant: gate CF >= par CF - 2 pts, v2, decision stratum): {s1}",
             f"  worst knee severity with par unsafe > 2 % (v2, decision): {worst}",
             f"SUCCESS-2 (at that severity: gate unsafe <= 1 % and gate CF >= pi CF + 5 pts, v2): {s2}"]
    (RES / "gate_verdict.txt").write_text("\n".join(lines) + "\n(source: results/theory/gate_cert.csv; exploratory)\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--seeds", default="0:400"); r.add_argument("--procs", type=int, default=4)
    z = sp.add_parser("analyze"); z.add_argument("--from-runs", nargs="*", help="runner output dirs (campaign)")
    a = ap.parse_args()
    dict(run=cmd_run, analyze=cmd_analyze)[a.cmd](a)
