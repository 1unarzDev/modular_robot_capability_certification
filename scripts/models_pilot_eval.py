"""Evaluate the models pilot (exploratory) against the capxfer v2 self-model on the pilot test sets.

Detection at matched 1 % false alarm (threshold = 99th percentile of the detector's own score on healthy test
episodes of the shape, as Stage A1), |g_hat - g| on the true weak wheel, mode-posterior ECE; direct-head and
compositional rho_rel quantiles (pinball, 90 % coverage) by child size, incl. 2x3 parents (unseen size).

  scripts/heavy.sh --gpu $PY scripts/models_pilot_eval.py --tags n4_s1,n2_s1 --compose n4_s1
Writes results/models/{detect,calib,cf,latency}.csv.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch

from cfcap.models import compose as CO
from cfcap.models import data as D
from cfcap.models import posterior as M
from cfcap.t1 import params as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results/models"
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", pathlib.Path.home() / ".cache/modular-runs/2026-11/cfcap")) / "models"
TEST = ("pair", "square", "2x3")
G_BINS = ((0.3, 0.45), (0.45, 0.6), (0.6, 0.75), (0.75, 0.9001))


def load_model(tag):
    ck = torch.load(OUT / f"model_{tag}.pt", map_location="cuda", weights_only=False)
    m = ck["meta"]
    model = M.MorphPosterior(h=m["h"], layers=m["layers"], direct=m["direct"]).cuda()
    model.load_state_dict(ck["state"])
    return model.eval(), m


@torch.no_grad()
def infer(model, ss, wheels=2048):
    outs = []
    bs = max(8, wheels // ss.n)
    for i in range(0, ss.N, bs):
        b = ss.batch(torch.arange(i, min(ss.N, i + bs), device="cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(b["wf"], b["af"], b["geom"], b["wm"], b["mask"], b["n_mod"], b["member"], b["cgeom"])
        o = {k: v.float() for k, v in o.items()}
        s = M.posterior_summary(o)
        s.update(ab=o["ab"], cf=o.get("cf"))
        outs.append(s)
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0] if outs[0][k] is not None}


def ece(conf, correct, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - correct[s].mean())
    return float(e)


def detect_rows(name, shape, p, ghat, mode, g, kind):
    """p (B,n,4), ghat (B,n) numpy."""
    score = (1 - p[..., 0]).max(1)
    h = kind == 0
    thr = float(np.quantile(score[h], 0.99))
    rows = []
    wk = kind == 1
    wheel = np.argmax(mode == 1, 1)
    gt = g[np.arange(len(g)), wheel]
    gh = ghat[np.arange(len(g)), wheel]
    loc = np.argmax(1 - p[..., 0], 1) == wheel
    for bn, sel in [("all", wk)] + [(f"{lo:.2f}-{min(hi, .9):.2f}", wk & (gt >= lo) & (gt < hi)) for lo, hi in G_BINS]:
        rows.append(dict(shape=shape, detector=name, g_bin=bn, n_weak=int(sel.sum()), n_healthy=int(h.sum()),
                         thr=round(thr, 4), false_alarm=round(float((score[h] > thr).mean()), 4),
                         detection=round(float((score[sel] > thr).mean()), 4),
                         localisation=round(float(loc[sel].mean()), 4),
                         ghat_err_median=round(float(np.median(np.abs(gh[sel] - gt[sel]))), 4),
                         ghat_err_le_0p05=round(float((np.abs(gh[sel] - gt[sel]) <= 0.05).mean()), 4)))
    return rows


def calib_rows(name, shape, p, mode, kind):
    pf, y = p.reshape(-1, 4), mode.reshape(-1)
    conf, pred = pf.max(1), pf.argmax(1)
    fault_p, fault_y = 1 - pf[:, 0], (y != 0).astype(float)
    faulted = y != 0
    return [dict(shape=shape, detector=name, ece_top=round(ece(conf, (pred == y).astype(float)), 4),
                 ece_fault=round(ece(fault_p, fault_y), 4), acc=round(float((pred == y).mean()), 4),
                 acc_faulted_wheels=round(float((pred[faulted] == y[faulted]).mean()), 4),
                 nll=round(float(-np.log(np.clip(pf[np.arange(len(y)), y], 1e-6, 1)).mean()), 4),
                 n_wheels=len(y))]


def cf_metrics(q, y, taus=M.QUANTILES):
    """q (E,L,Q), y (E,L) -> pinball, coverage of [q05,q95], median width."""
    t = np.asarray(taus)
    d = y[..., None] - q
    pin = np.maximum(t * d, (t - 1) * d).mean()
    cov = ((y >= q[..., 0]) & (y <= q[..., -1])).mean()
    return float(pin), float(cov), float(np.median(q[..., -1] - q[..., 0]))


def cf_rows(method, shape, q, y, size, faulted):
    rows = []
    for s in sorted(set(size.tolist())) + ["all", "faulted", "healthy"]:
        sel = {"all": np.ones(len(size), bool), "faulted": faulted, "healthy": ~faulted}.get(s, size == s)
        if not sel.any():
            continue
        pin, cov, wid = cf_metrics(q[sel], y[sel])
        rows.append(dict(shape=shape, method=method, child_size=s, n=int(sel.sum()), pinball=round(pin, 5),
                         cov90=round(cov, 4), width90_median=round(wid, 4),
                         frac_rho_lt_0p95=round(float((y[sel] < 0.95).mean()), 4)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="n4_s1")
    ap.add_argument("--compose", default="n4_s1")
    ap.add_argument("--n-compose", type=int, default=300)
    ap.add_argument("--S", type=int, default=16)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    par = P.base_params("plated")
    dr = P.theta_dr(P.load_theta("plated"))
    sets = {s: D.ShapeSet(OUT / "pilot" / f"test_{s}.npz", s) for s in TEST}
    det, cal, cfr, lat = [], [], [], []
    # marginal comparator for the rho_rel quantiles: per-leg quantiles of the pooled val children (sizes <= 4)
    vy = np.concatenate([np.load(OUT / "pilot" / f"val_{s}.npz")["rho_rel"].reshape(-1, 10)
                         for s in ("single", "pair", "L3", "square")])
    marg = np.quantile(vy, M.QUANTILES, axis=0).T                                    # (L,Q)
    posts = {}
    for shape, ss in sets.items():
        mode, g, kind = ss.mode.cpu().numpy(), ss.g.cpu().numpy(), ss.kind.cpu().numpy()
        bp, bg = (x.cpu().numpy() for x in ss.base)
        det += detect_rows("capxfer_v2", shape, bp, bg, mode, g, kind)
        cal += calib_rows("capxfer_v2", shape, bp, mode, kind)
        y = ss.rho_rel.cpu().numpy().reshape(-1, 10)
        size = np.array([len(ss.subsets[c]) for c in ss.child.cpu().numpy().reshape(-1)])
        wmod = np.asarray(ss.asm.wheel_module)
        fault_mod = np.stack([(mode[:, wmod == k] != 0).any(1) for k in range(ss.m)], 1)          # (B,m)
        ch = ss.child.cpu().numpy()
        faulted = np.array([fault_mod[e, list(ss.subsets[ch[e, c]])].any() for e, c in np.ndindex(*ch.shape)])
        cfr += cf_rows("marginal_val", shape, np.broadcast_to(marg, y.shape + (len(M.QUANTILES),)), y, size, faulted)
        for tag in a.tags.split(","):
            model, meta = load_model(tag)
            o = infer(model, ss)
            posts[(tag, shape)] = o
            p = o["p"].cpu().numpy()
            det += detect_rows(tag, shape, p, o["g_mean"].cpu().numpy(), mode, g, kind)
            cal += calib_rows(tag, shape, p, mode, kind)
            if "cf" in o:
                cfr += cf_rows(f"direct_{tag}", shape, o["cf"].cpu().numpy().reshape(-1, 10, len(M.QUANTILES)), y,
                               size, faulted)
            # latency: one episode, batch 1
            b = ss.batch(torch.arange(1, device="cuda"))
            ts = []
            with torch.no_grad():
                for k in range(30):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        model(b["wf"], b["af"], b["geom"], b["wm"], b["mask"], b["n_mod"], b["member"], b["cgeom"])
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
            lat.append(dict(method=f"amortized_{tag}", shape=shape, ms_per_episode=round(1e3 * np.median(ts[5:]), 2)))
    # compositional route on a subsample of each test shape (posterior samples -> T1 rollouts of the child)
    rng = np.random.default_rng(7)
    if a.compose:
        tag = a.compose
        for shape, ss in sets.items():
            o = posts[(tag, shape)]
            E = min(a.n_compose, ss.N)
            eps = rng.choice(ss.N, E, replace=False)
            wm = np.asarray(ss.asm.wheel_module)
            for method in ("compositional", "compositional_oracle_mode"):
                jobs, ys, sizes, fl = [], [], [], []
                S = a.S if method == "compositional" else max(4, a.S // 2)
                for e in eps:
                    post = dict(p=o["p"][e], ab=o["ab"][e], nu_mu=o["nu_mu"][e], nu_sd=o["nu_sd"][e])
                    orc = None if method == "compositional" else (ss.mode[e], ss.g[e])
                    for c in range(ss.child.shape[1]):
                        sub = ss.subsets[int(ss.child[e, c])]
                        ca = D.child_assembly(shape, sub, par)
                        widx = np.concatenate([np.nonzero(wm == k)[0] for k in sub])
                        jobs.append((ca, CO.sample_child_latents(ca, widx, post, S, rng, dr, mods=list(sub),
                                                                 oracle=orc)))
                        ys.append(ss.rho_rel[e, c].cpu().numpy())
                        sizes.append(len(sub))
                        fl.append(bool((ss.mode[e].cpu().numpy()[np.isin(wm, sub)] != 0).any()))
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                res = CO.compositional_quantiles(jobs, par)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                q = np.stack([r[0] for r in res])
                cfr += cf_rows(method, shape, q, np.stack(ys), np.array(sizes), np.array(fl))
                lat.append(dict(method=f"{method}_S{S}", shape=shape,
                                ms_per_episode=round(1e3 * dt / len(jobs), 2)))
                # the direct head on the same subsample, for a matched comparison
                if method == "compositional" and "cf" in o:
                    qd = o["cf"][torch.as_tensor(eps, device="cuda")].cpu().numpy().reshape(-1, 10, len(M.QUANTILES))
                    cfr += cf_rows(f"direct_{tag}_subsample", shape, qd, np.stack(ys), np.array(sizes), np.array(fl))
                print(shape, method, f"{dt:.1f}s", flush=True)
    for name, rows in (("detect", det), ("calib", cal), ("cf", cfr), ("latency", lat)):
        pd.DataFrame(rows).to_csv(RES / f"{name}.csv", index=False)
    d = pd.DataFrame(det)
    print(d[d.g_bin.isin(["all", "0.75-0.90"])].to_string(index=False))
    print(pd.DataFrame(cal).to_string(index=False))
    c = pd.DataFrame(cfr)
    print(c[c.child_size.astype(str).isin(["all", "faulted"])].to_string(index=False))
    print(pd.DataFrame(lat).to_string(index=False))


if __name__ == "__main__":
    main()
