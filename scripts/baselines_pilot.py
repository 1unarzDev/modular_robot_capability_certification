"""Baselines pilot (exploratory): six baselines on ~300 T1 episodes, each feeding conformal.core.SimultaneousLower.

  python scripts/baselines_pilot.py gen  [--per-shape 100]           # episodes -> $CFCAP_OUT/baselines/episodes.pt
  python scripts/baselines_pilot.py run  --baseline NAME [--max-min 14] [--shapes pair,square,2x3]
  python scripts/baselines_pilot.py ingest --runs DIR                 # rented `smc` runner outputs -> pred_*.pt
  python scripts/baselines_pilot.py eval                              # -> results/baselines/*.csv
  python scripts/baselines_pilot.py export                            # -> results/baselines/paper/ (paper contract)
  python scripts/baselines_pilot.py from-runs --runs ~/.cache/vast-out --hardware "RTX 3090"   # rented smc outputs
                                                                      #  -> pilot table + paper contract files

Episodes: parents pair / square / 2x3 (plated, heterogeneity HET_FULL, 0-2 faults), candidates = every connected
proper child, labels = rho_dec (T1 closed-loop rho over the nominal healthy reference) on 10 decision legs (cfcap.baselines.episode).
Baselines (cfcap.baselines): nominal, rls_interval, capxfer, enum_known, nuisance_smc, oracle.
Evaluation: 2-fold cross-fit split conformal (folds alternate within each shape), alpha = 0.1 simultaneous over every
(candidate, leg) of an episode; decisions commit iff lower >= tau, tau in {0.6, 0.7, 0.8}. capxfer's raw certificate
(its own Theorem 5 levels, no recalibration) is reported as `capxfer_raw`.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch

from cfcap.baselines import base
from cfcap.baselines import episode as E

OUT = pathlib.Path(os.environ.get("CFCAP_OUT", pathlib.Path.home() / ".cache/modular-runs/2026-11/cfcap")) / "baselines"
RES = pathlib.Path(__file__).resolve().parents[1] / "results/baselines"
SHAPES = ("pair", "square", "2x3")
ALPHA = 0.1


def make(name):
    if name == "nominal":
        from cfcap.baselines.simple import Nominal
        return Nominal()
    if name == "oracle":
        from cfcap.baselines.simple import Oracle
        return Oracle()
    if name == "rls_interval":
        from cfcap.baselines.rls import RLSInterval
        return RLSInterval()
    if name == "capxfer":
        from cfcap.baselines.capxfer_cert import CapxferCert
        return CapxferCert(alpha=ALPHA)
    if name == "enum_known":
        from cfcap.baselines.bayes import Enumeration
        return Enumeration()
    if name == "nuisance_smc":
        from cfcap.baselines.bayes import NuisanceSMC
        return NuisanceSMC()
    raise KeyError(name)


def cmd_gen(a):
    """Episodes j = 0..3*per_shape-1 of split `pilot`, exactly as the runner task `smc` regenerates them on a box
    (cfcap.baselines.smc.episode), so rented SMC rows join these episodes by eid."""
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eps = [E.pilot_episode(a.split, j) for j in range(len(SHAPES) * a.per_shape)]
    print(len(eps), f"{time.time() - t0:.1f}s", flush=True)
    E.save(eps, OUT / "episodes.pt")


def cmd_ingest(a):
    """Runner `smc` outputs (dec-schema units/*.npz under --runs, any number of shard dirs) -> pred_<method>.pt keyed
    by eid, for the episodes of the local bank (truth is checked against the bank)."""
    import glob
    eps = {e.eid: e for e in E.load(OUT / "episodes.pt")}
    preds = {}
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(a.runs), "**", "units", "*.npz"), recursive=True)):
        z = np.load(f)
        if "method" not in z:
            continue
        for meth in np.unique(z["method"]):
            for eid in np.unique(z["eid"][z["method"] == meth]):
                m = (z["method"] == meth) & (z["eid"] == eid)
                e = eps.get(int(eid))
                if e is None or str(z["split"][m][0]) != a.split:
                    continue
                C = e.n_cand
                assert m.sum() == C * base.L and np.allclose(z["truth"][m], e.truth.ravel(), atol=2e-3), (f, eid)
                preds.setdefault(str(meth), {})[int(eid)] = dict(
                    point=z["point"][m].reshape(C, base.L), scale=z["scale"][m].reshape(C, base.L),
                    lower=z["lower"][m].reshape(C, base.L), seconds=float(z["seconds"][m][0]),
                    info=dict(peak_vram_gb=float(z["peak_vram_gb"][m][0]), sims=float(z["sims"][m][0])))
    for name, d in preds.items():
        torch.save(d, OUT / f"pred_{name}{a.suffix}.pt")
        print(name, len(d))


POSTERIOR = dict(nominal="none", rls_interval="none", capxfer="none", enum_known="enum", nuisance_smc="smc",
                 oracle="oracle")


def cmd_export(a):
    """results/baselines/paper/dec_baselines.csv.gz (docs/board/paper.md 1a) and latency_baselines.csv (1e) from the
    local bank's pred_*.pt. Routes: "simultaneous" = 2-fold cross-fit SimultaneousLower at alpha (the pilot has no
    separate cal split; status exploratory), "raw" = the method's own bound (capxfer's certificate; the posterior
    5 % quantile for enum/SMC)."""
    import pandas as pd
    out = RES / "paper"
    out.mkdir(parents=True, exist_ok=True)
    eps_all = E.load(OUT / "episodes.pt")
    frames, lat = [], []
    for name in POSTERIOR:
        f = OUT / f"pred_{name}.pt"
        if not f.exists():
            continue
        preds = torch.load(f, weights_only=False)
        eps = [e for e in eps_all if e.eid in preds]
        W = max(e.n_cand for e in eps) * base.L
        truth = base.pad([e.truth for e in eps], W)
        point = np.where(np.isnan(truth), np.nan, base.pad([preds[e.eid]["point"] for e in eps], W))
        scale = base.pad([preds[e.eid]["scale"] for e in eps], W)
        shapes = np.array([e.shape for e in eps])
        folds = np.zeros(len(eps), int)
        for s_ in np.unique(shapes):
            idx = np.flatnonzero(shapes == s_)
            folds[idx] = np.arange(len(idx)) % 2
        sim_lo, _ = base.cross_fit_lower(point, scale, truth, folds, ALPHA)
        routes = {"simultaneous": sim_lo}
        if name in ("capxfer", "enum_known", "nuisance_smc"):
            routes["raw"] = base.pad([preds[e.eid]["lower"] for e in eps], W)
        for route, lower in routes.items():
            mname = "capxfer_raw" if (name == "capxfer" and route == "raw") else name
            for i, e in enumerate(eps):
                C = e.n_cand
                k = C * base.L
                hf = e.cand_has_fault()
                frames.append(pd.DataFrame(dict(
                    method=mname, route=route, posterior=POSTERIOR[name], train="T1", test_plant="T1",
                    split=e.split, status="exploratory", eid=e.eid, shape=e.shape,
                    cand=np.repeat(np.arange(C), base.L), kind=np.repeat([c.kind for c in e.cands], base.L),
                    cand_size=np.repeat([c.size for c in e.cands], base.L),
                    has_fault=np.repeat(hf.astype(int), base.L), leg=np.tile(np.arange(base.L), C),
                    truth=e.truth.ravel(), point=point[i, :k], scale=scale[i, :k], lower=lower[i, :k],
                    alpha=ALPHA, seconds=preds[e.eid]["seconds"])))
        for s_ in np.unique(shapes):
            m = shapes == s_
            sec = np.array([preds[e.eid]["seconds"] for e in eps])[m]
            lat.append(dict(method=name, shape=s_, n_wheels=int(eps[int(np.flatnonzero(m)[0])].parent_asm.n),
                            sec_per_decision=float(np.median(sec)), hardware=a.hardware, status="exploratory",
                            n=int(m.sum())))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "dec_baselines.csv.gz", index=False)
    pd.DataFrame(lat).to_csv(out / "latency_baselines.csv", index=False)
    print(len(df), "dec rows;", len(lat), "latency rows")


def cmd_run(a):
    eps = E.load(OUT / "episodes.pt")
    shapes = a.shapes.split(",")
    eps = [e for e in eps if e.shape in shapes]
    if a.per_shape:
        eps = [e for s in shapes for e in [x for x in eps if x.shape == s][:a.per_shape]]
    b = make(a.baseline)
    f = OUT / f"pred_{a.baseline}.pt"
    done = torch.load(f, weights_only=False) if f.exists() and not a.fresh else {}
    t_start = time.time()
    # interleave shapes so a time-limited run keeps every shape represented
    by = {s: [e for e in eps if e.shape == s] for s in shapes}
    order = [e for k in range(max(len(v) for v in by.values())) for s in shapes if k < len(by[s]) for e in [by[s][k]]]
    for e in order:
        if e.eid in done:
            continue
        if time.time() - t_start > a.max_min * 60:
            print("time limit", flush=True)
            break
        t0 = time.perf_counter()
        p = b(e)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        info = {k: v for k, v in p.info.items() if isinstance(v, (int, float, str))}
        done[e.eid] = dict(point=p.point, scale=p.scale, lower=p.lower, seconds=sec, info=info)
        if len(done) % 10 == 0:
            torch.save(done, f)
            print(a.baseline, len(done), e.shape, f"{sec:.2f}s", info, flush=True)
    torch.save(done, f)
    print("saved", f, len(done), flush=True)


def _arrays(eps, preds, key="point"):
    W = max(e.n_cand for e in eps) * base.L
    pt = base.pad([preds[e.eid][key] if preds[e.eid][key] is not None else np.full((e.n_cand, base.L), np.nan)
                   for e in eps], W)
    return pt, W


def cmd_eval(a):
    RES.mkdir(parents=True, exist_ok=True)
    eps_all = E.load(OUT / "episodes.pt")
    rows, timing = [], []
    for name in ("nominal", "rls_interval", "capxfer", "enum_known", "nuisance_smc", "oracle"):
        f = OUT / f"pred_{name}.pt"
        if not f.exists():
            continue
        preds = torch.load(f, weights_only=False)
        eps = [e for e in eps_all if e.eid in preds]
        W = max(e.n_cand for e in eps) * base.L
        truth = base.pad([e.truth for e in eps], W)
        hf = base.pad([np.repeat(e.cand_has_fault()[:, None], base.L, 1).astype(float) for e in eps], W)
        point = base.pad([preds[e.eid]["point"] for e in eps], W)
        scale = base.pad([preds[e.eid]["scale"] for e in eps], W)
        point = np.where(np.isnan(truth), np.nan, point)
        shapes = np.array([e.shape for e in eps])
        folds = np.zeros(len(eps), int)
        for s in SHAPES:
            idx = np.flatnonzero(shapes == s)
            folds[idx] = np.arange(len(idx)) % 2
        variants = {name: base.cross_fit_lower(point, scale, truth, folds, ALPHA)}
        if name == "capxfer":
            raw = base.pad([preds[e.eid]["lower"] for e in eps], W)
            variants["capxfer_raw"] = (np.where(np.isnan(truth), np.nan, raw), {})
        for vname, (lower, q) in variants.items():
            for shp in ("all",) + SHAPES:
                m = np.ones(len(eps), bool) if shp == "all" else shapes == shp
                if not m.any():
                    continue
                for r in base.metrics(lower, truth, hf, ep_mask=m):
                    rows.append(dict(baseline=vname, shape=shp, alpha=ALPHA, q_fold0=q.get(0, np.nan),
                                     q_fold1=q.get(1, np.nan), **r))
        sec = np.array([preds[e.eid]["seconds"] for e in eps])
        for shp in SHAPES:
            m = shapes == shp
            if m.any():
                nc = eps[int(np.flatnonzero(m)[0])].n_cand
                timing.append(dict(baseline=name, shape=shp, n_ep=int(m.sum()), n_cand=nc,
                                   sec_per_decision_median=float(np.median(sec[m])),
                                   sec_per_decision_mean=float(sec[m].mean()),
                                   sec_per_candidate_median=float(np.median(sec[m]) / nc)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "pilot_metrics.csv", index=False)
    pd.DataFrame(timing).to_csv(RES / "pilot_timing.csv", index=False)
    cols = ["baseline", "tau", "n_ep", "cert_frac", "cert_frac_faulted", "unsafe_commit_rate", "ep_any_unsafe",
            "ep_miscover", "q_fold0", "q_fold1"]
    print(df[df["shape"] == "all"][cols].round(4).to_string(index=False))
    print(pd.DataFrame(timing).round(4).to_string(index=False))


DEC_COLS = ("method", "route", "posterior", "train", "test_plant", "split", "status", "eid", "shape", "cand", "kind",
            "cand_size", "has_fault", "leg", "truth", "point", "scale", "lower", "alpha", "seconds")


def load_runs(runs):
    """Every dec-schema unit file (runner task `smc`) under `runs` -> one DataFrame (duplicates dropped)."""
    import glob
    import pandas as pd
    fs = sorted(glob.glob(os.path.join(os.path.expanduser(runs), "**", "units", "*.npz"), recursive=True))
    parts = []
    for f in fs:
        z = np.load(f)
        if "method" in z:
            parts.append(pd.DataFrame({k: z[k] for k in z.files}))
    if not parts:
        raise SystemExit(f"no smc unit files under {runs}")
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(["method", "split", "eid", "cand", "leg"], keep="last")


def cmd_from_runs(a):
    """Pilot table, latency and the paper contract files straight from rented `smc` outputs. Metrics per baseline,
    shape and stratum: "all" entries, and "decision" = candidates containing a faulted module with |rho_dec - tau| <
    0.1 (the paper headline stratum; dec rows carry has_fault / contains_fault and the rho_dec truth to form it):
    results/baselines/pilot_metrics.csv, pilot_timing.csv, paper/dec_baselines.csv.gz, paper/latency_baselines.csv.
    Simultaneous route: 2-fold cross-fit SimultaneousLower (alpha 0.1) within the split, folds alternating by eid
    within each parent shape (the pilot bank has no separate cal split; exploratory)."""
    import pandas as pd
    df = load_runs(a.runs)
    df = df[df.split == a.split]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "paper").mkdir(parents=True, exist_ok=True)
    rows, timing, dec = [], [], []
    for meth, d in df.groupby("method"):
        eids = np.sort(d.eid.unique())
        g = {e: x.sort_values(["cand", "leg"]) for e, x in d.groupby("eid")}
        W = max(len(x) for x in g.values())
        P_ = lambda col: base.pad([g[e][col].to_numpy(float) for e in eids], W)
        truth, point, scale, raw = P_("truth"), P_("point"), P_("scale"), P_("lower")
        hf = P_("has_fault")
        shapes = np.array([g[e]["shape"].iloc[0] for e in eids])
        folds = np.zeros(len(eids), int)
        for s_ in np.unique(shapes):
            idx = np.flatnonzero(shapes == s_)
            folds[idx] = np.arange(len(idx)) % 2
        variants = {}
        if meth == "capxfer":
            variants["capxfer"] = base.cross_fit_lower(raw, np.ones_like(raw), truth, folds, ALPHA)
            variants["capxfer_raw"] = (raw, {})
        else:
            variants[meth] = base.cross_fit_lower(point, scale, truth, folds, ALPHA)
            if meth in ("enum_known", "nuisance_smc"):
                variants[meth + "_q05raw"] = (raw, {})
        for vname, (lower, q) in variants.items():
            for shp in ("all",) + tuple(np.unique(shapes)):
                m = np.ones(len(eids), bool) if shp == "all" else shapes == shp
                for stratum, band in (("all", None), ("decision", 0.1)):
                    for r in base.metrics(lower, truth, hf, ep_mask=m, band=band):
                        rows.append(dict(baseline=vname, shape=shp, stratum=stratum, alpha=ALPHA,
                                         q_fold0=q.get(0, np.nan), q_fold1=q.get(1, np.nan), **r))
            if vname.endswith("_q05raw"):
                continue
            route = "raw" if vname == "capxfer_raw" else "simultaneous"
            for i, e in enumerate(eids):
                x = g[e].copy()
                x["method"], x["route"], x["lower"] = vname, route, lower[i, :len(x)]
                if meth == "capxfer":
                    x["point"], x["scale"] = np.nan, np.nan
                x["contains_fault"] = x["has_fault"]
                dec.append(x[list(DEC_COLS) + ["contains_fault"]])
        sec = np.array([g[e]["seconds"].iloc[0] for e in eids])
        for s_ in np.unique(shapes):
            m = shapes == s_
            nc = int(g[eids[np.flatnonzero(m)[0]]]["cand"].nunique())
            timing.append(dict(method=meth, shape=s_, n=int(m.sum()), n_cand=nc,
                               sec_per_decision=float(np.median(sec[m])), sec_mean=float(sec[m].mean()),
                               sec_per_candidate=float(np.median(sec[m]) / nc),
                               peak_vram_gb=float(d[d["shape"] == s_]["peak_vram_gb"].max()),
                               sims_per_episode=float(d[d["shape"] == s_].groupby("eid")["sims"].first().median())))
    met = pd.DataFrame(rows)
    met.to_csv(RES / "pilot_metrics.csv", index=False)
    tm = pd.DataFrame(timing)
    tm.to_csv(RES / "pilot_timing.csv", index=False)
    pd.concat(dec, ignore_index=True).to_csv(RES / "paper" / "dec_baselines.csv.gz", index=False)
    nw = {"pair": 8, "square": 16, "2x3": 24}
    lat = tm.assign(n_wheels=tm["shape"].map(nw), hardware=a.hardware, status="exploratory")
    lat[["method", "shape", "n_wheels", "sec_per_decision", "hardware", "status", "n"]].to_csv(
        RES / "paper" / "latency_baselines.csv", index=False)
    cols = ["baseline", "stratum", "tau", "n_dec", "cert_frac", "cert_frac_faulted", "unsafe_commit_rate",
            "ep_any_unsafe", "ep_miscover"]
    print(met[met["shape"] == "all"][cols].round(4).to_string(index=False))
    print(tm.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "run", "eval", "ingest", "export", "from-runs"])
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--runs", default="~/.cache/vast-out")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--hardware", default="RTX 4070 Ti SUPER (shared)")
    ap.add_argument("--per-shape", type=int, default=100)
    ap.add_argument("--baseline")
    ap.add_argument("--max-min", type=float, default=14.0)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run" and a.per_shape == 100:
        a.per_shape = 0
    dict(gen=cmd_gen, run=cmd_run, eval=cmd_eval, ingest=cmd_ingest, export=cmd_export, **{"from-runs": cmd_from_runs})[a.cmd](a)


if __name__ == "__main__":
    main()
