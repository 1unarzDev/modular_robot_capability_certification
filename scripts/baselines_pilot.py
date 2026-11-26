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
import re
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


def load_runs(runs, campaign=None):
    """Every dec-schema unit file (runner task `smc`) under `runs` -> one DataFrame (duplicates dropped). With
    `campaign`, only job directories of that campaign (path component starting with '<campaign>.', canary jobs
    excluded), and only complete units (the runner's .json sidecar exists next to the .npz)."""
    import glob
    import pandas as pd
    fs = sorted(glob.glob(os.path.join(os.path.expanduser(runs), "**", "units", "*.npz"), recursive=True))
    if campaign:
        fs = [f for f in fs if any(c.startswith(campaign + ".") and ".canary" not in c for c in f.split(os.sep))
              and os.path.exists(f[:-4] + ".json")]
    parts = []
    for f in fs:
        z = np.load(f)
        if "method" in z:
            parts.append(pd.DataFrame({k: z[k] for k in z.files}))
    if campaign and os.environ.get("CFCAP_BL_WITH_STATE"):
        # finished episodes inside units that never completed (the runner's atomic per-episode checkpoints)
        import pickle
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(runs), "**", "state", "*.state"), recursive=True)):
            if not any(c.startswith(campaign + ".") and ".canary" not in c for c in f.split(os.sep)):
                continue
            try:
                st = pickle.loads(open(f, "rb").read())
            except Exception:
                continue
            for rows in st.get("done", {}).values():
                parts.append(pd.DataFrame(rows))
    if not parts:
        raise SystemExit(f"no smc unit files under {runs}")
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(["method", "split", "eid", "cand", "leg"], keep="last")


ALPHAS = (0.05, 0.1)
TRIVIAL = ("detector_gated", "physics_gains", "copy_parent", "weakest_member", "model_free")


def model_free_rows(df, a, has_cal):
    """Observation-free predictor: per (candidate size, leg) the calibration median of rho_dec, scale = median - 5 %
    quantile + 0.01 (calibration = the cal split, or the other cross-fit fold). Built from any one method's rows
    (the truth and candidate columns are method-independent)."""
    import pandas as pd
    base_m = "nominal" if (df.method == "nominal").any() else df.method.iloc[0]
    d = df[df.method == base_m].copy()
    if has_cal:
        d["fold"] = (d.split == a.split).astype(int)
    else:
        ep = d.groupby(["split", "eid"])["shape"].first().reset_index()
        ep["fold"] = ep.groupby("shape").cumcount() % 2
        d = d.merge(ep[["split", "eid", "fold"]], on=["split", "eid"])
    out = []
    for f in sorted(d.fold.unique()):
        cal = d[d.fold != f]
        st = cal.groupby(["cand_size", "leg"]).truth.agg(med="median", q05=lambda x: x.quantile(0.05)).reset_index()
        te = d[d.fold == f].merge(st, on=["cand_size", "leg"], how="left")
        te["point"] = te["med"]
        te["scale"] = (te["med"] - te["q05"]).clip(lower=0) + 0.01
        te["lower"] = te["q05"]
        out.append(te.drop(columns=["med", "q05"]))
    r = pd.concat(out, ignore_index=True).drop(columns=["fold"])
    r["method"], r["posterior"], r["seconds"] = "model_free", "none", 0.0
    return r
RAW_NAME = dict(capxfer="capxfer_raw", enum_known="enum_known_raw", nuisance_smc="nuisance_smc_raw")


def cmd_from_runs(a):
    """Pilot / campaign tables, latency and the paper contract files straight from rented `smc` outputs.

    Calibration: with --cal-split present in the runs, every conformal layer is fitted on that split's episodes and
    evaluated on --split (generator cal/test ids, identical for every method incl. capxfer); otherwise 2-fold
    cross-fit within --split (folds alternate by eid within each parent shape; the pilot bank).
    Certifiers per method (results/baselines/<tag>metrics.csv, column `certifier`):
      sim_a{05,10}   SimultaneousLower at the method's own alpha 0.05 / 0.1 (capxfer: on its raw bound, scale 1);
      mondrian_a10   the same at 0.1 with q fitted per parent shape (robot-visible Mondrian class);
      raw            the method's own uncalibrated bound: capxfer's Theorem-5 certificate, the posterior 5 % quantile
                     for enum_known / nuisance_smc (review round 0 (a));
      cselect_q{05,10}  conformal selection at FDR q (Jin & Candes; review (b)), on (point - tau) / scale.
    Strata: all; decision (candidate contains a faulted module OR |rho_dec - tau| < 0.1; the paper headline);
    class0/1/2 (candidate truly healthy / weak only / coast-brake involved: per-fault-class behaviour, review (c));
    size<k> (candidate modules: score by body size, E3); lone_faulted (single-module candidates holding a fault:
    the population of capxfer's published 58-75 %).
    Trivial baselines (E3; cfcap/baselines/trivial.py + model_free here) go to paper/dec_trivial.csv, the rest to
    paper/dec_baselines.csv.gz.
    Shift: none, and faults2x (test episodes importance-weighted by 2**n_faults: doubled odds of each parent fault;
    calibration unchanged; review (c)).
    Paper files (results/baselines/paper/): dec_baselines.csv.gz with route simultaneous at alpha 0.1 (method name)
    and route raw (capxfer_raw, enum_known_raw, nuisance_smc_raw); latency_baselines.csv."""
    import pandas as pd
    df = load_runs(a.runs)
    has_cal = bool(a.cal_split) and (df.split == a.cal_split).any()
    df = df[df.split.isin([a.split] + ([a.cal_split] if has_cal else []))]
    df = pd.concat([df, model_free_rows(df, a, has_cal)], ignore_index=True)
    tag = a.tag
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "paper").mkdir(parents=True, exist_ok=True)
    rows, timing, dec, dec_triv = [], [], [], []
    for meth, d in df.groupby("method"):
        keys = sorted({(sp, int(e)) for sp, e in zip(d.split, d.eid)})
        g = {k: x.sort_values(["cand", "leg"]) for k, x in d.groupby(["split", "eid"])}
        keys = [k for k in keys if k in g]
        W = max(len(x) for x in g.values())
        P_ = lambda col: base.pad([g[k][col].to_numpy(float) for k in keys], W)
        truth, point, scale, raw, hf = P_("truth"), P_("point"), P_("scale"), P_("lower"), P_("has_fault")
        cls = P_("cand_fault_class") if "cand_fault_class" in d else np.where(hf > 0.5, 1.0, 0.0)
        csz = P_("cand_size")
        nf = np.array([g[k]["ep_n_faults"].iloc[0] if "ep_n_faults" in d else 0 for k in keys], float)
        shapes = np.array([g[k]["shape"].iloc[0] for k in keys])
        splits = np.array([k[0] for k in keys])
        if has_cal:
            folds = (splits == a.split).astype(int)
            ev = splits == a.split
        else:
            folds = np.zeros(len(keys), int)
            for s_ in np.unique(shapes):
                idx = np.flatnonzero(shapes == s_)
                folds[idx] = np.arange(len(idx)) % 2
            ev = np.ones(len(keys), bool)
        if meth == "capxfer":
            pt, sc = raw, np.ones_like(raw)
        else:
            pt, sc = point, scale
        cert = {}
        for al in ALPHAS:
            cert[f"sim_a{int(round(al * 100)):02d}"] = dict(lower=base.cross_fit_lower(pt, sc, truth, folds, al)[0])
        cert["mondrian_a10"] = dict(lower=base.cross_fit_lower_groups(pt, sc, truth, folds, 0.1, shapes)[0])
        if meth in RAW_NAME:
            cert["raw"] = dict(lower=raw)
        if meth not in ("capxfer", "oracle") and np.isfinite(pt).any():
            for qq in ALPHAS:
                cert[f"cselect_q{int(round(qq * 100)):02d}"] = dict(go={tau: base.conformal_selection(
                    pt, sc, truth, folds, tau, qq) for tau in base.TAUS})
        wts = {"none": None, "faults2x": base.fault_shift_weights(nf, 2.0)}
        for cname, c in cert.items():
            for shp in ("all",) + tuple(np.unique(shapes[ev])):
                m = ev & ((shapes == shp) if shp != "all" else True)
                strata = [("all", hf, None), ("decision", hf, 0.1)]
                if shp == "all":
                    strata += [(f"class{k}", (cls == k).astype(float), np.inf) for k in (0, 1, 2)]
                    strata += [(f"size{k}", (csz == k).astype(float), np.inf)
                               for k in np.unique(csz[np.isfinite(csz)]).astype(int)]
                    strata += [("lone_faulted", ((csz == 1) & (hf > 0.5)).astype(float), np.inf)]
                for sname, h, band in strata:
                    for wname, w in wts.items():
                        if wname != "none" and (shp != "all" or sname not in ("all", "decision")):
                            continue
                        for r in base.metrics(c.get("lower"), truth, h, ep_mask=m, band=band, ep_weight=w,
                                              go_by_tau=c.get("go")):
                            rows.append(dict(baseline=meth, certifier=cname, shape=shp, stratum=sname, shift=wname,
                                             eval_split=a.split, calibration=(a.cal_split if has_cal else "crossfit"),
                                             **r))
        for cname, mname, route in (("sim_a10", meth, "simultaneous"), ("raw", RAW_NAME.get(meth), "raw")):
            if cname not in cert or mname is None:
                continue
            lower = cert[cname]["lower"]
            for i, k in enumerate(keys):
                if not ev[i]:
                    continue
                x = g[k].copy()
                x["method"], x["route"], x["lower"] = mname, route, lower[i, :len(x)]
                if meth == "capxfer":
                    x["point"], x["scale"] = np.nan, np.nan
                x["alpha"] = 0.1
                x["contains_fault"] = x["has_fault"]
                extra = [c_ for c_ in ("ep_n_faults", "cand_fault_class") if c_ in x]
                (dec_triv if meth in TRIVIAL else dec).append(x[list(DEC_COLS) + ["contains_fault"] + extra])
        sec = np.array([g[k]["seconds"].iloc[0] for k in keys])
        for s_ in np.unique(shapes):
            m = shapes == s_
            nc = int(g[keys[int(np.flatnonzero(m)[0])]]["cand"].nunique())
            ds = d[d["shape"] == s_]
            timing.append(dict(method=meth, shape=s_, n=int(m.sum()), n_cand=nc,
                               sec_per_decision=float(np.median(sec[m])), sec_mean=float(sec[m].mean()),
                               sec_per_candidate=float(np.median(sec[m]) / nc),
                               peak_vram_gb=float(ds["peak_vram_gb"].max()) if "peak_vram_gb" in ds else np.nan,
                               sims_per_episode=float(ds.groupby("eid")["sims"].first().median())
                               if "sims" in ds else np.nan,
                               **{c_: float(ds.groupby(["split", "eid"])[c_].first().mean()) if c_ in ds else np.nan
                                  for c_ in ("truth_in_support", "truth_retained", "mass_true", "dropped_mass",
                                             "abstain")}))
    met = pd.DataFrame(rows)
    met.to_csv(RES / f"{tag}metrics.csv", index=False)
    tm = pd.DataFrame(timing)
    tm.to_csv(RES / f"{tag}timing.csv", index=False)
    mb = set().union(*[set(x.method) for x in dec]) if dec else set()
    mt = set().union(*[set(x.method) for x in dec_triv]) if dec_triv else set()
    assert mt <= set(TRIVIAL) and not (mb & mt), (mb, mt)   # paper loader: one file per (method, plant, split)
    if dec:
        pd.concat(dec, ignore_index=True).to_csv(RES / "paper" / "dec_baselines.csv.gz", index=False)
    if dec_triv:
        pd.concat(dec_triv, ignore_index=True).to_csv(RES / "paper" / "dec_trivial.csv", index=False)
    lat = tm.assign(n_wheels=[int(re.match(r"(\d+)m", s_).group(1)) * 4 if re.match(r"\d+m", s_) else
                              {"pair": 8, "square": 16, "2x3": 24}.get(s_, np.nan) for s_ in tm["shape"]],
                    hardware=a.hardware, status="exploratory")
    lat[["method", "shape", "n_wheels", "sec_per_decision", "hardware", "status", "n"]].to_csv(
        RES / "paper" / "latency_baselines.csv", index=False)
    cols = ["baseline", "certifier", "stratum", "shift", "tau", "n_dec", "cert_frac", "unsafe_commit_rate", "fdr",
            "ep_any_unsafe"]
    sel = met[(met["shape"] == "all") & met.stratum.isin(["all", "decision"]) & (met.tau == 0.8)]
    print(sel[cols].round(4).to_string(index=False))
    print(tm.round(3).to_string(index=False))


def cmd_smc_sweep(a):
    """Accuracy vs particles of nuisance-SMC on the sweep episodes (methods nuisance_smc_p<N>): per setting the
    s/episode, truth retention (true mode vector among the propagated draws), posterior share on the true mode vector,
    raw 5 % quantile coverage of the truth (all entries, and faulted-or-near-0.8 entries), and the mean |point -
    point_full| against the largest setting. Picks the cheapest setting within 1 pt of the largest on retention and
    coverage. -> results/baselines/smc_particle_sweep.csv"""
    import pandas as pd
    df = load_runs(a.runs)
    df = df[df.method.str.startswith("nuisance_smc_p")]
    if df.empty:
        raise SystemExit("no nuisance_smc_p<N> rows")
    df = df.assign(particles=df.method.str[len("nuisance_smc_p"):].astype(int))
    full = df.particles.max()
    ref = df[df.particles == full].set_index(["split", "eid", "cand", "leg"])["point"]
    rows = []
    for n_p, d in df.groupby("particles"):
        ep = d.groupby(["split", "eid"]).first()
        rel = (d.has_fault > 0.5) | ((d.truth - 0.8).abs() < 0.1)
        fin = np.isfinite(d.lower) & np.isfinite(d.truth)
        cov = (d.lower <= d.truth)[fin]
        j = d.set_index(["split", "eid", "cand", "leg"])["point"]
        common = j.index.intersection(ref.index)
        rows.append(dict(particles=int(n_p), n_ep=len(ep), s_per_episode_median=float(ep.seconds.median()),
                         s_per_episode_mean=float(ep.seconds.mean()), sims_median=float(ep.sims.median()),
                         truth_retained=float(ep.truth_retained.mean()), mass_true=float(ep.mass_true.mean()),
                         truth_in_support=float(ep.truth_in_support.mean()),
                         q05_coverage=float(cov.mean()), q05_coverage_relevant=float((d.lower <= d.truth)[fin & rel].mean()),
                         mean_abs_point_vs_full=float((j[common] - ref[common]).abs().mean()) if len(common) else np.nan,
                         peak_vram_gb=float(d.peak_vram_gb.max())))
    t = pd.DataFrame(rows).sort_values("particles")
    f = t[t.particles == full].iloc[0]
    ok = t[(t.truth_retained >= f.truth_retained - 0.01) & (t.q05_coverage >= f.q05_coverage - 0.01)
           & (t.q05_coverage_relevant >= f.q05_coverage_relevant - 0.01)]
    t["chosen"] = t.particles == ok.sort_values("s_per_episode_mean").particles.iloc[0]
    RES.mkdir(parents=True, exist_ok=True)
    t.to_csv(RES / "smc_particle_sweep.csv", index=False)
    print(t.round(4).to_string(index=False))


CONF_NAME = dict(detector_gated="detector_gated", physics_gains="ghat_physics", copy_parent="copy_parent",
                 weakest_member="min_over_members", model_free="metadata_nf", capxfer="capxfer", oracle="oracle",
                 nuisance_smc="nuisance_smc", enum_known="credible_lower")
CONF_CLAIM = {"conf_test": "C-MAIN", "conf_H4": "C-SCALE", "conf_H4b": "C-SCALE", "conf_het_test": "C-HET",
              "conf_H1": "C-HET", "conf_H2": "C-HET"}
CONF_METHODS = {"C-MAIN": ("always_go", "never_go", "metadata_nf", "detector_gated", "ghat_physics", "copy_parent",
                           "min_over_members", "capxfer", "oracle"),
                "C-SCALE": ("capxfer", "metadata_nf"), "C-HET": ("metadata_nf",)}
SCORING_OPENS = "2026-11-26T01:45:00+00:00"
BIG = 1e6


def cmd_conf_export(a):
    """Confirmatory decision rows (docs/board/paper.md 5a; docs/CONFIRMATORY.md) ->
    results/baselines/confirmatory/dec_conf_baselines.csv.gz.

    Inputs: runner `smc` unit files under --runs: test rows of the conf_* splits (claim by split: conf_test C-MAIN,
    indices 40-3,039 only; conf_H4 / conf_H4b C-SCALE; conf_het_test / conf_H1 / conf_H2 C-HET) and the exploratory
    `cal` split rows (generator cal 0-1999, b2-cheap-cal), which are the only calibration data (section 0.2).
    Per method: test `lower` = the method's own certified bound = SimultaneousLower at alpha 0.1 fitted on the
    method's cal rows (capxfer: on its raw Theorem-5 bound with scale 1; credible_lower = enum_known's Bayes q05,
    uncalibrated, as pre-registered; metadata_nf from cal (candidate size, leg) quantiles; always_go lower = +BIG,
    never_go -inf; cfBH = conformal selection at FDR 0.1 on enum_known's (point, scale) with the cal split's nulls:
    lower = +BIG if selected, else -inf). Cal rows (split cal, status exploratory, same claim, T1) carry point, scale
    and the 2-fold cross-fit lower, for the matched operating point. Truth = rho_dec with the fixed T1 reference
    (generator labels). Optional --t2-truth (csv with split, eid, cand, leg, truth) adds test_plant T2 rows with the
    same T1-calibrated bounds. Refuses to read confirmatory labels before 01:45Z unless --mock."""
    import datetime as dt
    import pandas as pd
    if not a.mock and dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(SCORING_OPENS):
        raise SystemExit(f"confirmatory labels stay closed until {SCORING_OPENS} (docs/CONFIRMATORY.md 0.1)")
    df = load_runs(a.runs, a.campaign or None)
    conf = df[df.split.isin(list(CONF_CLAIM))].copy()
    lo_t, hi_t = [int(x) for x in a.main_range.split("-")]
    conf = conf[~((conf.split == "conf_test") & ((conf.eid < lo_t) | (conf.eid > hi_t)))]
    cal = df[df.split == "cal"].copy()
    lo_c, hi_c = [int(x) for x in a.cal_range.split("-")]
    cal = cal[(cal.eid >= lo_c) & (cal.eid <= hi_c)]
    print("loaded", len(df), "rows;", conf.groupby("split").eid.nunique().to_dict(), "cal eps", cal.eid.nunique(),
          flush=True)
    out = []
    t2 = pd.read_csv(a.t2_truth) if a.t2_truth else None

    def arrays(d):
        d = d.sort_values(["split", "eid", "cand", "leg"])
        g = [x for _, x in d.groupby(["split", "eid"], sort=True)]
        W = max(len(x) for x in g)
        P_ = lambda col: base.pad([x[col].to_numpy(float) for x in g], W)
        return g, W, P_

    def emit(g, lower, point, scale, method, claim, split_status):
        for i, x in enumerate(g):
            y = x.copy()
            k = len(y)
            y["method"], y["claim"], y["lower"] = method, claim, lower[i, :k]
            y["point"], y["scale"] = point[i, :k], scale[i, :k]
            y["route"] = "raw" if method in ("credible_lower",) else (
                "none" if method in ("always_go", "never_go", "oracle") else "simultaneous")
            if split_status == "cal":
                y["status"] = "exploratory"
            y["contains_fault"] = y["has_fault"]
            out.append(y)

    for claim in ("C-MAIN", "C-SCALE", "C-HET"):
        splits = [sp for sp, c in CONF_CLAIM.items() if c == claim]
        te_all = conf[conf.split.isin(splits)]
        if te_all.empty:
            continue
        base_m = "nominal" if (te_all.method == "nominal").any() else te_all.method.iloc[0]
        for method in CONF_METHODS[claim]:
            src = {v: k for k, v in CONF_NAME.items()}.get(method)
            if method in ("always_go", "never_go", "metadata_nf"):
                src = base_m
            if method == "cfBH":
                src = "enum_known"
            te, ca = te_all[te_all.method == src], cal[cal.method == src]
            if te.empty or (ca.empty and method not in ("always_go", "never_go", "oracle")):
                print(f"{claim} {method}: skipped (test rows {len(te)}, cal rows {len(ca)})")
                continue
            gt, Wt, Pt = arrays(te)
            gc, Wc, Pc = arrays(ca) if not ca.empty else ([], 0, None)
            W = max(Wt, Wc)
            padto = lambda A, w: np.pad(A, ((0, 0), (0, W - A.shape[1])), constant_values=np.nan)
            tr_t, tr_c = padto(Pt("truth"), Wt), (padto(Pc("truth"), Wc) if gc else None)
            if method == "metadata_nf":
                st = ca.groupby(["cand_size", "leg"]).truth.agg(med="median", q05=lambda v: v.quantile(0.05))
                def mf(g_):
                    pts, scs = [], []
                    for x in g_:
                        j = st.reindex(list(zip(x.cand_size, x.leg)))
                        pts.append(j["med"].to_numpy(float))
                        scs.append((j["med"] - j["q05"]).clip(lower=0).to_numpy(float) + 0.01)
                    return padto(base.pad(pts, W), W), padto(base.pad(scs, W), W)
                pt_t, sc_t = mf(gt)
                pt_c, sc_c = mf(gc)
            elif method == "capxfer":
                pt_t, pt_c = padto(Pt("lower"), Wt), padto(Pc("lower"), Wc)
                sc_t, sc_c = np.ones_like(pt_t), np.ones_like(pt_c)
            elif method in ("always_go", "never_go"):
                v = BIG if method == "always_go" else -np.inf
                pt_t = np.where(np.isnan(tr_t), np.nan, v)
                emit(gt, pt_t, np.full_like(pt_t, np.nan), np.full_like(pt_t, np.nan), method, claim, "test")
                if gc:
                    pc = np.where(np.isnan(tr_c), np.nan, v)
                    emit(gc, pc, np.full_like(pc, np.nan), np.full_like(pc, np.nan), method, claim, "cal")
                continue
            else:
                pt_t, pt_c = padto(Pt("point"), Wt), padto(Pc("point"), Wc)
                sc_t, sc_c = padto(Pt("scale"), Wt), padto(Pc("scale"), Wc)
            # calibration on cal (fold 0), applied to test (fold 1); cal rows get a 2-fold cross-fit lower
            Pall = np.concatenate([pt_c, pt_t]); Sall = np.concatenate([sc_c, sc_t])
            Tall = np.concatenate([tr_c, tr_t])
            folds = np.r_[np.zeros(len(gc), int), np.ones(len(gt), int)]
            if method == "cfBH":
                # selection is per tau; one lower per row encodes it as the largest tau at which the entry was selected
                # (go at tau' <= that), -inf if never selected (scorer contract: finite = selected)
                lows = []
                for tau in base.TAUS:
                    gsel = base.conformal_selection(Pall, Sall, Tall, folds, tau, 0.1)[len(gc):]
                    lows.append(np.where(gsel, tau, -np.inf))
                lo_t = np.max(lows, 0)
                emit(gt, lo_t, np.full_like(lo_t, np.nan), np.full_like(lo_t, np.nan), method, claim, "test")
                continue
            if method == "oracle":
                lo_t, lo_c = tr_t, tr_c
            elif method == "credible_lower":
                lo_t, lo_c = padto(Pt("lower"), Wt), padto(Pc("lower"), Wc)
            else:
                lo_t = base.cross_fit_lower(Pall, Sall, Tall, folds, 0.1)[0][len(gc):]
                cf = np.zeros(len(gc), int)
                cf[1::2] = 1
                lo_c = base.cross_fit_lower(pt_c, sc_c, tr_c, cf, 0.1)[0] if gc else None
            emit(gt, lo_t, pt_t, sc_t, method, claim, "test")
            if gc:
                emit(gc, lo_c, pt_c, sc_c, method, claim, "cal")
    res = pd.concat(out, ignore_index=True)
    res["test_plant"] = np.where(res.split == "cal", "T1", res.test_plant)
    if t2 is not None:
        key = ["split", "eid", "cand", "leg"]
        t2r = res[res.split != "cal"].drop(columns=["truth"]).merge(t2[key + ["truth"]], on=key, how="inner")
        t2r["test_plant"] = "T2"
        res = pd.concat([res, t2r], ignore_index=True)
    cols = list(DEC_COLS) + ["claim", "shard", "contains_fault", "parent_n", "ep_n_faults", "cand_fault_class"]
    res = res[[c for c in cols if c in res.columns]]
    od = RES / "confirmatory"
    od.mkdir(parents=True, exist_ok=True)
    if os.environ.get("CFCAP_BL_WITH_STATE"):
        od = RES / "confirmatory_with_state"            # NOT read by the scorer unless the coordinator moves it
        od.mkdir(parents=True, exist_ok=True)
    f = od / ("dec_conf_baselines_MOCK.csv.gz" if a.mock else "dec_conf_baselines.csv.gz")
    res.to_csv(f, index=False, float_format="%.6g")
    print(f, len(res), res.groupby(["claim", "method", "split"]).size().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "run", "eval", "ingest", "export", "from-runs", "smc-sweep", "conf-export"])
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--runs", default="~/.cache/vast-out")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--hardware", default="RTX 4070 Ti SUPER (shared)")
    ap.add_argument("--cal-split", default="", help="from-runs: fit every conformal layer on this split")
    ap.add_argument("--tag", default="pilot_", help="from-runs: results/baselines/<tag>metrics.csv")
    ap.add_argument("--mock", action="store_true", help="conf-export on mock data (writes *_MOCK; no time gate)")
    ap.add_argument("--campaign", default="", help="conf-export: only this campaign's job dirs (complete units)")
    ap.add_argument("--main-range", default="40-3039", help="conf-export: conf_test eid range (C-MAIN)")
    ap.add_argument("--cal-range", default="0-1999", help="conf-export: exploratory cal eid range")
    ap.add_argument("--t2-truth", default="", help="conf-export: csv (split, eid, cand, leg, truth) of T2 labels")
    ap.add_argument("--per-shape", type=int, default=100)
    ap.add_argument("--baseline")
    ap.add_argument("--max-min", type=float, default=14.0)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run" and a.per_shape == 100:
        a.per_shape = 0
    dict(gen=cmd_gen, run=cmd_run, eval=cmd_eval, ingest=cmd_ingest, export=cmd_export, **{"from-runs": cmd_from_runs, "smc-sweep": cmd_smc_sweep, "conf-export": cmd_conf_export})[a.cmd](a)


if __name__ == "__main__":
    main()
