"""Export pulled `train` runs to the paper contracts (docs/board/paper.md §1a, 1d, 1e):

  results/models/paper/dec_models.csv      amortized_{marg,sim,comp}, blackbox_gnn, ensemble (headline diversity)
  results/models/paper/diversity.csv       every grid run x test morphology x metric
  results/models/paper/latency_models.csv  s/decision (episode) per method and shape

  $PY scripts/models_export.py RUN_ROOT [RUN_ROOT ...] [--headline n6] [--status exploratory]

A run dir is any directory with manifest.json of task `train` (its args give grid + shard -> arch, div, seed).
Label: rho_dec (truth column of the eval units). Lower bounds at ALPHA = 0.10, calibrated on the run's own eval-cal
rows (T1 cal), applied unchanged to test / H3 / H4. Routes:
  marginal      per-entry split conformal on (point - truth) / scale
  simultaneous  cfcap.conformal.core.SimultaneousLower (max-normalised over every candidate x leg of an episode)
  compositional simultaneous layer on the compositional quantiles (cal/test episodes that carry them)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cfcap.conformal import core  # noqa: E402
from cfcap.models import train as TR  # noqa: E402

ALPHA = 0.10
TAUS = (0.6, 0.7, 0.8)
KIND = {0: "self", 1: "child", 2: "merge"}
TOPO = ("single", "chain", "L", "path", "cyclic", "grid", "T", "star", "branched")
NAMED = {("single", 1): "single", ("chain", 2): "pair", ("L", 3): "L3", ("grid", 4): "square", ("grid", 6): "2x3"}


def shape_name(topo, n):
    t = TOPO[int(topo)]
    return NAMED.get((t, int(n)), f"{t}{int(n)}")


def find_runs(roots):
    runs = []
    for r in roots:
        for m in pathlib.Path(r).rglob("manifest.json"):
            try:
                man = json.loads(m.read_text())
            except ValueError:
                continue
            if man.get("task") != "train":
                continue
            argv = list(man.get("task_argv") or [])
            grid_name = argv[argv.index("--grid") + 1] if "--grid" in argv else "b1"
            shard = int(man.get("shard", 0))
            cfg = TR.grid(grid_name)[shard]
            runs.append(dict(dir=m.parent, cfg=cfg, grid=grid_name, host=man.get("host", "")))
    return runs


def load_unit(d, split):
    f = d / "units" / f"eval-{split}.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        df = pd.DataFrame({k: z[k] for k in z.files})
    meta = json.loads((d / "units" / f"eval-{split}.json").read_text()).get("meta", {})
    return df, meta


def to_matrix(df, point, scale):
    """Episode x (candidate, leg) matrices (NaN-padded) of point, scale, truth; rows keep df order via (eid, col)."""
    e = df.eid.to_numpy()
    ue, ei = np.unique(e, return_inverse=True)
    col = (df.cand.to_numpy() * 10 + df.leg.to_numpy())
    W = int(col.max()) + 1
    P_, S_, Y_ = (np.full((len(ue), W), np.nan) for _ in range(3))
    P_[ei, col], S_[ei, col], Y_[ei, col] = point, scale, df.truth.to_numpy()
    return P_, S_, Y_, ei, col


def lowers(cal, tests, point_fn, route):
    """Fit on cal (DataFrame) with point/scale from point_fn(df) -> (point, scale); return lower per test df."""
    pc, sc = point_fn(cal)
    ok = np.isfinite(pc) & np.isfinite(sc)
    out = []
    if route == "marginal":
        q = core.split_quantile((pc[ok] - cal.truth.to_numpy()[ok]) / np.maximum(sc[ok], 1e-6), ALPHA)
        for t in tests:
            p, s = point_fn(t)
            out.append(np.where(np.isfinite(p), p - q * np.maximum(s, 1e-6), np.nan))
        return out, q
    c = cal[ok]
    Pm, Sm, Ym, _, _ = to_matrix(c, pc[ok], sc[ok])
    sl = core.SimultaneousLower(ALPHA).fit(Pm, Sm, Ym)
    for t in tests:
        p, s = point_fn(t)
        out.append(np.where(np.isfinite(p), sl.lower(p, s), np.nan))
    return out, sl.q


direct = lambda df: (df.q50.to_numpy(float), np.maximum(df.q50 - df.q05, 1e-3).to_numpy(float))
comp = lambda df: (df.cq50.to_numpy(float), np.maximum(df.cq50 - df.cq05, 1e-3).to_numpy(float))


def dec_rows(method, route, posterior, splits, lower, point_fn, seconds_fn, status):
    out = []
    for (split, df), lo in zip(splits, lower):
        p, s = point_fn(df)
        keep = np.isfinite(p)
        d = df[keep]
        out.append(pd.DataFrame(dict(
            method=method, route=route, posterior=posterior, train="T1", test_plant="T1", split=split, status=status,
            eid=d.eid.to_numpy(), shape=[shape_name(t, n) for t, n in zip(d.parent_topo, d.parent_n)],
            family=d.cand_family.to_numpy(), cand=d.cand.to_numpy(), kind=d.kind.map(KIND).to_numpy(),
            cand_size=d.cand_size.to_numpy(), has_fault=d.has_fault.to_numpy(), leg=d.leg.to_numpy(),
            truth=d.truth.to_numpy(), point=p[keep], scale=s[keep], lower=np.where(np.isfinite(lo[keep]), lo[keep],
                                                                                    -np.inf),
            alpha=ALPHA, seconds=seconds_fn(split, d))))
    return out


def decision_metrics(lower, truth):
    lo = np.nan_to_num(lower, nan=-np.inf)
    cf, uc = [], []
    for tau in TAUS:
        go, capable = lo >= tau, truth >= tau
        cf.append(((go & capable).sum(), capable.sum()))
        uc.append(((go & ~capable).sum(), (~capable).sum()))
    r = lambda x: sum(a for a, _ in x) / max(1, sum(b for _, b in x))
    return r(cf), r(uc)


def pinball(q, y, taus=(0.05, 0.25, 0.5, 0.75, 0.95)):
    d = y[:, None] - q
    t = np.asarray(taus)
    return float(np.mean(np.maximum(t * d, (t - 1) * d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--headline", default="n6")
    ap.add_argument("--status", default="exploratory")
    ap.add_argument("--hardware", default="RTX 3090 (vast)")
    ap.add_argument("--out", default=str(ROOT / "results/models/paper"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(a.roots)
    print(f"{len(runs)} train runs")
    dec, div, lat = [], [], []
    ens = {}
    for run in runs:
        cfg, d = run["cfg"], run["dir"]
        units = {s: load_unit(d, s) for s in TR.EVAL_SPLITS}
        if units["cal"] is None or units["test"] is None:
            continue
        cal = units["cal"][0]
        evs = [(s, units[s][0]) for s in ("cal", "test", "H3", "H4") if units[s] is not None]
        bb = TR.ARCHS[cfg["arch"]].get("blackbox")
        sim_lo, _ = lowers(cal, [df for _, df in evs], direct, "simultaneous")
        inf_s = {s: units[s][1].get("infer_ms_per_episode", np.nan) / 1e3 for s, _ in evs}
        sec_direct = lambda split, df: np.full(len(df), inf_s[split])
        sec_comp = lambda split, df: df.comp_seconds.to_numpy()
        # diversity rows (every run)
        d_info = TR.DIVERSITY[cfg["div"]]
        for (split, df), lo in zip(evs, sim_lo):
            if split == "cal":
                continue
            for (topo, n), g in df.groupby(["parent_topo", "parent_n"]):
                idx = g.index.to_numpy()
                cf, uc = decision_metrics(lo[idx], g.truth.to_numpy())
                q = g[list(TR.QCOLS)].to_numpy(float)
                y = g.truth.to_numpy()
                held = int(split in ("H3", "H4") or n > d_info["max_n"] or (
                    d_info.get("topologies") is not None and TOPO[int(topo)] not in d_info["topologies"]))
                base = dict(arch=cfg["arch"], diversity=cfg["div"], n_train_families=TR.n_train_families(cfg["div"]),
                            max_train_n=d_info["max_n"], seed=cfg["seed"], test_morph=shape_name(topo, n),
                            test_n=int(n), heldout=held, status=a.status, split=split)
                for metric, v in (("cert_frac", cf), ("unsafe_commit", uc), ("pinball", pinball(q, y)),
                                  ("cov90", float(((y >= q[:, 0]) & (y <= q[:, -1])).mean()))):
                    div.append(dict(base, metric=metric, value=v))
            wf = d / f"wheels_{split}.npz"
            if wf.exists() and not bb:
                with np.load(wf) as z:
                    side = {k: z[k] for k in z.files}
                for n in np.unique(side["parent_n"]):
                    sub = {k: v[side["parent_n"] == n] for k, v in side.items()}
                    det = TR.detection(sub)
                    if "det_all" in det:
                        div.append(dict(arch=cfg["arch"], diversity=cfg["div"],
                                        n_train_families=TR.n_train_families(cfg["div"]), max_train_n=d_info["max_n"],
                                        seed=cfg["seed"], test_morph=f"n{int(n)}", test_n=int(n),
                                        heldout=int(split in ("H3", "H4") or n > d_info["max_n"]), status=a.status,
                                        split=split, metric="det_1pct", value=det["det_all"]))
        # decision rows: headline diversity, seed 1
        if cfg["div"] == a.headline and cfg["seed"] == 1 and cfg["arch"] in ("st", "bb"):
            if bb:
                dec += dec_rows("blackbox_gnn", "simultaneous", "none", evs, sim_lo, direct, sec_direct, a.status)
            else:
                mar_lo, _ = lowers(cal, [df for _, df in evs], direct, "marginal")
                dec += dec_rows("amortized_marg", "marginal", "amortized", evs, mar_lo, direct, sec_direct, a.status)
                dec += dec_rows("amortized_sim", "simultaneous", "amortized", evs, sim_lo, direct, sec_direct,
                                a.status)
                cevs = [(s, df) for s, df in evs if s in ("cal", "test")]
                if np.isfinite(cal.cq50).any():
                    c_lo, _ = lowers(cal, [df for _, df in cevs], comp, "compositional")
                    dec += dec_rows("amortized_comp", "compositional", "amortized", cevs, c_lo, comp, sec_comp,
                                    a.status)
                    for s, df in cevs:
                        ok = np.isfinite(df.cq50)
                        for (topo, n), g in df[ok].groupby(["parent_topo", "parent_n"]):
                            v = g.comp_seconds.dropna()
                            lat.append(dict(method="amortized_comp", shape=shape_name(topo, n), n_wheels=int(4 * n),
                                            sec_per_decision=float(v.median()), hardware=a.hardware,
                                            status=a.status, n=int(g.eid.nunique()), split=s))
                for s, df in evs:
                    lat.append(dict(method="amortized_sim", shape="all", n_wheels=-1, sec_per_decision=inf_s[s],
                                    hardware=a.hardware, status=a.status, n=int(df.eid.nunique()), split=s))
        if cfg["div"] == a.headline and cfg["arch"] == "st":
            ens[cfg["seed"]] = evs
    # ensemble over st seeds of the headline diversity (point = mean median, scale = mean half-width + seed spread)
    if len(ens) >= 2:
        seeds = sorted(ens)
        base = ens[seeds[0]]
        for k in range(len(base)):
            dfs = [ens[s][k][1] for s in seeds]
            q50 = np.stack([d_.q50.to_numpy(float) for d_ in dfs])
            hw = np.stack([(d_.q50 - d_.q05).to_numpy(float) for d_ in dfs])
            df = base[k][1].copy()
            df["q50"], df["q05"] = q50.mean(0), q50.mean(0) - (hw.mean(0) + q50.std(0))
            base[k] = (base[k][0], df)
        e_lo, _ = lowers(base[0][1], [df for _, df in base], direct, "simultaneous")
        dec += dec_rows("ensemble", "simultaneous", "amortized", base, e_lo, direct,
                        lambda s, df: np.full(len(df), np.nan), a.status)
    if dec:
        pd.concat(dec).to_csv(out / "dec_models.csv.gz", index=False)
    pd.DataFrame(div).to_csv(out / "diversity.csv", index=False)
    loc = ROOT / "results/models/latency.csv"
    if loc.exists():                                  # local pilot timings (batch 1, own pilot data)
        for r in pd.read_csv(loc).itertuples():
            n = {"pair": 2, "square": 4, "2x3": 6}.get(r.shape, 0)
            lat.append(dict(method=f"pilot_{r.method}", shape=r.shape, n_wheels=4 * n,
                            sec_per_decision=r.ms_per_episode / 1e3, hardware="RTX 4070 Ti SUPER (local pilot)",
                            status="exploratory", n=0))
    pd.DataFrame(lat).to_csv(out / "latency_models.csv", index=False)
    print(f"dec rows {sum(len(x) for x in dec)}  diversity rows {len(div)}  latency rows {len(lat)}  -> {out}")


if __name__ == "__main__":
    main()
