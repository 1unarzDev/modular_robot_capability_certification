"""Nuisance pilot driver and analysis (exploratory). Primary label rho_dec (truth/q05/q50/q95), secondary rho_rel (*_rel). Core: cfcap/nuisance_post.py; runner task: cfcap/nuisance_task.py.

Pre-registration and thresholds: docs/board/nuisance.md.

    scripts/heavy.sh --gpu $PY scripts/nuisance_pilot.py run --shape pair --episodes 8 --tag p1
    $PY scripts/nuisance_pilot.py collect DIR...      # runner outputs (units/*.npz + .json) -> results/nuisance CSVs
    $PY scripts/nuisance_pilot.py analyze
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results/nuisance"
TAUS = (0.7, 0.8)


def run(a):
    from cfcap import nuisance_post as NP
    RES.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    dec_all, ep_all = [], []
    for ps in a.probes.split(","):
        for b0 in range(0, a.episodes, a.batch):
            if (time.time() - t_start) / 60 > a.max_minutes:
                print("time budget reached", flush=True)
                break
            eps = list(range(b0, min(a.episodes, b0 + a.batch)))
            dec, ep = NP.run_batch(a.shape, ps, eps, seed=a.seed, particles=a.particles, draws=a.draws,
                                   max_stages=a.max_stages, engine=a.engine, chunk=a.chunk)
            d = pd.DataFrame(dec); d.insert(0, "tag", a.tag); dec_all.append(d)
            e = pd.DataFrame(ep); e.insert(0, "tag", a.tag); ep_all.append(e)
            print(e[["probe", "episode", "faults", "stages", "smc_s_per_ep", "label_s_per_ep"]].assign(
                dll=e.ll_true - e.ll_best, mass=e.post_mass_true_modes).to_string(index=False), flush=True)
            pd.concat(dec_all).to_csv(RES / f"decisions_{a.tag}_{a.shape}.csv", index=False)
            pd.concat(ep_all).to_csv(RES / f"episodes_{a.tag}_{a.shape}.csv", index=False)
    print("wall min", (time.time() - t_start) / 60)


def collect(a):
    """Runner outputs -> results CSVs (decisions from the unit npz, episodes from the sidecars' meta)."""
    RES.mkdir(parents=True, exist_ok=True)
    dec, eps = [], []
    for d in a.dirs:
        for f in sorted(glob.glob(f"{d}/**/units/*.npz", recursive=True)):
            z = np.load(f)
            df = pd.DataFrame({k: z[k] for k in z.files}); df.insert(0, "tag", a.tag); dec.append(df)
            meta = json.loads(pathlib.Path(f[:-4] + ".json").read_text()).get("meta", {})
            for r in meta.get("episodes", []):
                eps.append(dict(r, tag=a.tag))
    pd.concat(dec).to_csv(RES / f"decisions_{a.tag}_all.csv", index=False)
    pd.DataFrame(eps).to_csv(RES / f"episodes_{a.tag}_all.csv", index=False)
    print(len(dec), "units", len(eps), "episodes")


# ----------------------------------------------------------------------------- analysis
def boot_share(df, B=2000, seed=0):
    """Share of ambiguous decisions with an episode bootstrap (episodes keyed by tag+shape+episode)."""
    g = df.groupby("ekey").agg(a=("amb", "sum"), n=("amb", "size"))
    a, n = g.a.to_numpy(), g.n.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    bs = a[idx].sum(1) / n[idx].sum(1)
    return a.sum() / n.sum(), np.quantile(bs, 0.025), np.quantile(bs, 0.975), int(a.sum()), int(n.sum()), len(g)


def amb_het(dec, status="exploratory"):
    """paper contract 1c: results/nuisance/paper/amb_het.csv (probe v2, tau pooled, per delta and subset)."""
    d = dec[dec.probe == "v2"]
    long = [d.assign(tau=t, amb=((d.q05 < t) & (t <= d.q95)).astype(int)) for t in TAUS]
    L = pd.concat(long, ignore_index=True)
    rows = []
    for (dl,), g in L.groupby(["delta"]):
        for sub, gg in (("all", g), ("faulted_cand", g[g.has_fault == 1])):
            if len(gg) == 0:
                continue
            s, lo, hi, na, nd, ne = boot_share(gg)
            rows.append(dict(source="T1 SMC posterior", delta=dl, subset=sub, n_dec=nd, amb_share=s, lo=lo, hi=hi,
                             status=status, plant="T1", n_ep=ne))
    (RES / "paper").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RES / "paper" / "amb_het.csv", index=False)


def analyze(a):
    dec = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("decisions_*.csv"))], ignore_index=True)
    eps = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("episodes_*.csv"))], ignore_index=True)
    for df in (dec, eps):
        if "delta" not in df:
            df["delta"] = 0.08
        df["delta"] = df["delta"].round(3)
        df["ekey"] = df.tag + "/" + df["shape"] + "/" + df.delta.astype(str) + "/" + df.episode.astype(str)
    if a.tags:
        dec, eps = dec[dec.tag.isin(a.tags.split(","))], eps[eps.tag.isin(a.tags.split(","))]
    amb_het(dec, a.status)
    dec, eps = dec[dec.delta == 0.08], eps[eps.delta == 0.08]
    rows = []
    long = []
    for tau in TAUS:
        d = dec.copy()
        d["tau"] = tau
        d["amb"] = ((d.q05 < tau) & (tau <= d.q95)).astype(int)
        d["cover"] = ((d.truth >= d.q05) & (d.truth <= d.q95)).astype(int)
        if "q05_rel" in d:                                   # secondary label (capxfer comparison only)
            d["amb_rel"] = ((d.q05_rel < tau) & (tau <= d.q95_rel)).astype(int)
        long.append(d)
    L = pd.concat(long, ignore_index=True)
    # restrict probe comparisons to episodes run under every probe set present for that tag/shape
    for keys, name in ((["probe"], "pooled"), (["probe", "shape"], "by_shape"), (["probe", "tau"], "by_tau"),
                       (["probe", "shape", "tau"], "by_shape_tau"), (["probe", "kind"], "by_kind"),
                       (["probe", "has_fault"], "by_has_fault"), (["probe", "kind", "size"], "by_kind_size")):
        for k, g in L.groupby(keys):
            s, lo, hi, na, nd, ne = boot_share(g)
            k = k if isinstance(k, tuple) else (k,)
            rows.append(dict(breakdown=name, **dict(zip(keys, k)), amb_share=s, amb_lo=lo, amb_hi=hi, n_amb=na,
                             n_dec=nd, n_ep=ne, coverage=g.cover.mean(),
                             amb_share_rel=g.amb_rel.mean() if "amb_rel" in g else np.nan))
    summ = pd.DataFrame(rows)
    summ.to_csv(RES / "summary.csv", index=False)
    # probe reductions vs v2 on matched episodes
    red = []
    base = L[L.probe == "v2"]
    for ps in sorted(set(L.probe) - {"v2"}):
        other = L[L.probe == ps]
        common = set(base.ekey) & set(other.ekey)
        b = base[base.ekey.isin(common)].groupby("ekey").amb.sum()
        o = other[other.ekey.isin(common)].groupby("ekey").amb.sum().reindex(b.index)
        rng = np.random.default_rng(1)
        idx = rng.integers(0, len(b), size=(2000, len(b)))
        bb, oo = b.to_numpy()[idx].sum(1), o.to_numpy()[idx].sum(1)
        rr = 1 - oo / np.maximum(bb, 1)
        red.append(dict(probe=ps, n_ep=len(b), amb_v2=int(b.sum()), amb_probe=int(o.sum()),
                        rel_reduction=1 - o.sum() / max(b.sum(), 1), red_lo=np.quantile(rr, 0.025),
                        red_hi=np.quantile(rr, 0.975), testable=int(b.sum() >= 50)))
    pd.DataFrame(red).to_csv(RES / "probe_reduction.csv", index=False)
    ev = eps.assign(found=(eps.ll_true - eps.ll_best <= 10).astype(int)).groupby(["shape", "probe"]).agg(
        n_ep=("episode", "size"), found_share=("found", "mean"), mass_true_med=("post_mass_true_modes", "median"),
        stages=("stages", "mean"), forced=("forced", "sum"), smc_s=("smc_s_per_ep", "mean"),
        label_s=("label_s_per_ep", "mean"), total_s=("total_s_per_ep", "mean"),
        sims=("smc_sims_per_ep", "mean")).reset_index()
    ev.to_csv(RES / "cost_validity.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ[summ.breakdown.isin(["pooled", "by_shape", "by_tau", "by_shape_tau", "by_kind", "by_has_fault"])]
              .to_string(index=False))
        print(pd.DataFrame(red).to_string(index=False))
        print(ev.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze", "collect"])
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--tags", default="", help="analyze only these tags")
    ap.add_argument("--status", default="exploratory", choices=["exploratory", "confirmatory"])
    ap.add_argument("--shape", default="pair")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--particles", type=int, default=256)
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--probes", default="v2,v2+clamp,v2+tug")
    ap.add_argument("--max-stages", type=int, default=150)
    ap.add_argument("--max-minutes", type=float, default=12.0)
    ap.add_argument("--engine", default="graph")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=909)
    ap.add_argument("--tag", default="p")
    a = ap.parse_args()
    dict(run=run, analyze=analyze, collect=collect)[a.cmd](a)


if __name__ == "__main__":
    main()

