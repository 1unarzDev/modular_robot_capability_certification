"""Size-generalisation curve and held-out-split table from stage-2 evaluations (plan §2b/§2c; exploratory).

  $PY scripts/models_scaling.py ~/.cache/vast-out/<models-eval2 dirs> [--tags st-n6-1,bb-n6-1]

Models were trained on parents of <= 6 modules (mecanum only). For every model tag and route (direct head with the
simultaneous layer; compositional route with its simultaneous layer), lower bounds are fitted on T1 cal rows and
applied unchanged to every other split. Rows of results/models/paper/scaling.csv:
  method, route, split, test_n (parent modules; "all" for a split-level row), stratum (all | relevant),
  metric in {cov90, entry_cov, cert_frac, unsafe_commit}, value, lo, hi (episode-bootstrap 95 % percentile CI,
  B = 1000, resampling episodes), n_ep, n_rows, status.
  cov90      truth inside the [q05, q95] interval (uncalibrated quantiles)
  entry_cov  truth >= conformal lower bound (alpha 0.1)
  cert_frac / unsafe_commit  pooled over tau in {0.6, 0.7, 0.8} as in docs/board/paper.md.
Splits: test (N 1-6), H4 (7-10), H4b (11-16), H3, H1/H2 (module types), the confirmatory split; whichever exist.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import models_export as X  # noqa: E402
from cfcap.models import train as TR  # noqa: E402

TAUS = (0.6, 0.7, 0.8)


def per_episode_stats(df, lo, qlo, qhi):
    """Per-episode sufficient statistics so the bootstrap resamples whole episodes."""
    y = df.truth.to_numpy()
    lo = np.nan_to_num(lo, nan=-np.inf)
    d = pd.DataFrame(dict(eid=df.eid.to_numpy(), cov=(y >= qlo) & (y <= qhi), ecov=lo <= y))
    for t in TAUS:
        go, cap = lo >= t, y >= t
        d[f"gc{t}"], d[f"c{t}"] = go & cap, cap
        d[f"gi{t}"], d[f"i{t}"] = go & ~cap, ~cap
    return d.groupby("eid").sum()


def metrics(s):
    n = s["cov"].count()
    rows = s.sum()
    tot = max(1.0, float(sum(rows[f"c{t}"] + rows[f"i{t}"] for t in TAUS)) / len(TAUS))
    cf = sum(rows[f"gc{t}"] for t in TAUS) / max(1, sum(rows[f"c{t}"] for t in TAUS))
    uc = sum(rows[f"gi{t}"] for t in TAUS) / max(1, sum(rows[f"i{t}"] for t in TAUS))
    return dict(cov90=rows["cov"] / tot, entry_cov=rows["ecov"] / tot, cert_frac=cf, unsafe_commit=uc), n


def boot(s, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in ("cov90", "entry_cov", "cert_frac", "unsafe_commit")}
    arr = s.to_numpy()
    for _ in range(B):
        bs = pd.DataFrame(arr[rng.integers(len(arr), size=len(arr))], columns=s.columns)
        m, _ = metrics(bs)
        for k, v in m.items():
            vals[k].append(v)
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--tags", default="st-n6-1,bb-n6-1")
    ap.add_argument("--status", default="exploratory")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "results/models/paper/scaling.csv"))
    a = ap.parse_args()
    e2 = X.load_eval2(a.roots)
    # confirmatory splits stay closed here (they are scored only by scripts/analysis/confirmatory.py after 01:45Z)
    e2 = {k: {s_: v for s_, v in d.items() if not s_.startswith("conf_")} for k, d in e2.items()}
    rows = []
    for tag in a.tags.split(","):
        by = e2.get(tag)
        if not by or "cal" not in by:
            print(f"skip {tag}: no cal rows")
            continue
        cal = by["cal"]
        others = [(s, df) for s, df in by.items() if s != "cal"]
        routes = [("direct_sim", X.direct, ("q05", "q95"))]
        if np.isfinite(cal.cq50).any():
            routes.append(("compositional", X.comp, ("cq05", "cq95")))
        for route, fn, (a05, a95) in routes:
            los, _ = X.lowers(cal, [df for _, df in others], fn, "simultaneous")
            for (split, df), lo in zip(others, los):
                ok = np.isfinite(fn(df)[0])
                df, lo = df[ok].reset_index(drop=True), lo[ok]
                rel = TR.relevant(df)
                groups = [("all", slice(None))] + [(int(n), (df.parent_n == n).to_numpy()) for n in
                                                    sorted(df.parent_n.unique())]
                for test_n, sel in groups:
                    for stratum, mask in (("all", np.ones(len(df), bool)), ("relevant", rel)):
                        m_ = mask if isinstance(sel, slice) else (mask & sel)
                        if not m_.any():
                            continue
                        g = df[m_]
                        s = per_episode_stats(g, lo[m_], g[a05].to_numpy(), g[a95].to_numpy())
                        val, n_ep = metrics(s)
                        ci = boot(s, a.B)
                        for k, v in val.items():
                            rows.append(dict(method=tag, route=route, split=split, test_n=test_n, stratum=stratum,
                                             metric=k, value=float(v), lo=ci[k][0], hi=ci[k][1], n_ep=int(n_ep),
                                             n_rows=int(m_.sum()), status=a.status))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
