"""H5 with uncertainty (critic A): does composition degrade less than the direct head on bodies larger than training?

  $PY scripts/models_h5.py ~/.cache/vast-out [--B 2000]

Data: the Stage-B train runs' own eval-test units (generator test episodes 0-299, label rho_dec; the compositional
route was run on the first 100 episodes, so the comparison is restricted to those composed episodes, paired). Models:
every available set-transformer seed of diversity n4 (trained on parents <= 4 modules: N 5-6 is unseen) and n6
(trained <= 6; the 2x3 parents are in-family). Per run and pooled over the seeds of a diversity level:
  groups   unseen (parent N > max_train_n), 2x3 parents (grid, 6 modules), all composed
  strata   all candidates | faulted candidates (headline) | healthy candidates
  metrics  cov90_direct, cov90_comp, d_cov90 = comp - direct, pinball_direct, pinball_comp, d_pinball
  CI       paired episode bootstrap (B resamples of episodes, both routes on the same resample), percentile 95 %
Writes results/models/paper/h5_ci.csv (exploratory). The pilot H5 numbers (own data, rho_rel, one seed, n = 600
children, no CI; results/models/cf.csv) are superseded by these.
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

TAUS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
GRID = X.TOPO.index("grid")


def per_episode(df):
    q = df[list(TR.QCOLS)].to_numpy(float)
    c = df[["c" + k for k in TR.QCOLS]].to_numpy(float)
    y = df.truth.to_numpy(float)
    pin = lambda qq: np.maximum(TAUS * (y[:, None] - qq), (TAUS - 1) * (y[:, None] - qq)).mean(1)
    e = pd.DataFrame(dict(eid=df.eid.to_numpy(), cov_d=(y >= q[:, 0]) & (y <= q[:, -1]),
                          cov_c=(y >= c[:, 0]) & (y <= c[:, -1]), pin_d=pin(q), pin_c=pin(c), n=1))
    return e.groupby("eid").sum()


def stats(s):
    t = s.sum()
    n = max(t["n"], 1)
    return dict(cov90_direct=t.cov_d / n, cov90_comp=t.cov_c / n, d_cov90=(t.cov_c - t.cov_d) / n,
                pinball_direct=t.pin_d / n, pinball_comp=t.pin_c / n, d_pinball=(t.pin_c - t.pin_d) / n)


def boot(s, B, seed=0):
    rng = np.random.default_rng(seed)
    a = s.to_numpy()
    cols = s.columns
    vals = {k: [] for k in stats(s)}
    for _ in range(B):
        r = pd.DataFrame(a[rng.integers(len(a), size=len(a))], columns=cols)
        for k, v in stats(r).items():
            vals[k].append(v)
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results/models/paper/h5_ci.csv"))
    a = ap.parse_args()
    data = {}
    for run in X.find_runs(a.roots):
        c = run["cfg"]
        if c["arch"] != "st" or c["div"] not in ("n4", "n6"):
            continue
        u = X.load_unit(run["dir"], "test")
        if u is None:
            continue
        df = u[0]
        df = df[np.isfinite(df.cq50)].reset_index(drop=True)
        if len(df):
            data.setdefault((c["div"], c["seed"]), df.assign(seed=c["seed"], eidk=df.eid + 0))
    rows = []
    for div in ("n4", "n6"):
        max_n = TR.DIVERSITY[div]["max_n"]
        runs = {k[1]: v for k, v in data.items() if k[0] == div}
        if not runs:
            continue
        variants = [(f"seed{s}", {s: d}) for s, d in sorted(runs.items())] + \
                   ([("pooled", runs)] if len(runs) > 1 else [])
        for label, rs in variants:
            df = pd.concat([d.assign(eid=d.eid + 100000 * s) for s, d in rs.items()], ignore_index=True)
            groups = {"unseen": df.parent_n > max_n, "2x3": (df.parent_topo == GRID) & (df.parent_n == 6),
                      "all": np.ones(len(df), bool)}
            for gname, gm in groups.items():
                for stratum, sm in (("all", np.ones(len(df), bool)), ("faulted", df.has_fault == 1),
                                    ("healthy", df.has_fault == 0)):
                    m = np.asarray(gm) & np.asarray(sm)
                    if m.sum() == 0:
                        continue
                    s = per_episode(df[m])
                    v, ci = stats(s), boot(s, a.B)
                    for k, x in v.items():
                        rows.append(dict(diversity=div, max_train_n=max_n, seeds=",".join(map(str, sorted(rs))),
                                         variant=label, group=gname, stratum=stratum, metric=k, value=float(x),
                                         lo=ci[k][0], hi=ci[k][1], n_ep=len(s), n_rows=int(m.sum()),
                                         label="rho_dec", status="exploratory"))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} rows -> {out}; runs {sorted(data)}")


if __name__ == "__main__":
    main()
