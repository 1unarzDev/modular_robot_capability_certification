"""Exploratory calibrated baselines table (critic A, docs/reviews/weakpoints_A.md: H7 / H15 untested exploratorily).

Input: the returned exploratory `smc` runner rows of the generator `cal` split (cfcap-baselines-b2* b2-cheap-cal,
~/.cache/vast-out; confirmatory campaigns are never read). No test split of the cheap baselines has returned, so every
calibration is 2-fold cross-fit over these episodes (folds = eid parity): each fold's bounds are fitted on the other
fold only.
Per method (nominal, oracle, rls_interval, capxfer, enum_known, detector_gated, physics_gains, copy_parent,
weakest_member, model_free = cal (candidate size, leg) quantiles):
  sim_a10   SimultaneousLower at alpha = 0.1 on (point, scale) (capxfer: its Theorem-5 bound, scale 1);
  matched   the paper scorer's matched operating point U* = 1 % on the decision-relevant stratum
            (scripts/analysis/common.py matched_lower, margin fitted on the other fold).
Metrics per (method, certifier, stratum in all / rel / lone_faulted, tau pooled and per tau): certified fraction of
capable decisions, unsafe-commit rate among incapable decisions, share of episodes with an unsafe commit; 95 %
percentile bootstrap over episodes (B = 1000, seed 0).
Outputs: results/baselines/cal_table.csv (metrics with CIs) and results/baselines/paper/dec_baselines_cal.csv.gz
(decision rows, split cal, status exploratory, route simultaneous; lower = the cross-fit sim_a10 bound).
"""
from __future__ import annotations

import glob
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as CM                                  # noqa: E402  (paper scorer: matched point, strata)
from cfcap.baselines import base                     # noqa: E402

RES = ROOT / "results" / "baselines"
TAUS = base.TAUS
B_BOOT = 1000


def load(runs):
    fs = [f for f in glob.glob(os.path.join(os.path.expanduser(runs), "**", "units", "*.npz"), recursive=True)
          if "cfcap-baselines-b2" in f]
    parts = []
    for f in fs:
        z = np.load(f)
        if "method" in z and "split" in z:
            d = pd.DataFrame({k: z[k] for k in z.files})
            parts.append(d[d.split == "cal"])
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(["method", "split", "eid", "cand", "leg"], keep="last")


def add_model_free(df):
    d = df[df.method == "nominal"].copy()
    d["fold"] = d.eid % 2
    out = []
    for f in (0, 1):
        st = d[d.fold != f].groupby(["cand_size", "leg"]).truth.agg(med="median", q05=lambda x: x.quantile(0.05))
        te = d[d.fold == f].merge(st.reset_index(), on=["cand_size", "leg"], how="left")
        te["point"], te["scale"] = te["med"], (te["med"] - te["q05"]).clip(lower=0) + 0.01
        te["lower"] = te["q05"]
        out.append(te.drop(columns=["med", "q05", "fold"]))
    r = pd.concat(out, ignore_index=True)
    r["method"] = "model_free"
    return pd.concat([df, r], ignore_index=True)


def sim_lower(d, method):
    d = d.sort_values(["eid", "cand", "leg"])
    g = [x for _, x in d.groupby("eid")]
    W = max(len(x) for x in g)
    P = lambda c: base.pad([x[c].to_numpy(float) for x in g], W)
    tr = P("truth")
    if method == "capxfer":
        pt, sc = P("lower"), np.ones((len(g), W))
    else:
        pt, sc = P("point"), P("scale")
    folds = np.array([int(x.eid.iloc[0]) % 2 for x in g])
    lo, _ = base.cross_fit_lower(pt, sc, tr, folds, 0.1)
    return pd.concat([x.assign(lower_sim=lo[i, :len(x)]) for i, x in enumerate(g)], ignore_index=True)


def matched(d, method):
    d = d.copy()
    if method == "capxfer":
        d["point"], d["scale"] = np.nan, np.nan          # lower-only: the scorer shifts the bound itself
    out = []
    for f in (0, 1):
        fit, ev = d[d.eid % 2 != f].reset_index(drop=True), d[d.eid % 2 == f].reset_index(drop=True)
        lo, k = CM.matched_lower(fit, ev, stratum="rel")
        out.append(ev.assign(lower_matched=lo))
    return pd.concat(out, ignore_index=True)


def counts(d, low_col, taus):
    """Per episode arrays (n_cap, n_cert_cap, n_inc, n_unsafe) for the rows of d at the given taus."""
    rows = []
    for tau in taus:
        lo = d[low_col].to_numpy(float)
        tr = d["truth"].to_numpy(float)
        ok = np.isfinite(tr)
        go = ok & np.isfinite(lo) & (lo >= tau)
        cap, inc = ok & (tr >= tau), ok & (tr < tau)
        rows.append(pd.DataFrame(dict(eid=d.eid.to_numpy(), cap=cap, cc=go & cap, inc=inc, un=go & inc,
                                      tau=tau)))
    return pd.concat(rows, ignore_index=True)


def strata_mask(d, name, tau):
    if name == "all":
        return np.ones(len(d), bool)
    if name == "rel":
        return ((d.has_fault > 0.5) | ((d.truth - tau).abs() < CM.REL_BAND)).to_numpy()
    if name == "lone_faulted":
        return ((d.cand_size == 1) & (d.has_fault > 0.5) & (d.kind == "child")).to_numpy()
    raise ValueError(name)


def summarise(d, low_col, stratum, taus):
    parts = []
    for tau in taus:
        m = strata_mask(d, stratum, tau)
        parts.append(counts(d[m], low_col, (tau,)))
    c = pd.concat(parts, ignore_index=True)
    ep = c.groupby("eid")[["cap", "cc", "inc", "un"]].sum()
    eids = ep.index.to_numpy()
    rng = np.random.default_rng(0)
    A = ep.to_numpy(float)

    def stat(a):
        return (a[:, 1].sum() / a[:, 0].sum() if a[:, 0].sum() else np.nan,
                a[:, 3].sum() / a[:, 2].sum() if a[:, 2].sum() else np.nan,
                float((a[:, 3] > 0).mean()))
    pt = stat(A)
    bs = np.array([stat(A[rng.integers(0, len(A), len(A))]) for _ in range(B_BOOT)])
    lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
    return dict(n_ep=len(eids), n_capable=int(A[:, 0].sum()), n_incapable=int(A[:, 2].sum()),
                cert_frac=pt[0], cert_lo=lo[0], cert_hi=hi[0], unsafe_rate=pt[1], unsafe_lo=lo[1], unsafe_hi=hi[1],
                ep_any_unsafe=pt[2], ep_lo=lo[2], ep_hi=hi[2])


def main(runs="~/.cache/vast-out"):
    df = add_model_free(load(runs))
    rows, dec = [], []
    for meth, d in df.groupby("method"):
        d = matched(sim_lower(d, meth), meth)
        for cert, col in (("sim_a10", "lower_sim"), ("matched_U1", "lower_matched")):
            for stratum in ("all", "rel", "lone_faulted"):
                for taus, tl in ((TAUS, "pooled"),) + tuple(((t,), str(t)) for t in TAUS):
                    r = summarise(d, col, stratum, taus)
                    rows.append(dict(method=meth, certifier=cert, stratum=stratum, tau=tl, **r))
        x = d.copy()
        x["lower"], x["route"], x["status"], x["alpha"] = x["lower_sim"], "simultaneous", "exploratory", 0.1
        if meth == "capxfer":
            x["point"], x["scale"] = np.nan, np.nan
        x["contains_fault"] = x["has_fault"]
        dec.append(x)
    t = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "paper").mkdir(parents=True, exist_ok=True)
    t.to_csv(RES / "cal_table.csv", index=False)
    cols = ["method", "route", "posterior", "train", "test_plant", "split", "status", "eid", "shape", "cand", "kind",
            "cand_size", "has_fault", "leg", "truth", "point", "scale", "lower", "alpha", "seconds", "contains_fault"]
    pd.concat(dec, ignore_index=True)[cols].to_csv(RES / "paper" / "dec_baselines_cal.csv.gz", index=False,
                                                    float_format="%.6g")
    show = t[(t.tau == "pooled")]
    with pd.option_context("display.width", 200):
        print(show[["method", "certifier", "stratum", "n_ep", "n_incapable", "cert_frac", "cert_lo", "cert_hi",
                    "unsafe_rate", "unsafe_hi", "ep_any_unsafe", "ep_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
