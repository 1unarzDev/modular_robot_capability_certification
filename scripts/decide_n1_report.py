"""N1 report (docs/board/decision.md pre-registration): per-difficulty-bin episode unsafe rate with episode-bootstrap
CI, healthy abstention, useful commits, screen-miss; matched-probe-time comparison.

    $PY scripts/decide_n1_report.py [--tag t0_n1]

Reads results/decision/<tag>_{episodes,meta}.csv; writes <tag>_bins.csv, <tag>_overall.csv, <tag>_matched.csv and
<tag>_verdict.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "decision"
CERTS = ("sim0.1", "simM0.1", "simF0.1", "raw", "bayes05")


def boot_hi(x, B=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t0_n1")
    a = ap.parse_args()
    d = pd.read_csv(R / f"{a.tag}_episodes.csv")
    meta = pd.read_csv(R / f"{a.tag}_meta.csv")
    d = d[d.predictor == "posterior"].copy()
    d["any_unsafe"] = (d.n_unsafe > 0).astype(float)
    sub = d[d.policy == "v2"]
    # n_screen tercile on all episodes (for reporting; the certifier used calibration-fold terciles)
    cuts = np.quantile(meta.n_screen, [1 / 3, 2 / 3])
    sub = sub.assign(tercile=np.searchsorted(cuts, sub.n_screen, side="right"))
    rows = []
    for cert in CERTS:
        s = sub[sub.cert == cert]
        for (mb, t), g in s.groupby(["mass_bin", "tercile"]):
            rows.append(dict(cert=cert, mass_bin=mb, tercile=t, n_ep=len(g), ep_any_unsafe=g.any_unsafe.mean(),
                             ep_any_unsafe_hi=boot_hi(g.any_unsafe), useful=g.n_served.sum() / max(g.oracle_served.sum(), 1),
                             abstain=g.abstain.mean()))
    bins = pd.DataFrame(rows)
    bins.to_csv(R / f"{a.tag}_bins.csv", index=False)
    ov = []
    for (pol, cert), g in d[d.cert.isin(CERTS)].groupby(["policy", "cert"]):
        h = g[g.n_faults == 0]
        ov.append(dict(policy=pol, cert=cert, n_ep=len(g), ep_any_unsafe=g.any_unsafe.mean(),
                       ep_any_unsafe_hi=boot_hi(g.any_unsafe),
                       unsafe_commit_rate=g.n_unsafe.sum() / max(g.n_assigned.sum(), 1),
                       useful=g.n_served.sum() / max(g.oracle_served.sum(), 1), abstain=g.abstain.mean(),
                       healthy_abstain=h.abstain.mean() if len(h) else np.nan,
                       regret=g.regret.mean(), regret_se=g.regret.std(ddof=1) / np.sqrt(len(g)),
                       probe_s=g.probe_s.mean(), certified=g.n_certified.sum()))
    ov = pd.DataFrame(ov)
    ov.to_csv(R / f"{a.tag}_overall.csv", index=False)
    m = ov[ov.policy.str.contains("@") | ov.policy.isin(["v2", "fix:clamp", "fix:clamp+moddiff"])]
    m.to_csv(R / f"{a.tag}_matched.csv", index=False)
    sm = bins[bins.cert == "simM0.1"]
    v2 = ov[ov.policy == "v2"].set_index("cert")
    verdict = dict(
        screen_miss=float(1 - meta.truth_in.mean()), screen_miss_final=float(1 - meta.truth_in_final.mean()),
        empty_screen=float((meta.unknown.fillna("") == "empty_screen").mean()),
        truncated_share=float((meta.mass_dropped > 0).mean()),
        bins_pass=bool((sm.ep_any_unsafe <= 0.10).all() and (sm.ep_any_unsafe_hi <= 0.15).all()),
        healthy_abstain_simM=float(v2.loc["simM0.1", "healthy_abstain"]),
        useful_simM=float(v2.loc["simM0.1", "useful"]),
        marginal_sim_ep_unsafe=float(v2.loc["sim0.1", "ep_any_unsafe"]))
    verdict["bar_pass"] = bool(verdict["bins_pass"] and verdict["healthy_abstain_simM"] <= 0.05
                               and verdict["useful_simM"] >= 0.95)
    verdict["marginal_pass"] = bool(verdict["marginal_sim_ep_unsafe"] <= 0.10)
    (R / f"{a.tag}_verdict.json").write_text(json.dumps(verdict, indent=1))
    pd.set_option("display.width", 250)
    print(bins.round(4).to_string(index=False))
    print(ov.round(4).to_string(index=False))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
