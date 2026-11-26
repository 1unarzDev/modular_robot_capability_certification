"""Exploratory T0 decision-layer numbers cited in the text (read-only formatting).

n1b5_summary.csv: hard screen at nominal parameters (k = 5); n1bK_summary.csv: knee-tolerant calibrated screen
(fresh seeds 3900-4499); hos_K02_summary.csv: screen and certificate calibrated on severities 0-2, tested on
severity 3. Predictor env_cal (end-to-end calibrated envelope posterior) throughout.
"""
from __future__ import annotations

import pandas as pd

import common as C


def run(res, paper):
    N = C.Numbers("expl_numbers.py")
    d = res / "decision"
    files = {"nominal": "n1b5_summary.csv", "knee": "n1bK_summary.csv", "hos": "hos_K02_summary.csv"}
    for k, f in files.items():
        p = d / f
        if not p.exists():
            N.ph(f"screen.{k}.useful", f"missing {f}")
            continue
        s = pd.read_csv(p)
        s = s[s.predictor == "env_cal"].set_index("policy")
        src = f"results/decision/{f}"
        N.pct(f"screen.{k}.useful", s.loc["evoi", "useful"], src)
        N.pct(f"screen.{k}.eep", s.loc["evoi", "ep_any_unsafe"], src)
        N.num(f"screen.{k}.neps", s.loc["evoi", "n_ep"], src)
        if k == "knee":
            N.num("probe.extra.evoi", s.loc["evoi", "probe_s"], src, fmt="{:.1f}")
            N.num("probe.extra.dominance", s.loc["dom", "probe_s"], src, fmt="{:.1f}")
            N.pct("probe.share.evoi", s.loc["evoi", "probed_share"], src, d=0)
    return N
