"""Decisions under the model-free envelope class vs the physics fit (exploratory; review item 2, research N1 follow-up).

    scripts/heavy.sh $PY scripts/decide_envelope.py --seeds 0:480 --workers 4

Episodes: cfcap/theory_task.py:gate_episode(seed) (theory E1: knee-law truth with severity seed % 4, two-constant
fit and identified-set low end at v2 and v2+clamp). Reuses theory's $CFCAP_OUT/theory/gate/gate_*.pkl when present,
else computes them into $CFCAP_OUT/decision/gate_<s0>_<s1>.pkl. Evaluation: cfcap.decide.envelope (3-fold
leak-free cross-fit). Outputs results/decision/envelope_{rows,summary,by_law}.csv.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = Path(os.environ.get("CFCAP_OUT", "/tmp"))


def _work(seed):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from cfcap.t0 import qs
    from cfcap.theory_task import gate_episode
    P, th = qs.load_params(), qs.load_theta()
    return pd.DataFrame(gate_episode(seed, P, th))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0:480")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split(":"))
    theirs = sorted(glob.glob(str(OUT / "theory" / "gate" / "gate_*.pkl")))
    mine = OUT / "decision" / f"gate_{s0}_{s1}.pkl"
    if theirs:
        df = pd.concat([pd.read_pickle(f) for f in theirs])
        df = df[(df.seed >= s0) & (df.seed < s1)]
        print(f"theory gate pickles: {len(theirs)} files, {df.seed.nunique()} seeds")
    elif mine.exists():
        df = pd.read_pickle(mine)
    else:
        with get_context("spawn").Pool(a.workers) as p:
            df = pd.concat(p.map(_work, range(s0, s1), chunksize=2))
        mine.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(mine)
    if "conv" in df:
        df = df.groupby("seed").filter(lambda g: bool(g.conv.all()))     # drop non-converged knee episodes
    from cfcap.decide import envelope as En
    from cfcap.t0 import qs
    eps = En.episodes_from_gate(df, qs.LEG_NAMES)
    rows = En.run(eps)
    res = REPO / "results" / "decision"
    rows.to_csv(res / "envelope_rows.csv", index=False)
    summ = En.summarise(rows)
    summ.to_csv(res / "envelope_summary.csv", index=False)
    by = rows.assign(knee=rows.law > 0).groupby("knee", group_keys=True).apply(En.summarise, include_groups=False)
    by.to_csv(res / "envelope_by_law.csv")
    pd.set_option("display.width", 250)
    print(f"{len(eps)} episodes")
    print(summ.round(4).to_string(index=False))
    print(by.round(4).to_string())


if __name__ == "__main__":
    main()
