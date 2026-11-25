"""Offline decision evaluation of T1 `decide` runner outputs (exploratory until pre-registered).

    $PY scripts/decide_t1_eval.py --tag t1 DIR [DIR ...]

DIR: runner OUT directories (pulled boxes) holding decide_eps/*.pkl. Every episode is evaluated with
cfcap.decide.evaluate.run_policies (cross-fitted folds = episode index parity). Outputs results/decision/<tag>_*.csv
and the paper contract file results/decision/paper/probe_<tag>.csv (test_plant T1).
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
import time
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cfcap.decide import evaluate as Ev        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tag", default="t1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--policies", default="v2,all,random,eig,epig,evoi,always_split")
    ap.add_argument("--v2-s", type=float, default=12.5, help="T1 v2 routine duration charged to every policy")
    a = ap.parse_args()
    eps = []
    for d in a.dirs:
        for f in sorted(glob.glob(str(Path(d) / "**" / "decide_eps" / "*.pkl"), recursive=True)):
            eps += pickle.loads(Path(f).read_bytes())
    seen, uniq = set(), []
    for e in eps:
        if e["meta"]["seed"] not in seen:
            seen.add(e["meta"]["seed"])
            uniq.append(e)
    print(f"{len(uniq)} episodes", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        rows = Ev.run_policies(uniq, a.policies.split(","), log=lambda s: print(s, f"{time.time() - t0:.0f}s"),
                               pmap=lambda f, xs: pool.map(f, xs, chunksize=2))
    df, summ = Ev.summarise(rows)
    out = REPO / "results" / "decision"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.tag}_episodes.csv", index=False)
    summ.to_csv(out / f"{a.tag}_summary.csv", index=False)
    Ev.write_paper(df, uniq, out / "paper" / f"probe_{a.tag}.csv", v2_s=a.v2_s, test_plant="T1")
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
