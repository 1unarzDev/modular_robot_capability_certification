"""Decision-layer pilot on T0 (CPU, exploratory): predictors x probe policies x certifiers on seeded episodes.

    scripts/heavy.sh $PY scripts/decide_pilot.py --episodes 400 --workers 4

Episodes: cfcap.decide.t0data.episode(seed), seeds 909000 + j (decision namespace), parents of 2-4 modules, generator
fault prior; tasks from problem.make_tasks with seed stream 919000 + j (independent of the truth). Folds j % 2
(cross-fitted conformal). Outputs results/decision/pilot_t0_{episodes,summary}.csv and the episode cache under
$CFCAP_OUT/decision/.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cfcap.decide import evaluate as Ev        # noqa: E402
from cfcap.decide import problem as Pb         # noqa: E402

SEED0, TASK_SEED0 = 909000, 919000


def _gen(j):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from cfcap.decide import t0data as T
    e = T.episode(SEED0 + j)
    e["tasks"] = Pb.make_tasks(np.random.default_rng(TASK_SEED0 + j), e["legs"])
    e["fold"] = j % 2
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--policies", default="v2,all,random,eig,epig,evoi,always_split")
    ap.add_argument("--max-probes", type=int, default=2)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--tag", default="pilot_t0")
    a = ap.parse_args()
    out = REPO / "results" / "decision"
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("CFCAP_OUT", "/tmp")) / "decision" / f"{a.tag}_eps{a.episodes}.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        if cache.exists():
            eps = pickle.loads(cache.read_bytes())
        else:
            eps = pool.map(_gen, range(a.episodes), chunksize=4)
            cache.write_bytes(pickle.dumps(eps))
        t_gen = time.time() - t0
        print(f"episodes {len(eps)} in {t_gen:.0f} s", flush=True)
        rows = Ev.run_policies(eps, a.policies.split(","), alpha=0.1, alpha_crc=(0.1, 0.05, 0.01), S=a.S,
                               max_probes=a.max_probes, log=lambda s: print(s, f"{time.time() - t0:.0f}s", flush=True),
                               pmap=lambda f, xs: pool.map(f, xs, chunksize=4))
    df, summ = Ev.summarise(rows)
    import pandas as pd
    meta = pd.DataFrame([e["meta"] for e in eps])
    df.to_csv(out / f"{a.tag}_episodes.csv", index=False)
    summ.to_csv(out / f"{a.tag}_summary.csv", index=False)
    meta.to_csv(out / f"{a.tag}_meta.csv", index=False)
    from cfcap.t0 import probesets as PS
    Ev.write_paper(df, eps, out / "paper" / f"probe_{a.tag}.csv", v2_s=PS.set_time("v2"), test_plant="T0")
    # probe usage per policy (posterior predictor, sim certifier)
    sub = df[(df.predictor == "posterior") & (df.cert == "sim0.1")]
    use = sub.groupby("policy").probes.apply(lambda s: s.str.split("+").explode().replace("", np.nan).dropna()
                                             .str.replace(r"split:\d+", "split", regex=True).value_counts().to_dict())
    pd.Series(use).to_csv(out / f"{a.tag}_probe_use.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summ.round(4).to_string())
    print("meta:", meta[["P", "ess", "truth_in", "n_screen", "C", "gen_s"]].describe().round(2).to_string())
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
