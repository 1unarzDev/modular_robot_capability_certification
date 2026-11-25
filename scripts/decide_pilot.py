"""Decision-layer pilot on T0 (CPU, exploratory): predictors x probe policies x certifiers on seeded episodes.

    scripts/heavy.sh $PY scripts/decide_pilot.py --episodes 400 --workers 4

Episodes: cfcap.decide.t0data.episode(seed), seeds 909000 + j (decision namespace), parents of 2-4 modules, generator
fault prior; tasks from problem.make_tasks with seed stream 919000 + j (independent of the truth). Folds j % 2
(cross-fitted conformal, evaluate.fold_plan). Options: --k-screen 5 (legacy hard screen; empty screen -> unknown -> abstain) or cal (conformalized screen at --alpha-s on seeds 929000+), --M nuisance draws, --p-max particle budget (over budget -> unknown -> abstain), --probe-lik marginal|draw. Outputs results/decision/pilot_t0_{episodes,summary}.csv and the episode cache under
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


CFG = {}


def _gen(j):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from cfcap.decide import t0data as T
    c = CFG or dict(seed0=SEED0, folds=2, k_screen=None, M=T.M_NUIS, p_max=T.P_MAX, probe_lik=None)
    e = T.episode(c["seed0"] + j, k_screen=c["k_screen"], M=c["M"], p_max=c["p_max"], probe_lik=c["probe_lik"])
    e["tasks"] = Pb.make_tasks(np.random.default_rng(TASK_SEED0 + (c["seed0"] - SEED0) + j), e["legs"])
    e["fold"] = j % c["folds"]
    return e


def _init(cfg):
    CFG.update(cfg)


def _zcal(j):
    import torch
    torch.set_num_threads(1)
    from cfcap.decide import t0data as T
    return T.episode(CAL_SEED0 + j, z_only=True)


CAL_SEED0 = 929000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--policies", default="v2,fix:clamp,fix:clamp+moddiff,all,random,eig,epig,evoi,always_split")
    ap.add_argument("--max-probes", type=int, default=2)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--tag", default="pilot_t0")
    ap.add_argument("--paper-cert", default="simF0.1")
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--k-screen", default="5", help="screen threshold, or 'cal' = conformalized (alpha_s)")
    ap.add_argument("--alpha-s", type=float, default=0.01)
    ap.add_argument("--n-cal-screen", type=int, default=600)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--p-max", type=int, default=192)
    ap.add_argument("--probe-lik", default="marginal")
    a = ap.parse_args()
    out = REPO / "results" / "decision"
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("CFCAP_OUT", "/tmp")) / "decision" / f"{a.tag}_eps{a.episodes}.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    k_screen = None
    if a.k_screen == "cal":
        import pandas as pd
        from cfcap.conformal.core import split_quantile
        zc = REPO / "results" / "decision" / "screen_calibration.csv"
        if zc.exists() and len(pd.read_csv(zc)) >= a.n_cal_screen:
            z = pd.read_csv(zc).z_true.to_numpy()[:a.n_cal_screen]
        else:
            with get_context("spawn").Pool(a.workers) as p0:
                d = pd.DataFrame(p0.map(_zcal, range(a.n_cal_screen), chunksize=8))
            d.to_csv(zc, index=False)
            z = d.z_true.to_numpy()
        k_screen = split_quantile(z, a.alpha_s)
        print(f"conformalized screen: K = {k_screen:.3f} (alpha_s {a.alpha_s}, n_cal {len(z)}, seeds {CAL_SEED0}+)")
    else:
        k_screen = float(a.k_screen)
    cfg = dict(seed0=a.seed0, folds=a.folds, k_screen=k_screen, M=a.M, p_max=a.p_max, probe_lik=a.probe_lik)
    with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
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
    Ev.write_paper(df, eps, out / "paper" / f"probe_{a.tag}.csv", v2_s=PS.set_time("v2"), test_plant="T0", cert=a.paper_cert)
    # probe usage per policy (posterior predictor, sim certifier)
    sub = df[(df.predictor == "posterior") & (df.cert == a.paper_cert)]
    use = sub.groupby("policy").probes.apply(lambda s: s.str.split("+").explode().replace("", np.nan).dropna()
                                             .str.replace(r"split:\d+", "split", regex=True).value_counts().to_dict())
    pd.Series(use).to_csv(out / f"{a.tag}_probe_use.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summ.round(4).to_string())
    print("meta:", meta[["P", "ess", "truth_in", "n_screen", "C", "gen_s"]].describe().round(2).to_string())
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
