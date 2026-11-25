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
    import torch
    torch.set_num_threads(1)
    from cfcap.t0 import qs
    from cfcap.theory_task import gate_episode
    P, th = qs.load_params(), qs.load_theta()
    return pd.DataFrame(gate_episode(seed, P, th))


def _work_z(seed):
    import torch
    torch.set_num_threads(1)
    from cfcap.t0 import qs
    from cfcap.decide import envelope_post as EP
    P, th = qs.load_params(), qs.load_theta()
    return EP.episode(seed, P, th, z_only=True)


def _work_post(args):
    seed, lik, cover, ks = args
    import torch
    torch.set_num_threads(1)
    from cfcap.t0 import qs
    from cfcap.decide import envelope_post as EP
    P, th = qs.load_params(), qs.load_theta()
    return pd.DataFrame(EP.episode(seed, P, th, lik=lik, cover=cover, k_screen=ks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0:480")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--post", action="store_true", help="fault modes from a posterior (cfcap.decide.envelope_post)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--lik", default="v2", help="envelope_post fault likelihood: v2 | robust")
    ap.add_argument("--cover", type=float, default=0.99, help="envelope_post kept posterior mass")
    ap.add_argument("--screen-cal", default=None, help="s0:s1 calibration seeds for the knee-tolerant conformal screen")
    ap.add_argument("--alpha-s", type=float, default=0.01)
    ap.add_argument("--screen-laws", default="0,1,2,3", help="severities used to fit the screen threshold")
    ap.add_argument("--laws", default="0,1,2,3", help="evaluate only seeds with seed % 4 in this set")
    ap.add_argument("--cal-seeds", default=None, help="s0:s1 external certificate-calibration seeds (held-out law)")
    ap.add_argument("--cal-laws", default="0,1,2")
    ap.add_argument("--test-plant", default=None)
    ap.add_argument("--status", default="exploratory")
    ap.add_argument("--keep-nonconv", action="store_true",
                    help="keep episodes whose knee fixed point did not converge (truth = least-residual point)")
    a = ap.parse_args()
    tag = a.tag or ("envpost" if a.post else "envelope")
    s0, s1 = map(int, a.seeds.split(":"))
    ks = None
    if a.screen_cal:
        from cfcap.conformal.core import split_quantile
        c0, c1 = map(int, a.screen_cal.split(":"))
        zc = REPO / "results" / "decision" / f"screen_cal_knee_{c0}_{c1}.csv"
        if zc.exists():
            zd = pd.read_csv(zc)
        else:
            with get_context("spawn").Pool(a.workers) as p:
                zd = pd.DataFrame(p.map(_work_z, range(c0, c1), chunksize=4))
            zd.to_csv(zc, index=False)
        zd = zd[zd.conv & zd.law.isin([int(x) for x in a.screen_laws.split(",")])]
        ks = split_quantile(zd.z_true.to_numpy(), a.alpha_s)
        print(f"knee-tolerant screen: K = {ks:.3f} (alpha_s {a.alpha_s}, n_cal {len(zd)}, seeds {c0}:{c1}, "
              f"laws {a.screen_laws})")
        print(zd.groupby("law").z_true.quantile([0.5, 0.9, 0.99]).unstack().round(2).to_string())
    theirs = [] if a.post else sorted(glob.glob(str(OUT / "theory" / "gate" / "gate_*.pkl")))
    lawtag = "" if a.laws == "0,1,2,3" else "L" + a.laws.replace(",", "") + "_"
    mine = OUT / "decision" / (f"gatepost_{a.lik}_{a.cover}_{'' if ks is None else f'K{ks:.2f}_'}{lawtag}{s0}_{s1}.pkl"
                               if a.post else f"gate_{s0}_{s1}.pkl")
    if a.post and ks is None and a.lik == "v2" and a.cover == 0.99 and (OUT / "decision" / f"gatepost_{s0}_{s1}.pkl").exists():
        mine = OUT / "decision" / f"gatepost_{s0}_{s1}.pkl"
    if theirs:
        df = pd.concat([pd.read_pickle(f) for f in theirs])
        df = df[(df.seed >= s0) & (df.seed < s1)]
        print(f"theory gate pickles: {len(theirs)} files, {df.seed.nunique()} seeds")
    elif mine.exists():
        df = pd.read_pickle(mine)
    else:
        with get_context("spawn").Pool(a.workers) as p:
            laws = {int(x) for x in a.laws.split(",")}
            sel = [s_ for s_ in range(s0, s1) if s_ % 4 in laws]
            jobs = [(s_, a.lik, a.cover, ks) for s_ in sel] if a.post else sel
            df = pd.concat(p.map(_work_post if a.post else _work, jobs, chunksize=2))
        mine.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(mine)
    if "conv" in df and not a.keep_nonconv:
        df = df.groupby("seed").filter(lambda g: bool(g.conv.all()))     # drop non-converged knee episodes
    from cfcap.decide import envelope as En
    from cfcap.t0 import qs
    eps = En.episodes_from_gate(df, qs.LEG_NAMES)
    cal_eps = None
    if a.cal_seeds:
        c0, c1 = map(int, a.cal_seeds.split(":"))
        claws = {int(x) for x in a.cal_laws.split(",")}
        cp = OUT / "decision" / f"gatepost_{a.lik}_{a.cover}_{'' if ks is None else f'K{ks:.2f}_'}cal{a.cal_laws.replace(',', '')}_{c0}_{c1}.pkl"
        if cp.exists():
            cdf = pd.read_pickle(cp)
        else:
            with get_context("spawn").Pool(a.workers) as p:
                cdf = pd.concat(p.map(_work_post, [(s_, a.lik, a.cover, ks) for s_ in range(c0, c1) if s_ % 4 in claws],
                                      chunksize=2))
            cdf.to_pickle(cp)
        cdf = cdf.groupby("seed").filter(lambda g: bool(g.conv.all()))
        cal_eps = En.episodes_from_gate(cdf, qs.LEG_NAMES)
        print(f"external calibration: {len(cal_eps)} episodes, laws {a.cal_laws}")
    rows = En.run(eps, cal_eps=cal_eps)
    res = REPO / "results" / "decision"
    rows.to_csv(res / f"{tag}_rows.csv", index=False)
    summ = En.summarise(rows)
    summ.to_csv(res / f"{tag}_summary.csv", index=False)
    by = rows.assign(knee=rows.law > 0).groupby("knee", group_keys=True).apply(En.summarise, include_groups=False)
    by.to_csv(res / f"{tag}_by_law.csv")
    pr = []
    for pred in ("env_cal", "env", "par_cal"):
        for b in ("clamp", "dom", "v2"):
            m, lo, hi, n = En.paired(rows, pred, "evoi", b)
            pr.append(dict(predictor=pred, a="evoi", b=b, mean_diff=m, lo95=lo, hi95=hi, n=n))
        m, lo, hi, n = En.paired(rows, pred, "dom", "clamp")
        pr.append(dict(predictor=pred, a="dom", b="clamp", mean_diff=m, lo95=lo, hi95=hi, n=n))
    pr = pd.DataFrame(pr)
    pr.to_csv(res / f"{tag}_paired.csv", index=False)
    print(pr.round(3).to_string(index=False))
    if a.keep_nonconv:
        return
    En.write_paper(rows, res / "paper" / f"probe_t0{tag}.csv", status=a.status,
                   test_plant=a.test_plant or ("T0-knee-post" if a.post else "T0-knee"))
    pd.set_option("display.width", 250)
    print(f"{len(eps)} episodes")
    print(summ.round(4).to_string(index=False))
    print(by.round(4).to_string())


if __name__ == "__main__":
    main()
