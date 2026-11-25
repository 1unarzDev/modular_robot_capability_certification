"""A5 misspecification pilot: T2 (cfcap.t2) vs T1 (cfcap.t1) child rho_rel at matched latents, with an ablation over
the T2 effects. Targets and thresholds: docs/board/stress.md (written before the run). Exploratory.

    scripts/heavy.sh --gpu $PY scripts/stress_A5.py t1 --arm P0 --n 100 --shard 0      # latents + T1 (GPU, seconds)
    scripts/heavy.sh $PY scripts/stress_A5.py t2 --arm P0 --n 100 --shard 0            # T2 ablation (numba, CPU)
    $PY scripts/stress_A5.py analyze                                                    # -> results/stress/A5_*.csv
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import pathlib
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", pathlib.Path.home() / ".cache/modular-runs/2026-11/cfcap")) / "stress"
RES = ROOT / "results/stress"
SHAPES = ("single", "pair", "L3", "square", "2x3")
CONDS = ("base", "compliance", "load_transfer", "aniso", "aniso_mu", "all-aniso_mu", "all")
TAUS = (0.7, 0.8)
SEED0 = 905000


def faults_k(B, n, rng, modes=(1, 2, 3), g_range=(0.3, 0.9)):
    mode = np.zeros((B, n), np.int64)
    g = np.ones((B, n), np.float32)
    for b in range(B):
        k = int(rng.integers(0, 3))
        for w in rng.choice(n, size=k, replace=False):
            mode[b, w] = int(rng.choice(modes))
            if mode[b, w] == 1:
                g[b, w] = rng.uniform(*g_range)
    return mode, g


def _lat_np(lat):
    import torch
    cv = lambda v: v.detach().cpu().numpy() if torch.is_tensor(v) else v          # noqa: E731
    out = {}
    for k, v in lat.items():
        if isinstance(v, dict):
            out[k] = {kk: cv(vv) for kk, vv in v.items()}
        else:
            out[k] = cv(v)
    return out


def _flat(d, pre=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(_flat(v, f"{pre}{k}."))
        elif v is not None:
            o[pre + k] = np.asarray(v)
    return o


def _unflat(z, pre):
    lat = {"dom": {}, "payload": None}
    for k in z.files:
        if not k.startswith(pre):
            continue
        key = k[len(pre):]
        if key.startswith("dom."):
            v = z[k]
            lat["dom"][key[4:]] = float(v) if v.ndim == 0 else v
        elif key.startswith("payload."):
            lat["payload"] = lat["payload"] or {}
            lat["payload"][key[8:]] = z[k]
        else:
            v = z[k]
            lat[key] = float(v) if v.ndim == 0 else v
    return lat


def stage_t1(arm, n, shard, device="cuda"):
    """T1 stage (GPU, or torch CPU with --device cpu): latents + T1 labels -> npz."""
    import torch
    from cfcap.t1 import capability as cap1, params as P, plant
    par = P.base_params("plated")
    dr = P.theta_dr(P.load_theta("plated"))
    het = plant.HET_FULL if arm == "P0" else dataclasses.replace(plant.HET_FULL, payload=(0.0, 10.0))
    seed = SEED0 + 1000 * (arm == "P1") + shard
    rng = np.random.default_rng(seed)
    targets = []
    for s in SHAPES:
        asm = P.named_assembly(s, par)
        targets.append((asm, plant.sample_latents(asm, n, rng, het, dr=dr, faults=faults_k(n, asm.n, rng),
                                                  device=device)))
    t0 = time.time()
    o1 = cap1.capability(targets, par, relative=True, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    out = {"time_T1": time.time() - t0, "seed": seed}
    for s, (asm, lat), o in zip(SHAPES, targets, o1):
        out[f"{s}/T1/rho_rel"] = o["rho_rel"].cpu().numpy()
        out[f"{s}/T1/rho"] = o["rho"].cpu().numpy()
        out[f"{s}/nfault"] = (lat["mode"].cpu().numpy() > 0).sum(1)
        out.update({f"{s}/lat/{k}": v for k, v in _flat(_lat_np(lat)).items()})
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"A5_{arm}_n{n}_s{shard}_t1.npz"
    np.savez_compressed(f, **out)
    print(json.dumps(dict(file=str(f), time_T1=out["time_T1"], sims=2 * 10 * n * len(SHAPES))))


def stage_t2(arm, n, shard, conds=CONDS, backend="numba"):
    """CPU stage: T2 labels for every condition on the stage-t1 latents -> npz (no GPU)."""
    from cfcap.t1 import params as P
    from cfcap.t2 import build, capability as cap2
    par = P.base_params("plated")
    z = np.load(OUT / f"A5_{arm}_n{n}_s{shard}_t1.npz")
    seed = int(z["seed"])
    targets = [(P.named_assembly(s, par), _unflat(z, f"{s}/lat/")) for s in SHAPES]
    out = {}
    for c in conds:
        t0 = time.time()
        o2 = cap2.capability(targets, par, build.ABLATIONS[c], relative=True, backend=backend, seed=seed)
        out[f"time_{c}"] = time.time() - t0
        for s, o in zip(SHAPES, o2):
            out[f"{s}/{c}/rho_rel"] = o["rho_rel"]
            out[f"{s}/{c}/rho"] = o["rho"]
        print(f"{c}: {out[f'time_{c}']:.1f}s ({2 * 10 * n * len(SHAPES)} sims)", flush=True)
    tag = "all" if tuple(conds) == CONDS else "-".join(conds)
    np.savez_compressed(OUT / f"A5_{arm}_n{n}_s{shard}_t2_{tag}.npz", **out)


def _load(arm):
    data = {}
    for f1 in sorted(glob.glob(str(OUT / f"A5_{arm}_*_t1.npz"))):
        f2s = sorted(glob.glob(f1.replace("_t1.npz", "_t2_*.npz")))
        if not f2s:
            continue
        zs = [np.load(f1)] + [np.load(f) for f in f2s]
        for z in zs:
            for k in z.files:
                if k.startswith("time") or "/lat/" in k or k == "seed":
                    continue
                data.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v, 0) for k, v in data.items()}


def _boot(flip, nboot=1000, seed=0):
    """Episode-cluster bootstrap CI of the mean of flip (B, D)."""
    rng = np.random.default_rng(seed)
    B = flip.shape[0]
    m = flip.mean(1)
    bs = [m[rng.integers(0, B, B)].mean() for _ in range(nboot)]
    return np.percentile(bs, [2.5, 97.5])


def analyze():
    import csv
    RES.mkdir(parents=True, exist_ok=True)
    rows, qrows = [], []
    verdict = {}
    for arm in ("P0", "P1"):
        D = _load(arm)
        if not D:
            continue
        conds = sorted({k.split("/")[1] for k in D if k.endswith("/rho_rel") and "/T1/" not in k}, key=CONDS.index)
        for c in conds:
            for sub in ("children",) + SHAPES:
                shapes = ("pair", "L3", "square", "2x3") if sub == "children" else (sub,)
                for fsel in ("all", "faulted"):
                    flips, ds = [], []
                    for s in shapes:
                        r1, r2 = D[f"{s}/T1/rho_rel"], D[f"{s}/{c}/rho_rel"]
                        sel = np.ones(len(r1), bool) if fsel == "all" else D[f"{s}/nfault"] > 0
                        r1, r2 = r1[sel], r2[sel]
                        fl = np.concatenate([(r1 >= t) != (r2 >= t) for t in TAUS], 1)
                        flips.append(fl)
                        ds.append(r2 - r1)
                    fl = np.concatenate(flips, 0)
                    d = np.concatenate(ds, 0)
                    lo, hi = _boot(fl.astype(float))
                    q = np.percentile(d, [1, 5, 50, 95, 99])
                    rows.append(dict(arm=arm, cond=c, child=sub, sel=fsel, episodes=fl.shape[0], decisions=fl.size,
                                     flip=fl.mean(), flip_lo=lo, flip_hi=hi, share_d05=(np.abs(d) > 0.05).mean(),
                                     share_d02=(np.abs(d) > 0.02).mean(), d_mean=d.mean(),
                                     d_q01=q[0], d_q05=q[1], d_q50=q[2], d_q95=q[3], d_q99=q[4],
                                     d_absmax=np.abs(d).max()))
                    if sub == "children" and fsel == "all" and c == "all":
                        verdict[arm] = dict(flip=float(fl.mean()), flip_ci=[float(lo), float(hi)],
                                            share_d05=float((np.abs(d) > 0.05).mean()))
        # per-leg shift (all, children)
        from cfcap.t1.capability import LEGS
        for c in conds:
            for li, leg in enumerate(LEGS):
                d = np.concatenate([D[f"{s}/{c}/rho_rel"][:, li] - D[f"{s}/T1/rho_rel"][:, li]
                                    for s in ("pair", "L3", "square", "2x3")])
                f = np.concatenate([np.stack([(D[f"{s}/T1/rho_rel"][:, li] >= t) != (D[f"{s}/{c}/rho_rel"][:, li] >= t)
                                              for t in TAUS], 1) for s in ("pair", "L3", "square", "2x3")])
                qrows.append(dict(arm=arm, cond=c, leg=leg, flip=f.mean(), share_d05=(np.abs(d) > 0.05).mean(),
                                  d_q05=np.percentile(d, 5), d_q50=np.median(d), d_q95=np.percentile(d, 95)))
    for name, rr in (("A5_flips.csv", rows), ("A5_by_leg.csv", qrows)):
        with open(RES / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            for r in rr:
                w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    v = verdict.get("P0")
    if v:
        v["lead_misspecification"] = v["flip"] > 0.05
        v["secondary_d05_gt_5pct"] = v["share_d05"] > 0.05
        v["evaluation_only"] = v["flip"] <= 0.01 and v["share_d05"] <= 0.01
    (RES / "A5_verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("t1", "t2", "analyze"))
    ap.add_argument("--arm", default="P0", choices=("P0", "P1"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd == "t1":
        stage_t1(a.arm, a.n, a.shard, a.device)
    elif a.cmd == "t2":
        stage_t2(a.arm, a.n, a.shard, tuple(a.conds.split(",")))
    else:
        analyze()
