"""A5 follow-up: would a v2 (+ clamp) observation have caught the T1 -> T2 flips? Exploratory.

For every A5 episode (P0 shards), the child is probed with the generator's v2 and clamp routines
(cfcap.gen.simrun.probe_tables, same jitter streams) under T1 (cfcap.t1.plant, torch CPU) and T2 `all` (same latents,
same effect draws as A5). Misfit statistic of the robot-visible IMU channels, both derived from the 50 Hz twist the
same way (gz = w, a = dv/dt + w x v, so the two plants' accelerometer definitions cannot differ):
    D = mean over samples and channels of ((T2 - T1) / sigma)^2,  sigma = capxfer IMU noise (0.002, 0.02, 0.02),
plus the duty channel with an assumed 0.01 duty noise floor. With latents KNOWN (an oracle misfit test, an upper
bound on what any T1-model check could detect), a noisy T2 log differs from the T1 prediction by excess D; detection
at ~3 sigma of the chi-square: D > 3 sqrt(2 / N_samples). Rows -> results/stress/A5_clamp.csv, summary -> stdout.

    scripts/heavy.sh $PY scripts/stress_A5_clamp.py
"""
import csv
import glob
import json
import os
import pathlib
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stress_A5 as A5  # noqa: E402

SIG = np.array([0.002, 0.02, 0.02])
SIG_DUTY = 0.01


def imu_from_twist(tw):
    v = tw
    dv = np.zeros_like(v)
    dv[:, 1:] = np.diff(v, axis=1) / 0.02
    ax = dv[..., 0] - v[..., 2] * v[..., 1]
    ay = dv[..., 1] + v[..., 2] * v[..., 0]
    return np.stack([v[..., 2], ax, ay], -1)


def misfit(a, b, da, db):
    T = min(a.shape[1], b.shape[1])
    di = ((imu_from_twist(a[:, :T]) - imu_from_twist(b[:, :T])) / SIG) ** 2
    dd = ((da[:, :T] - db[:, :T]) / SIG_DUTY) ** 2
    N_i, N_d = 3 * T, dd.shape[2] * T
    return di.reshape(len(a), -1).mean(1), dd.reshape(len(a), -1).mean(1), N_i, N_d


def main():
    import torch
    from cfcap.gen import simrun as SR
    from cfcap.t1 import params as P, plant
    from cfcap.t2 import build, capability as cap2
    par = P.base_params("plated")
    R = np.load(A5.OUT / "A5_refs.npz")
    rows = []
    for f1 in sorted(glob.glob(str(A5.OUT / "A5_P0_*_t1.npz"))):
        f2 = f1.replace("_t1.npz", "_t2_all.npz")
        if not os.path.exists(f2):
            continue
        z1, z2 = np.load(f1), np.load(f2)
        seed = int(z1["seed"])
        for si, s in enumerate(A5.SHAPES):
            if s == "single":
                continue
            asm = P.named_assembly(s, par)
            lat = A5._unflat(z1, f"{s}/lat/")
            B = lat["mode"].shape[0]
            r1 = z1[f"{s}/T1/rho"] / R[f"{s}/T1"][None]
            r2 = z2[f"{s}/all/rho"] / R[f"{s}/all"][None]
            flip = np.zeros(B, bool)
            for t in A5.TAUS:
                flip |= ((r1 >= t) != (r2 >= t)).any(1)
            stat = {}
            for kind in ("v2", "clamp"):
                rng = np.random.default_rng([seed, si, 17 + ("v2", "clamp").index(kind)])
                tabs = np.stack([SR.probe_tables(kind, asm, par, rng)[0] for _ in range(B)])
                T = SR.probe_T(kind)
                lt = {k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in lat.items()
                      if k not in ("dom", "payload")}
                lt["dom"] = {k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in lat["dom"].items()}
                lt["payload"] = None
                R1 = plant.simulate_bucket([dict(asm=asm, lat=lt, cmd=tabs)], par, T, device="cpu")[0]
                # T2: effect seed of A5 target index si (cfcap.t2.capability.capability: seed + 1000003 i)
                R2 = cap2.simulate(asm, lat, par, tabs, T, build.ALL, seed=seed + 1000003 * si)
                stat[kind] = misfit(R1["twist"].numpy().astype(float), R2["twist"], R1["duty"].numpy().astype(float),
                                    R2["duty"])
            for b in range(B):
                row = dict(shard_seed=seed, shape=s, ep=b, n_fault=int((lat["mode"][b] > 0).sum()), flip=int(flip[b]))
                det = {}
                for kind, (di, dd, Ni, Nd) in stat.items():
                    row[f"D_imu_{kind}"], row[f"D_duty_{kind}"] = float(di[b]), float(dd[b])
                    det[kind] = (di[b] > 3 * np.sqrt(2 / Ni), dd[b] > 3 * np.sqrt(2 / Nd))
                row["det_v2_imu"], row["det_v2_any"] = int(det["v2"][0]), int(det["v2"][0] or det["v2"][1])
                row["det_v2clamp_imu"] = int(det["v2"][0] or det["clamp"][0])
                row["det_v2clamp_any"] = int(any(det["v2"]) or any(det["clamp"]))
                rows.append(row)
            print(s, "done", flush=True)
    out = A5.RES / "A5_clamp.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fl = np.array([r["flip"] for r in rows], bool)
    summ = {}
    for k in ("det_v2_imu", "det_v2_any", "det_v2clamp_imu", "det_v2clamp_any"):
        d = np.array([r[k] for r in rows], bool)
        summ[k] = dict(flipped=float(d[fl].mean()) if fl.any() else None, not_flipped=float(d[~fl].mean()),
                       n_flipped=int(fl.sum()), n=len(rows))
    for k in ("D_imu_v2", "D_imu_clamp"):
        d = np.array([r[k] for r in rows])
        summ[k + "_median"] = dict(flipped=float(np.median(d[fl])) if fl.any() else None,
                                   not_flipped=float(np.median(d[~fl])))
    (A5.RES / "A5_clamp_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
