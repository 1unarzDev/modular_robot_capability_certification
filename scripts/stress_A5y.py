"""A5y: does the child-rate yaw probe flag the T1 -> T2 (all+motor) yaw shifts of >= 3-module children? Exploratory.
Targets (written before the run): docs/board/stress.md "A5y targets".

    scripts/heavy.sh $PY scripts/stress_A5y.py run --n 200    # -> $CFCAP_OUT/stress/A5y.npz
    $PY scripts/stress_A5y.py analyze                         # -> results/stress/A5y_auc.csv, A5y_rows.csv
"""
import argparse
import csv
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stress_A5 as A5    # noqa: E402

SHAPES = ("L3", "square", "2x3")
SEED0 = 908000
COND = "all+motor"
KINDS = ("v2", "clamp", "yawc", "wheelrate")


def run(n):
    from cfcap.t1 import capability as cap1, params as P, plant
    from cfcap.t2 import build, capability as cap2, misfit as MF
    par = P.base_params("plated")
    th = P.load_theta("plated")
    dr = P.theta_dr(th)
    eff = build.ABLATIONS[COND]
    rng = np.random.default_rng(SEED0)
    out = {}
    t0 = time.time()
    for si, s in enumerate(SHAPES):
        asm = P.named_assembly(s, par)
        lat = plant.sample_latents(asm, n, rng, plant.HET_FULL, dr=dr, faults=A5.faults_k(n, asm.n, rng), device="cpu")
        lr = plant.sample_latents(asm, 1, np.random.default_rng(0), plant.HET_OFF, domain=P.point_domain(th, 1, "cpu"),
                                  device="cpu")
        ref1 = cap1.capability([(asm, lr)], par, device="cpu")[0]["rho"].cpu().numpy()[0]
        out[f"{s}/T1"] = cap1.capability([(asm, lat)], par, device="cpu")[0]["rho"].cpu().numpy() / ref1[None]
        seed = SEED0 + 1000003 * si
        ref2 = cap2.rho_ref(asm, par, "plated", eff)
        out[f"{s}/T2"] = cap2.capability([(asm, lat)], par, eff, seed=seed)[0]["rho"] / ref2[None]
        out[f"{s}/T2_fixed"] = out[f"{s}/T2"] * ref2[None] / ref1[None]       # N3: rho_abs^T2 / rho_ref^T1
        mf = MF.probe_misfit(asm, lat, par, eff, seed, kinds=KINDS, rng=np.random.default_rng([SEED0, si, 5]))
        for k in KINDS:
            out[f"{s}/D_imu_{k}"], out[f"{s}/D_duty_{k}"] = mf[k]
        out[f"{s}/yawc_track"] = mf["yawc_track"]
        out[f"{s}/n_fault"] = (lat["mode"].numpy() > 0).sum(1)
        out[f"{s}/faults"] = np.array([json.dumps([[int(w), int(lat["mode"][b, w]), round(float(lat["g"][b, w]), 3)]
                                                   for w in np.nonzero(lat["mode"][b].numpy())[0]]) for b in range(n)])
        print(s, f"{time.time() - t0:.0f}s", flush=True)
    A5.OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(A5.OUT / "A5y.npz", **out)


def _auc(a, b):
    if not len(a) or not len(b):
        return None
    return float((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean())


def analyze():
    from cfcap.t1.capability import LEGS
    z = np.load(A5.OUT / "A5y.npz")
    yaw = np.array([l.startswith("yaw") for l in LEGS])
    res = []
    for label in ("fixed", "own"):
        res += _analyze_label(z, yaw, label)
    A5.RES.mkdir(parents=True, exist_ok=True)
    with open(A5.RES / "A5y_auc.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    for r in res:
        print(r)


def _analyze_label(z, yaw, label):
    from cfcap.t1.capability import LEGS  # noqa: F401
    rows = []
    for s in SHAPES:
        r1 = z[f"{s}/T1"]
        r2 = z[f"{s}/T2_fixed"] if label == "fixed" else z[f"{s}/T2"]
        fl = np.zeros_like(r1, bool)
        for t in A5.TAUS:
            fl |= (r1 >= t) != (r2 >= t)
        sh = np.abs(r2 - r1) > 0.05
        for b in range(len(r1)):
            r = dict(shape=s, ep=b, n_fault=int(z[f"{s}/n_fault"][b]), faults=str(z[f"{s}/faults"][b]),
                     yaw_flip=int(fl[b, yaw].any()), yaw_shift05=int(sh[b, yaw].any()), any_flip=int(fl[b].any()),
                     any_shift05=int(sh[b].any()), yaw_absd_max=float(np.abs(r2 - r1)[b, yaw].max()))
            for k in KINDS:
                r[f"D_imu_{k}"] = float(z[f"{s}/D_imu_{k}"][b])
                r[f"D_duty_{k}"] = float(z[f"{s}/D_duty_{k}"][b])
            r["D_imu_clamp+yawc"] = r["D_imu_clamp"] + r["D_imu_yawc"]
            r["D_wheelrate"] = r["D_imu_wheelrate"] + r["D_duty_wheelrate"]
            r["yawc_track"] = float(z[f"{s}/yawc_track"][b]) if f"{s}/yawc_track" in z.files else np.nan
            rows.append(r)
    stats = ("D_imu_v2", "D_imu_clamp", "D_imu_yawc", "D_imu_clamp+yawc", "D_duty_yawc", "yawc_track", "D_wheelrate",
             "D_imu_wheelrate", "D_duty_wheelrate")
    res = []
    for target, label_, faulted_neg in (("a", "yaw_flip|yaw_shift05", False), ("a_flips", "yaw_flip", True),
                                       ("b", "any_flip|any_shift05", False), ("b_flips", "any_flip", True)):
        pos = np.array([any(r[k] for k in label_.split("|")) for r in rows], bool)
        neg = ~pos
        if faulted_neg:
            neg &= np.array([r["n_fault"] > 0 for r in rows])
        row = dict(ref_label=label, target=target, label=label_, negatives="faulted" if faulted_neg else "all", positives=int(pos.sum()),
                   n_neg=int(neg.sum()))
        for k in stats:
            d = np.array([r[k] for r in rows])
            row[f"auc_{k}"] = _auc(d[pos], d[neg])
        res.append(row)
    if label == "fixed":
        with open(A5.RES / "A5y_rows.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "analyze"))
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    run(a.n) if a.cmd == "run" else analyze()
