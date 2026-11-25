"""E2 / A4: composition defect delta_comp for merges of never-co-assembled modules, in T1 and T2. Exploratory.
Targets (written before the run): docs/board/stress.md "E2 / A4 targets". Spec: docs/board/research.md E2.

    scripts/heavy.sh $PY scripts/stress_A4_merge.py run --arm P0 --pairs 250 --shard 0   # -> $CFCAP_OUT/stress/A4_P0_s0.npz
    (P0 and P5, shards 0 and 1: 500 pairs per arm)
    $PY scripts/stress_A4_merge.py analyze        # -> results/stress/A4_merge.csv, A4_merge_rows.csv

Pair: G1, G2 random 1-3-module polyominoes, module yaws U{0..3}, <= 1 fault each (p = 0.5; weak g ~ U(0.3, 0.95) /
coast / brake), HET_FULL latents, one theta draw from the v2 DR shared by the pair. Merges: up to 6 face-touching
placements of all of G2 against all of G1 (cfcap.t0.candidates.merge_placements / merge). Per merge and leg:
  T1 truth   the merged body: per-wheel / per-module latents carried, the indeterminate load split REDRAWN for the
             merged geometry (HET_FULL load_split), payload (arm P5: 0-5 kg at a random G1 deck point) carried;
  T1 comp    (b) the composition from the separate assemblies' true latents: merged geometry, least-norm load split,
             the known payload;
  T2 truth   cfcap.t2 `all` on the merged body (same latents as T1 truth).
Label rho_dec = rho / rho_ref(merged geometry) with each plant's own nominal healthy reference (theta point, no
heterogeneity, no fault; T2 reference = one draw of its effect nuisances).
"""
import argparse
import csv
import json
import os
import pathlib
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", pathlib.Path.home() / ".cache/modular-runs/2026-11/cfcap")) / "stress"
RES = ROOT / "results/stress"
TAUS = (0.6, 0.7, 0.8)
SEED0 = 907000
MAX_MERGES = 6


def polyomino(rng, k):
    cells = [(0, 0)]
    while len(cells) < k:
        x, y = cells[int(rng.integers(len(cells)))]
        dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(rng.integers(4))]
        if (x + dx, y + dy) not in cells:
            cells.append((x + dx, y + dy))
    mx, my = min(c[0] for c in cells), min(c[1] for c in cells)
    return [(x - mx, y - my) for x, y in cells]


def one_fault(n, rng):
    mode = np.zeros((1, n), np.int64)
    g = np.ones((1, n), np.float32)
    if rng.uniform() < 0.5:
        w = int(rng.integers(n))
        mode[0, w] = int(rng.choice([1, 2, 3]))
        if mode[0, w] == 1:
            g[0, w] = rng.uniform(0.3, 0.95)
    return mode, g


def _clone_dom(dom):
    import torch
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in dom.items()}


def merged_latents(l1, l2, n, load_z, payload):
    import torch
    cat = lambda k: torch.cat([l1[k], l2[k]], 1)                          # noqa: E731
    lat = dict(dom=_clone_dom(l1["dom"]), mode=cat("mode"), g=cat("g"), stall_w=cat("stall_w"),
               noload_w=cat("noload_w"), yaw_err=cat("yaw_err"), vbatt_off=cat("vbatt_off"), mu_mod=cat("mu_mod"),
               load_z=torch.as_tensor(load_z[None], dtype=torch.float32), gyro_bias=l1["gyro_bias"],
               acc_bias=l1["acc_bias"], drop_p=l1["drop_p"], payload=payload)
    return lat


def run(arm, pairs, shard=0):
    import torch
    from capxfer import assembly as A, sim
    from cfcap.t0 import candidates as C
    from cfcap.t1 import capability as cap1, params as P, plant
    from cfcap.t2 import build, capability as cap2
    par = P.base_params("plated")
    th = P.load_theta("plated")
    dr = P.theta_dr(th)
    rng = np.random.default_rng([SEED0, arm == "P5", shard])
    het = plant.HET_FULL
    truth, comp, refl, meta = [], [], [], []
    for pi in range(pairs):
        k1, k2 = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        c1, c2 = polyomino(rng, k1), polyomino(rng, k2)
        y1, y2 = rng.integers(0, 4, k1), rng.integers(0, 4, k2)
        g1 = A.build(f"p{pi}g1", [(P.MODULE_IDS[q], i, j, int(y)) for q, ((i, j), y) in enumerate(zip(c1, y1))], par)
        g2 = A.build(f"p{pi}g2", [(P.MODULE_IDS[k1 + q], i, j, int(y)) for q, ((i, j), y) in enumerate(zip(c2, y2))],
                     par)
        dom = sim.sample_domain(1, rng, dr, device="cpu")
        l1 = plant.sample_latents(g1, 1, rng, het, domain=_clone_dom(dom), faults=one_fault(g1.n, rng), device="cpu")
        l2 = plant.sample_latents(g2, 1, rng, het, domain=_clone_dom(dom), faults=one_fault(g2.n, rng), device="cpu")
        pl = C.merge_placements(c1, c2)
        if not pl:
            continue
        sel = rng.choice(len(pl), size=min(MAX_MERGES, len(pl)), replace=False)
        host = int(rng.integers(k1))
        off = rng.uniform(-0.08, 0.08, 2)
        pm = float(rng.uniform(0.0, 5.0)) if arm == "P5" else 0.0
        for mi, s in enumerate(sel):
            t = pl[int(s)]
            cand = C.merge(g1, tuple(range(k1)), g2, t, par)
            asm = cand.asm
            n = asm.n
            pay = None
            if pm > 0:
                xy = np.asarray(asm.module_pos)[host] + off
                pay = dict(mass=torch.tensor([pm]), x=torch.tensor([float(xy[0])]), y=torch.tensor([float(xy[1])]))
            lt = merged_latents(l1, l2, n, rng.normal(size=n) * het.load_split, pay)
            lc = merged_latents(l1, l2, n, np.zeros(n), pay)
            lr = plant.sample_latents(asm, 1, np.random.default_rng(0), plant.HET_OFF,
                                      domain=P.point_domain(th, 1, "cpu"), device="cpu")
            lr["yaw_err"] = None
            truth.append((asm, lt)); comp.append((asm, lc)); refl.append((asm, lr))
            meta.append(dict(pair=shard * 100000 + pi, merge=mi, n_mod=k1 + k2, n_wheel=n, k1=k1, k2=k2,
                             n_fault=int((lt["mode"] > 0).sum()), payload_kg=pm, t=json.dumps(list(map(int, t)))))
    print(arm, len(meta), "merges", flush=True)
    t0 = time.time()
    out = {}
    lr_ = [(a, dict(l, yaw_err=torch.zeros_like(l["mode"], dtype=torch.float32))) for a, l in refl]
    r1t = np.stack([o["rho"].cpu().numpy()[0] for o in cap1.capability(truth, par, device="cpu")])
    r1c = np.stack([o["rho"].cpu().numpy()[0] for o in cap1.capability(comp, par, device="cpu")])
    ref1 = np.stack([o["rho"].cpu().numpy()[0] for o in cap1.capability(lr_, par, device="cpu")])
    print("T1", f"{time.time() - t0:.0f}s", flush=True)
    r2t = np.stack([o["rho"][0] for o in cap2.capability(truth, par, build.ALL, seed=SEED0)])
    ref2 = np.stack([o["rho"][0] for o in cap2.capability(refl, par, build.ALL, seed=SEED0 + 1)])
    print("T2", f"{time.time() - t0:.0f}s", flush=True)
    out.update(T1_truth=r1t / ref1, T1_comp=r1c / ref1, T2_truth=r2t / ref2, ref1=ref1, ref2=ref2)
    for k in meta[0]:
        out[f"meta_{k}"] = np.array([m[k] for m in meta])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"A4_{arm}_s{shard}.npz", **out)


def _boot(fl, cl, nb=1000, seed=0):
    """Cluster (pair) bootstrap of the mean of fl (rows) -> (lo, hi)."""
    rng = np.random.default_rng(seed)
    u = np.unique(cl)
    idx = {c: np.nonzero(cl == c)[0] for c in u}
    s = np.array([fl[idx[c]].sum() for c in u])
    m = np.array([len(idx[c]) for c in u])
    bs = []
    for _ in range(nb):
        k = rng.integers(0, len(u), len(u))
        bs.append(s[k].sum() / m[k].sum())
    return np.percentile(bs, [2.5, 97.5])


def analyze():
    from cfcap.t1.capability import LEGS
    rows, summ = [], []
    for arm in ("P0", "P5"):
        fs = sorted(OUT.glob(f"A4_{arm}_s*.npz"))
        if not fs:
            continue
        zs = [np.load(f) for f in fs]
        z = {k: np.concatenate([q[k] for q in zs]) for k in zs[0].files}
        M = len(z["meta_pair"])
        for i in range(M):
            for l, leg in enumerate(LEGS):
                r = dict(arm=arm, **{k[5:]: (z[k][i].item() if hasattr(z[k][i], "item") else z[k][i])
                                     for k in z if k.startswith("meta_")}, leg=leg,
                         rho_dec_T1_comp=float(z["T1_comp"][i, l]), rho_dec_T1_truth=float(z["T1_truth"][i, l]),
                         rho_dec_T2_truth=float(z["T2_truth"][i, l]))
                rows.append(r)
        for plant in ("T1", "T2"):
            tr, pr = z[f"{plant}_truth"], z["T1_comp"]
            for grp, sel in (("all", np.ones(M, bool)), ("n_mod<=3", z["meta_n_mod"] <= 3),
                             ("n_mod4-6", z["meta_n_mod"] >= 4), ("faulted", z["meta_n_fault"] > 0)):
                if not sel.any():
                    continue
                fl = np.stack([(pr[sel] >= t) != (tr[sel] >= t) for t in TAUS], -1)    # (m, L, 3)
                per_row = fl.reshape(fl.shape[0], -1).mean(1)
                lo, hi = _boot(per_row, z["meta_pair"][sel])
                d = np.abs(tr[sel] - pr[sel]).ravel()
                summ.append(dict(arm=arm, plant=plant, group=grp, merges=int(sel.sum()),
                                 pairs=len(np.unique(z["meta_pair"][sel])), decisions=fl.size, flip=float(fl.mean()),
                                 flip_lo=float(lo), flip_hi=float(hi), absd_q50=float(np.percentile(d, 50)),
                                 absd_q95=float(np.percentile(d, 95)), absd_q99=float(np.percentile(d, 99)),
                                 share_d05=float((d > 0.05).mean()),
                                 success=bool(hi < (0.01 if plant == "T1" else 0.02)),
                                 kill=bool(plant == "T1" and fl.mean() > 0.02)))
    RES.mkdir(parents=True, exist_ok=True)
    for name, rr in (("A4_merge_rows.csv", rows), ("A4_merge.csv", summ)):
        with open(RES / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    for r in summ:
        print(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "analyze"))
    ap.add_argument("--arm", default="P0", choices=("P0", "P5"))
    ap.add_argument("--pairs", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    run(a.arm, a.pairs, a.shard) if a.cmd == "run" else analyze()
