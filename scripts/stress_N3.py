"""N3: fixed-reference relabelling of the misspecification tests (targets: docs/board/stress.md "N3 targets").

rho_dec^fixed = rho_abs^T2 / rho_ref^T1(shape, leg); the own-reference label rho_abs^T2 / rho_ref^T2 side by side.
Re-scores A5 (P0), A5m, A5y and E2 offline; rho_abs is recovered where only the own-reference label was stored
(times cfcap.t2.capability.rho_ref with the settings of the original run).

    $PY scripts/stress_N3.py      # -> results/stress/N3_flips.csv, N3_merge.csv, N3_verdict.json
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
import stress_A5 as A5    # noqa: E402

from cfcap.t1.capability import LEGS  # noqa: E402

LT = np.array(["yaw" if l.startswith("yaw") else "diag" if l.startswith("diag") else "axis" for l in LEGS])
TAUS = (0.7, 0.8)


def dom_class(shape):
    big = shape not in ("single", "pair")
    return np.array(["dominated" if t == "axis" else "undominated" if (t == "diag" or big) else "other" for t in LT])


def boot(fl_ep, nb=1000, seed=0):
    """fl_ep (B, D) flips per episode -> 95 % CI of the mean (episode bootstrap)."""
    rng = np.random.default_rng(seed)
    if fl_ep.size == 0:
        return (np.nan, np.nan)
    s, m = fl_ep.sum(1), fl_ep.shape[1]
    bs = [s[rng.integers(0, len(s), len(s))].sum() / (len(s) * m) for _ in range(nb)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


def sets():
    """(source, cond, shape) -> (T1 rho_dec (B, L), T2 own-ref rho_dec, T2 fixed-ref rho_dec)."""
    from cfcap.t1 import params as P
    from cfcap.t2 import build, capability as cap2
    par = P.base_params("plated")
    R = np.load(A5.OUT / "A5_refs.npz")
    out = {}
    # A5 P0 (raw rho stored)
    D = A5._load("P0")
    for c in ("all",):
        for s in A5.SHAPES:
            r1 = D[f"{s}/T1/rho"] / R[f"{s}/T1"][None]
            out[("A5", c, s)] = (r1, D[f"{s}/{c}/rho"] / R[f"{s}/{c}"][None], D[f"{s}/{c}/rho"] / R[f"{s}/T1"][None])
    # A5m (own-ref rho_dec stored; T2 refs = rho_ref default 8 draws)
    z = np.load(A5.OUT / "A5m_rows.npz")
    for c in ("all", "motor", "all+motor"):
        for s in A5.SHAPES:
            asm = P.named_assembly(s, par)
            ref2 = cap2.rho_ref(asm, par, "plated", build.ABLATIONS[c])
            own = z[f"{s}/{c}/rho_dec"]
            out[("A5m", c, s)] = (z[f"{s}/T1/rho_dec"], own, own * ref2[None] / R[f"{s}/T1"][None])
    # A5y
    f = A5.OUT / "A5y.npz"
    if f.exists():
        y = np.load(f)
        for s in ("L3", "square", "2x3"):
            asm = P.named_assembly(s, par)
            ref2 = cap2.rho_ref(asm, par, "plated", build.ABLATIONS["all+motor"])
            own = y[f"{s}/T2"]
            out[("A5y", "all+motor", s)] = (y[f"{s}/T1"], own, own * ref2[None] / R[f"{s}/T1"][None])
    return out


def main():
    S = sets()
    rows = []
    fl_store = {}
    for (src, c, s), (r1, own, fix) in S.items():
        dc = dom_class(s)
        for label, r2 in (("own", own), ("fixed", fix)):
            fl = np.concatenate([(r1 >= t) != (r2 >= t) for t in TAUS], 1)          # (B, 2L)
            dl = np.concatenate([r2 - r1, r2 - r1], 1)
            ltt, dcc = np.concatenate([LT, LT]), np.concatenate([dc, dc])
            for lt in ("axis", "diag", "yaw"):
                m = ltt == lt
                lo, hi = boot(fl[:, m].astype(float))
                rows.append(dict(source=src, cond=c, label=label, shape=s, leg_type=lt, dominance=dcc[m][0],
                                 decisions=int(m.sum() * len(fl)), flips=int(fl[:, m].sum()), flip=float(fl[:, m].mean()),
                                 flip_lo=lo, flip_hi=hi, share_d05=float((np.abs(dl[:, m]) > 0.05).mean()),
                                 d_mean=float(dl[:, m].mean())))
            for cl in ("dominated", "undominated", "other"):
                fl_store.setdefault((src, c, label, cl), []).append(fl[:, dcc == cl])
    # pooled by dominance class (episode bootstrap over the pooled episodes of all shapes)
    pooled, verdict = [], {}
    for (src, c, label, cl), parts in fl_store.items():
        parts = [p for p in parts if p.size]
        if not parts:
            continue
        D = max(p.shape[1] for p in parts)
        # pad per-episode rates to a common width via rates: bootstrap on per-episode flip counts / decisions
        s_ = np.concatenate([p.sum(1) for p in parts]).astype(float)
        m_ = np.concatenate([np.full(len(p), p.shape[1]) for p in parts]).astype(float)
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(1000):
            k = rng.integers(0, len(s_), len(s_))
            bs.append(s_[k].sum() / m_[k].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        r = dict(source=src, cond=c, label=label, dominance=cl, decisions=int(m_.sum()), flips=int(s_.sum()),
                 flip=float(s_.sum() / m_.sum()), flip_lo=float(lo), flip_hi=float(hi))
        pooled.append(r)
    for src in ("A5m", "A5m+A5y"):
        for label in ("own", "fixed"):
            def get(cl):
                if src == "A5m":
                    return [r for r in pooled if r["source"] == "A5m" and r["cond"] == "all+motor" and r["label"] == label
                            and r["dominance"] == cl]
                parts = fl_store.get(("A5m", "all+motor", label, cl), []) + fl_store.get(("A5y", "all+motor", label, cl), [])
                parts = [p for p in parts if p.size]
                s_ = np.concatenate([p.sum(1) for p in parts]).astype(float)
                m_ = np.concatenate([np.full(len(p), p.shape[1]) for p in parts]).astype(float)
                rng = np.random.default_rng(0)
                bs = [s_[k].sum() / m_[k].sum() for k in (rng.integers(0, len(s_), len(s_)) for _ in range(1000))]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                return [dict(flip=float(s_.sum() / m_.sum()), flip_lo=float(lo), flip_hi=float(hi), decisions=int(m_.sum()))]
            u, d = get("undominated")[0], get("dominated")[0]
            verdict[f"{src}/{label}"] = dict(undominated=u["flip"], undominated_ci=[u["flip_lo"], u["flip_hi"]],
                                             dominated=d["flip"], dominated_ci=[d["flip_lo"], d["flip_hi"]],
                                             H12_supported=bool(u["flip_hi"] > 0.01 and d["flip"] <= 0.005),
                                             H12_falsified=bool(u["flip_hi"] <= 0.01))
    # E2 merges with the fixed reference
    mrows = []
    for arm in ("P0", "P5"):
        fs = sorted(f for f in A5.OUT.glob(f"A4_{arm}_s*.npz") if not f.name.endswith("_pert.npz"))
        if not fs:
            continue
        zs = [np.load(f) for f in fs]
        z = {k: np.concatenate([q[k] for q in zs]) for k in zs[0].files}
        fixed = z["T2_truth"] * z["ref2"] / z["ref1"]
        for label, tr in (("own", z["T2_truth"]), ("fixed", fixed)):
            fl = np.stack([(z["T1_comp"] >= t) != (tr >= t) for t in (0.6, 0.7, 0.8)], -1)
            mrows.append(dict(arm=arm, label=label, merges=len(tr), flip=float(fl.mean()),
                              share_d05=float((np.abs(tr - z["T1_comp"]) > 0.05).mean()),
                              mean_d=float((tr - z["T1_comp"]).mean())))
    A5.RES.mkdir(parents=True, exist_ok=True)
    for name, rr in (("N3_flips.csv", rows), ("N3_pooled.csv", pooled), ("N3_merge.csv", mrows)):
        with open(A5.RES / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    (A5.RES / "N3_verdict.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    for r in pooled:
        print(r["source"], r["cond"], r["label"], r["dominance"], r["flips"], "/", r["decisions"],
              f"{r['flip']:.4f} [{r['flip_lo']:.4f}, {r['flip_hi']:.4f}]")
    for r in mrows:
        print(r)


if __name__ == "__main__":
    main()
