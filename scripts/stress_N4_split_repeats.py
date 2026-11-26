"""N4 split-variance check (exploratory; critic A): repeat the cal/test split of the exploratory N4 pairs with fresh
random halves and report the T1 per-pair miscoverage of comp_enum (posterior lower minus the conformal margin fitted
on the cal half, alpha = 0.1), per arm. Frozen margins and confirmatory files are not touched.

    $PY scripts/stress_N4_split_repeats.py      # -> results/stress/N4_split_repeats.csv
"""
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stress_A4_merge as A4          # noqa: E402
import stress_N4 as N4                # noqa: E402

REPS = 20


def main():
    rows = []
    for arm in ("P0", "P5"):
        zs = [np.load(f) for f in sorted(A4.OUT.glob(f"N4_{arm}_s*.npz"))]
        z = {k: np.concatenate([q[k] for q in zs]) for k in zs[0].files}
        pair, lo, tr = z["pair"], z["enum_lower"], z["truth_T1"]
        lo = np.where(np.isfinite(lo), lo, -np.inf)
        up = np.unique(pair)
        short = {p: float(np.maximum(lo[pair == p] - tr[pair == p], 0).max()) for p in up}
        miss = {}
        for r in range(REPS + 1):
            if r == 0:
                cal = up[up % 2 == 0]                                    # the reported parity split (one direction)
            else:
                cal = np.random.default_rng([4242, r, arm == "P5"]).permutation(up)[:len(up) // 2]
            test = np.setdiff1d(up, cal)
            q = N4._conf_q(short, cal)
            mis = float(np.mean([(tr[pair == p] < (lo[pair == p] - q)).any() for p in test]))
            rows.append(dict(arm=arm, rep=r, split="parity" if r == 0 else "random", q=q, test_pairs=len(test),
                             miscov_pair=mis))
            if r:
                miss.setdefault(arm, []).append(mis)
        m = np.array(miss[arm])
        rows.append(dict(arm=arm, rep="summary", split=f"random x{REPS}", q=np.nan, test_pairs=len(up) // 2,
                         miscov_pair=float(m.mean()), mean=float(m.mean()), sd=float(m.std(ddof=1)),
                         q05=float(np.percentile(m, 5)), q95=float(np.percentile(m, 95)),
                         share_above_010=float((m > 0.10).mean())))
        print(arm, rows[-1])
    out = A4.RES / "N4_split_repeats.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
