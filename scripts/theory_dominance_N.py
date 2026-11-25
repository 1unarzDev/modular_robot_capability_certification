"""Extrapolated (undominated) share of child decisions vs parent size N <= 16 (closed-form speed coordinate, Prop. 2b).

Random parents of N modules are grown cell by cell (uniform frontier growth, random module yaws), so large N does not
need a full polyomino enumeration. For each parent, children are sampled stratified by size: for every child size
k in 1..N-1, up to K_PER_SIZE distinct connected k-subsets (random frontier growth). A decision (child, leg) is
*extrapolated* for a probe set if some wheel's needed speed exceeds that wheel's maximum probed speed in the parent
(X-pattern closed form = T0 IK, `theory_coverage.py closed`). The torque coordinate is not included (see E1).

Shares are reported (a) uniform over child sizes 1..N-1 (each size weighted equally), (b) weighted by the sampled
children, and per leg kind; 95 % CIs by bootstrap over parents.
  python scripts/theory_dominance_N.py   -> results/theory/dominance_vs_N.csv, dominance_vs_N_by_size.csv
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "theory"
NS = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16)
N_PARENTS = 150
K_PER_SIZE = 6
LEG_KIND = np.array(["surge"] * 2 + ["sway"] * 2 + ["diag"] * 4 + ["yaw"] * 2)
SETS = ("v2", "v2+clamp", "v2+clamp+yawhi")


def grow(rng, n, cells=None, allowed=None):
    """Random connected set of n grid cells (from `allowed` if given)."""
    if allowed is not None:
        allowed = list(allowed)
        start = allowed[int(rng.integers(len(allowed)))]
        S, aset = {start}, set(allowed)
    else:
        S, aset = {(0, 0)}, None
    while len(S) < n:
        fr = sorted({(x + dx, y + dy) for x, y in S for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))} - S)
        if aset is not None:
            fr = [c for c in fr if c in aset]
        if not fr:
            return None
        S.add(fr[int(rng.integers(len(fr)))])
    return sorted(S)


def main():
    from capxfer import assembly as A
    from cfcap.t0 import qs, probesets as PS, candidates as C
    P, th = qs.load_params(), qs.load_theta()
    R = qs.sim_nominal(P)["radius"]
    rng = np.random.default_rng(16)
    rows = []
    for N in NS:
        for pi in range(N_PARENTS):
            cells = grow(rng, N)
            mods = [(str(rng.choice(("M1", "M2", "M3", "M4"))), i, j, int(rng.integers(4))) for i, j in cells]
            parent = A.build(f"p{N}", mods, P)
            pl = qs.Plant(parent, P, th)
            v2 = np.abs(pl.ik(PS.v2_twists(parent))).max(0)
            cl = np.abs(np.stack([l[1] for l in PS.probe_legs(pl, "clamp")])).max(0)
            w = 2 * PS.v2_twists(parent)[4, 2]
            yh = np.abs(pl.ik(np.array([[0, 0, w], [0, 0, -w]]))).max(0)
            vmax = {"v2": v2, "v2+clamp": np.maximum(v2, cl), "v2+clamp+yawhi": np.maximum.reduce([v2, cl, yh])}
            pcells = C.cells_of(parent)
            idx_of = {c: k for k, c in enumerate(pcells)}
            for k in range(1, N):
                seen = set()
                for _ in range(4 * K_PER_SIZE):
                    if len(seen) >= K_PER_SIZE:
                        break
                    sub = grow(rng, k, allowed=pcells)
                    if sub is None or tuple(sub) in seen:
                        continue
                    seen.add(tuple(sub))
                    mi = [idx_of[c] for c in sub]
                    child = A.build("c", [parent.modules[m] for m in mi], P)
                    src = C.wheel_index(parent, mi)
                    om = np.abs(qs.ik(child, qs.decision_legs(child), R))       # (10, n_c)
                    rec = dict(N=N, parent=pi, child_size=k)
                    for s in SETS:
                        und = (om > vmax[s][src][None] * (1 + 1e-9)).any(1)
                        for lk in ("surge", "sway", "diag", "yaw"):
                            rec[f"{s}|{lk}"] = und[LEG_KIND == lk].mean()
                        rec[f"{s}|all"] = und.mean()
                    rows.append(rec)
        print("N", N, len(rows), flush=True)
    D = pd.DataFrame(rows)
    out, by = [], []
    boot = np.random.default_rng(1)
    for N, d in D.groupby("N"):
        per_size = d.groupby(["parent", "child_size"]).mean(numeric_only=True)
        for s in SETS:
            for lk in ("all", "surge", "sway", "diag", "yaw"):
                col = f"{s}|{lk}"
                ps_uniform = per_size[col].groupby(level="parent").mean()        # uniform over child sizes
                ps_sample = d.groupby("parent")[col].mean()
                bs = [ps_uniform.sample(len(ps_uniform), replace=True, random_state=int(boot.integers(1e9))).mean()
                      for _ in range(400)]
                out.append(dict(N=N, probe_set=s, leg_kind=lk, extrapolated_share_uniform_size=ps_uniform.mean(),
                                ci_lo=np.quantile(bs, 0.025), ci_hi=np.quantile(bs, 0.975),
                                extrapolated_share_sampled=ps_sample.mean(), n_parents=len(ps_uniform),
                                n_children=len(d)))
        for (k,), g in d.groupby(["child_size"]):
            for s in SETS:
                by.append(dict(N=N, child_size=k, probe_set=s, extrapolated_share=g[f"{s}|all"].mean(),
                               yaw=g[f"{s}|yaw"].mean(), diag=g[f"{s}|diag"].mean(), n_children=len(g)))
    RES.mkdir(parents=True, exist_ok=True)
    O = pd.DataFrame(out)
    O.to_csv(RES / "dominance_vs_N.csv", index=False)
    pd.DataFrame(by).to_csv(RES / "dominance_vs_N_by_size.csv", index=False)
    print(O[O.leg_kind.isin(["all", "yaw"])].pivot_table(index="N", columns=["probe_set", "leg_kind"],
                                                          values="extrapolated_share_uniform_size").round(3))


if __name__ == "__main__":
    main()
