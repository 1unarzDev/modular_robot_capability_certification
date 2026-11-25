"""Tiny mock CSVs that follow every input contract (docs/board/paper.md), for tests and for a dry paper build.

    python scripts/analysis/mock.py OUTDIR          # writes OUTDIR/<role>/paper/*.csv
    python scripts/analysis/make_all.py --results OUTDIR --paper /tmp/cfcap-paper-mock

Mock numbers are random and must never reach paper/numbers.tex: make_all refuses a --results dir containing
MOCK.txt unless --paper points outside the repo's paper/.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

SHAPES = ("pair", "square", "2x3")
METHODS = [  # (method, route, posterior, train, noise sd of point, scale)
    ("nominal", "none", "none", "T1", 0.10, np.nan),
    ("rls_interval", "marginal", "none", "T1", 0.06, 0.06),
    ("capxfer", "raw", "none", "T1", 0.05, np.nan),
    ("nuisance_smc", "simultaneous", "smc", "T1", 0.02, 0.02),
    ("amortized_sim", "simultaneous", "amortized", "wideDR", 0.025, 0.025),
    ("amortized_comp", "compositional", "amortized", "wideDR", 0.03, 0.03),
    ("oracle", "none", "oracle", "T1", 0.0, 1e-6),
]


def dec_rows(rng, method, route, post, train, sd, sc, plant, split, n_ep=24, eid0=0):
    rows = []
    for e in range(n_ep):
        shape = SHAPES[e % 3]
        n_c = {"pair": 3, "square": 6, "2x3": 9}[shape]
        for c in range(n_c):
            kind = "merge" if c == n_c - 1 else ("self" if c == 0 else "child")
            size = 1 + (c % 4)
            hf = int(rng.random() < 0.4)
            for leg in range(10):
                truth = float(np.clip(rng.uniform(0.5, 1.0) - 0.2 * hf * rng.random(), 0, 1))
                shift = 0.03 if plant == "T2" else 0.0
                point = truth + shift + sd * rng.standard_normal()
                s = sc if np.isfinite(sc) else np.nan
                lower = point - (1.645 * s if np.isfinite(s) else 0.08)
                rows.append(dict(method=method, route=route, posterior=post, train=train, test_plant=plant,
                                 split=split, status="exploratory", eid=eid0 + e, shape=shape,
                                 family=f"fam{size}", cand=c, kind=kind, cand_size=size, has_fault=hf, leg=leg,
                                 truth=truth, truth_rel=min(1.0, truth / 0.97), rho_ref=0.97,
                                 point=point, scale=s, lower=lower, alpha=0.1,
                                 seconds=0.003 if post == "amortized" else 4.0))
    return rows


def write(out: pathlib.Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = pathlib.Path(out)
    (out / "MOCK.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "MOCK.txt").write_text("mock data; never cite\n")
    for role, sel in (("baselines", ("nominal", "rls_interval", "capxfer", "nuisance_smc", "oracle")),
                      ("models", ("amortized_sim", "amortized_comp"))):
        rows = []
        for m in METHODS:
            if m[0] not in sel:
                continue
            for plant in ("T1", "T2"):
                for split in ("cal", "test", "H3", "H4"):
                    if plant == "T2" and split == "cal":
                        continue
                    r = np.random.default_rng(hash((m[0], plant, split)) % 2**32)
                    rows += dec_rows(r, *m, plant, split, eid0={"cal": 0, "test": 1000, "H3": 2000,
                                                                 "H4": 3000}[split])
        p = out / role / "paper"
        p.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(p / f"dec_{role}.csv", index=False)
    # probe policies
    rows = []
    for pol, red, t in (("fixed_v2", 0.0, 12), ("random", 0.1, 20), ("eig", 0.3, 20), ("epig", 0.4, 20),
                        ("evoi", 0.6, 18), ("always_split", 0.5, 42)):
        for e in range(30):
            nb = int(rng.integers(0, 6))
            na = int(round(nb * (1 - red)))
            nd, ncap = 60, int(rng.integers(30, 55))
            ncr, nir = int(rng.integers(5, 15)), int(rng.integers(3, 10))
            rows.append(dict(policy=pol, split="test", status="exploratory", eid=e, shape=SHAPES[e % 3],
                             n_probes=1 if pol == "fixed_v2" else 2, probe_time_s=t, action="commit", n_dec=nd,
                             n_capable=ncap, n_incapable=nd - ncap, n_certified=ncap - na,
                             n_unsafe=int(rng.random() < 0.05), n_amb_before=nb, n_amb_after=na,
                             utility=float(ncap - na - 0.01 * t), wall_s=0.5, test_plant="T1",
                             n_capable_rel=ncr, n_incapable_rel=nir, n_certified_rel=max(0, ncr - na),
                             n_unsafe_rel=int(rng.random() < 0.05)))
    p = out / "probe" / "paper"
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p / "probe_policies.csv", index=False)
    # ambiguity vs heterogeneity
    rows = []
    for d in (0.02, 0.04, 0.08, 0.12, 0.16):
        for src, a in (("theory_linear", 0.012 * d), ("T0", 0.011 * d), ("T1", 0.014 * d + 1e-4)):
            rows.append(dict(source=src, delta=d, subset="all", n_dec=20000, amb_share=a, lo=a * 0.8,
                             hi=a * 1.2, status="exploratory", plant="T1" if src == "T1" else "T0"))
    p = out / "nuisance" / "paper"
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p / "amb_het.csv", index=False)
    # diversity grid
    rows = []
    for div, nf, mx in (("n2", 2, 2), ("n4", 8, 4), ("n6", 40, 6)):
        for seed_ in (1, 2, 3):
            for tm, tn, ho in (("pair", 2, 0), ("square", 4, 0), ("2x3", 6, 0), ("T5", 5, 1), ("bb3x3", 9, 1)):
                v = 0.95 - 0.04 * max(0, tn - mx) + 0.01 * rng.standard_normal()
                for metric, val in (("cert_frac", v), ("unsafe_commit", 0.01 + 0.005 * max(0, tn - mx)),
                                    ("pinball", 0.004 + 0.001 * max(0, tn - mx))):
                    rows.append(dict(arch="st", diversity=div, n_train_families=nf, max_train_n=mx, seed=seed_,
                                     test_morph=tm, test_n=tn, heldout=ho, metric=metric, value=val,
                                     status="exploratory", split="test"))
    p = out / "models" / "paper"
    pd.DataFrame(rows).to_csv(p / "diversity.csv", index=False)
    rows = [dict(method=m, shape=s, n_wheels=w, sec_per_decision=t * w / 8, hardware="RTX 3090",
                 status="exploratory", n=100)
            for m, t in (("amortized_sim", 0.003), ("nuisance_smc", 4.0), ("enum_known", 3.9))
            for s, w in (("pair", 8), ("square", 16), ("2x3", 24))]
    pd.DataFrame(rows).to_csv(p / "latency.csv", index=False)
    return out


if __name__ == "__main__":
    print(write(pathlib.Path(sys.argv[1])))
