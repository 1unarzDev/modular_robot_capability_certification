"""Independent exploratory fixed-reference audit; no stress-analysis imports.

Run (CPU only): source env.sh; PYTHONDONTWRITEBYTECODE=1 scripts/heavy.sh "$PY" scripts/codex/n3_fixedref.py
Only writes results/codex/n3_fixedref.csv. Saved NPZ archives supplement lossy CSV
exports: A5 absolute rho, and A4 original per-merge ref1/ref2. No corpus access.
A4 geometry keys were not saved: its shape strata explicitly denote module count.
Dominated = surge/sway. Undominated = diagonals, plus yaw for >=3-module
children. Yaw for smaller shapes is unclassified.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cfcap-n3-fixedref-numba")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

SHAPES = ("single", "pair", "L3", "square", "2x3")
SIZES = dict(zip(SHAPES, (1, 2, 3, 4, 6)))
LEGS = ("surge+", "surge-", "sway+", "sway-", "yaw+", "yaw-", "diag++", "diag+-", "diag-+", "diag--")
TAUS = (0.7, 0.8)


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def leg_type(leg):
    return "yaw" if leg.startswith("yaw") else "diag" if leg.startswith("diag") else "axis"


def summarize(dataset, arm, cond, blocks, sources, nboot):
    """Paired cluster bootstrap, resampling whole episodes/pairs with all decisions.

    A cluster contributes its sums and counts, so unequal numbers of A4 merges
    retain the decision-weighted estimand. Same draws for both labels and metrics.
    """
    a = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    rows = []
    for shape in ["all", "children", "3+mod", *sorted(set(a["shape"]))]:
        if shape == "all":
            sm = np.ones(len(a["r1"]), bool)
        elif shape == "children":
            sm = a["size"] > 1
        elif shape == "3+mod":
            sm = a["size"] >= 3
        else:
            sm = a["shape"] == shape
        for lt in ("any", "axis", "diag", "yaw"):
            lm = sm & ((a["leg_type"] == lt) if lt != "any" else True)
            for dom in ("any", "dominated", "undominated", "unclassified"):
                mask = lm & ((a["dominance"] == dom) if dom != "any" else True)
                if not mask.any():
                    continue
                _, cl = np.unique(a["cluster"][mask], return_inverse=True)
                nc = cl.max() + 1
                counts = np.bincount(cl)
                # Multinomial draws are equivalent to sampling nc clusters with replacement.
                weights = np.random.default_rng(90209).multinomial(nc, np.full(nc, 1 / nc), size=nboot)
                denom = weights @ counts
                r1, old, fixed = (a[k][mask] for k in ("r1", "old", "fixed"))
                for tau_name, taus in (("pooled", TAUS), ("0.7", (0.7,)), ("0.8", (0.8,))):
                    row = dict(status="exploratory", dataset=dataset, arm=arm, condition=cond,
                               shape=shape, leg_type=lt, dominance=dom, tau=tau_name,
                               clusters=nc, leg_rows=len(r1), decisions=len(r1) * len(taus),
                               bootstrap_replicates=nboot, bootstrap_seed=90209,
                               sources=sources,
                               shape_note="module-count strata; geometry not saved" if dataset == "A4" else "named geometry")
                    for label, r2 in (("own", old), ("fixed", fixed)):
                        flip = np.mean([(r1 >= t) != (r2 >= t) for t in taus], axis=0)
                        shift = (np.abs(r2 - r1) > 0.05).astype(float)
                        unsafe = np.mean([(r1 >= t) & (r2 < t) for t in taus], axis=0)
                        for metric, values in (("flip", flip), ("share_absd_gt_005", shift), ("unsafe", unsafe)):
                            samples = (weights @ np.bincount(cl, weights=values, minlength=nc)) / denom
                            lo, hi = np.quantile(samples, (0.025, 0.975))
                            row[f"{label}_{metric}"] = values.mean()
                            row[f"{label}_{metric}_lo"] = lo
                            row[f"{label}_{metric}_hi"] = hi
                    # Paired change is useful when own and fixed classifications differ.
                    delta = np.mean([(r1 >= t) != (fixed >= t) for t in taus], axis=0) - np.mean(
                        [(r1 >= t) != (old >= t) for t in taus], axis=0)
                    bs = (weights @ np.bincount(cl, weights=delta, minlength=nc)) / denom
                    row.update(flip_change=delta.mean(), flip_change_lo=np.quantile(bs, .025),
                               flip_change_hi=np.quantile(bs, .975))
                    rows.append(row)
    return rows


def block(r1, old, fixed, shape, size, clusters):
    r1, old, fixed = [np.asarray(x, float) for x in (r1, old, fixed)]
    assert r1.shape == old.shape == fixed.shape and r1.ndim == 2 and r1.shape[1] == len(LEGS)
    assert all(np.isfinite(x).all() for x in (r1, old, fixed))
    n = len(r1)
    sizes = np.broadcast_to(size, (n,))
    lt = np.tile([leg_type(l) for l in LEGS], n)
    sz = np.repeat(sizes, len(LEGS))
    dom = np.where(lt == "axis", "dominated",
                   np.where((lt == "diag") | ((lt == "yaw") & (sz >= 3)), "undominated", "unclassified"))
    return dict(r1=r1.ravel(), old=old.ravel(), fixed=fixed.ravel(),
                shape=np.repeat(np.broadcast_to(shape, (n,)), len(LEGS)), size=sz,
                cluster=np.repeat(clusters, len(LEGS)), leg_type=lt, dominance=dom)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=Path(os.environ.get("CFCAP_OUT", Path.home() / ".cache/modular-runs/2026-11/cfcap")) / "stress")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    assert args.bootstrap > 0
    archive = args.archive
    from cfcap.t1 import params as P
    from cfcap.t1.reference import rho_ref_many
    from cfcap.t2 import build, capability as C2
    print("CPU slot acquired; checking inputs and nominal references", flush=True)

    # Verify recovery inputs before any reference simulation. Missing data is fatal.
    mpath = ROOT / "results/stress/A5m_rows.csv"
    apath = ROOT / "results/stress/A4_merge_rows.csv"
    motor_rows, merge_rows = csv_rows(mpath), csv_rows(apath)
    a5files = sorted(archive.glob("A5_P*_t1.npz"))
    a4files = sorted(p for p in archive.glob("A4_P*_s*.npz") if not p.stem.endswith("_pert"))
    if not a5files or not a4files:
        raise SystemExit("STOP: A5 absolute-rho archives or A4 per-merge reference archives missing; CSVs cannot recover them.")
    for f in a4files:
        with np.load(f) as z:
            for key in ("ref1", "ref2", "meta_pair", "meta_merge"):
                if key not in z:
                    raise SystemExit(f"STOP: {f} lacks {key}; A4 CSV omits geometry (module cells/yaws), absolute rho and references.")
    par = P.base_params("plated")
    refs = dict(zip(SHAPES, rho_ref_many([P.named_assembly(s, par) for s in SHAPES], device="cpu")))
    saved = np.load(archive / "A5_refs.npz")
    for s in SHAPES:
        np.testing.assert_allclose(refs[s], saved[f"{s}/T1"], rtol=2e-5, atol=2e-6)
    result = []
    # A5m: the producer used default rho_ref settings (seed=12345, 8 draws).
    for cond in ("all", "motor", "all+motor"):
        blocks = []
        for s in SHAPES:
            selected = [r for r in motor_rows if r["shape"] == s]
            eps = sorted(set(int(r["ep"]) for r in selected))
            lookup = {(int(r["ep"]), r["leg"]): r for r in selected}
            assert len(lookup) == len(selected) == len(eps) * len(LEGS)
            r1 = np.array([[float(lookup[e, l]["rho_dec_T1"]) for l in LEGS] for e in eps])
            old = np.array([[float(lookup[e, l][f"{cond}.rho_dec"]) for l in LEGS] for e in eps])
            # The original T1 normalization is identical to the checked table above.
            t2ref = C2.rho_ref(P.named_assembly(s, par), par, "plated", build.ABLATIONS[cond])
            fixed = old * t2ref / refs[s]
            blocks.append(block(r1, old, fixed, s, SIZES[s], [f"{s}/{e}" for e in eps]))
        sources = f"{mpath.relative_to(ROOT)};cfcap/t1/reference.py;{archive / 'A5_refs.npz'};cfcap/t2/capability.py:rho_ref(defaults)"
        result.extend(summarize("A5m", "P0", cond, blocks, sources, args.bootstrap))
        print(f"A5m {cond} completed", flush=True)
    # A5: absolute scores from each matched archive, independent of stress loaders.
    for arm in ("P0", "P1"):
        files = [f for f in a5files if f.name.startswith(f"A5_{arm}_")]
        if not files:
            continue
        by_cond, source_files = {}, []
        for f in files:
            one = np.load(f)
            partners = sorted(archive.glob(f.name.replace("_t1.npz", "_t2_*.npz")))
            if not partners:
                raise SystemExit(f"STOP: matched T2 absolute rho missing for {f}")
            source_files.extend([str(f), *map(str, partners)])
            for partner in partners:
                two = np.load(partner)
                conds = sorted({k.split('/')[1] for k in two.files if k.endswith('/rho')})
                for cond in conds:
                    for s in SHAPES:
                        raw1, raw2 = one[f"{s}/T1/rho"], two[f"{s}/{cond}/rho"]
                        r1 = raw1 / refs[s]
                        old = raw2 / saved[f"{s}/{cond}"]
                        fixed = raw2 / refs[s]
                        by_cond.setdefault(cond, []).append(block(r1, old, fixed, s, SIZES[s],
                            [f"{f.stem}/{s}/{i}" for i in range(len(r1))]))
        for cond, blocks in by_cond.items():
            result.extend(summarize("A5", arm, cond, blocks,
                ';'.join(source_files + [str(archive / 'A5_refs.npz'), 'cfcap/t1/reference.py']), args.bootstrap))
        print(f"A5 {arm} completed", flush=True)
    # A4: original ref1 is exactly its nominal T1 geometry reference by producer
    # definition, not a T2 reference. Recover rho_abs=old*ref2 and divide by ref1.
    # Match every exported CSV value to its archive using arm/pair/merge/leg.
    lookup = {(r['arm'], int(r['pair']), int(r['merge']), r['leg']): r for r in merge_rows}
    assert len(lookup) == len(merge_rows)
    matched = 0
    for arm in ("P0", "P5"):
        blocks, source_files = [], []
        for f in a4files:
            if not f.name.startswith(f"A4_{arm}_"):
                continue
            z = np.load(f)
            r1, old = z['T1_truth'], z['T2_truth']
            for i, (pair, merge) in enumerate(zip(z['meta_pair'], z['meta_merge'])):
                for j, leg in enumerate(LEGS):
                    r = lookup[arm, int(pair), int(merge), leg]
                    np.testing.assert_allclose([float(r['rho_dec_T1_truth']), float(r['rho_dec_T2_truth'])],
                                               [r1[i, j], old[i, j]], rtol=1e-12, atol=1e-12)
                    matched += 1
            assert np.all(z['ref1'] > 0) and np.all(z['ref2'] > 0)
            fixed = old * z['ref2'] / z['ref1']
            sizes = z['meta_n_mod']
            blocks.append(block(r1, old, fixed, [f"merged_n_mod_{n}" for n in sizes], sizes,
                                [f"{arm}/{p}" for p in z['meta_pair']]))
            source_files.append(str(f))
        result.extend(summarize("A4", arm, "all", blocks,
                               ';'.join([str(apath.relative_to(ROOT)), *source_files]), args.bootstrap))
        print(f"A4 {arm} completed", flush=True)
    assert matched == len(merge_rows)
    out = ROOT / "results/codex/n3_fixedref.csv"
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"Wrote {out}", flush=True)


if __name__ == '__main__':
    main()
