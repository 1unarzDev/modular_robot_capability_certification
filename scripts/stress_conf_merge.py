"""C-MERGE exporter: confirmatory stress_merge units -> results/stress/confirmatory/dec_conf_merge.csv.gz
(docs/board/paper.md §5a; docs/CONFIRMATORY.md §2). Margins are FROZEN from exploratory data
(results/stress/N4_margins.json: q_cf_all from the E2 exploratory pairs, q_pert_by_nmod from E2's T1 perturbations).
Calibration rows (split = cal, status = exploratory, test_plant = T1) come from the exploratory N4 pairs.

    $PY scripts/stress_conf_merge.py --runs DIR [DIR ...]      # only at / after the scoring window (01:45Z)
"""
import argparse
import csv
import glob
import gzip
import json
import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results/stress"
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", pathlib.Path.home() / ".cache/modular-runs/2026-11/cfcap")) / "stress"
L = 10
CONF_PAIRS = 500
COLS = ("claim", "shard", "arm", "method", "route", "posterior", "train", "test_plant", "split", "status", "eid",
        "shape", "family", "cand", "kind", "cand_size", "has_fault", "leg", "truth", "point", "scale", "lower", "alpha",
        "seconds")


def _methods(z, i, arm, M):
    qp = M[arm]["q_pert_by_nmod"]
    qcf = M[arm]["q_cf_all"]
    n = int(z["n_mod"][i])
    enum_lo = np.array([z[f"enum_lower.{l}"][i] for l in range(L)]) if "enum_lower.0" in z else z["enum_lower"][i]
    get = (lambda k: np.array([z[f"{k}.{l}"][i] for l in range(L)])) if "enum_lower.0" in z else (lambda k: z[k][i])
    pt, sc = get("enum_point"), get("enum_scale")
    lcf = enum_lo - qcf
    return {"comp_enum": (pt, sc, lcf, "compositional", "enum"),
            "comp_enum_size": (pt, sc, lcf - float(qp.get(str(n), 0.0)), "compositional", "enum"),
            "copy_parent": (get("copy_parent"), np.full(L, np.nan), get("copy_parent"), "raw", "none"),
            "detector_gated": (get("detector_gated"), np.full(L, np.nan), get("detector_gated"), "raw", "none"),
            "comp_oracle": (get("comp_oracle") if "comp_oracle.0" in z or "comp_oracle" in z else np.full(L, np.nan),
                            np.full(L, np.nan), np.full(L, np.nan), "raw", "oracle")}


def main(runs, mock=False):
    import datetime
    if not mock and datetime.datetime.now(datetime.timezone.utc) < datetime.datetime(2026, 11, 26, 1, 45,
                                                                                     tzinfo=datetime.timezone.utc):
        raise SystemExit("refusing: confirmatory labels are not opened before 2026-11-26 01:45Z (use --mock on "
                         "non-confirmatory runs)")
    M = json.loads((RES / "N4_margins.json").read_text())
    (RES / "confirmatory").mkdir(parents=True, exist_ok=True)
    out = RES / "confirmatory" / ("dec_conf_merge_mock.csv.gz" if mock else "dec_conf_merge.csv.gz")
    n = 0
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLS))
        w.writeheader()
        # confirmatory rows
        for d in runs:
            for f in sorted(glob.glob(os.path.join(d, "**/units/merge-*.npz"), recursive=True)):
                if f.endswith("-spare.npz"):
                    continue                                      # shard 0 timing spare: exploratory, never scored
                shard = int(json.loads((pathlib.Path(f).parents[1] / "manifest.json").read_text())["shard"])
                z = np.load(f)
                for i in range(len(z["pair"])):
                    if int(z["pair"][i]) >= CONF_PAIRS:          # A-POWER (77b17b0 §11): pairs 0-499 per arm only
                        continue
                    arm = str(z["arm"][i])
                    for plant in ("T1", "T2"):
                        tr = np.array([z[f"truth_{plant}.{l}"][i] for l in range(L)])
                        for m, (pt, sc, lo, route, post) in _methods(z, i, arm, M).items():
                            for l in range(L):
                                w.writerow(dict(claim="C-MERGE", shard=shard, arm=arm, method=m, route=route,
                                                posterior=post, train="T1", test_plant=plant, split=f"conf_merge_{arm}",
                                                status="confirmatory", eid=int(z["pair"][i]), shape="", family="",
                                                cand=int(z["merge"][i]), kind="merge", cand_size=int(z["n_mod"][i]),
                                                has_fault=int(z["n_fault"][i] > 0), leg=l, truth=tr[l], point=pt[l],
                                                scale=sc[l], lower=lo[l], alpha=0.1, seconds=np.nan))
                                n += 1
        # calibration rows from the exploratory N4 pairs (T1 truth)
        for arm in ("P0", "P5"):
            for f in sorted(OUT.glob(f"N4_{arm}_s*.npz")):
                z = np.load(f)
                for i in range(len(z["pair"])):
                    meth = _methods(z, i, arm, M)
                    for m in ("comp_enum_size", "copy_parent"):
                        pt, sc, lo, route, post = meth[m]
                        for l in range(L):
                            w.writerow(dict(claim="C-MERGE", shard=-1, arm=arm, method=m, route=route, posterior=post,
                                            train="T1", test_plant="T1", split="cal", status="exploratory",
                                            eid=int(z["pair"][i]), shape="", family="", cand=int(z["merge"][i]),
                                            kind="merge", cand_size=int(z["n_mod"][i]),
                                            has_fault=int(z["n_fault"][i] > 0), leg=l, truth=z["truth_T1"][i, l],
                                            point=pt[l], scale=sc[l], lower=lo[l], alpha=0.1, seconds=np.nan))
    print("wrote", out, n, "confirmatory rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--mock", action="store_true", help="non-confirmatory runs only (smoke tests)")
    a = ap.parse_args()
    main(a.runs, a.mock)
