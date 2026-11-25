"""C-GATE (T2) margins and exporter (docs/CONFIRMATORY.md §3 + A-GATE; docs/board/paper.md §5a).

    $PY scripts/stress_conf_gate.py margins --runs DIR [...]   # exploratory knee cal units -> results/stress/gate_margins.json
    $PY scripts/stress_conf_gate.py export --runs DIR [...]    # confirmatory units -> confirmatory/dec_conf_gate.csv.gz
                                                               #   (refuses before 2026-11-26 01:45Z unless --mock)

Lower bounds are in rho_dec units: l0 = rho_x / rho_nom (the robot's T0 two-constant model relative to its own nominal
healthy prediction); truth = T2 rho_abs / rho_ref^T1. Margins (E1's q_sim: the conformal (1 - alpha) quantile of the
per-episode max of (l0 - truth), alpha = 0.1, cfcap.conformal.core.split_quantile), an episode being (index, severity):
  robust   pi (v2clamp) - max(0, q), q pooled over cal severities 0-3          (A-GATE primary)
  lawcal   par (v2) - q, q pooled over cal severities 0-2 only (s = 3 held out)
  par      par (probe) - q, q on cal severity 0 only (trusts the two-constant law)
  raw      par (probe), q = 0;   pi   pi (probe), q = 0
`dominated` = the closed-form Prop. 2b flag of the row's probe set (cfcap/t2/gate.py).
"""
import argparse
import csv
import datetime
import glob
import gzip
import json
import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results/stress"
ALPHA = 0.1
PKEY = {"v2": "v2", "v2+clamp": "v2clamp"}


def _load(runs, spare=False):
    parts = []
    for d in runs:
        for f in sorted(glob.glob(os.path.join(d, "**/units/gate-*.npz"), recursive=True)):
            if f.endswith("-spare.npz") and not spare:
                continue
            z = dict(np.load(f))
            man = pathlib.Path(f).parents[1] / "manifest.json"
            z["shard"] = np.full(len(z["leg"]), json.loads(man.read_text()).get("shard", -1) if man.exists() else -1)
            parts.append(z)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None


def _q(z, l0, mask):
    from cfcap.conformal.core import split_quantile
    key = z["index"][mask] * 10 + z["severity"][mask]
    s = l0[mask] - z["truth"][mask]
    u, inv = np.unique(key, return_inverse=True)
    mx = np.full(len(u), -np.inf)
    np.maximum.at(mx, inv, s)
    return float(split_quantile(mx, ALPHA))


def margins(runs):
    z = _load(runs)
    out = {}
    for probe, pk in PKEY.items():
        par = z[f"rho_par_{pk}"] / z["rho_nom"]
        pi = z[f"rho_pi_{pk}"] / z["rho_nom"]
        sev = z["severity"]
        out[probe] = dict(q_par=_q(z, par, sev == 0), q_par_pooled_s012=_q(z, par, sev <= 2),
                          q_pi_pooled=max(0.0, _q(z, pi, np.ones_like(sev, bool))))
    out["_source"] = dict(runs=[str(r) for r in runs], episodes=int(len(np.unique(z["index"]))), alpha=ALPHA,
                          note="exploratory knee cal episodes (stress_gate --split cal --root 92599,97)")
    (RES / "gate_margins.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


COLS = ("claim", "shard", "severity", "probe", "dominated", "method", "route", "posterior", "train", "test_plant",
        "split", "status", "eid", "shape", "family", "cand", "kind", "cand_size", "has_fault", "leg", "truth", "point",
        "scale", "lower", "alpha", "seconds")
KIND = {0: "self", 1: "child", 2: "merge"}


def export(runs, mock=False):
    if not mock and datetime.datetime.now(datetime.timezone.utc) < datetime.datetime(2026, 11, 26, 1, 45,
                                                                                     tzinfo=datetime.timezone.utc):
        raise SystemExit("refusing: confirmatory labels are not opened before 2026-11-26 01:45Z")
    M = json.loads((RES / "gate_margins.json").read_text())
    z = _load(runs)
    (RES / "confirmatory").mkdir(parents=True, exist_ok=True)
    out = RES / "confirmatory" / ("dec_conf_gate_mock.csv.gz" if mock else "dec_conf_gate.csv.gz")
    n = 0
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLS))
        w.writeheader()
        for probe, pk in PKEY.items():
            par = z[f"rho_par_{pk}"] / z["rho_nom"]
            pi = z[f"rho_pi_{pk}"] / z["rho_nom"]
            m = M[probe]
            meths = {"par": (par, par - m["q_par"]), "raw": (par, par), "pi": (pi, pi)}
            if probe == "v2+clamp":
                meths["robust"] = (pi, pi - m["q_pi_pooled"])
            if probe == "v2":
                meths["lawcal"] = (par, par - m["q_par_pooled_s012"])
            dom = z[f"dom_{pk}"].astype(int)
            for meth, (pt, lo) in meths.items():
                for i in range(len(z["leg"])):
                    w.writerow(dict(claim="C-GATE", shard=int(z["shard"][i]), severity=int(z["severity"][i]),
                                    probe=probe, dominated=int(dom[i]), method=meth,
                                    route="simultaneous" if meth in ("par", "robust", "lawcal") else "raw",
                                    posterior="none" if meth in ("pi", "robust") else "par", train="T1",
                                    test_plant="T2", split="conf_test", status="confirmatory", eid=int(z["index"][i]),
                                    shape="", family="", cand=int(z["cand"][i]), kind=KIND.get(int(z["kind"][i]), ""),
                                    cand_size=int(z["cand_size"][i]), has_fault=int(z["has_fault"][i]),
                                    leg=int(z["leg"][i]), truth=float(z["truth"][i]), point=float(pt[i]),
                                    scale=np.nan, lower=float(lo[i]), alpha=ALPHA, seconds=np.nan))
                    n += 1
    print("wrote", out, n, "rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("margins", "export"))
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    margins(a.runs) if a.cmd == "margins" else export(a.runs, a.mock)
