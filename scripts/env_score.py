"""C-ENV mission scoring: env_missions runner rows -> one row per (method, mission) in the scorer format of
docs/board/paper.md §5b (claim, method, family, building, mission, shard, committed, failed, success, oracle_success,
reconfig, time_s, status).

    python scripts/env_score.py RUN_DIR [RUN_DIR ...] --out results/env/confirmatory/missions_conf_env.csv.gz \
        [--lower amortized_comp=PATH] [--lower amortized_comp_envcal=PATH] [--lower raw=PATH] [--claim C-ENV]

Truth is the per-segment ok of the rows (footprint, load and fixed-T1-reference rho_dec >= req). Plans are (a, b):
body a drives phase 0, body b phase 1 (`cfcap.env.missions.plan_options`). A method certifies segment (c, s) iff
footprint and load hold (known geometry) and its lower bound >= req; it commits the cheapest fully certified plan
(cost: 0 parent as-is, T_RECONF per reconfiguration before the start and at the site; ties -> more modules), else
abstains.
  oracle           bound = truth (commits iff some plan truly succeeds);
  nominal          bound = 1 (nominal healthy plug-in: ignores faults, floor and payload);
  biggest_body     the single body with the most modules that fits the narrowest passage and carries the load,
                   committed iff its nominal bound clears every segment;
  greedy_feasible  the smallest footprint- and load-feasible single body whose nominal bound clears each segment;
  <name>=PATH      any method with per-row lower bounds: a CSV / npz with columns family, mission, cand, seg, lower
                   (rho_dec units, for the segment's leg under its condition); e.g. amortized_comp, raw (models).
Do not run on confirmatory rows before 2026-11-26 01:45Z (docs/CONFIRMATORY.md).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cfcap.env import buildings as B  # noqa: E402

from cfcap.env.missions import T_RECONF  # noqa: E402
FAMNAME = {f"conf-{x}": x for x in ("E-H1", "E-H2", "E-H3", "E-H4")}
FAMNAME["conf-ID"] = "ID"
CODE2FAM = {v: k for k, v in B.FAM_CODE.items()}
EXCLUDE = {("conf-ID", 1499)}      # docs/CONFIRMATORY.md A-ENV: constructed before scoring; ID = conf-ID 0-1,498


def _manifest_shard(unit_file):
    """Runner shard id of a unit file: the run directory's manifest.json (units/../manifest.json)."""
    import json
    m = os.path.join(os.path.dirname(os.path.dirname(unit_file)), "manifest.json")
    if os.path.exists(m):
        v = json.load(open(m)).get("shard")
        if v is not None:
            return int(v)
    return None


def load_rows(dirs):
    """Runner rows of every unit under dirs. Every row carries the runner `shard` (the row column written by
    cfcap.env.task, else the run manifest); a unit whose shard cannot be determined is an error, never -1, so the
    scorer's canary-shard exclusion is verifiable."""
    parts = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "units", "*.npz"), recursive=True)):
            z = np.load(f)
            df = pd.DataFrame({k: z[k] for k in z.files})
            ms = _manifest_shard(f)
            if "shard" not in df:
                if ms is None:
                    raise SystemExit(f"{f}: no shard column and no manifest.json shard")
                df["shard"] = ms
            elif ms is not None and len(df) and not (df["shard"] == ms).all():
                raise SystemExit(f"{f}: row shard {sorted(df['shard'].unique())} != manifest shard {ms}")
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plans(mask, site):
    C = len(mask)
    out = [(a, a) for a in range(C)]
    if site >= 0:
        out += [(a, b) for a in range(C) for b in range(C)
                if a != b and ((mask[b] & ~mask[a]) == 0 or (mask[a] & ~mask[b]) == 0)]
    return out


def choose(ok, kind, nmod, phase, mask, site, singles_only=False):
    """Cheapest plan whose segments are all ok under `ok` (C, S); None if none."""
    best, key = None, None
    for a, b in plans(mask, site):
        if singles_only and a != b:
            continue
        if not (ok[a, phase == 0].all() and ok[b, phase == 1].all()):
            continue
        cost = (0 if kind[a] == 0 else T_RECONF) + (0 if a == b else T_RECONF)
        k = (cost, -min(nmod[a], nmod[b]))
        if key is None or k < key:
            best, key = (a, b), k
    return best, (key[0] if key else 0.0)


def score_mission(g, lowers):
    C, S = int(g.cand.max()) + 1, int(g.seg.max()) + 1
    grid = lambda col, dt=float: g.pivot(index="cand", columns="seg", values=col).to_numpy(dt)
    per_c = g.drop_duplicates("cand").sort_values("cand")
    per_s = g.drop_duplicates("seg").sort_values("seg")
    kind, nmod = per_c.cand_kind.to_numpy(), per_c.cand_n_mod.to_numpy()
    mask = per_c.cand_mask.to_numpy().astype(np.int64)
    fpl = per_c.cand_footprint_ok.to_numpy(bool) & per_c.cand_load_ok.to_numpy(bool)
    phase = per_s.seg_phase.to_numpy()
    site = float(g.site.iloc[0])
    truth_ok = grid("seg_ok", bool)
    geo = grid("fp_ok", bool) & grid("load_ok", bool)
    req = grid("req")
    nominal_ok = geo & (1.0 >= req)
    out = {}
    o_plan, o_cost = choose(truth_ok, kind, nmod, phase, mask, site)
    out["oracle"] = (o_plan, o_cost)
    out["nominal"] = choose(nominal_ok, kind, nmod, phase, mask, site)
    feas = np.nonzero(fpl)[0]
    if feas.size:
        c = feas[np.argmax(nmod[feas])]
        out["biggest_body"] = (((c, c), 0.0 if kind[c] == 0 else T_RECONF) if nominal_ok[c].all() else (None, 0.0))
        cs = feas[np.argsort(nmod[feas], kind="stable")]
        pick = next((c for c in cs if nominal_ok[c].all()), None)
        out["greedy_feasible"] = (((pick, pick), 0.0 if kind[pick] == 0 else T_RECONF) if pick is not None
                                  else (None, 0.0))
    else:
        out["biggest_body"] = out["greedy_feasible"] = (None, 0.0)
    key = (int(g.family.iloc[0]), int(g.mission.iloc[0]))
    for name, lw in lowers.items():
        sub = lw.get(key)
        if sub is None:
            continue
        lb = np.full((C, S), -np.inf)
        lb[sub[:, 0].astype(int), sub[:, 1].astype(int)] = sub[:, 2]
        out[name] = choose(geo & (np.nan_to_num(lb, nan=-np.inf) >= req), kind, nmod, phase, mask, site)
    o_success = o_plan is not None
    rows = []
    for meth, (plan, cost) in out.items():
        committed = plan is not None
        failed = committed and not (truth_ok[plan[0], phase == 0].all() and truth_ok[plan[1], phase == 1].all())
        reconfig = committed and not (plan[0] == plan[1] and kind[plan[0]] == 0)
        rows.append(dict(method=meth, committed=int(committed), failed=int(failed),
                         success=int(committed and not failed), oracle_success=int(o_success),
                         reconfig=int(reconfig), time_s=float(cost) if committed else 0.0))
    return rows


def load_lower(path):
    df = pd.read_csv(path) if path.endswith((".csv", ".gz")) else pd.DataFrame(dict(np.load(path)))
    if df.family.dtype == object:
        df["family"] = df.family.map(B.FAM_CODE)
    return {k: v[["cand", "seg", "lower"]].to_numpy(float) for k, v in df.groupby(["family", "mission"])}


def _one_shard(g):
    sh = g["shard"].unique()
    if len(sh) != 1:
        raise SystemExit(f"mission {int(g.mission.iloc[0])}: rows from shards {sorted(sh)}")
    return int(sh[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lower", action="append", default=[])
    ap.add_argument("--claim", default=None, help="keep rows of this claim only (C-ENV / exploratory)")
    a = ap.parse_args(argv)
    df = load_rows(a.dirs)
    if a.claim is not None and "claim" in df:
        df = df[df["claim"] == a.claim]
    lowers = {k: load_lower(v) for k, v in (x.split("=", 1) for x in a.lower)}
    out = []
    for (fam, mis), g in df.groupby(["family", "mission"]):
        fname = CODE2FAM[int(fam)]
        if (fname, int(mis)) in EXCLUDE:
            continue
        for r in score_mission(g, lowers):
            b = int(g.building.iloc[0])
            out.append(dict(claim=str(g["claim"].iloc[0]) if "claim" in g else "exploratory", family=FAMNAME.get(fname, fname),
                            building=f"{fname}-{b}", mission=int(mis),
                            shard=_one_shard(g),
                            status=str(g["status"].iloc[0]) if "status" in g else "exploratory", **r))
    res = pd.DataFrame(out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res.to_csv(a.out, index=False)
    if len(res):
        print(res.groupby(["family", "method"])[["committed", "failed", "success", "oracle_success", "reconfig"]]
              .mean().round(3).to_string())


if __name__ == "__main__":
    main()
