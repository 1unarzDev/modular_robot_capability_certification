"""Empirical mission statistics of the 900 HM3D-derived building records (MRMN real_buildings study, read-only).

Data: Habitat-Matterport 3D (HM3D; Ramakrishnan et al., 2021), derived records only (grid windows, cells, sites,
counts), used under the Matterport academic terms in the source directory's DATA_TERMS.md. Nothing here identifies,
or tries to identify, any building: building ids are dropped on load and never written out.

Field meanings (MRMN `modular_robot_benchmarks/real_buildings.py`, `draw_missions` / `evaluate_mission`):
  * route_steps: moves of the shortest 8-connected lattice route start -> goal (RESOLUTION = 0.1 m grid,
    `building_mesh.py`), drawn in [20, 250];
  * route_m: metric length of that route (0.1 m per axis move, 0.1*sqrt2 per diagonal move);
  * start / goal / passage / station_cell: window-local grid cells; the passage is the first route cell the compact
    body cannot reach (the narrow passage that forces the shape change); exactly one per mission by construction;
  * result.methods.<method>.sites: the reconfiguration sites the planner placed (one per `reconfigure` segment);
  * draw.levels: building levels (each mission lies on one level).

Derived per-mission statistics (the calibration targets of `cfcap.env.buildings`):
  route_m, route_steps, m_per_step (route_m / steps: diagonal share of the route), tortuosity (route_m / straight
  start-goal distance), passage_pos (straight-line fraction start->passage of start->passage->goal), station_m
  (station to passage distance), n_sites (geometry_coupled reconfiguration sites; 0 where unplanned). Per building:
  levels, missions.

    python -m cfcap.env.hm3d_stats        # -> results/env/hm3d_stats.json
"""
from __future__ import annotations

import json
import math
import os
import pathlib

import numpy as np

SRC = pathlib.Path(os.environ.get("MRMN_REAL_BUILDINGS", pathlib.Path.home() / "modular_robot_morphology_aware_navigation"
                                  / "studies/real_buildings/real_buildings_20261115.json"))
RESOLUTION = 0.1                       # m per grid cell (MRMN building_mesh.RESOLUTION)
SITE_METHOD = "geometry_coupled"
MISSION_STATS = ("route_m", "route_steps", "m_per_step", "tortuosity", "passage_pos", "station_m", "n_sites")
BUILDING_STATS = ("levels", "missions")
OUT = pathlib.Path(__file__).resolve().parents[2] / "results" / "env" / "hm3d_stats.json"
CITATION = ("Ramakrishnan, S. K., et al. (2021). Habitat-Matterport 3D Dataset (HM3D): 1000 large-scale 3D "
            "environments for embodied AI. NeurIPS Datasets and Benchmarks.")


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) * RESOLUTION


def load_records(path=SRC):
    """(missions: {stat: array}, buildings: {stat: array}); anonymous (no building ids)."""
    d = json.loads(pathlib.Path(path).read_text())
    ms = {k: [] for k in MISSION_STATS}
    bs = {k: [] for k in BUILDING_STATS}
    for b in d["results"]:
        if b.get("draw"):
            bs["levels"].append(int(b["draw"]["levels"]))
        bs["missions"].append(len(b.get("missions") or []))
        for m in b.get("missions") or []:
            s, g, p, st = m["start"][:2], m["goal"], m["passage"], m["station_cell"]
            eu = _d(s, g)
            sp, pg = _d(s, p), _d(p, g)
            ms["route_m"].append(float(m["route_m"]))
            ms["route_steps"].append(int(m["route_steps"]))
            ms["m_per_step"].append(float(m["route_m"]) / max(int(m["route_steps"]), 1))
            ms["tortuosity"].append(float(m["route_m"]) / max(eu, RESOLUTION))
            ms["passage_pos"].append(sp / (sp + pg) if sp + pg > 0 else 1.0)
            ms["station_m"].append(_d(st, p))
            meth = m.get("result", {}).get("methods", {}).get(SITE_METHOD, {})
            ms["n_sites"].append(len(meth.get("sites", [])) if meth.get("planned") else 0)
    return ({k: np.asarray(v, float) for k, v in ms.items()}, {k: np.asarray(v, float) for k, v in bs.items()})


def summary(x):
    q = np.linspace(0, 1, 101)
    return dict(n=int(x.size), mean=float(x.mean()), std=float(x.std(ddof=1)), min=float(x.min()), max=float(x.max()),
                quantiles=[float(v) for v in np.quantile(x, q)])


def fit(path=SRC):
    ms, bs = load_records(path)
    X = np.stack([np.log(ms["route_m"]), np.log(ms["tortuosity"]), ms["m_per_step"], ms["passage_pos"],
                  np.log(np.maximum(ms["station_m"], RESOLUTION))], 1)
    # Gaussian KDE (Scott bandwidth) of the joint route features: the generator samples its route knobs from it
    bw = X.shape[0] ** (-1.0 / (X.shape[1] + 4)) * X.std(0, ddof=1)
    out = dict(source="MRMN studies/real_buildings/real_buildings_20261115.json (HM3D-derived records, read-only)",
               citation=CITATION, terms="Matterport academic EULA (DATA_TERMS.md); no building identification",
               resolution_m=RESOLUTION, site_method=SITE_METHOD, n_missions=int(ms["route_m"].size),
               n_buildings=int(bs["levels"].size),
               mission={k: summary(v) for k, v in ms.items()},
               building={k: summary(v) for k, v in bs.items()},
               pmf={"n_sites": _pmf(ms["n_sites"]), "levels": _pmf(bs["levels"]), "missions": _pmf(bs["missions"])},
               kde=dict(features=["log_route_m", "log_tortuosity", "m_per_step", "passage_pos", "log_station_m"],
                        bandwidth=[float(v) for v in bw], points=np.round(X, 5).tolist()))
    return out


def _pmf(x):
    v, c = np.unique(x.astype(int), return_counts=True)
    return {str(int(a)): float(b) / x.size for a, b in zip(v, c)}


def load_fitted(path=OUT):
    return json.loads(pathlib.Path(path).read_text())


def main():
    s = fit()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(s, indent=1))
    for k, v in s["mission"].items():
        q = v["quantiles"]
        print(f"{k:12s} n={v['n']} mean={v['mean']:.3f} sd={v['std']:.3f} q05={q[5]:.3f} q50={q[50]:.3f} q95={q[95]:.3f}")
    print("pmf", s["pmf"])


if __name__ == "__main__":
    main()
