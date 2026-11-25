"""Geometry of the one Stage A2 episode drawn in Fig. 1 (teaser) -> scripts/analysis/assets/teaser_episode.json.

The episode is seed 88 of the A2 engine (cfcap/t0/stagea.run_episode, unchanged; rows of it are in
results/stageA/A2_episodes.csv and results/theory/coverage_ambiguous.csv). It is picked because the parent has six
modules and one faulted wheel, its only body with an undecided leg after v2 is the faulted module alone, and that
body's identified sets contain the true capability. The episode is replayed (same rng stream) with a read-only hook on
candidates.sample_merges so the merge shapes are the ones the engine scored; no number is computed here, only
geometry (catalog build of capxfer/assembly.py). Blender and the figure script read the JSON, not the code.

    source env.sh; $PY scripts/analysis/teaser_episode.py      # ~10 s
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

SEED = 88
OUT = pathlib.Path(__file__).resolve().parent / "assets" / "teaser_episode.json"


def asm_json(a):
    return dict(name=a.name, modules=[[m, int(i), int(j), int(k)] for m, i, j, k in a.modules],
                wheel_pos=np.round(a.wheel_pos, 6).tolist(), w_hat=np.round(a.w_hat, 6).tolist(),
                r_hat=np.round(a.r_hat, 6).tolist(), wheel_module=a.wheel_module.tolist(),
                wheel_corner=list(a.wheel_corner), module_pos=np.round(a.module_pos, 6).tolist(),
                lidar=a.lidar.astype(bool).tolist())


def main():
    from cfcap.t0 import candidates as C, qs, stagea
    from capxfer.sim import COAST, BRAKE, WEAK
    P, th = qs.load_params(), qs.load_theta()
    got = {}
    orig = C.sample_merges

    def hook(parent, other, params, rng, n_merge=10):
        out = orig(parent, other, params, rng, n_merge)
        got.update(parent=parent, other=other, merges=out)
        return out

    C.sample_merges = hook
    try:
        rec, _ = stagea.run_episode(SEED, P, th)
    finally:
        C.sample_merges = orig
    parent, other = got["parent"], got["other"]
    # the true faults, replayed exactly as scripts/theory_coverage.py:truth_of does
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from theory_coverage import truth_of
    p2, het, _, tm, tg = truth_of(SEED, P)
    assert p2.modules == parent.modules
    mode = {WEAK: "weak", COAST: "coast", BRAKE: "brake"}
    faults = [dict(wheel=int(w), module=int(parent.wheel_module[w]), corner=parent.wheel_corner[w],
                   mode=mode[int(tm[w])], g=float(tg[w])) for w in np.flatnonzero(tm)]
    kids = C.children(parent, P)
    out = dict(seed=SEED, source="cfcap/t0/stagea.run_episode (A2), replayed", het=bool(het),
               pitch_m=float(P["module"]["module_pitch_m"]), side_m=float(P["module"]["side_x_m"]),
               half_wheelbase_m=float(P["module"]["half_wheelbase_m"]), half_track_m=float(P["module"]["half_track_m"]),
               wheel_radius_m=float(P["wheel"]["radius_m"]), roller_pattern=P["wheel"]["roller_pattern"],
               rollers=int(P["wheel"]["rollers"]), lidar_modules=list(P["module"]["lidar_modules"]),
               parent=asm_json(parent), faults=faults, other=asm_json(other),
               children=[dict(name=c.name, modules=list(map(int, c.modules))) for c in kids],
               merges=[dict(name=m.name, piece=list(map(int, m.modules)),
                            cells=[[int(i), int(j)] for _, i, j, _ in m.asm.modules],
                            n_other=int(m.n_other)) for m in got["merges"]],
               a2=dict(n_children=len(kids), n_merges=len(got["merges"]), fault_type=rec["fault_type"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(OUT, out["a2"], faults)


if __name__ == "__main__":
    main()
