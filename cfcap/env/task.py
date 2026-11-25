"""Runner task: HM3D-calibrated building missions with closed-loop T1 truth per (mission, candidate, segment).

    python -m cfcap.runner env_missions --task-spec cfcap.env.task:TASK --shard 0 --n-shards 1 \
        --families test,E-H1 --missions 2000 --per-unit 100

One unit = missions [lo, hi) of one family (ids `env-<family>-<lo>-<hi>`; units of all families are dealt round-robin
over the shards). A unit is a pure function of (family, mission index, plate): buildings by building seed
(`cfcap.env.buildings`), bodies from the generator episode stream (`cfcap.env.missions.body_episode`), labels by the
generator's `simrun.label_episodes` under each segment condition. Rows: one per (mission, candidate, segment), the
mission decision record of `cfcap.env.missions.record` flattened (docs/board/env.md, Interface).
"""
from __future__ import annotations

import time

import numpy as np

from cfcap.runner import Task

FAM_ARG = "train,cal,test,E-H1,E-H2,E-H3,E-H4"


def _args(p):
    p.add_argument("--families", default=FAM_ARG)
    p.add_argument("--missions", type=int, default=1000, help="missions per family")
    p.add_argument("--per-unit", type=int, default=100)
    p.add_argument("--plate", default="plated")
    p.add_argument("--engine", default="compile")
    p.add_argument("--max-wheels", type=int, default=1 << 17)


def _units(a, shard, n_shards):
    us = []
    for f in a.families.split(","):
        us += [f"env-{f}-{lo:06d}-{min(lo + a.per_unit, a.missions):06d}" for lo in range(0, a.missions, a.per_unit)]
    return us[shard::n_shards]


def flatten(recs):
    """Mission records -> columnar rows (mission, candidate, segment)."""
    from cfcap.env import buildings as B
    from cfcap.env import missions as M
    cols = {k: [] for k in ("family", "mission", "building", "body_index", "route_m", "narrowest", "payload_kg",
                            "cand", "cand_kind", "cand_n_mod", "cand_width", "cand_load_ok", "cand_footprint_ok",
                            "mission_ok", "phase0_ok", "phase1_ok", "seg", "n_seg", "seg_leg", "seg_kind",
                            "seg_phase", "seg_cond", "seg_mu", "seg_loaded", "seg_door_w", "seg_narrow", "seg_v_req",
                            "req", "truth", "rho_ref", "fp_ok", "load_ok", "seg_ok", "relevant")}
    for r in recs:
        C, S = r["req"].shape
        if C == 0 or S == 0:
            continue
        cc, ss = np.meshgrid(np.arange(C), np.arange(S), indexing="ij")
        cc, ss = cc.ravel(), ss.ravel()
        full = lambda v, dt: np.full(C * S, v, dt)
        cols["family"].append(full(B.FAM_CODE[r["family"]], np.int8))
        cols["mission"].append(full(r["mission"], np.int32))
        cols["building"].append(full(r["building"], np.int32))
        cols["body_index"].append(full(r["body_index"], np.int64))
        cols["route_m"].append(full(r["route_m"], np.float32))
        cols["narrowest"].append(full(r["narrowest"], np.float32))
        cols["payload_kg"].append(full(r["payload_kg"], np.float32))
        cols["cand"].append(cc.astype(np.int16))
        for k in ("cand_kind", "cand_n_mod", "cand_width", "cand_load_ok", "cand_footprint_ok", "mission_ok"):
            cols[k].append(r[k][cc])
        cols["phase0_ok"].append(r["phase_ok"][cc, 0])
        cols["phase1_ok"].append(r["phase_ok"][cc, 1])
        cols["seg"].append(ss.astype(np.int16))
        cols["n_seg"].append(full(S, np.int16))
        for k in ("seg_leg", "seg_kind", "seg_phase", "seg_cond", "seg_mu", "seg_loaded", "seg_door_w", "seg_narrow",
                  "seg_v_req"):
            cols[k].append(r[k][ss])
        cols["req"].append(r["req"][cc, ss])
        cols["truth"].append(r["seg_truth"][cc, ss])
        cols["rho_ref"].append(r["rho_ref"][cc, r["seg_leg"][ss]])
        for k in ("fp_ok", "load_ok", "seg_ok"):
            cols[k].append(r[k][cc, ss])
        cols["relevant"].append(M.relevant(r)[cc, ss])
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}


def _run(unit, ctx):
    from cfcap.env import buildings as B
    from cfcap.env import missions as M
    a = ctx.args
    _, fam_lo_hi = unit.split("-", 1)
    fam, lo, hi = fam_lo_hi.rsplit("-", 2)
    t0 = time.perf_counter()
    ctx.progress(0.0, phase=f"env {fam} {lo}-{hi}")
    ms = [B.make_mission(fam, i) for i in range(int(lo), int(hi))]
    t1 = time.perf_counter()
    stats = {}
    recs = M.label_missions(ms, plate=a.plate, engine=a.engine, max_wheels=a.max_wheels, stats=stats)
    t2 = time.perf_counter()
    rows = flatten(recs)
    ctx.meta.update(dict(family=fam, missions=len(ms), rows=int(rows["req"].size),
                         relevant=int(rows["relevant"].sum()), feasible=int((rows["fp_ok"] & rows["load_ok"]).sum()),
                         mission_ok_any=int(sum(bool(r["mission_ok"].any()) for r in recs)),
                         build_s=t1 - t0, label_s=t2 - t1, conds=int(sum(len(r["cond_mu"]) for r in recs)),
                         cands=int(sum(len(r["cand_kind"]) for r in recs))))
    ctx.progress(1.0)
    return rows


def _summary(recs, a):
    m = [r.get("meta", {}) for r in recs]
    out = {"env.missions": sum(x.get("missions", 0) for x in m), "env.rows": sum(x.get("rows", 0) for x in m),
           "env.relevant": sum(x.get("relevant", 0) for x in m), "env.feasible": sum(x.get("feasible", 0) for x in m),
           "env.label_s": sum(x.get("label_s", 0) for x in m)}
    for f in a.families.split(","):
        out[f"env.relevant.{f}"] = sum(x.get("relevant", 0) for x in m if x.get("family") == f)
    return out


TASK = Task("env_missions", _units, _run, _args, _summary, ("numpy", "torch"),
            "HM3D-calibrated building missions: per (mission, candidate, segment) T1 truth vs requirement")
