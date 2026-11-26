"""Choose and replay the one confirmatory C-MAIN episode drawn in Fig. 1 -> scripts/analysis/assets/teaser_episode.json.

Selection (deterministic, from the confirmatory rows only; nothing here is scored or reported as a rate):
  rows   results/models/confirmatory/dec_conf_models.csv.gz, claim C-MAIN, split conf_test, the scored prefix
         (confirmatory.CMAIN_IDX), method Ours (confirmatory.OURS), test plant T1;
  jobs   results/annotations/task_library.csv jobs any body may take (min_modules = 1) that need surge, a diagonal
         and yaw; tau of leg l = the job's r_<l> (rho_dec units, the paper's label), legs in common.LEGS order;
  verdict per (body, leg): certify if lower >= tau, probe if point >= tau > lower, else refuse; a body certifies a
         job if every needed leg certifies, refuses if any leg refuses, else probes;
  keep   six-module parents with a faulted wheel and >= 1 merge, (episode, job) pairs with no unsafe certify
         (certified leg with truth < tau) and all three body verdicts present;
  rank   most certified bodies, then lowest eid, then task id. Panel (c) shows the episode's refused body and its
         smallest probe body (lowest candidate index on ties).
The episode's geometry is replayed read-only from the generator (cfcap.gen.episode.make_episode, plate 'plated', the
same call baselines/episode.py:from_gen makes; the candidate order is checked against the rows' kind / size / fault
flag). Blender (render_modules.py) and the figure (fig_teaser.py) read the JSON, not the code.

    source env.sh; $PY scripts/analysis/teaser_episode.py      # ~40 s (reads the 3.3 M-row export once)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

OUT = HERE / "assets" / "teaser_episode.json"
DEC = "models/confirmatory/dec_conf_models.csv.gz"
TASKS = "annotations/task_library.csv"
PLATE = "plated"
COLS = ["claim", "method", "split", "test_plant", "eid", "shape", "parent_n", "cand", "kind", "cand_size",
        "has_fault", "leg", "truth", "point", "lower"]


def cmain_rows(res):
    import confirmatory as CF
    d = pd.read_csv(res / DEC, usecols=COLS)
    d = d[(d.claim == "C-MAIN") & (d.split == "conf_test") & (d.method == CF.OURS) & (d.test_plant == "T1")
          & d.eid.between(CF.CMAIN_IDX[0], CF.CMAIN_IDX[1] - 1)]
    return d.reset_index(drop=True)


def job_taus(row):
    """{leg index: tau} for one task-library row."""
    return {i: float(row[f"r_{l}"]) for i, l in enumerate(C.LEGS) if pd.notna(row.get(f"r_{l}"))}


def leg_verdict(lower, point, tau):
    lo = np.nan_to_num(np.asarray(lower, float), nan=-np.inf)
    return np.where(lo >= tau, "certify", np.where(np.asarray(point, float) >= tau, "probe", "refuse"))


def body_verdicts(rows, taus):
    """rows: one episode's rows. -> DataFrame (cand, leg, tau, verdict, unsafe) on the job's legs, and per-body."""
    x = rows[rows.leg.isin(list(taus))].copy()
    x["tau"] = x.leg.map(taus)
    x["verdict"] = leg_verdict(x.lower, x.point, x.tau)
    x["unsafe"] = (x.verdict == "certify") & (x.truth < x.tau)
    b = x.groupby("cand").verdict.agg(
        lambda s: "refuse" if (s == "refuse").any() else ("certify" if (s == "certify").all() else "probe"))
    return x, b


def select(res):
    d = cmain_rows(res)
    t = pd.read_csv(res / TASKS)
    kinds = lambda r: {l.rstrip("+-") for l in (C.LEGS[i] for i in job_taus(r))}
    jobs = [r for _, r in t[t.min_modules == 1].iterrows() if {"surge", "diag", "yaw"} <= kinds(r)]
    self_f = d[d.kind == "self"].groupby("eid").has_fault.first()
    n_m = d[d.kind == "merge"].groupby("eid").cand.nunique().reindex(self_f.index).fillna(0)
    pn = d.groupby("eid").parent_n.first()
    eids = [e for e in self_f.index if self_f[e] == 1 and n_m[e] >= 1 and pn[e] == 6]
    best = None
    for e in eids:
        rows = d[d.eid == e]
        for j in jobs:
            x, b = body_verdicts(rows, job_taus(j))
            if x.unsafe.any() or set(b) != {"certify", "probe", "refuse"}:
                continue
            key = (-(b == "certify").sum(), e, j.task_id)
            if best is None or key < best[0]:
                best = (key, e, j)
    _, e, j = best
    rows = d[d.eid == e]
    x, b = body_verdicts(rows, job_taus(j))
    size = rows.groupby("cand").cand_size.first()
    probe = sorted(b[b == "probe"].index, key=lambda c: (size[c], c))[0]
    refuse = sorted(b[b == "refuse"].index, key=lambda c: (size[c], c))[0]
    return e, j, (int(refuse), int(probe)), rows, len(d.eid.unique()), len(eids)


def asm_json(a):
    return dict(name=a.name, modules=[[m, int(i), int(j), int(k)] for m, i, j, k in a.modules],
                wheel_pos=np.round(a.wheel_pos, 6).tolist(), w_hat=np.round(a.w_hat, 6).tolist(),
                r_hat=np.round(a.r_hat, 6).tolist(), wheel_module=a.wheel_module.tolist(),
                wheel_corner=list(a.wheel_corner), module_pos=np.round(a.module_pos, 6).tolist(),
                lidar=a.lidar.astype(bool).tolist())


def faults_of(spec):
    from capxfer.sim import BRAKE, COAST, WEAK
    mode = {WEAK: "weak", COAST: "coast", BRAKE: "brake"}
    a = spec.asm
    out = []
    for w in np.flatnonzero(spec.mode):
        k = int(a.wheel_module[w])
        rel = (np.asarray(a.wheel_pos[w]) - np.asarray(a.module_pos[k])) / float(a.meta["pitch"])
        out.append(dict(wheel=int(w), module=k, corner=a.wheel_corner[w], mode=mode[int(spec.mode[w])],
                        g=round(float(spec.g[w]), 4), rel=np.round(rel, 4).tolist()))
    return out


def main(res=C.RES):
    from cfcap.gen import episode as GE
    eid, job, shown, rows, n_eps, n_pool = select(res)
    gep = GE.make_episode(PLATE, "conf_test", eid)
    par = GE.params(PLATE)
    kinds = {v: k for k, v in GE.KIND.items()}
    got = rows.groupby("cand").agg(kind=("kind", "first"), size=("cand_size", "first"), hf=("has_fault", "first"))
    cands = []
    for ci, c in enumerate(gep.cands):
        faulted = any(gep.asms[r].mode[4 * k:4 * k + 4].any() for r, k in c.modules)
        assert (kinds[c.kind], len(c.modules), int(faulted)) == tuple(got.loc[ci]), f"candidate {ci} mismatch"
        cands.append(dict(cand=ci, kind=kinds[c.kind], modules=[[int(r), int(k)] for r, k in c.modules],
                          cells=[[int(i), int(j)] for i, j in GE.cells(c.asm)]))
    assert len(cands) == len(got)
    pa = gep.parent
    from cfcap.gen import simrun as SR
    seg = SR.probes.Schedule(1.0, pids=("P1", "P2", "P5")).segments      # simrun.probe_T('v2') schedule
    probe = dict(kind="v2", twist_s=round(sum(s.t1 - s.t0 for s in seg), 3), total_s=float(SR.probe_T("v2")),
                 segments=[dict(pid=s.pid, sign=int(s.leg), t0=s.t0, t1=s.t1, twist=np.round(s.twist, 3).tolist())
                           for s in seg])
    out = dict(
        source=f"{DEC} (C-MAIN conf_test, Ours, T1) + generator replay cfcap.gen.episode.make_episode"
               f"('{PLATE}', 'conf_test', {eid})",
        rule=__doc__.split("Selection")[1].split("The episode's")[0].strip(),
        pool=dict(n_episodes=n_eps, n_six_faulted_merge=n_pool),
        eid=int(eid), split="conf_test", plate=PLATE, shape=str(rows["shape"].iloc[0]),
        job=dict(task_id=job.task_id, title=job.title, domain=job.domain, taus={C.LEGS[i]: v for i, v in
                                                                                 job_taus(job).items()}),
        shown_bodies=dict(refuse=shown[0], probe=shown[1]), probe=probe,
        pitch_m=float(par["module"]["module_pitch_m"]), side_m=float(par["module"]["side_x_m"]),
        half_wheelbase_m=float(par["module"]["half_wheelbase_m"]),
        half_track_m=float(par["module"]["half_track_m"]), wheel_radius_m=float(par["wheel"]["radius_m"]),
        roller_pattern=par["wheel"]["roller_pattern"], rollers=int(par["wheel"]["rollers"]),
        lidar_modules=list(par["module"]["lidar_modules"]),
        parent=asm_json(pa.asm), parent_cells=[[int(i), int(j)] for i, j in GE.cells(pa.asm)],
        faults=faults_of(pa),
        others=[dict(role=int(o.role), cells=[[int(i), int(j)] for i, j in GE.cells(o.asm)], faults=faults_of(o))
                for o in gep.asms[1:]],
        cands=cands)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(OUT, eid, job.task_id, "shown bodies", shown, "pool", n_pool, "of", n_eps)


if __name__ == "__main__":
    main()
