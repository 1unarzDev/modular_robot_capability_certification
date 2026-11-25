"""Stage B generator pilot (exploratory): throughput, bytes/episode, label cost share and candidates per episode.

  python scripts/gen_pilot.py --episodes 2000            # GPU pilot (one train shard, plated) + CPU tables
  python scripts/gen_pilot.py --cpu-only                 # only the candidate / family tables

Writes results/generator/{pilot.csv, pilot_blocks.csv, candidates_by_split.csv, families.csv}; bulk shard to
$CFCAP_OUT/generator/pilot/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cfcap.gen import episode as E  # noqa: E402
from cfcap.gen import families as F  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results/generator"
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", pathlib.Path.home() / ".cache/modular-runs/2026-11/cfcap")) / "generator"


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cpu_tables(n_per=300):
    rows = [dict(n=r[0], family_hash=r[1], topology=r[2], split=r[3], cells=" ".join(f"{x},{y}" for x, y in r[4]),
                 n_fixed=r[5]) for r in F.family_table(6)]
    write_csv(RES / "families.csv", rows)
    out = []
    for split in ("train", "H3", "H4", "H6", "H7"):
        t0 = time.perf_counter()
        eps = [E.make_episode("plated", split, i) for i in range(n_per)]
        dt = time.perf_counter() - t0
        nc = np.array([len(e.cands) for e in eps])
        kinds = np.array([[sum(c.kind == k for c in e.cands) for k in (0, 1, 2)] for e in eps])
        cm = np.array([sum(len(c.modules) for c in e.cands) for e in eps])
        nf = np.array([sum(any(e.asms[r].mode[4 * k:4 * k + 4].any() for r, k in c.modules) for c in e.cands)
                       for e in eps])
        held = np.array([sum(c.fam_split != F.FAM_TRAIN for c in e.cands) for e in eps])
        out.append(dict(split=split, episodes=n_per, parent_modules_mean=round(np.mean([e.parent.m for e in eps]), 2),
                        cands_mean=round(nc.mean(), 2), cands_median=float(np.median(nc)),
                        cands_p90=float(np.percentile(nc, 90)), cands_max=int(nc.max()),
                        children_mean=round(kinds[:, 1].mean(), 2), merges_mean=round(kinds[:, 2].mean(), 2),
                        cand_modules_mean=round(cm.mean(), 1), faulted_cand_share=round(nf.sum() / nc.sum(), 3),
                        heldout_family_cand_share=round(held.sum() / nc.sum(), 3),
                        payload_share=round(np.mean([e.payload is not None for e in eps]), 3),
                        make_ms_per_episode=round(1e3 * dt / n_per, 2)))
    write_csv(RES / "candidates_by_split.csv", out)
    for r in out:
        print(json.dumps(r))


def gpu_pilot(n, block, kinds, engine):
    import torch
    from cfcap.gen import shard as SH
    blocks = []

    def log(msg):
        print(msg, flush=True)
        if msg.startswith("[gen] plated"):
            p = msg.split()
            blocks.append(dict(block=len(blocks), make_s=float(p[5][:-1]), probe_s=float(p[7][:-1]),
                               label_s=float(p[9][:-1])))
    t0 = time.perf_counter()
    info = SH.write_shard(OUT / "pilot", "plated", "train", 0, n, kinds, engine=engine, block=block, force=True, log=log)
    wall = time.perf_counter() - t0
    st = info["stats"]
    gen = info["gen_seconds"]
    lab, prb = st["label"], st["probe"]
    D = SH.load(OUT / "pilot/plated/train" / info["file"])
    nbytes = {k: v.nbytes for k, v in D.items()}
    raw = sum(nbytes.values())
    obs_raw = sum(v for k, v in nbytes.items() if k.startswith("obs_"))
    steady = blocks[1:] if len(blocks) > 1 else blocks
    steady_s = sum(b["make_s"] + b["probe_s"] + b["label_s"] for b in steady)
    steady_eps = block * len(steady)
    rr = D["cand_rho_rel"]
    row = dict(gpu=torch.cuda.get_device_name(0), engine=engine, kinds="+".join(kinds), episodes=n, block=block,
               wall_s=round(wall, 1), gen_s=gen, write_s=info["write_seconds"],
               episodes_per_s=round(n / wall, 2), episodes_per_s_steady=round(steady_eps / steady_s, 2),
               make_s=round(st["make_s"], 1), probe_s=round(st["probe_s"], 1), label_s=round(st["label_s"], 1),
               assemble_s=round(st["assemble_s"], 1), label_share=round(st["label_s"] / wall, 3),
               label_share_steady=round(sum(b["label_s"] for b in steady) / steady_s, 3),
               label_prep_s=round(lab["prep_s"], 1), label_sim_s=round(lab["sim_s"], 1),
               label_rows=lab["rows"], label_pad_share=round(1 - lab["rows"] / lab["rows_padded"], 3),
               label_leg_trials_per_s=round(lab["rows"] / (lab["prep_s"] + lab["sim_s"]), 0),
               probe_prep_s=round(prb["prep_s"], 1), probe_sim_s=round(prb["sim_s"], 1),
               cands=info["n_cand"], cands_per_episode=round(info["n_cand"] / n, 2),
               label_leg_trials_per_episode=round(lab["rows"] / n, 1),
               bytes_per_episode=round(info["bytes"] / n, 0), raw_bytes_per_episode=round(raw / n, 0),
               obs_raw_share=round(obs_raw / raw, 3),
               rho_rel_min=round(float(rr.min()), 3), rho_rel_max=round(float(rr.max()), 3),
               rho_rel_lt_0p8_share=round(float((rr < 0.8).mean()), 4),
               peak_vram_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
    write_csv(RES / "pilot.csv", [row])
    write_csv(RES / "pilot_blocks.csv", blocks)
    write_csv(RES / "pilot_bytes.csv", [dict(array=k, dtype=str(D[k].dtype), shape="x".join(map(str, D[k].shape)),
                                             raw_bytes_per_episode=round(v / n, 1)) for k, v in sorted(nbytes.items())])
    print(json.dumps(row, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--probes", default="v2")
    ap.add_argument("--engine", default="compile")
    ap.add_argument("--cpu-only", action="store_true")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    cpu_tables()
    if not a.cpu_only:
        gpu_pilot(a.episodes, a.block, tuple(a.probes.split(",")), a.engine)


if __name__ == "__main__":
    main()
