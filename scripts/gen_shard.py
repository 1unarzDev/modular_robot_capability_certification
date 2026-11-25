"""Stage B generator shard: episodes [k*n, (k+1)*n) of one (plate, split) -> <out>/<plate>/<split>/shard_<k>.npz.

Needs only torch + numpy (+ scipy unused), PYTHONPATH with cfcap and the capxfer repo root, and CAPXFER_THETA_DIR with
theta_{plated,plateless}.json (no corpus, pandas or PyYAML). Resumable: a complete shard (sidecar sha256 matches) is
skipped, so a requeued job re-runs only missing shards. Schema: docs/board/generator.md.

  python scripts/gen_shard.py --split train --shard 0 --episodes 2000 --out $CFCAP_OUT/generator/data

Rented RTX 3090: inputs are pushed to /workspace/in/{cfcap (git archive of HEAD), capxfer (source + params + theta/)}.
Preferred entry is the runner adapter (cfcap/compute_tasks.py GEN_SHARD; one unit = one generator shard):
  cd /workspace/in/cfcap && PYTHONPATH=/workspace/in/cfcap:/workspace/in/capxfer \
    CAPXFER_THETA_DIR=/workspace/in/capxfer/theta python -m cfcap.runner gen_shard --shard $K --n-shards $N \
    --split train --gen-shards $G --episodes 2000
Direct (same inputs):
  ... python scripts/gen_shard.py --plate plated --split train --shard $K --shards 4 --episodes 2000 \
    --out /workspace/out/$CRANE_JOB_ID
(--shards s runs shards k .. k+s-1 in one process, sharing torch.compile warm-up.)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cfcap.gen import episode as E  # noqa: E402
from cfcap.gen import shard as SH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=sorted(E.SPLITS))
    ap.add_argument("--plate", default="plated", choices=E.PLATES)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=1, help="run shards shard .. shard+shards-1")
    ap.add_argument("--episodes", type=int, required=True, help="episodes per shard (fixed per split)")
    ap.add_argument("--probes", default="v2", help="comma list of v2,clamp,tug")
    ap.add_argument("--engine", default="compile", choices=("compile", "graph", "eager"))
    ap.add_argument("--block", type=int, default=256, help="episodes simulated together")
    ap.add_argument("--max-wheels", type=int, default=1 << 17, help="B*n per label run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    kinds = tuple(a.probes.split(","))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for k in range(a.shard, a.shard + a.shards):
        t0 = time.time()
        info = SH.write_shard(out, a.plate, a.split, k, a.episodes, kinds, engine=a.engine, block=a.block,
                              force=a.force, max_wheels=a.max_wheels)
        done.append(k)
        prog = dict(plate=a.plate, split=a.split, shards_done=done, last=info["file"], last_seconds=time.time() - t0,
                    job=os.environ.get("CRANE_JOB_ID"), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (out / "progress.json").write_text(json.dumps(prog, indent=1))
        print(json.dumps({k_: v for k_, v in info.items() if k_ != "stats"}), flush=True)


if __name__ == "__main__":
    main()
