"""Generator pilot for the broadening splits (exploratory): H4b (11-16 modules), het_test / H1 / H2 (module types),
conf_test. One shard per split, 2+ blocks; steady rate = blocks after the first (torch.compile warm-up).

  python scripts/gen_pilot2.py            -> results/generator/pilot2.csv, bulk $CFCAP_OUT/generator/pilot2/
"""
from __future__ import annotations

import csv
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cfcap.gen import shard as SH  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results/generator"
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", pathlib.Path.home() / ".cache/modular-runs/2026-11/cfcap")) / "generator"
PLAN = [("het_test", 400, 200), ("H1", 200, 100), ("H2", 200, 100), ("H4b", 120, 40)]


def main():
    rows = []
    for split, n, block in PLAN:
        blocks = []

        def log(msg):
            if msg.startswith("[gen] plated"):
                p = msg.split()
                blocks.append(float(p[5][:-1]) + float(p[7][:-1]) + float(p[9][:-1]))
                print(msg, flush=True)
        t0 = time.perf_counter()
        info = SH.write_shard(OUT / "pilot2", "plated", split, 0, n, ("v2",), engine="compile", block=block,
                              force=True, log=log)
        wall = time.perf_counter() - t0
        D = SH.load(OUT / "pilot2/plated" / split / info["file"])
        st = info["stats"]
        steady = blocks[1:]
        r = dict(gpu=torch.cuda.get_device_name(0), split=split, episodes=n, block=block, wall_s=round(wall, 1),
                 s_per_episode_steady=round(sum(steady) / (block * len(steady)), 4),
                 episodes_per_s_steady=round(block * len(steady) / sum(steady), 2),
                 first_block_s=round(blocks[0], 1), label_share=round(st["label_s"] / max(1e-9, st["label_s"] +
                                                                                       st["probe_s"] + st["make_s"]), 3),
                 bytes_per_episode=round(info["bytes"] / n), cands_per_episode=round(info["n_cand"] / n, 2),
                 parent_modules_mean=round(float(np.mean(np.diff(D["asm_mod_off"])[D["ep_asm_off"][:-1]])), 2),
                 max_cand_modules=int(D["cand_n_mod"].max()) if "cand_n_mod" in D else int(np.diff(D["cand_cm_off"]).max()),
                 rho_dec_lt_0p8=round(float((D["cand_rho_dec"] < 0.8).mean()), 4),
                 rho_dec_min=round(float(D["cand_rho_dec"].min()), 3),
                 peak_vram_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
        if "mod_type" in D:
            r["types_present"] = "".join(str(t) for t in sorted(set(D["mod_type"].tolist())))
        print(r, flush=True)
        rows.append(r)
        keys = sorted({k for rr in rows for k in rr}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        with open(RES / "pilot2.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
