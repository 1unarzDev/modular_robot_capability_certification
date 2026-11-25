"""Models pilot data (exploratory): T1 episodes for the amortized posterior, until cfcap/gen is ready.

train/val: single, pair, L3, square (parents <= 4 modules); test: pair, square, 2x3 (2x3 = size generalisation).
Heterogeneity HET_FULL, jittered v2 probe, 2 candidate children per episode with closed-loop rho_rel. Test sets also
carry the capxfer v2 self-model outputs (3 seeds, as Stage A1 D2).

  scripts/heavy.sh --gpu $PY scripts/models_pilot_gen.py
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import time

import numpy as np

from cfcap.models import data as D

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(os.environ.get("CFCAP_OUT", pathlib.Path.home() / ".cache/modular-runs/2026-11/cfcap")) / \
    "models/pilot"
PLAN = {"train": {"single": 6000, "pair": 8000, "L3": 8000, "square": 8000},
        "val": {"single": 500, "pair": 500, "L3": 500, "square": 500},
        "test": {"pair": 2000, "square": 2000, "2x3": 2000}}


def baseline():
    spec = importlib.util.spec_from_file_location("stageA1_hidden", ROOT / "scripts/stageA1_hidden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SelfModelV2()


def main():
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    OUT.mkdir(parents=True, exist_ok=True)
    sm = baseline()
    t0 = time.time()
    for si, (split, shapes) in enumerate(PLAN.items()):
        for sj, (shape, B) in enumerate(shapes.items()):
            f = OUT / f"{split}_{shape}.npz"
            if f.exists():
                continue
            rng = np.random.default_rng(1000 * si + sj + 20261125)
            d = D.generate(shape, max(16, int(B * scale)), rng, baseline=sm if split == "test" else None)
            np.savez(f, **d)
            bad = int(np.isnan(d["rho_rel"]).any(-1).sum())
            print(f"{split} {shape} B={len(d['kind'])} T={d['wf'].shape[1]} nan_children={bad} "
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
