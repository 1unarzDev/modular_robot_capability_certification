"""T1 parameters: capxfer's measured params + identified theta, loadable without the field corpus or pandas.

The rented-box path needs only torch/numpy (+ the capxfer source on PYTHONPATH). capxfer.assembly imports `yaml` at
module level; when PyYAML is missing a stub is registered and params come from `module_measured.json`, a verbatim JSON
snapshot of capxfer `params/module_measured.yaml` (the capxfer file itself is never edited).

Corpus-matched overrides (as capxfer.corpus.replay.params_for does): 6533 output counts/rev and the per-module mass of
the plate state (configs.csv: plated single configs 12.72-12.99 kg, pair 25.85/2; plateless 5.03-5.30 kg, pair 10.47/2).
"""
from __future__ import annotations

import copy
import functools
import json
import os
import pathlib
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent
CPR = 6533.0
MODULE_MASS = {"plated": 12.9, "plateless": 5.2}
MODULE_IDS = tuple(f"M{k}" for k in range(1, 9))

# capxfer.corpus.sysid.BOUNDS (copied so the plant never imports the pandas-based corpus adapter)
SYSID_BOUNDS = dict(mu=(0.3, 1.2), s0=(0.003, 0.06), c_roller=(0.05, 3.0), crr=(0.0005, 0.03), latency=(0.0, 0.08),
                    stall_scale=(0.5, 1.6), noload_scale=(0.8, 1.3), jeff_scale=(0.1, 3.0), izz_scale=(0.5, 1.6),
                    v_batt=(10.5, 12.6))


def _ensure_yaml():
    try:
        import yaml  # noqa: F401
    except ImportError:                                   # rented box without PyYAML: capxfer.assembly only imports it
        sys.modules["yaml"] = types.ModuleType("yaml")


_ensure_yaml()


@functools.lru_cache(maxsize=None)
def _raw_params():
    root = os.environ.get("CAPXFER_ROOT")
    if root:
        f = pathlib.Path(root) / "params/module_measured.yaml"
        try:
            import yaml
            if f.exists() and hasattr(yaml, "safe_load"):
                return yaml.safe_load(f.read_text())
        except ImportError:
            pass
    return json.loads((HERE / "module_measured.json").read_text())


def base_params(plate="plated"):
    p = copy.deepcopy(_raw_params())
    p["motor"]["counts_per_output_rev"] = CPR
    p["module"]["mass_kg"] = {k: MODULE_MASS[plate] for k in MODULE_IDS}
    return p


def theta_dir():
    d = os.environ.get("CAPXFER_THETA_DIR")
    return pathlib.Path(d) if d else pathlib.Path.home() / ".cache/modular-runs/2026-11/capxfer/sysid"


@functools.lru_cache(maxsize=None)
def load_theta(plate="plated"):
    return json.loads((theta_dir() / f"theta_{plate}.json").read_text())["theta"]


def theta_dr(theta, rel=0.25):
    """capxfer.corpus.selfmodel.train_dr (the v2 self-model's training DR), reproduced without the corpus import."""
    from capxfer import sim
    dr = dict(sim.DR_MEASURED)
    for k, (lo, hi) in SYSID_BOUNDS.items():
        v = float(theta[k])
        dr[k] = (max(lo, v * (1 - rel)), min(hi, v * (1 + rel)))
    dr["mu"] = (0.35, 1.0)
    dr["latency"] = (max(0.0, theta["latency"] - 0.02), theta["latency"] + 0.02)
    dr["v_batt"] = (10.8, 12.6)
    dr.update(kp_scale=(1.0, 1.0), ki_scale=(1.0, 1.0), radius_scale=(0.99, 1.01), mass_scale=(0.95, 1.05),
              tau_bd=(0.2, 1.5), cg_off=0.015, wheel_yaw_err=0.02)
    return dr


THETA_KEYS = ("mu", "s0", "c_roller", "crr", "latency", "stall_scale", "noload_scale", "izz_scale", "kp_scale",
              "ki_scale", "radius_scale", "v_batt", "jeff_scale")


def point_domain(theta, B, device="cpu", tau_bd=0.75):
    """capxfer.corpus.replay.point_domain: every trial at theta (no DR)."""
    import torch
    f = lambda v: torch.full((B,), float(v), dtype=torch.float32, device=device)
    th = dict(theta, kp_scale=1.0, ki_scale=1.0, radius_scale=1.0)
    d = {k: f(th.get(k, 1.0)) for k in THETA_KEYS}
    d.update(tau_bd=f(tau_bd), mass_scale=f(1.0), cg_dx=f(0.0), cg_dy=f(0.0), wheel_yaw_err_std=0.0)
    return d


# Named assemblies (grid cells, all module yaws 0)
SHAPES = {
    "single": [(0, 0)],
    "pair": [(0, 0), (1, 0)],
    "L3": [(0, 0), (1, 0), (0, 1)],
    "square": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "2x3": [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
    "2x4": [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
}
SIZE_SHAPE = {1: "single", 2: "pair", 3: "L3", 4: "square", 6: "2x3", 8: "2x4"}


def named_assembly(name, params):
    from capxfer import assembly as asmlib
    cells = SHAPES[name]
    return asmlib.build(name, [(MODULE_IDS[k], i, j, 0) for k, (i, j) in enumerate(cells)], params)
