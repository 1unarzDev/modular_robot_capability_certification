#!/usr/bin/env python3
"""Generate the cfcap (study 09) field run sheet for the next modular session.

Deterministic: the seed is fixed below, so re-running gives the same file. Trials are blocked by physical
shape (to minimise reconfiguration) and randomised within each shape block. The spec is
~/crane-program/datacollection/09_cfcap.md; the probe routine, ratio_* and the fault encoding
(fault_type / fault_actuator / fault_level) are those of 05_capxfer.md.

Usage: python3 scripts/make_runsheet.py [out.csv]
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

SEED = 20261125
OUT = Path.home() / "crane-program/datacollection/09_cfcap_runsheet.csv"

COLUMNS = [
    "seq", "round", "block", "priority", "shape", "n_modules", "morph", "slot_map", "corpus_layout", "role",
    "scenario_id", "fault_type", "fault_actuator", "fault_level", "fault_type_2", "fault_actuator_2",
    "fault_level_2", "chain_id", "routine", "cmd_channel", "payload_spec", "prior_partners", "notes",
]

# ---- shapes: module -> (col, row) on the 0.32 m grid; col along assembly +x, row along +y; all yaw 0 ----
R6 = {"M1": (0, 0), "M2": (1, 0), "M3": (0, 1), "M4": (1, 1), "M5": (0, 2), "M6": (1, 2)}


def _sub(parent: dict, *mods: str) -> dict:
    c0 = min(parent[m][0] for m in mods)
    r0 = min(parent[m][1] for m in mods)
    return {m: (parent[m][0] - c0, parent[m][1] - r0) for m in mods}


SHAPES = {
    "R6": R6,                                   # 2 wide x 3 long parent
    "SQ4": _sub(R6, "M1", "M2", "M3", "M4"),    # = corpus square4 positions
    "PR2": _sub(R6, "M5", "M6"),                # pair along x, like the corpus pair
    "L3a": _sub(R6, "M1", "M2", "M3"),          # = corpus L3 cell pattern
    "L3b": _sub(R6, "M4", "M5", "M6"),          # L3 rotated 180 deg
    "Z4": {"M5": (1, 0), "M6": (2, 0), "M3": (0, 1), "M4": (1, 1)},          # held-out merge shape
    "P6": {"M1": (1, 0), "M5": (2, 0), "M6": (3, 0), "M2": (0, 1), "M3": (1, 1), "M4": (2, 1)},  # held-out
}
CORPUS_LAYOUT = {"SQ4": "square4", "S1": "single"}


def single(m: str) -> dict:
    return {m: (0, 0)}


def ordered(shape: dict) -> list[str]:
    return sorted(shape, key=lambda m: (shape[m][1], shape[m][0], m))


def morph_str(shape: dict) -> str:
    return ";".join(f"{m}@{shape[m][0]},{shape[m][1]},0" for m in ordered(shape))


def slot_map_str(shape: dict) -> str:
    return ";".join(f"m{i}={m}" for i, m in enumerate(ordered(shape), 1))


def partners(shape: dict) -> dict:
    out = {}
    for m, (c, r) in shape.items():
        out[m] = sorted(k for k, (c2, r2) in shape.items() if abs(c - c2) + abs(r - r2) == 1)
    return out


# ---- fault scenarios: (module, corner, type, g); g only for 201 ----
SCENARIOS = {
    "F00": [],
    "F01": [("M2", "rr", 201, 0.55)],
    "F02": [("M4", "fl", 201, 0.70)],
    "F03": [("M6", "fr", 201, 0.80)],
    "F04": [("M3", "rl", 202, None)],
    "F05": [("M5", "rr", 203, None)],
    "F06": [("M1", "fl", 201, 0.70), ("M1", "rr", 202, None)],
    "F07": [("M2", "fr", 201, 0.55), ("M5", "fl", 203, None)],
    "F08": [("M4", "rl", 202, None), ("M6", "rl", 201, 0.80)],
}
CHAIN_SCENARIOS = ["F01", "F05", "F07", "F08"]
C2_SCEN = {"Z4": ["F00", "F00", "F02", "F05", "F07", "F08"], "P6": ["F00", "F00", "F01", "F03", "F06", "F07"]}


def held_out_scenarios(rng: random.Random, n: int = 6) -> dict:
    """C5 scenarios: new g levels and new wheel draws, 1-2 faults each."""
    out = {}
    mods = list(R6)
    for i in range(1, n + 1):
        k = 2 if i in (4, 5) else 1
        faults, used = [], set()
        while len(faults) < k:
            m, c = rng.choice(mods), rng.choice(["fl", "fr", "rl", "rr"])
            if (m, c) in used:
                continue
            used.add((m, c))
            t = rng.choice([201, 201, 202, 203])
            faults.append((m, c, t, rng.choice([0.60, 0.65, 0.75]) if t == 201 else None))
        out[f"H{i:02d}"] = faults
    return out


def restrict(faults, shape) -> list:
    return [f for f in faults if f[0] in shape]


def fault_key(faults) -> tuple:
    return tuple(sorted(faults, key=lambda f: (f[0], f[1])))


def make_row(rnd, block, prio, shape_name, shape, role, scen, faults, routine, chain="", payload="", notes=""):
    slot_of = {m: i for i, m in enumerate(ordered(shape), 1)}
    fr = sorted(restrict(faults, shape), key=lambda f: (slot_of[f[0]], f[1]))
    assert len(fr) <= 2
    cols = {}
    for j, suffix in ((0, ""), (1, "_2")):
        if j < len(fr):
            m, c, t, g = fr[j]
            cols["fault_type" + suffix] = t
            cols["fault_actuator" + suffix] = f"m{slot_of[m]}_{c}"
            cols["fault_level" + suffix] = f"{g:.2f}" if g is not None else ""
        else:
            cols["fault_type" + suffix] = 0 if j == 0 else ""
            cols["fault_actuator" + suffix] = ""
            cols["fault_level" + suffix] = ""
    channel = {"DL": "body", "P_clamp": "body", "P_mdiff": "wheel_speed", "P_tug": "wheel_speed",
               "FREESPIN": "wheel_duty"}[routine]
    return {
        "round": rnd, "block": block, "priority": prio, "shape": shape_name, "n_modules": len(shape),
        "morph": morph_str(shape), "slot_map": slot_map_str(shape),
        "corpus_layout": CORPUS_LAYOUT.get(shape_name, ""), "role": role, "scenario_id": scen, **cols,
        "chain_id": chain, "routine": routine, "cmd_channel": channel, "payload_spec": payload,
        "prior_partners": "", "notes": notes,
    }


HEAVY = {"P_tug", "P_mdiff"}


def shuffle_constrained(rows: list, rng: random.Random) -> list:
    """Shuffle; no more than 2 consecutive P_tug/P_mdiff (motor cool-down rule)."""
    for _ in range(10000):
        r = rows[:]
        rng.shuffle(r)
        run, ok = 0, True
        for x in r:
            run = run + 1 if x["routine"] in HEAVY else 0
            if run > 2:
                ok = False
                break
        if ok:
            return r
    raise RuntimeError("could not satisfy shuffle constraint")


def child_rows(rnd, block, prio, shape_name, scen_ids, scen_table, healthy_reps=2, role="target"):
    shape = SHAPES[shape_name]
    rows, seen = [], set()
    for s in scen_ids:
        fr = restrict(scen_table[s], shape)
        k = fault_key(fr)
        if not fr or k in seen:
            continue
        seen.add(k)
        rows.append(make_row(rnd, block, prio, shape_name, shape, role, s, fr, "DL"))
    rows += [make_row(rnd, block, prio, shape_name, shape, role, "F00", [], "DL") for _ in range(healthy_reps)]
    return rows


def build_rows(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    hold = held_out_scenarios(random.Random(seed + 1))
    allscen = {**SCENARIOS, **hold}
    F = [s for s in SCENARIOS if s != "F00"]
    seq: list[dict] = []

    # ---------------- R1 ----------------
    # C0: stand free-spin on all six modules, then lone healthy decision legs and the P_tug pattern check
    seq += [make_row("R1", "C0", 1, "S1", single(m), "char", "F00", [], "FREESPIN",
                     notes="module on stand, wheels off the floor") for m in R6]
    c0 = [make_row("R1", "C0", 1, "S1", single(m), "target", "F00", [], "DL") for m in R6]
    c0 += [make_row("R1", "C0", 1, "S1", single(m), "target", "F00", [], "DL") for m in ("M5", "M6")]
    c0.append(make_row("R1", "C0", 1, "S1", single("M1"), "char", "F00", [], "P_tug",
                       notes="pattern check: pass if TRUTH displacement <=0.10 m and |yaw change| <=0.20 rad"))
    seq += shuffle_constrained(c0, rng)

    # R6 block: C1a parent decision legs x2, C3 probes x1, then C4 payload rows (priority 2) at the end
    r6 = SHAPES["R6"]
    blk = [make_row("R1", "C1a", 1, "R6", r6, "obs", s, allscen[s], "DL") for s in SCENARIOS for _ in range(2)]
    blk += [make_row("R1", "C3", 1, "R6", r6, "obs", s, allscen[s], p) for s in SCENARIOS
            for p in ("P_clamp", "P_mdiff", "P_tug")]
    seq += shuffle_constrained(blk, rng)
    pay = "M2:2.0;M4:2.0"
    c4 = [make_row("R1", "C4", 2, "R6", r6, "obs", s, allscen[s], "DL", payload=pay) for s in ("F01", "F02")
          for _ in range(2)]
    seq += shuffle_constrained(c4, rng)

    # C1b: physical split chains (continuous injection across a real 6>4+2 split)
    chains = CHAIN_SCENARIOS[:]
    rng.shuffle(chains)
    for i, s in enumerate(chains, 1):
        cid = f"CH{i:02d}"
        seq.append(make_row("R1", "C1b", 1, "R6", r6, "obs", s, allscen[s], "DL", chain=cid,
                            notes="inject; run; keep injection active"))
        seq.append(make_row("R1", "C1b", 1, "SQ4", SHAPES["SQ4"], "target", s, allscen[s], "DL", chain=cid,
                            notes="after physical split 6>4+2 (log in modular_reconfig.csv); no re-injection"))
        last = i == len(chains)
        seq.append(make_row("R1", "C1b", 1, "PR2", SHAPES["PR2"], "target", s, allscen[s], "DL", chain=cid,
                            notes="no re-injection" + ("" if last else "; then recombine 4+2>6")))

    # C1c: children of the 4+2 split, blockwise (software re-injection per trial)
    for sh in ("SQ4", "PR2"):
        seq += shuffle_constrained(child_rows("R1", "C1c", 1, sh, F, allscen), rng)

    # ---------------- R2 ----------------
    for sh in ("L3a", "L3b"):
        seq += shuffle_constrained(child_rows("R2", "C1d", 1, sh, F, allscen), rng)
    # C1e lone faulted modules (x2) + C4 lone payload rows, one module on the floor at a time
    for m in R6:
        blk, seen = [], set()
        for s in F:
            fr = restrict(allscen[s], single(m))
            if fr and fault_key(fr) not in seen:
                seen.add(fault_key(fr))
                blk += [make_row("R2", "C1e", 1, "S1", single(m), "target", s, fr, "DL") for _ in range(2)]
        seq += shuffle_constrained(blk, rng)
        if m in ("M2", "M4"):
            own = "F01" if m == "M2" else "F02"
            p = [make_row("R2", "C4", 2, "S1", single(m), "target", s, allscen[s], "DL", payload=f"{m}:2.0")
                 for s in ("F00", own) for _ in range(2)]
            seq += shuffle_constrained(p, rng)
    # C2 held-out merge shapes
    for sh in ("Z4", "P6"):
        rows = [make_row("R2", "C2", 1, sh, SHAPES[sh], "target", s, allscen[s], "DL") for s in C2_SCEN[sh]]
        seq += shuffle_constrained(rows, rng)

    # ---------------- R3 ----------------
    H = list(hold)
    blk = [make_row("R3", "C5", 2, "R6", r6, "obs", s, allscen[s], p) for s in H
           for p in ("DL", "P_clamp", "P_mdiff", "P_tug")]
    seq += shuffle_constrained(blk, rng)
    for sh in ("SQ4", "PR2", "L3a", "L3b"):
        rows = child_rows("R3", "C5", 2, sh, H, allscen, healthy_reps=0)
        if rows:
            seq += shuffle_constrained(rows, rng)

    # seq numbers and prior docking partners (derived from the shape order; the operator corrects if order changes)
    last_morph: dict = {}
    cur_part: dict = {}
    prev_part: dict = {}
    for i, row in enumerate(seq, 1):
        row["seq"] = i
        shape = {}
        for tok in row["morph"].split(";"):
            m, pos = tok.split("@")
            c, r, _ = pos.split(",")
            shape[m] = (int(c), int(r))
        part = partners(shape)
        for m in shape:
            if last_morph.get(m) != row["morph"]:
                if m in last_morph:
                    prev_part[m] = cur_part[m]
                last_morph[m] = row["morph"]
                cur_part[m] = part[m]
        row["prior_partners"] = ";".join(
            f"{m}:{'+'.join(prev_part.get(m, [])) or '-'}" for m in ordered(shape))
    return seq


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLUMNS})


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    rows = build_rows()
    write(rows, out)
    from collections import Counter
    print(out, len(rows), "rows")
    print(Counter((r["round"], r["block"]) for r in rows))
