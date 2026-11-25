"""Ingest and validate cfcap (study 09) field CSVs; build (observation -> measured configuration) pairs.

Spec: ~/crane-program/datacollection/09_cfcap.md. The probe routine, ratio_* and the fault encoding
(fault_type / fault_actuator / fault_level, slot-based `m<slot>_<corner>`) are 05_capxfer.md's; cfcap adds a
second fault (`*_2`), graded fault levels, the morphology graph and the wheel-level probe routines.

Two inputs share one validator:
- the run sheet (`09_cfcap_runsheet.csv`), kind="runsheet";
- the per-trial summary (`modular_cfcap.csv`) = run-sheet columns + corpus columns + measured columns,
  kind="summary".

`build_pairs` returns one record per (observation trial, target trial) where the target configuration's faults are
the observation's faults restricted to (split) or extended from (merge) the shared modules. Pairing uses modules and
faults, never `scenario_id` alone, because software faults are reproducible across assemblies.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

RUNSHEET_COLUMNS = [
    "seq", "round", "block", "priority", "shape", "n_modules", "morph", "slot_map", "corpus_layout", "role",
    "scenario_id", "fault_type", "fault_actuator", "fault_level", "fault_type_2", "fault_actuator_2",
    "fault_level_2", "chain_id", "routine", "cmd_channel", "payload_spec", "prior_partners", "notes",
]
CORPUS_COLUMNS = ["session_id", "trial_id", "outcome_code", "abort_code", "config_id", "duration_s"]
MEASURED_COLUMNS = ["portion_code", "plate_state", "aborted"]
RATIO_COLUMNS = [f"ratio_{a}_{s}" for a in ("surge", "sway", "yaw", "diag") for s in ("pos", "neg")]
VBATT_COLUMNS = [f"vbatt_M{k}" for k in range(1, 7)]
OPTIONAL_SUMMARY = RATIO_COLUMNS + VBATT_COLUMNS + ["vbatt_src", "disp_max_m", "abort_reason"]

ROUNDS = {"R1", "R2", "R3"}
BLOCKS = {"C0", "C1a", "C1b", "C1c", "C1d", "C1e", "C2", "C3", "C4", "C5"}
ROLES = {"obs", "target", "char"}
ROUTINE_CHANNEL = {"DL": "body", "P_clamp": "body", "P_mdiff": "wheel_speed", "P_tug": "wheel_speed",
                   "FREESPIN": "wheel_duty"}
CORNERS = ("fl", "fr", "rl", "rr")
FAULT_TYPES = {0, 201, 202, 203}
LIFTER_FREE = {"M1"}          # M1 is the lidar module, no lifter (physical_context.md)
MODULE_RE = re.compile(r"^M[1-9]$")
SCEN_RE = re.compile(r"^[FH]\d\d$")
PORTION_CFCAP = 216


@dataclass(frozen=True)
class Fault:
    module: str
    corner: str
    ftype: int
    level: float | None

    def key(self):
        return (self.module, self.corner, self.ftype, None if self.level is None else round(self.level, 3))


# ------------------------------------------------------------------ parsers
def parse_morph(s: str) -> dict[str, tuple[int, int, int]]:
    """'M1@0,0,0;M2@1,0,0' -> {module: (col, row, yaw_quarter_turns)}. Raises ValueError."""
    out: dict[str, tuple[int, int, int]] = {}
    if not s or not s.strip():
        raise ValueError("empty morph")
    for tok in s.split(";"):
        try:
            m, pos = tok.strip().split("@")
            c, r, y = (int(v) for v in pos.split(","))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"bad morph token {tok!r}") from e
        if not MODULE_RE.match(m):
            raise ValueError(f"bad module id {m!r}")
        if m in out:
            raise ValueError(f"module {m} twice")
        if y not in (0, 1, 2, 3):
            raise ValueError(f"yaw quarter-turns must be 0..3, got {y}")
        out[m] = (c, r, y)
    cells = [(c, r) for c, r, _ in out.values()]
    if len(set(cells)) != len(cells):
        raise ValueError("two modules on one grid cell")
    # 4-connectivity
    todo, seen = [cells[0]], {cells[0]}
    cellset = set(cells)
    while todo:
        c, r = todo.pop()
        for n in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
            if n in cellset and n not in seen:
                seen.add(n)
                todo.append(n)
    if len(seen) != len(cells):
        raise ValueError("morph is not 4-connected")
    return out


def parse_slot_map(s: str) -> dict[str, str]:
    """'m1=M1;m2=M3' -> {'m1': 'M1', 'm2': 'M3'}."""
    out = {}
    for tok in s.split(";"):
        k, v = tok.strip().split("=")
        out[k] = v
    return out


def parse_faults(row: dict, slot_map: dict[str, str]) -> list[Fault]:
    """05 encoding, plus the cfcap second fault (*_2). Returns faults keyed by module id."""
    faults = []
    for suf in ("", "_2"):
        t = (row.get("fault_type" + suf) or "").strip()
        a = (row.get("fault_actuator" + suf) or "").strip()
        lv = (row.get("fault_level" + suf) or "").strip()
        if t in ("", "0"):
            if a or lv:
                raise ValueError(f"fault_actuator{suf}/fault_level{suf} set without a fault type")
            continue
        ftype = int(t)
        if ftype not in FAULT_TYPES:
            raise ValueError(f"fault_type{suf} {ftype} not in {sorted(FAULT_TYPES)}")
        m = re.match(r"^(m\d)_(fl|fr|rl|rr)$", a)
        if not m:
            raise ValueError(f"fault_actuator{suf} {a!r} is not m<slot>_<corner>")
        if m.group(1) not in slot_map:
            raise ValueError(f"fault_actuator{suf} slot {m.group(1)} not in slot_map")
        level = None
        if ftype == 201:
            level = float(lv)
            if not 0.0 < level < 1.0:
                raise ValueError(f"fault_level{suf} {level} outside (0,1)")
        elif lv:
            raise ValueError(f"fault_level{suf} only applies to 201")
        faults.append(Fault(slot_map[m.group(1)], m.group(2), ftype, level))
    if len({(f.module, f.corner) for f in faults}) != len(faults):
        raise ValueError("two faults on one wheel")
    if (row.get("fault_type", "").strip() in ("", "0")) and (row.get("fault_type_2", "").strip() not in ("", "0")):
        raise ValueError("fault_type_2 set while fault_type is 0")
    return faults


def parse_payload(s: str) -> dict[str, float]:
    out = {}
    if not s or not s.strip():
        return out
    for tok in s.split(";"):
        m, kg = tok.split(":")
        out[m.strip()] = float(kg)
    return out


def _num(v) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    x = float(v)
    if math.isnan(x):
        return None
    return x


# ------------------------------------------------------------------ validation
def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def validate(rows: list[dict], kind: str = "summary") -> list[str]:
    """Return a list of human-readable errors (empty = valid)."""
    if kind not in ("runsheet", "summary"):
        raise ValueError(kind)
    errs: list[str] = []
    if not rows:
        return ["no rows"]
    need = RUNSHEET_COLUMNS + (CORPUS_COLUMNS + MEASURED_COLUMNS if kind == "summary" else [])
    missing = [c for c in need if c not in rows[0]]
    if missing:
        return [f"missing columns: {missing}"]

    seqs: dict[int, list[dict]] = {}
    scen_union: dict[str, dict[tuple[str, str], tuple]] = {}
    parsed = []
    plate_by_session: dict[str, str] = {}
    for i, row in enumerate(rows):
        tag = f"row {i + 1} (seq {row.get('seq')})"
        try:
            seq = int(row["seq"])
        except ValueError:
            errs.append(f"{tag}: seq not an integer")
            continue
        seqs.setdefault(seq, []).append(row)
        for col, allowed in (("round", ROUNDS), ("block", BLOCKS), ("role", ROLES), ("routine", ROUTINE_CHANNEL)):
            if row[col] not in allowed:
                errs.append(f"{tag}: {col}={row[col]!r} not in {sorted(allowed)}")
        if row["priority"] not in ("1", "2", "3"):
            errs.append(f"{tag}: priority {row['priority']!r}")
        if row["routine"] in ROUTINE_CHANNEL and row["cmd_channel"] != ROUTINE_CHANNEL[row["routine"]]:
            errs.append(f"{tag}: cmd_channel {row['cmd_channel']!r} != {ROUTINE_CHANNEL[row['routine']]} for "
                        f"{row['routine']}")
        if not SCEN_RE.match(row["scenario_id"] or ""):
            errs.append(f"{tag}: scenario_id {row['scenario_id']!r}")
        try:
            morph = parse_morph(row["morph"])
            slots = parse_slot_map(row["slot_map"])
        except ValueError as e:
            errs.append(f"{tag}: {e}")
            continue
        if sorted(slots.values()) != sorted(morph):
            errs.append(f"{tag}: slot_map modules {sorted(slots.values())} != morph modules {sorted(morph)}")
            continue
        if sorted(slots) != sorted(f"m{k}" for k in range(1, len(morph) + 1)):
            errs.append(f"{tag}: slot_map keys must be m1..m{len(morph)}")
        if str(len(morph)) != str(row["n_modules"]).strip():
            errs.append(f"{tag}: n_modules {row['n_modules']} != {len(morph)} modules in morph")
        try:
            faults = parse_faults(row, slots)
        except ValueError as e:
            errs.append(f"{tag}: {e}")
            continue
        if row["routine"] == "FREESPIN" and faults:
            errs.append(f"{tag}: FREESPIN runs healthy only")
        if row["scenario_id"] == "F00" and faults:
            errs.append(f"{tag}: F00 must be healthy")
        try:
            pay = parse_payload(row["payload_spec"])
        except ValueError:
            errs.append(f"{tag}: payload_spec {row['payload_spec']!r}")
            pay = {}
        for m, kg in pay.items():
            if m not in morph:
                errs.append(f"{tag}: payload on {m}, not in morph")
            if m in LIFTER_FREE:
                errs.append(f"{tag}: payload on {m}, which has no lifter")
            if not 0.0 < kg <= 2.0:
                errs.append(f"{tag}: payload {kg} kg outside (0, 2.0] (lifter rating)")
        if row["prior_partners"]:
            for tok in row["prior_partners"].split(";"):
                if ":" not in tok or tok.split(":")[0] not in morph:
                    errs.append(f"{tag}: prior_partners token {tok!r}")
        if row["chain_id"] and not re.match(r"^CH\d\d$", row["chain_id"]):
            errs.append(f"{tag}: chain_id {row['chain_id']!r}")
        # scenario consistency: one fault per wheel per scenario
        u = scen_union.setdefault(row["scenario_id"], {})
        for f in faults:
            prev = u.get((f.module, f.corner))
            if prev is not None and prev != f.key():
                errs.append(f"{tag}: scenario {row['scenario_id']} wheel {f.module}_{f.corner} has {f.key()} "
                            f"here but {prev} elsewhere")
            u[(f.module, f.corner)] = f.key()

        if kind == "summary":
            if row["aborted"] not in ("0", "1"):
                errs.append(f"{tag}: aborted {row['aborted']!r}")
            if str(row["portion_code"]).strip() != str(PORTION_CFCAP):
                errs.append(f"{tag}: portion_code must be {PORTION_CFCAP}")
            if row["plate_state"] not in ("plated", "plateless"):
                errs.append(f"{tag}: plate_state {row['plate_state']!r}")
            ps = plate_by_session.setdefault(row["session_id"], row["plate_state"])
            if ps != row["plate_state"]:
                errs.append(f"{tag}: plate_state changes within session {row['session_id']}")
            for c in OPTIONAL_SUMMARY:
                if c in row and c not in ("vbatt_src", "abort_reason"):
                    try:
                        v = _num(row[c])
                    except ValueError:
                        errs.append(f"{tag}: {c}={row[c]!r} not numeric")
                        continue
                    if c in RATIO_COLUMNS and v is not None and row["routine"] != "DL":
                        errs.append(f"{tag}: {c} only for routine DL")
                    if c in VBATT_COLUMNS and v is not None and c.split("_")[1] not in morph:
                        errs.append(f"{tag}: {c} given for a module not in the assembly")
            if row.get("vbatt_src", "") not in ("", "log", "meter"):
                errs.append(f"{tag}: vbatt_src {row['vbatt_src']!r}")
        parsed.append((row, morph, faults))

    # every row's faults == its scenario's faults restricted to its modules
    for row, morph, faults in parsed:
        want = {k: v for k, v in scen_union[row["scenario_id"]].items() if k[0] in morph}
        got = {(f.module, f.corner): f.key() for f in faults}
        if want != got:
            errs.append(f"seq {row['seq']}: faults {sorted(got.values())} != scenario {row['scenario_id']} "
                        f"restricted to its modules {sorted(want.values())}")
    # seq: unique in the run sheet; in the summary a repeat is allowed only after an aborted attempt
    for seq, rs in seqs.items():
        if kind == "runsheet" and len(rs) > 1:
            errs.append(f"seq {seq} repeated in run sheet")
        if kind == "summary" and len(rs) > 1 and any(r["aborted"] != "1" for r in rs[:-1]):
            errs.append(f"seq {seq} repeated without an earlier aborted attempt")
    # chains: one scenario and one session each
    chains: dict[str, set] = {}
    for row, _, _ in parsed:
        if row["chain_id"]:
            chains.setdefault(row["chain_id"], set()).add(
                (row["scenario_id"], row.get("session_id", "")))
    for c, s in chains.items():
        if len(s) != 1:
            errs.append(f"chain {c} spans several scenarios/sessions: {sorted(s)}")
    return errs


# ------------------------------------------------------------------ pairing
def _record(row: dict) -> dict:
    morph = parse_morph(row["morph"])
    faults = parse_faults(row, parse_slot_map(row["slot_map"]))
    return {"row": row, "morph": morph, "mods": frozenset(morph),
            "faults": frozenset(f.key() for f in faults)}


def _rel_positions(morph: dict, mods) -> dict:
    mods = sorted(mods)
    c0 = min(morph[m][0] for m in mods)
    r0 = min(morph[m][1] for m in mods)
    return {m: (morph[m][0] - c0, morph[m][1] - r0, morph[m][2]) for m in mods}


def _usable(row: dict) -> bool:
    return row.get("aborted", "0") != "1" and row.get("outcome_code", "1") in ("1", "")


class ProspectiveGuardError(RuntimeError):
    pass


def split_roles(rows: list[dict], predictions_path: str | Path | None = None) -> tuple[list[dict], list[dict]]:
    """Return (obs rows, target rows). C5 targets are withheld until the C5 predictions file exists."""
    obs = [r for r in rows if r["role"] == "obs" and _usable(r)]
    tgt = [r for r in rows if r["role"] == "target" and r["routine"] == "DL" and _usable(r)]
    if any(r["block"] == "C5" for r in tgt):
        if predictions_path is None or not Path(predictions_path).exists():
            tgt = [r for r in tgt if r["block"] != "C5"]
    return obs, tgt


def load_c5_targets(rows: list[dict], predictions_path: str | Path) -> list[dict]:
    """Strict accessor for the confirmatory block: raises unless predictions were written first."""
    if not Path(predictions_path).exists():
        raise ProspectiveGuardError(f"write C5 predictions to {predictions_path} before reading C5 targets")
    return [r for r in rows if r["block"] == "C5" and r["role"] == "target" and _usable(r)]


def build_pairs(rows: list[dict], predictions_path: str | Path | None = None,
                obs_roles=("obs", "target")) -> list[dict]:
    """Pairs (observation trial -> measured target configuration).

    kind = "split": target modules strictly inside the observed assembly, same relative positions;
           "rearrange": target modules inside the observed assembly, different arrangement;
           "merge": observed modules strictly inside the target assembly (observation from a smaller piece).
    Faults must match exactly on the shared modules (software-injected faults are reproducible), and plate state
    must match when given. Merge observations may come from any DL row (children and singles are observations
    for a merge); split observations are role "obs" rows (parent DL and probes).
    """
    obs_rows, targets = split_roles(rows, predictions_path)
    merge_src = [r for r in rows if r["role"] in obs_roles and r["routine"] == "DL" and _usable(r)]
    T = [_record(r) for r in targets]
    out = []

    def same_plate(a, b):
        pa, pb = a["row"].get("plate_state", ""), b["row"].get("plate_state", "")
        return not pa or not pb or pa == pb

    def mk(kind, o, t):
        orow, trow = o["row"], t["row"]
        rec = {"kind": kind, "obs_seq": orow["seq"], "obs_trial": orow.get("trial_id", ""),
               "obs_shape": orow["shape"], "obs_routine": orow["routine"], "target_seq": trow["seq"],
               "target_trial": trow.get("trial_id", ""), "target_shape": trow["shape"],
               "target_morph": trow["morph"], "scenario_obs": orow["scenario_id"],
               "scenario_target": trow["scenario_id"],
               "same_chain": bool(orow["chain_id"]) and orow["chain_id"] == trow["chain_id"],
               "same_session": orow.get("session_id", "") == trow.get("session_id", ""),
               "confirmatory": trow["block"] == "C5",
               "obs_payload": orow.get("payload_spec", ""), "target_payload": trow.get("payload_spec", "")}
        for c in RATIO_COLUMNS:
            rec["target_" + c] = _num(trow.get(c))
        return rec

    for o in (_record(r) for r in obs_rows):
        for t in T:
            if t["row"] is o["row"] or not same_plate(o, t):
                continue
            if t["mods"] < o["mods"] or (t["mods"] == o["mods"] and t["morph"] != o["morph"]):
                if {f for f in o["faults"] if f[0] in t["mods"]} != t["faults"]:
                    continue
                same = _rel_positions(o["morph"], t["mods"]) == _rel_positions(t["morph"], t["mods"])
                out.append(mk("split" if same else "rearrange", o, t))
    # merge targets: the held-out merge shapes (C2) and, exploratory, the R6 parent decision legs
    MT = [t for t in T if t["row"]["block"] == "C2"]
    MT += [_record(r) for r in rows if r["shape"] == "R6" and r["routine"] == "DL" and _usable(r)
           and not r["payload_spec"] and r["block"] != "C5"]
    for o in (_record(r) for r in merge_src):
        for t in MT:
            if t["row"] is o["row"] or not same_plate(o, t) or not o["mods"] < t["mods"]:
                continue
            if {f for f in t["faults"] if f[0] in o["mods"]} != o["faults"]:
                continue
            out.append(mk("merge", o, t))
    return out


def merge_cover(target_row: dict, rows: list[dict]) -> bool:
    """True if usable DL observations with matching faults jointly cover every module of a merge target."""
    t = _record(target_row)
    covered = set()
    for r in rows:
        if r is target_row or r["routine"] != "DL" or not _usable(r):
            continue
        o = _record(r)
        if o["mods"] < t["mods"] and {f for f in t["faults"] if f[0] in o["mods"]} == o["faults"]:
            covered |= o["mods"]
    return covered == set(t["mods"])


# ------------------------------------------------------------------ 05 (capxfer) rows, cfcap's derived view
CAPXFER_LAYOUTS = {
    "pair": ({"M1": (0, 0), "M3": (1, 0)}, ["M1", "M3"]),
    "L3": ({"M2": (0, 0), "M3": (1, 0), "M4": (0, 1)}, ["M2", "M3", "M4"]),
    "square4": ({"M1": (0, 0), "M2": (1, 0), "M3": (0, 1), "M4": (1, 1)}, ["M1", "M2", "M3", "M4"]),
}


def from_capxfer_row(row: dict) -> dict:
    """Map a 05_capxfer_runsheet/summary row into cfcap columns ("cfcap's derived view"; capxfer's own analysis
    is unchanged). Lone rows become targets, assembly rows observations; the scenario id is synthetic."""
    lay = row["layout"]
    if lay == "single":
        pos, order = {row["module_1"]: (0, 0)}, [row["module_1"]]
    else:
        pos, order = CAPXFER_LAYOUTS[lay]
    morph = ";".join(f"{m}@{pos[m][0]},{pos[m][1]},0" for m in order)
    slot_map = ";".join(f"m{i}={m}" for i, m in enumerate(order, 1))
    out = {c: "" for c in RUNSHEET_COLUMNS}
    out.update({
        "seq": f"05-{row['seq']}", "round": row["round"], "block": "C1e" if lay == "single" else "C1a",
        "priority": "1", "shape": {"single": "S1", "pair": "PR2", "L3": "L3a", "square4": "SQ4"}[lay],
        "n_modules": str(len(order)), "morph": morph, "slot_map": slot_map,
        "corpus_layout": lay, "role": "target" if lay == "single" else "obs",
        "fault_type": row.get("fault_type", "0") or "0", "fault_actuator": row.get("fault_actuator", ""),
        "fault_level": row.get("fault_level", ""), "routine": "DL", "cmd_channel": "body",
        "notes": "capxfer 11-24 row, cfcap's derived view (no diagonal legs)",
    })
    for k in ("session_id", "trial_id", "aborted", "outcome_code", "plate_state", *RATIO_COLUMNS):
        if k in row:
            out[k] = row[k]
    faults = parse_faults(out, parse_slot_map(slot_map))
    out["scenario_id"] = "F00" if not faults else "X" + "".join(
        f"{f.module}{f.corner}{f.ftype}" for f in faults)[:12]
    return out
