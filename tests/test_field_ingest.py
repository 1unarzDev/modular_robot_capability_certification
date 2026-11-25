"""Mock-row tests for the cfcap field ingest (cfcap/field/ingest.py) and the run-sheet generator."""
import copy
import importlib.util
import random
from pathlib import Path

import pytest

from cfcap.field import ingest

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_runsheet", ROOT / "scripts" / "make_runsheet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_generator()


def runsheet():
    return [{k: str(v) for k, v in r.items()} for r in GEN.build_rows()]


def mock_summary(rows, seed=0):
    rng = random.Random(seed)
    out = []
    for i, r in enumerate(rows):
        s = dict(r)
        s.update({"session_id": "S20261201", "trial_id": f"T{i:04d}", "outcome_code": "1", "abort_code": "0",
                  "config_id": "", "duration_s": "16.5", "portion_code": "216", "plate_state": "plated",
                  "aborted": "0", "vbatt_src": "log", "disp_max_m": "", "abort_reason": ""})
        for c in ingest.RATIO_COLUMNS:
            s[c] = f"{rng.uniform(0.5, 1.0):.3f}" if r["routine"] == "DL" else ""
        mods = ingest.parse_morph(r["morph"])
        for c in ingest.VBATT_COLUMNS:
            s[c] = "12.4" if c.split("_")[1] in mods else ""
        out.append(s)
    return out


# ---------------------------------------------------------------- generator
def test_runsheet_deterministic_and_valid():
    a, b = runsheet(), runsheet()
    assert a == b
    assert ingest.validate(a, "runsheet") == []
    assert len(a) == len({r["seq"] for r in a})


def test_runsheet_shape_blocking_and_cooldown():
    rows = runsheet()
    # shapes come in contiguous blocks: count shape changes, must be small
    changes = sum(1 for x, y in zip(rows, rows[1:]) if x["morph"] != y["morph"])
    singles_changes = sum(1 for x, y in zip(rows, rows[1:]) if x["morph"] != y["morph"] and x["shape"] == "S1")
    assert changes - singles_changes <= 25
    run = 0
    for r in rows:
        run = run + 1 if r["routine"] in ("P_tug", "P_mdiff") else 0
        assert run <= 2


def test_must_have_counts():
    rows = runsheet()
    must = [r for r in rows if r["priority"] == "1"]
    assert 130 <= len(must) <= 145


# ---------------------------------------------------------------- parsers
def test_parse_morph_errors():
    assert ingest.parse_morph("M1@0,0,0;M2@1,0,0") == {"M1": (0, 0, 0), "M2": (1, 0, 0)}
    for bad in ("M1@0,0,0;M2@2,0,0", "M1@0,0,0;M2@0,0,0", "M1@0,0,4", "X1@0,0,0", ""):
        with pytest.raises(ValueError):
            ingest.parse_morph(bad)


def test_parse_faults_two_and_levels():
    row = {"fault_type": "201", "fault_actuator": "m2_rr", "fault_level": "0.55",
           "fault_type_2": "203", "fault_actuator_2": "m1_fl", "fault_level_2": ""}
    f = ingest.parse_faults(row, {"m1": "M5", "m2": "M2"})
    assert {(x.module, x.corner, x.ftype, x.level) for x in f} == {("M2", "rr", 201, 0.55), ("M5", "fl", 203, None)}
    with pytest.raises(ValueError):
        ingest.parse_faults({**row, "fault_level": ""}, {"m1": "M5", "m2": "M2"})
    with pytest.raises(ValueError):
        ingest.parse_faults({**row, "fault_level_2": "0.5"}, {"m1": "M5", "m2": "M2"})
    with pytest.raises(ValueError):
        ingest.parse_faults({**row, "fault_actuator_2": "m2_rr", "fault_type_2": "202"}, {"m1": "M5", "m2": "M2"})


# ---------------------------------------------------------------- validation catches mistakes
def _first(rows, **kw):
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


def test_validate_catches_errors():
    rows = runsheet()
    bad = copy.deepcopy(rows)
    _first(bad, block="C1c", shape="SQ4", scenario_id="F01")["fault_level"] = "0.60"   # scenario conflict
    _first(bad, routine="P_tug", block="C3")["cmd_channel"] = "body"                  # channel mismatch
    r = _first(bad, block="C4", shape="S1")
    r["payload_spec"] = r["payload_spec"].replace("M2", "M1")                            # M1 has no lifter / absent
    _first(bad, block="C2", shape="P6")["morph"] = "M1@0,0,0;M2@5,5,0"                   # disconnected
    errs = ingest.validate(bad, "runsheet")
    text = "\n".join(errs)
    assert "has ('M2', 'rr', 201, 0.6)" in text or "0.6" in text
    assert "cmd_channel" in text
    assert "no lifter" in text or "not in morph" in text
    assert "not 4-connected" in text


def test_validate_missing_fault_in_child():
    rows = runsheet()
    bad = copy.deepcopy(rows)
    r = _first(bad, block="C1c", shape="SQ4", scenario_id="F07")
    r["fault_type"], r["fault_actuator"], r["fault_level"] = "0", "", ""
    errs = ingest.validate(bad, "runsheet")
    assert any("restricted to its modules" in e for e in errs)


def test_summary_valid_and_rerun_rule():
    s = mock_summary(runsheet())
    assert ingest.validate(s, "summary") == []
    dup = copy.deepcopy(s[20])
    s2 = s[:21] + [dup] + s[21:]
    assert any("repeated" in e for e in ingest.validate(s2, "summary"))
    s2[20]["aborted"] = "1"
    assert not any("repeated" in e for e in ingest.validate(s2, "summary"))
    s3 = copy.deepcopy(s)
    _first(s3, routine="P_clamp")["ratio_surge_pos"] = "0.9"
    _first(s3, shape="PR2")["vbatt_M1"] = "12.0"
    s3[5]["plate_state"] = "plateless"
    t = "\n".join(ingest.validate(s3, "summary"))
    assert "only for routine DL" in t and "not in the assembly" in t and "plate_state changes" in t


# ---------------------------------------------------------------- pairs
def test_pairs_split_merge_and_guard(tmp_path):
    s = mock_summary(runsheet())
    pairs = ingest.build_pairs(s)
    kinds = {p["kind"] for p in pairs}
    assert {"split", "merge"} <= kinds
    # F07 parent (M2 fr weak 0.55 + M5 fl brake) -> SQ4 child carries only the M2 fault, PR2 only the M5 fault
    f07 = [p for p in pairs if p["kind"] == "split" and p["scenario_obs"] == "F07" and p["obs_shape"] == "R6"]
    assert {p["target_shape"] for p in f07} >= {"SQ4", "PR2", "L3a", "L3b", "S1"}
    # chain pairs are flagged
    assert any(p["same_chain"] for p in pairs)
    # a healthy child pairs with every parent whose faults lie outside it (software reproducibility)
    pr2_healthy = [p for p in pairs if p["kind"] == "split" and p["target_shape"] == "PR2"
                   and p["scenario_target"] == "F00" and p["obs_shape"] == "R6"]
    assert {p["scenario_obs"] for p in pr2_healthy} >= {"F00", "F01", "F02", "F04", "F06"}
    # merges into the held-out shapes are present; P6 is covered by children observations
    assert any(p["kind"] == "merge" and p["target_shape"] == "P6" for p in pairs)
    p6 = _first(s, block="C2", shape="P6", scenario_id="F07")
    assert ingest.merge_cover(p6, s)
    # C5 targets are withheld until predictions exist
    assert not any(p["confirmatory"] for p in pairs)
    with pytest.raises(ingest.ProspectiveGuardError):
        ingest.load_c5_targets(s, tmp_path / "pred.csv")
    (tmp_path / "pred.csv").write_text("x\n")
    pairs2 = ingest.build_pairs(s, predictions_path=tmp_path / "pred.csv")
    assert any(p["confirmatory"] for p in pairs2)
    assert len(ingest.load_c5_targets(s, tmp_path / "pred.csv")) > 0


def test_aborted_rows_excluded():
    s = mock_summary(runsheet())
    for r in s:
        if r["shape"] == "PR2":
            r["aborted"] = "1"
    assert not any(p["target_shape"] == "PR2" for p in ingest.build_pairs(s))


def test_capxfer_rows_derived_view():
    rows = [
        {"seq": "1", "round": "R1", "layout": "single", "module_1": "M1", "fault_type": "202",
         "fault_actuator": "m1_fl", "fault_level": ""},
        {"seq": "2", "round": "R2", "layout": "square4", "module_1": "M1", "fault_type": "202",
         "fault_actuator": "m1_fl", "fault_level": ""},
        {"seq": "3", "round": "R2", "layout": "L3", "module_1": "M2", "fault_type": "201",
         "fault_actuator": "m3_rr", "fault_level": "0.45"},
    ]
    conv = [ingest.from_capxfer_row(r) for r in rows]
    assert conv[2]["morph"] == "M2@0,0,0;M3@1,0,0;M4@0,1,0"
    assert ingest.parse_faults(conv[2], ingest.parse_slot_map(conv[2]["slot_map"]))[0].module == "M4"
    pairs = ingest.build_pairs(conv)
    assert [(p["obs_shape"], p["target_shape"], p["kind"]) for p in pairs] == [("SQ4", "S1", "split")]
