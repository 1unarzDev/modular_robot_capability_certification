"""scripts/codex_annotate.py (batch codex pipeline, stubbed codex) and scripts/validate_annotations.py (validators)."""
import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


CA = _load("codex_annotate")
VA = _load("validate_annotations")

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {
    "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["case_id", "label", "x"],
                               "properties": {"case_id": {"type": "string"},
                                              "label": {"type": "string", "enum": ["a", "b"]},
                                              "x": {"type": "number", "minimum": 0, "maximum": 1}}}}}}

STUB = textwrap.dedent('''
    import json, os, sys
    args = sys.argv[1:]
    out = args[args.index("-o") + 1]
    prompt = sys.stdin.read()
    cases = [json.loads(l) for l in prompt.splitlines() if l.startswith("{")]
    mode = os.environ.get("STUB_MODE", "ok")
    log = os.environ["STUB_LOG"]
    with open(log, "a") as fh:
        fh.write(json.dumps([c["case_id"] for c in cases]) + "\\n")
    n_calls = sum(1 for _ in open(log))
    if mode == "fail_first" and n_calls == 1:
        sys.stderr.write("upstream 503\\n"); sys.exit(1)
    items = [dict(case_id=c["case_id"], label="a" if c["v"] < 5 else "b", x=c["v"] / 10) for c in cases]
    if mode == "drop_one" and n_calls == 1:
        items = items[1:]
    if mode == "bad_item" and n_calls == 1:
        items[0]["label"] = "zzz"
    if mode == "bad_json" and n_calls == 1:
        open(out, "w").write("not json"); print(json.dumps({"type": "turn.completed"})); sys.exit(0)
    if mode == "quota_first" and n_calls == 1:
        print(json.dumps({"type": "turn.failed", "error": {"message": "You've hit your usage limit. Try again at 6:52 PM."}}))
        sys.exit(1)
    if mode == "always_fail":
        sys.exit(2)
    open(out, "w").write(json.dumps({"items": items}))
    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 10,
                                                         "output_tokens": 5, "reasoning_output_tokens": 1}}))
''')


@pytest.fixture
def env(tmp_path, monkeypatch):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    monkeypatch.setenv("CODEX_BIN", f"{sys.executable} {stub}")
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "log.txt"))
    sch = tmp_path / "schema.json"
    sch.write_text(json.dumps(SCHEMA))
    cases = [dict(case_id=f"c{i}", v=i) for i in range(10)]
    return tmp_path, sch, cases


def _run(env, **kw):
    tmp, sch, cases = env
    args = dict(chunk=3, parallel=2, max_parallel=4, retries=3, res_dir=tmp / "res", out_dir=tmp / "out",
                backoff0=0.0, log=lambda s: None)
    args.update(kw)
    return CA.run("t", args.pop("cases", cases), "Annotate:\n{{CASES}}\nn={{N}} {{FOO}}", sch, vars_={"FOO": "bar"},
                  **args)


def _records(tmp):
    return [json.loads(l) for l in (tmp / "res" / "t.jsonl").read_text().splitlines()]


def test_schema_errors():
    it = SCHEMA["properties"]["items"]["items"]
    assert CA.schema_errors(dict(case_id="a", label="a", x=0.5), it) == []
    assert CA.schema_errors(dict(case_id="a", label="c", x=0.5), it)
    assert CA.schema_errors(dict(case_id="a", label="a", x=1.5), it)
    assert CA.schema_errors(dict(case_id="a", label="a"), it)
    assert CA.schema_errors(dict(case_id="a", label="a", x=0.1, y=1), it)
    assert CA.schema_errors(True, {"type": "number"})                     # bool is not a number
    assert CA.schema_errors(None, {"anyOf": [{"type": "number"}, {"type": "null"}]}) == []


def test_render_and_usage():
    txt = CA.render("{{N}} {{IDS}} {{K}}\n{{CASES}}", [dict(case_id="a", v=1)], "case_id", {"K": "kk"})
    assert txt.startswith('1 ["a"] kk') and '{"case_id":"a","v":1}' in txt
    u = CA.parse_usage('{"type":"turn.completed","usage":{"input_tokens":7,"output_tokens":2}}\nnoise')
    assert u["input_tokens"] == 7 and u["output_tokens"] == 2


def test_run_all_valid(env):
    tmp = env[0]
    out = _run(env)
    assert out["n_done"] == 10 and out["n_failed"] == 0
    recs = _records(tmp)
    assert sorted(r["case_id"] for r in recs) == sorted(f"c{i}" for i in range(10))
    assert all(r["label"] == ("a" if int(r["case_id"][1:]) < 5 else "b") for r in recs)
    s = CA.summarize_calls(tmp / "res" / "t_calls.csv")
    assert s["calls"] == 4 and s["valid_items"] == 10 and s["input_tokens"] == 400
    prompt = next((tmp / "out" / "t").glob("chunk*.prompt.txt")).read_text()
    assert "n=3 bar" in prompt or "n=1 bar" in prompt


def test_resume_skips_done(env):
    tmp = env[0]
    _run(env)
    n_log = len((tmp / "log.txt").read_text().splitlines())
    out = _run(env)
    assert out["calls"] == 0 and len((tmp / "log.txt").read_text().splitlines()) == n_log
    more = env[2] + [dict(case_id="c10", v=3)]
    out = _run(env, cases=more)
    assert out["calls"] == 1 and len(_records(tmp)) == 11


@pytest.mark.parametrize("mode", ["fail_first", "drop_one", "bad_item", "bad_json"])
def test_retry_recovers(env, monkeypatch, mode):
    monkeypatch.setenv("STUB_MODE", mode)
    tmp = env[0]
    out = _run(env, parallel=1, max_parallel=1)
    assert out["n_done"] == 10 and out["n_failed"] == 0
    recs = _records(tmp)
    assert len(recs) == len({r["case_id"] for r in recs}) == 10                 # no duplicates
    import csv
    rows = list(csv.DictReader(open(tmp / "res" / "t_calls.csv")))
    assert rows[0]["status"] in ("fail", "partial") and any(r["attempt"] == "2" for r in rows)


def test_quota_reset_parse():
    from datetime import datetime
    now = datetime(2026, 11, 25, 14, 40)
    t = CA.quota_reset("You've hit your usage limit ... try again at 6:52 PM.", now)
    assert datetime.fromtimestamp(t) == datetime(2026, 11, 25, 18, 54)
    t2 = CA.quota_reset("usage limit, try again at 1:05 PM", now)            # already past -> tomorrow
    assert datetime.fromtimestamp(t2).day == 26
    assert CA.quota_reset("all good", now) is None
    assert CA.quota_reset("502 Bad Gateway: Previous response owner account is unavailable", now) is not None


def test_quota_pauses_without_using_retries(env, monkeypatch):
    import time as _t
    monkeypatch.setenv("STUB_MODE", "quota_first")
    monkeypatch.setattr(CA, "quota_reset", lambda text, now=None: _t.time() + 0.3 if "usage limit" in text else None)
    tmp = env[0]
    out = _run(env, parallel=1, max_parallel=1, retries=1)          # retries=1: a counted failure would lose cases
    assert out["n_done"] == 10 and out["n_failed"] == 0
    import csv
    rows = list(csv.DictReader(open(tmp / "res" / "t_calls.csv")))
    assert rows[0]["status"] == "quota" and all(r["attempt"] == "1" for r in rows)


def test_always_fail_gives_up(env, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "always_fail")
    out = _run(env, retries=2)
    assert out["n_done"] == 0 and out["n_failed"] == 10


def test_limiter_aimd():
    L = CA.Limiter(6, 8, raise_after=2, backoff0=1.0)
    for _ in range(4):
        L.ok()
    assert L.limit == 8
    L.ok(); L.ok()
    assert L.limit == 8
    assert L.fail() == 1.0 and L.limit == 4
    assert L.fail() == 2.0 and L.limit == 2
    L.fail(); L.fail()
    assert L.limit == 1


# ----------------------------------------------------------------------------- validators
def _case_t0():
    return dict(case_id="x", source="t0_decision", kinds=["unsafe_cert"],
                truth=dict(wheels=[dict(wheel="m0_fl", mode="weak", g=0.4, stall=1.0, noload=1.0, in_candidate=1),
                                   dict(wheel="m0_fr", mode="ok", g=1.0, stall=0.95, noload=1.0, in_candidate=1),
                                   dict(wheel="m1_fl", mode="coast", g=1.0, stall=1.0, noload=1.0, in_candidate=0)],
                           modules_mu={"m0": 1.05}, motor_spread_halfwidth=0.08),
                candidate=dict(wheels=["m0_fl", "m0_fr", "m0_rl", "m0_rr"]),
                legs={"sway+": dict(r=0.8, truth=0.62, mean_v2=0.7, sd_v2=0.05, lower_v2=0.65, p_leg_v2=0.3)})


def _cl(kind, target="", op="eq", value=1, mode=""):
    return dict(kind=kind, target=target, op=op, value=value, mode=mode, text="")


def test_check_claims():
    ix = VA.truth_index(_case_t0(), dict(truth_in_screen=1, truth_mass=0.2, probe_resolves={"clamp": 1, "tug": 0}))
    ok = [_cl("wheel_mode", "m0_fl", mode="weak"), _cl("wheel_gain", "m0_fl", "lt", 0.5),
          _cl("wheel_mode", "m0_rl", mode="ok"),                            # unlisted wheel = healthy
          _cl("leg_truth", "sway+", "lt", 0.8), _cl("leg_margin", "sway+", "gt", 0.1),
          _cl("candidate_has_fault", value=1), _cl("truth_in_screen", value=1), _cl("truth_mass", op="lt", value=0.5),
          _cl("wheel_in_candidate", "m0_fl", value=1), _cl("post_sd", "sway+", "lt", 0.1),
          _cl("probe_resolves", "clamp", value=1), _cl("probe_resolves", "v2+tug", value=0),
          _cl("module_mu", "m0", "gt", 1.0), _cl("wheel_stall", "m0_fr", "lt", 0.96),
          _cl("spread_masked", "m0_fl", value=0)]
    bad = [_cl("wheel_mode", "m0_fr", mode="weak"), _cl("wheel_gain", "m0_fl", "gt", 0.5),
           _cl("leg_truth", "sway+", "gt", 0.8), _cl("wheel_in_candidate", "m1_fl", value=1),
           _cl("probe_resolves", "tug", value=1)]
    assert [VA.check_claim(ix, k)[0] for k in ok] == [True] * len(ok)
    assert [VA.check_claim(ix, k)[0] for k in bad] == [False] * len(bad)
    assert VA.check_claim(ix, _cl("wheel_gain", "m9_zz", "lt", 0.5))[0] is None
    assert VA.check_claim(ix, _cl("leg_truth", "yaw+", "lt", 0.5))[0] is None
    assert VA.check_claim(ix, _cl("condition_flip", "base", value=1))[0] is None


def test_signatures():
    c = _case_t0()
    ix = VA.truth_index(c, dict(truth_in_screen=0, truth_mass=0.0, probe_resolves={"split": 1}))
    assert VA._sig_screen_miss(ix) and not VA._sig_spread_masked(ix) and VA._sig_genuine_incapable(ix)
    assert VA._sig_split_only(ix) and not VA._sig_unexcited(ix)
    c5 = dict(case_id="y", source="a5_flip", kinds=["flip_down"], truth=dict(wheels=[]), candidate=dict(wheels=[]),
              legs={"yaw+": dict(tau=0.8, t1=0.81, t2=0.79)})
    ix5 = VA.truth_index(c5, dict(condition_flip={"base": 0, "aniso_mu": 1, "all-aniso_mu": 0}))
    assert VA._sig_friction_scatter(ix5) and not VA._sig_kernel_numerics(ix5) and VA._sig_near_threshold(ix5)
    assert VA.check_claim(ix5, _cl("leg_shift", "yaw+", "lt", 0.0))[0]


def test_taxonomy_table_parse(tmp_path):
    p = tmp_path / "T.md"
    p.write_text("| `screen_miss` | `screen_miss` | x |\n| `other` | none | y |\n")
    cats = VA.load_taxonomy(p)
    assert cats["screen_miss"] is VA._sig_screen_miss and cats["other"] is None


def test_task_r_from_physics():
    m = dict(surge_mps=0.2, sway_mps=0.0, diag_mps=0.0, yaw_dps=0.0, surge_dirs=["surge+", "surge-"], sway_dirs=[],
             diag_dirs=[], yaw_dirs=[])
    r = VA.task_r(dict(m, team_modules=1))
    assert np.isclose(r["surge+"], 0.2 / (0.3 * VA.RHO_REF)) and np.isclose(r["surge-"], r["surge+"])
    assert np.isnan(r["sway+"]) and np.isnan(r["yaw+"])
    r2 = VA.task_r(dict(m, surge_mps=0.0, yaw_dps=45.0, yaw_dirs=["yaw+"], team_modules=6))
    assert np.isclose(r2["yaw+"], np.deg2rad(45) / (1.0 * VA.yaw_scale_n(6) * VA.RHO_REF)) and np.isnan(r2["yaw-"])
    r3 = VA.task_r(dict(m, surge_mps=0.0, diag_mps=0.1, diag_dirs=["diag++"], team_modules=2))
    assert np.isclose(r3["diag++"], 0.1 / (0.21 * np.sqrt(2) * VA.RHO_REF)) and np.isnan(r3["diag--"])


def test_task_consistency():
    m = dict(surge_mps=0.2, sway_mps=0.0, diag_mps=0.0, yaw_dps=0.0, surge_dirs=["surge+"], sway_dirs=[],
             diag_dirs=[], yaw_dirs=[])
    row = dict(title="Shelf pick", description="Slide sideways along the shelf, then reverse out.",
               environment=dict(aisle_width_m=0.3, constraints=[]), payload_kg=12, team_modules=2)
    fl = VA.consistency(row, m)
    assert "text needs sway, no sway" in fl and "text needs reverse, no rearward leg" in fl
    assert VA.consistency(dict(row, description="Drive forward without sliding sideways."), dict(m, surge_dirs=["surge+"])) == ["team too small for payload"]
    assert "team too small for payload" in fl
    ok = dict(m, sway_mps=0.1, sway_dirs=["sway+", "sway-"], surge_dirs=["surge+", "surge-"])
    assert VA.consistency(dict(row, payload_kg=8), ok) == []


def test_faithfulness():
    ix = VA.truth_index(_case_t0(), {})
    f = VA.faithful(ix, dict(blamed_wheels=["m0_fl"], blamed_legs=["sway+"], blames_uncertainty=False))
    assert f["wheel_precision"] == 1.0 and f["leg_precision"] == 1.0 and f["faithful"] and f["wheel_recall"] == 1.0
    f2 = VA.faithful(ix, dict(blamed_wheels=["m0_fr"], blamed_legs=["sway+"], blames_uncertainty=False))
    assert f2["wheel_precision"] == 0.0 and not f2["faithful"]
    f3 = VA.faithful(ix, dict(blamed_wheels=[], blamed_legs=[], blames_uncertainty=False))
    assert not f3["faithful"]
