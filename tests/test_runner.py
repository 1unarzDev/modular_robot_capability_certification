"""cfcap.runner (shard wrapper) and scripts/make_jobs.py (job emission). Fast: the selftest task with tiny units."""
import json
import math
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from cfcap import runner as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_jobs as MJ  # noqa: E402


def run_main(args, monkeypatch, env=None):
    for k in ("CRANE_JOB_ID", "CRANE_RUN_DIR", "CRANE_OUT_ROOT", "CRANE_INPUTS_DIR"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    return R.main(args)


def spawn(args, env):
    e = {**os.environ, "PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}", **env}
    for k in ("CRANE_JOB_ID", "CRANE_RUN_DIR", "CRANE_OUT_ROOT"):
        if k not in env:
            e.pop(k, None)
    return subprocess.Popen([sys.executable, "-m", "cfcap.runner", *args], env=e, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def wait_for(pred, timeout=30.0):
    t = time.time()
    while time.time() - t < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def read(p):
    return json.loads(Path(p).read_text())


ST = ["selftest", "--units", "4", "--rows", "50", "--unit-s", "0.05", "--steps", "2"]


def test_complete_outputs_manifest_metrics(tmp_path, monkeypatch):
    out, run = tmp_path / "out", tmp_path / "run"
    assert run_main([*ST, "--out", str(out)], monkeypatch, {"CRANE_RUN_DIR": str(run)}) == 0
    m = read(out / "manifest.json")
    assert m["status"] == "complete" and m["units_done"] == 4 and m["rows"] == 200
    import numpy as np
    for f in m["files"]:
        z = np.load(out / f["path"])
        assert set(z.files) == {"x", "y", "i"} and len(z["x"]) == 50
        assert R.sha256_file(out / f["path"]) == f["sha256"]
    met = read(run / "metrics.json")
    assert met["units_done"] == 4 and met["interrupted"] == 0 and len(met) <= 1000
    assert all(isinstance(v, float) and math.isfinite(v) for v in met.values())
    assert read(run / "progress.json")["done"] == 4
    assert (run / "artifacts" / "manifest.json").is_file()        # a small record next to the run


def test_deterministic_across_shards_and_reruns(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    assert run_main([*ST, "--out", str(a)], monkeypatch) == 0
    assert run_main([*ST, "--out", str(b)], monkeypatch) == 0
    assert [f["sha256"] for f in read(a / "manifest.json")["files"]] == \
           [f["sha256"] for f in read(b / "manifest.json")["files"]]
    c = tmp_path / "c"
    assert run_main([*ST, "--shard", "1", "--n-shards", "2", "--out", str(c)], monkeypatch) == 0
    assert read(c / "manifest.json")["files"][0]["sha256"] != read(a / "manifest.json")["files"][0]["sha256"]


def test_resume_skips_finished_units(tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run_main([*ST, "--out", str(out), "--max-units", "2"], monkeypatch) == R.EXIT_YIELD
    m = read(out / "manifest.json")
    assert m["status"] == "interrupted" and m["units_done"] == 2
    first = {f["unit"]: (out / f["path"]).stat().st_mtime_ns for f in m["files"]}
    assert run_main([*ST, "--out", str(out)], monkeypatch) == 0
    m2 = read(out / "manifest.json")
    assert m2["status"] == "complete" and m2["units_done"] == 4
    assert all((out / "units" / f"{u}.npz").stat().st_mtime_ns == t for u, t in first.items())
    assert read(out / "metrics.json")["units_resumed"] == 2


def test_torn_unit_is_redone(tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run_main([*ST, "--out", str(out)], monkeypatch) == 0
    f = read(out / "manifest.json")["files"][1]
    (out / f["path"]).write_bytes(b"garbage")
    assert run_main([*ST, "--out", str(out)], monkeypatch) == 0
    assert R.sha256_file(out / f["path"]) == f["sha256"]
    assert read(out / "metrics.json")["units_resumed"] == 3


def test_args_mismatch_refused(tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run_main([*ST, "--out", str(out)], monkeypatch) == 0
    assert run_main([*ST, "--rows", "51", "--out", str(out)], monkeypatch) == R.EXIT_USAGE
    assert run_main([*ST, "--rows", "51", "--out", str(out), "--fresh"], monkeypatch) == 0


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGUSR1])
def test_signal_checkpoints_mid_unit_and_exits_75(tmp_path, sig):
    out, run = tmp_path / "out", tmp_path / "run"
    args = ["selftest", "--units", "3", "--rows", "20", "--unit-s", "3", "--steps", "30", "--out", str(out)]
    p = spawn(args, {"CRANE_RUN_DIR": str(run)})
    assert wait_for(lambda: any((out / "state").glob("*.state")) if (out / "state").is_dir() else False)
    time.sleep(0.3)
    p.send_signal(sig)
    log = p.communicate(timeout=30)[0]
    assert p.returncode == R.EXIT_YIELD, log
    assert read(out / "manifest.json")["status"] == "interrupted"
    assert read(run / "metrics.json")["interrupted"] == 1
    assert list((out / "state").glob("*.state"))                   # the unit's own checkpoint survives
    p = spawn(args, {"CRANE_RUN_DIR": str(run)})
    log = p.communicate(timeout=60)[0]
    assert p.returncode == 0, log
    assert read(run / "metrics.json")["selftest.resumed_mid_unit"] >= 1
    assert not list((out / "state").glob("*.state"))


def test_unit_that_ignores_stop_is_cut_after_grace(tmp_path):
    mod = tmp_path / "stubborn_task.py"
    mod.write_text(textwrap.dedent("""
        import time
        from cfcap.runner import Task
        def run(unit, ctx):
            time.sleep(30)
            return {"a": [1]}
        TASK = Task("stubborn", lambda a, s, n: ["u0"], run)
        """))
    out = tmp_path / "out"
    env = {"PYTHONPATH": f"{ROOT}:{tmp_path}", "CFCAP_GRACE_S": "1"}
    p = spawn(["stubborn", "--task-spec", "stubborn_task:TASK", "--out", str(out)], env)
    assert wait_for(lambda: (out / "manifest.json").is_file())
    time.sleep(0.3)
    t = time.time()
    p.send_signal(signal.SIGTERM)
    log = p.communicate(timeout=40)[0]
    assert p.returncode == R.EXIT_YIELD, log
    assert time.time() - t < 10
    assert read(out / "manifest.json")["units_done"] == 0


def test_kill9_leaves_no_torn_outputs(tmp_path):
    out = tmp_path / "out"
    args = ["selftest", "--units", "6", "--rows", "2000", "--unit-s", "0.4", "--steps", "4", "--out", str(out)]
    p = spawn(args, {})
    assert wait_for(lambda: len(list((out / "units").glob("*.json"))) >= 2 if (out / "units").is_dir() else False)
    p.kill()
    p.wait()
    for j in (out / "units").glob("*.json"):
        rec = read(j)
        assert R.sha256_file(out / "units" / f"{rec['unit']}.npz") == rec["sha256"]
    p = spawn(args, {})
    assert p.communicate(timeout=60) and p.returncode == 0
    assert read(out / "manifest.json")["units_done"] == 6


def test_failing_unit_exit_1_keeps_done_units(tmp_path, monkeypatch):
    out = tmp_path / "out"
    rc = run_main([*ST, "--out", str(out), "--fail-unit", "st-s000-u002"], monkeypatch)
    assert rc == R.EXIT_TASK
    m = read(out / "manifest.json")
    assert m["status"] == "failed" and m["units_done"] == 2
    assert read(out / "metrics.json")["interrupted"] == 1


def test_out_dir_resolution(tmp_path):
    env = {"CRANE_JOB_ID": "crane/cfcap-b1.gen.s003", "CRANE_OUT_ROOT": str(tmp_path / "w"),
           "CRANE_RUN_DIR": str(tmp_path / "run")}
    assert R.resolve_out(None, "t", 3, env) == tmp_path / "w" / "crane" / "cfcap-b1.gen.s003"
    assert R.resolve_out(None, "t", 3, {"CRANE_JOB_ID": "../../x", "CRANE_OUT_ROOT": "/o"}) == Path("/o/x")
    run_only = {"CRANE_RUN_DIR": str(tmp_path / "run")}
    if not Path("/workspace/out").is_dir():
        assert R.resolve_out(None, "t", 3, run_only) == tmp_path / "run" / "artifacts" / "out"
        assert R.resolve_out(None, "t", 3, {"CFCAP_OUT": "/c"}) == Path("/c/runs/t/shard0003")
    assert R.resolve_out("/x", "t", 3, env) == Path("/x")


def test_clean_metrics():
    m = R.clean_metrics({"a": 1, "b": float("nan"), "c": float("inf"), "bad name": 2, "d": True, "e": "x",
                         "f": 2.5, **{f"k{i}": i for i in range(2000)}})
    assert m["a"] == 1.0 and m["f"] == 2.5 and "b" not in m and "c" not in m and "bad name" not in m
    assert "d" not in m and "e" not in m and len(m) == 1000


def test_unknown_and_missing_role_tasks(monkeypatch, capsys):
    assert run_main(["nope"], monkeypatch) == R.EXIT_USAGE
    monkeypatch.setitem(R.TASKS, "ghost", "cfcap.ghost_role_module:TASK")
    monkeypatch.setitem(R.OWNER, "ghost", "ghostrole")
    assert run_main(["ghost"], monkeypatch) == R.EXIT_USAGE
    assert "ghostrole" in capsys.readouterr().err


def test_missing_module_exit_3(monkeypatch):
    t = R.Task("needs", lambda a, s, n: ["u"], lambda u, c: None, requires=("no_such_module_xyz",))
    monkeypatch.setitem(R.TASKS, "needs", "x:y")
    monkeypatch.setattr(R, "load_task", lambda name, spec=None: t)
    assert run_main(["needs"], monkeypatch) == R.EXIT_INPUT


def test_chunk_helpers():
    r = R.shard_range(1000, 3, 8)
    assert (r.start, r.stop) == (375, 500)
    assert sum(len(R.shard_range(1001, k, 7)) for k in range(7)) == 1001
    us = R.chunk_units("ep", r, 40)
    assert us[0] == "ep-0000375-0000415" and R.parse_chunk(us[-1]) == range(495, 500)
    assert all(R.UNIT_ID.match(u) for u in us)
    assert R.unit_seed("a", 1) == R.unit_seed("a", 1) != R.unit_seed("a", 2)


# ------------------------------------------------------------------ make_jobs

PLAN = {"campaign": "cfcap-test", "submitter": "09-cfcap", "stop": "stop when the stage-1 grid is complete",
        "groups": []}
INFO = {"cfcap_commit": "a" * 40, "capxfer_commit": "b" * 40, "capxfer_stage_commit": "c" * 40}


def group(**kw):
    g = {"name": "gen", "task": "gen_shard", "lane": "rented", "profile": "torch", "shards": 3,
         "units_per_shard": 10, "unit_s": 60, "overhead_s": 120, "vram_gb": 6, "pilot": "local 4070 run",
         "purpose": "Generate T1 reference episodes for calibration and test splits of the cfcap study."}
    g.update(kw)
    return g


@pytest.fixture
def staged(tmp_path, monkeypatch):
    monkeypatch.setattr(MJ, "STAGE", tmp_path / "stage")
    (tmp_path / "stage" / "cfcap").mkdir(parents=True)
    (tmp_path / "stage" / "capxfer").mkdir(parents=True)
    return tmp_path / "stage"


def test_rented_gpu_jobs(staged):
    jobs, warn = MJ.group_jobs(PLAN, group(), INFO, draft=False)
    assert len(jobs) == 3 and not warn
    j = jobs[0]
    assert j["kind"] == "gpu_command" and j["est_seconds"] == 720.0
    assert j["needs"]["gpu"] is True and j["needs"]["vram_gb"] == 6.0
    assert j["needs"]["tags"] == ["cloud-ok", "torch", "rented"]
    assert j["spec"]["inputs_rsync"] == [str(staged / "cfcap"), str(staged / "capxfer")]
    assert "/workspace/in/cfcap" in j["spec"]["run"] and "--shard 0 --n-shards 3" in j["spec"]["run"]
    assert "/workspace/inputs" not in j["spec"]["run"]
    assert all(MJ.JOB_ID.match(x["job_id"]) for x in jobs) and len({x["job_id"] for x in jobs}) == 3
    assert "expect" not in j


def test_jax_free_and_limits(staged):
    j = MJ.group_jobs(PLAN, group(profile="jax"), INFO, False)[0][0]
    assert j["needs"]["tags"] == ["cloud-ok", "jax", "rented"]
    f = MJ.group_jobs(PLAN, group(lane="free", profile="cpu", vram_gb=None), INFO, False)[0][0]
    assert f["kind"] == "py_command" and "tags" not in f["needs"] and "gpu" not in f["needs"]
    assert "inputs_rsync" not in f["spec"] and {r["name"] for r in f["repos"]} == {"cfcap", "capxfer_src"}
    assert "$PWD/src" in f["spec"]["run"]
    with pytest.raises(MJ.PlanError, match="60 min"):
        MJ.group_jobs(PLAN, group(lane="free", profile="cpu", units_per_shard=70), INFO, False)
    with pytest.raises(MJ.PlanError, match="240 min"):
        MJ.group_jobs(PLAN, group(units_per_shard=300), INFO, False)
    long = MJ.group_jobs(PLAN, group(units_per_shard=130), INFO, False)[0][0]
    assert long["expect"]["stop"] and long["expect"]["metrics"]


def test_draft_needs_flag(staged):
    with pytest.raises(MJ.PlanError, match="pilot"):
        MJ.group_jobs(PLAN, group(pilot=None), INFO, False)
    jobs, warn = MJ.group_jobs(PLAN, group(pilot=None), INFO, True)
    assert jobs and any("DRAFT" in w for w in warn)


def test_campaign_plans_parse(staged):
    for p in sorted((ROOT / "scripts" / "campaigns").glob("*.json")):
        plan = json.loads(p.read_text())
        for g in plan["groups"]:
            MJ.group_jobs(plan, {**g, "extra_inputs": []}, INFO, draft=True)   # stage-2 inputs exist only later


def test_gen_shard_adapter_maps_shards(tmp_path, monkeypatch):
    """The gen_shard adapter: runner shard k of N -> generator shards shard_range(G, k, N), via write_shard."""
    import types
    calls = []

    def write_shard(out, plate, split, k, n, kinds, **kw):
        calls.append((out, plate, split, k, n, kinds))
        return {"file": f"shard_{k:05d}.npz", "episodes": n, "n_cand": 3 * n, "bytes": 10, "gen_seconds": 1.0}

    fake = types.ModuleType("cfcap.gen.shard")
    fake.write_shard = write_shard
    monkeypatch.setitem(sys.modules, "cfcap.gen.shard", fake)
    out = tmp_path / "out"
    args = ["gen_shard", "--shard", "1", "--n-shards", "3", "--split", "test", "--gen-shards", "10",
            "--episodes", "50", "--out", str(out)]
    assert run_main(args, monkeypatch) == 0
    assert [c[3] for c in calls] == [3, 4, 5] and all(c[0] == out / "gen" and c[4] == 50 for c in calls)
    met = read(out / "metrics.json")
    assert met["gen.episodes"] == 150 and met["gen.candidates"] == 450
    calls.clear()
    assert run_main(args, monkeypatch) == 0 and calls == []          # resumed: nothing regenerated
