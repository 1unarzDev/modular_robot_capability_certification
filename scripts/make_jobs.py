#!/usr/bin/env python3
"""Emit a research-cluster jobs JSON (rc-job/1) for a cfcap campaign plan.

    python scripts/make_jobs.py stage                       # export cfcap@HEAD + capxfer source + theta JSONs
    python scripts/make_jobs.py emit PLAN.json -o jobs.json [--validate] [--draft] [--allow-dirty]
    python scripts/make_jobs.py validate jobs.json          # the coordinator's own checks (read-only, node+esbuild)

Nothing here submits. The main session submits: `python3 ~/research-cluster/bin/research submit -p <project> jobs.json`.

A plan (scripts/campaigns/*.json) is {campaign, submitter, project, stop, groups: [...]}; each group is one task over
`shards` jobs:
  name, task, stage            the runner task (cfcap/runner.py TASKS) and the dependency stage (1 first)
  lane                         rented  gpu_command on a rented 3090, profile torch or jax; inputs pushed by crane-vast
                               cpu     py_command on rented CPU boxes (and GPU boxes' spare cores), inputs pushed
                               free    py_command on free workers only (no cloud-ok tag: never rents); code comes as
                                       repo bundles, so no /workspace paths
  profile                      torch | jax | cpu (the vast.json profile; the needs.tags entry autoscale reads)
  shards, units_per_shard      the job count and the runner units per job
  unit_s, overhead_s           measured wall per unit on the target box (s) and the per-job fixed cost (imports,
                               compile, warm-up); est_seconds = overhead_s + units_per_shard * unit_s, no padding
  pilot                        where unit_s comes from (command, output path, host). Without it the group is a draft.
  vram_gb, task_args, seed, priority, purpose, extra_inputs (stage-2 inputs: pulled-back weights, T2 episodes)

Box contract (compute session, 2026-11-25): code goes as pinned repos (`repos`: cfcap@HEAD and the staged capxfer
repo; `research submit` uploads the bundles; the worker checks them out at <work>/src/<name>). inputs_rsync carries
data only (theta JSONs, stage-2 outputs): each path P lands at /workspace/in/<basename P>, and crane-vast re-pushes
changed inputs to live boxes; the runner waits up to 6 min for them (--require-inputs). Big outputs go to
/workspace/out/$CRANE_JOB_ID/ (the runner's default there), pulled to ~/.cache/vast-out/<box>/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HOME = Path.home()
CFCAP = Path(__file__).resolve().parents[1]
CAPXFER_ROOT = Path(os.environ.get("CAPXFER_ROOT", HOME / "worktrees/capxfer-cluster/capxfer"))
THETA_DIR = Path(os.environ.get("CAPXFER_THETA_DIR", HOME / ".cache/modular-runs/2026-11/capxfer/sysid"))
CFCAP_OUT = Path(os.environ.get("CFCAP_OUT", HOME / ".cache/modular-runs/2026-11/cfcap"))
STAGE = CFCAP_OUT / "compute" / "stage"
RC = HOME / "research-cluster"
VAST_JSON = HOME / "crane-program/compute/vast.json"

CAPXFER_PARTS = ("capxfer", "params", "configs")        # source and the params it reads; no results, no corpus
THETA_FILES = ("theta_plated.json", "theta_plateless.json")
JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
MAX_JOB_S = {"rented": 240 * 60, "cpu": 240 * 60, "free": 60 * 60}   # vast.json max_job_min; innovation 60 min
MIN_JOB_S = 5 * 60
EXPENSIVE_S = 2 * 3600                                   # guard.expensive_compute_h: needs expect
PRICE = {"torch": 0.18, "jax": 0.18, "cpu": 0.10}        # $/box-h planning prices (board README; vast.json caps 0.20)
PROFILES = {"torch", "jax", "cpu"}


class PlanError(Exception):
    pass


def git(root: Path, *a: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)
    if r.returncode:
        raise PlanError(f"git {' '.join(a)} in {root}: {r.stderr.strip()}")
    return r.stdout.strip()


# ------------------------------------------------------------------ stage: what boxes receive

def stage(allow_dirty: bool = False, log=print) -> dict:
    """Code goes to every box as pinned repo bundles (`research submit` uploads them; the worker checks each one out
    at <work dir>/src/<name>): ~/cfcap at HEAD, and STAGE/capxfer, a small git repo of capxfer's package, params,
    configs and theta/*.json (the real capxfer worktree is ~100k files). Data go by inputs_rsync: STAGE/theta (the
    theta JSONs) lands at /workspace/in/theta on rented boxes; the free lane reads the copy inside the capxfer repo."""
    head = git(CFCAP, "rev-parse", "HEAD")
    dirty = git(CFCAP, "status", "--porcelain", "--untracked-files=no", "--", "cfcap", "scripts")
    if dirty and not allow_dirty:
        raise PlanError(f"~/cfcap has uncommitted tracked changes ({dirty.splitlines()[0]}); jobs run HEAD "
                        f"{head[:12]}: commit first or pass --allow-dirty")
    if (STAGE / "cfcap").exists():          # the old code-by-rsync export: code now goes as a repo bundle
        shutil.rmtree(STAGE / "cfcap")

    cx = STAGE / "capxfer"
    cx.mkdir(parents=True, exist_ok=True)
    for part in CAPXFER_PARTS:
        src = CAPXFER_ROOT / part
        if not src.exists():
            raise PlanError(f"capxfer source {src} missing")
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "__pycache__", "--exclude", "*.pyc",
                        f"{src}/", f"{cx / part}/"], check=True)
    (cx / "theta").mkdir(exist_ok=True)
    for f in THETA_FILES:
        if not (THETA_DIR / f).is_file():
            raise PlanError(f"theta file {THETA_DIR / f} missing")
        shutil.copy2(THETA_DIR / f, cx / "theta" / f)
        (STAGE / "theta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(THETA_DIR / f, STAGE / "theta" / f)
    try:
        cx_head = git(CAPXFER_ROOT, "rev-parse", "HEAD")
        cx_dirty = bool(git(CAPXFER_ROOT, "status", "--porcelain", "--untracked-files=no", "--", *CAPXFER_PARTS))
    except PlanError:
        cx_head, cx_dirty = None, None
    info = {"cfcap_commit": head, "capxfer_commit": cx_head, "capxfer_dirty": cx_dirty,
            "theta": {f: (THETA_DIR / f).stat().st_size for f in THETA_FILES}}
    (cx / "STAGE.json").write_text(json.dumps(info, indent=1) + "\n")
    if not (cx / ".git").exists():
        git(cx, "init", "-q")
    git(cx, "add", "-A")
    if git(cx, "status", "--porcelain"):
        subprocess.run(["git", "-C", str(cx), "-c", "user.name=cfcap-stage", "-c", "user.email=stage@localhost",
                        "commit", "-q", "-m", f"stage capxfer {str(cx_head)[:12]} for cfcap {head[:12]}"], check=True)
    info["capxfer_stage_commit"] = git(cx, "rev-parse", "HEAD")
    size = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file() and ".git" not in p.parts)
    log(f"staged {STAGE}: cfcap@{head[:12]}, capxfer@{str(cx_head)[:12]}{' (dirty)' if cx_dirty else ''}, "
        f"{size / 1e6:.1f} MB")
    return info


# ------------------------------------------------------------------ jobs

# Batch canaries (the coordinator gates each batch = project/campaign/kind@refs: 2 jobs lease until one is valid).
# Per task, the flag overrides that shrink one job to ~1-3 min; None drops the flag. A group's `canary_task_args`
# (dict of overrides, or a full list) wins; `canary_est_s` gives its measured wall. The runner's --canary-units 1
# caps every canary at its shard's first unit.
CANARY_OVERRIDES = {
    "selftest": {"--units": "2", "--unit-s": "5", "--steps": "5"},
    "gen_shard": {"--gen-shards": "1", "--episodes": "50", "--block": "50"},
    "nuisance": {"--shapes": "pair", "--probe-sets": "v2", "--episodes": "pair=1", "--ep-per-unit": "pair=1",
                 "--particles": "64", "--max-stages": "20", "--sweep-deltas": None, "--sweep-episodes": None},
    "t0_a2": {"--seeds": "1000:1002", "--per-unit": "2"},
}
CANARY_EST_S = 240.0


def override_args(args: list, over: dict) -> list:
    """task_args with flag values replaced ({flag: value}; None drops the flag and its value; new flags appended)."""
    out, seen, i = [], set(), 0
    args = [str(a) for a in args]
    while i < len(args):
        a = args[i]
        flag, eq, val = a.partition("=")
        has_val = bool(eq) or (i + 1 < len(args) and not args[i + 1].startswith("--"))
        step = 1 if eq or not has_val else 2
        if flag.startswith("--") and flag in over:
            seen.add(flag)
            if over[flag] is not None:
                out += [flag, str(over[flag])]
        else:
            out += args[i:i + step]
        i += step
    for f, v in over.items():
        if f not in seen and v is not None:
            out += [f, str(v)]
    return out


def run_line(g: dict, lane: str, shard: int, n: int, task_args_list: list | None = None, runner_extra: str = "") -> str:
    ta = (g.get("task_args") or []) if task_args_list is None else task_args_list
    task_args = " ".join(shlex.quote(str(a)) for a in ta)
    common = (f"-m cfcap.runner {shlex.quote(g['task'])} --shard {shard} --n-shards {n} --seed {int(g.get('seed', 0))}"
              + (f" {task_args}" if task_args else "") + (f" {runner_extra}" if runner_extra else ""))
    # Repos are checked out at $PWD/src/<name> (worker.py: work/src/<name>, cwd = the work dir). The box's backend
    # is `bash -lc {spec[run]}`: str.format runs once on that template, so a {repos[...]} inside spec.run would reach
    # bash literally; $PWD/src is the same path, spelled so it works.
    code = ("S=$PWD/src; test -e $S/cfcap/cfcap/runner.py || { echo 'cfcap repo not checked out at '$S >&2; exit 3; }; "
            "export PYTHONPATH=$S/cfcap:$S/capxfer CAPXFER_ROOT=$S/capxfer ")
    if lane in ("free", "cpu") and not g.get("extra_inputs"):
        # CPU lanes read theta from the capxfer repo (no input push, so any matching worker can run them).
        # Free workers' python3 may lack numpy/torch: prefer CFCAP_PY, then the capxfer venv when the host has one
        # (innovation), else python3; the runner exits 3 on a missing module. No input push: theta from the repo.
        return (code + "CAPXFER_THETA_DIR=$S/capxfer/theta OMP_NUM_THREADS=${CRANE_CPUS:-1} "
                "MKL_NUM_THREADS=${CRANE_CPUS:-1}; PY=python3; "
                "for p in \"${CFCAP_PY:-}\" \"$HOME/crane_explain/capxfer/.venv/bin/python\"; "
                "do [ -n \"$p\" ] && [ -x \"$p\" ] && PY=$p && break; done; "
                f"exec $PY {common}")
    req = ",".join(["theta", *[Path(x).expanduser().name for x in g.get("extra_inputs") or []]])
    return (code + "CAPXFER_THETA_DIR=/workspace/in/theta CRANE_INPUTS_DIR=/workspace/in; "
            f"exec python3 {common} --require-inputs {req}")


def group_jobs(plan: dict, g: dict, info: dict, draft: bool, first_only: bool = False) -> tuple[list[dict], list[str]]:
    """first_only: shard 0 alone, with the same id and body as in the full emit (a canary per group: two broken
    outcomes in a row trip a batch). A later full submit returns `exists` for it and queues the rest."""
    warn: list[str] = []
    lane, profile = g.get("lane", "rented"), g.get("profile", "torch")
    if lane not in MAX_JOB_S or profile not in PROFILES:
        raise PlanError(f"group {g['name']}: lane {lane!r} / profile {profile!r} unknown")
    if lane == "rented" and profile == "cpu":
        raise PlanError(f"group {g['name']}: lane rented is a GPU lane; use lane cpu for profile cpu")
    if not g.get("pilot"):
        if not draft:
            raise PlanError(f"group {g['name']}: no pilot evidence for unit_s (pass --draft to emit a draft)")
        warn.append(f"{g['name']}: DRAFT, unit_s {g.get('unit_s')} is not measured")
    n, ups, unit_s = int(g["shards"]), int(g["units_per_shard"]), float(g["unit_s"])
    est = round(float(g.get("overhead_s", 60)) + ups * unit_s, 1)
    if est > MAX_JOB_S[lane]:
        raise PlanError(f"group {g['name']}: {est / 60:.0f} min per job > {MAX_JOB_S[lane] // 60} min ({lane}); "
                        "raise shards or lower units_per_shard")
    if est < MIN_JOB_S:
        warn.append(f"{g['name']}: {est / 60:.1f} min per job is under 5 min; pack more units per shard")
    # lane cpu: py_command on rented CPU boxes (tags cloud-ok, cpu), or with `ride: "torch"` on the spare cores of
    # torch boxes (tags cloud-ok, torch, no needs.gpu: crane-vast counts it as torch backlog, 8 per box).
    cpu_tags = ["cloud-ok", g.get("ride") or "cpu"] + (["rented"] if g.get("extra_inputs") else [])
    tags = {"rented": ["cloud-ok", profile, "rented"], "cpu": cpu_tags, "free": []}[lane]
    needs: dict = {"os": ["linux"]}
    if tags:
        needs["tags"] = tags
    if lane == "rented":
        needs.update(gpu=True, vram_gb=float(g.get("vram_gb", 8)))
    if g.get("min_ram_gb"):
        needs["min_ram_gb"] = float(g["min_ram_gb"])
    repos = [{"name": "cfcap", "url": str(CFCAP), "ref": info["cfcap_commit"]},
             {"name": "capxfer", "url": str(STAGE / "capxfer"), "ref": info["capxfer_stage_commit"]}]
    inputs: list[str] = []
    if lane in ("free", "cpu") and not g.get("extra_inputs"):
        pass
    elif lane == "free":
        if g.get("extra_inputs"):
            raise PlanError(f"group {g['name']}: the free lane has no input push; extra_inputs need a rented lane")
    else:
        inputs = [str(STAGE / "theta"), *[str(Path(p).expanduser()) for p in g.get("extra_inputs") or []]]
        for p in inputs:
            if not Path(p).exists():
                if draft and p != str(STAGE / "theta"):
                    warn.append(f"{g['name']}: input {p} does not exist yet (stage-2 input)")
                    continue
                raise PlanError(f"group {g['name']}: input {p} does not exist (run `make_jobs.py stage` first?)")
    purpose = " ".join(str(g["purpose"]).split())
    jobs = []
    for k in range(1 if first_only else n):
        jid = f"{plan['campaign']}.{g['name']}.s{k:03d}"
        if not JOB_ID.match(jid):
            raise PlanError(f"job id {jid!r} is not an rc-job/1 id")
        spec = {"schema": "cfcap.job/1", "task": g["task"], "group": g["name"], "stage": int(g.get("stage", 1)),
                "shard": k, "n_shards": n, "task_args": list(g.get("task_args") or []),
                "cfcap_commit": info["cfcap_commit"], "capxfer_commit": info.get("capxfer_commit"),
                "run": run_line(g, lane, k, n)}
        if plan.get("lineage"):
            spec["lineage"] = plan["lineage"]
        if inputs:
            spec["inputs_rsync"] = inputs
        job = {"schema": "rc-job/1", "job_id": jid, "kind": "gpu_command" if lane == "rented" else "py_command",
               "purpose": f"{purpose} Shard {k + 1} of {n}.", "est_seconds": est,
               "submitter": plan["submitter"], "campaign": plan["campaign"],
               "priority": int(plan.get("priority_base", 100)) + int(g.get("priority", 0)),
               "needs": needs, "seeds": [0], "spec": spec, "outputs": {"class": "summary"}, "max_attempts": 3}
        job["repos"] = repos
        if est >= EXPENSIVE_S:
            job["expect"] = {"owner": plan.get("owner", plan["submitter"]), "metrics": ["units_done", "rows"],
                             "stop": plan["stop"]}
        jobs.append(job)
    return jobs, warn


def batch_canaries(plan: dict, jobs: list[dict], groups: dict, log=print) -> list[dict]:
    """One canary per batch (campaign, kind, refs) at priority 0: the group's shard-0 job with its task args shrunk
    (CANARY_OVERRIDES / canary_task_args) and --canary-units 1, so the batch opens after minutes, not hours."""
    out = []
    by_kind: dict[str, list[dict]] = {}
    for j in jobs:
        by_kind.setdefault(j["kind"], []).append(j)
    for kind, js in by_kind.items():
        base = min(js, key=lambda j: (j["est_seconds"], j["spec"]["shard"]))
        g = groups[base["spec"]["group"]]
        ov = g.get("canary_task_args")
        if isinstance(ov, list):
            args = [str(a) for a in ov]
        else:
            known = CANARY_OVERRIDES.get(g["task"], {})
            if not known and not ov:
                log(f"warning: canary {kind}: no shrink rule for task {g['task']!r}; it runs one unit of "
                    f"{g['name']} (~{float(g['unit_s']) / 60:.0f} min): give the group canary_task_args")
            args = override_args(g.get("task_args") or [], {**known, **(ov or {})})
        est = float(g.get("canary_est_s") or min(CANARY_EST_S, base["est_seconds"]))
        lane = g.get("lane", "rented")
        # n_shards 1: shard 0 of the shrunk args always has units (e.g. gen_shard --gen-shards 1).
        spec = {**base["spec"], "task_args": args, "canary": True, "shard": 0, "n_shards": 1,
                "run": run_line(g, lane, 0, 1, args, "--canary-units 1")}
        c = {**base, "job_id": f"{plan['campaign']}.canary.{kind.replace('_', '-')}", "priority": 0,
             "est_seconds": est, "spec": spec,
             "purpose": f"Batch canary for {plan['campaign']} ({kind}): a minutes-long run of task {g['task']} from "
                        f"group {g['name']} that must finish valid before the batch's long jobs lease."}
        c.pop("expect", None)
        out.append(c)
    return out


def lineage_of(campaign: str) -> str:
    """cfcap-nuis2 -> cfcap-nuis, cfcap-baselines-b2r -> cfcap-baselines, cfcap-gen-ref2 -> cfcap-gen-ref."""
    return re.sub(r"(-b\d+|\d+)r?$", "", campaign) or campaign


def emit(plan: dict, draft: bool = False, allow_dirty: bool = False, restage: bool = True, log=print,
         first_only: bool = False, canary: bool = True, campaign: str | None = None,
         cfcap_ref: str | None = None) -> dict:
    """campaign: re-emit the plan under another campaign name (job ids follow it: a new batch, `--resubmit-as`).
    cfcap_ref: pin this cfcap commit instead of HEAD. Every job of the file carries the same refs (one batch per kind)."""
    # spec.lineage: a stable name across resubmits and new commits (the coordinator opens a new batch at once when
    # its lineage already has an open batch). The plan's own campaign, minus -bN / N / r version suffixes.
    plan = {**plan, "lineage": plan.get("lineage") or lineage_of(plan.get("campaign", ""))}
    if campaign:
        plan = {**plan, "campaign": campaign}
    for k in ("campaign", "submitter", "groups", "stop"):
        if k not in plan:
            raise PlanError(f"plan lacks {k!r}")
    info = stage(allow_dirty, log=log) if restage else json.loads((STAGE / "capxfer" / "STAGE.json").read_text())
    if not restage:
        info["capxfer_stage_commit"] = git(STAGE / "capxfer", "rev-parse", "HEAD")
        info["cfcap_commit"] = git(CFCAP, "rev-parse", "HEAD")
    if cfcap_ref:
        info["cfcap_commit"] = git(CFCAP, "rev-parse", "--verify", f"{cfcap_ref}^{{commit}}")
    jobs, warns, rows = [], [], []
    for g in sorted(plan["groups"], key=lambda g: (int(g.get("stage", 1)), int(g.get("priority", 0)))):
        js, w = group_jobs(plan, g, info, draft, first_only)
        jobs += js
        warns += w
        bh = sum(j["est_seconds"] for j in js) / 3600
        lane = g.get("lane", "rented")
        price = 0.0 if lane == "free" else PRICE[g.get("profile", "torch")]
        rows.append((int(g.get("stage", 1)), g["name"], g["task"], lane, g.get("profile", "torch"), len(js),
                     js[0]["est_seconds"] / 60 if js else 0, bh, bh * price, "pilot" if g.get("pilot") else "DRAFT"))
    refs = {json.dumps(j["repos"], sort_keys=True) for j in jobs}
    if len(refs) > 1:
        raise PlanError(f"{len(refs)} different repo ref sets in one campaign file; one batch per kind needs one")
    if canary and jobs:
        cs = batch_canaries(plan, jobs, {g["name"]: g for g in plan["groups"]}, log=log)
        jobs = cs + jobs
        for c in cs:
            log(f"canary {c['job_id']}: priority 0, est {c['est_seconds'] / 60:.1f} min, "
                f"args {' '.join(c['spec']['task_args'])}")
    if len(jobs) > 500:
        warns.append(f"{len(jobs)} jobs: research submit takes at most 500 per call; split the file")
    log(f"{'stage':>5} {'group':<22} {'task':<10} {'lane':<6} {'prof':<5} {'jobs':>4} {'min/job':>7} {'box-h':>6} "
        f"{'$':>6}  est")
    for r in rows:
        log(f"{r[0]:>5} {r[1]:<22} {r[2]:<10} {r[3]:<6} {r[4]:<5} {r[5]:>4} {r[6]:>7.1f} {r[7]:>6.2f} {r[8]:>6.2f}  {r[9]}")
    log(f"total {len(jobs)} jobs, {sum(r[7] for r in rows):.1f} box-h, ${sum(r[8] for r in rows):.2f} "
        "(+ ~10 min boot per rented box)")
    for w in warns:
        log(f"warning: {w}")
    key = "draft_jobs" if any(r[9] == "DRAFT" for r in rows) else "jobs"   # a draft cannot be submitted by accident
    return {key: jobs, "campaign": plan["campaign"], "stage_info": info, "warnings": warns}


# ------------------------------------------------------------------ validate (the coordinator's own code, read-only)

VALIDATOR = r"""
import { checkPurpose, checkEst, checkSubmitter, checkCampaign } from "%(src)s/purpose";
import { parseNeeds } from "%(src)s/match";
import { readFileSync } from "node:fs";
const JOB_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$/, SAFE_NAME = /^[A-Za-z0-9._-]{1,64}$/;
const REPO_NAME = /^[A-Za-z0-9._/-]{1,200}$/, REF = /^[A-Za-z0-9._/@^~-]{1,200}$/, METRIC = /^[A-Za-z0-9._:/-]{1,128}$/;
const doc = JSON.parse(readFileSync(process.argv[2], "utf8"));
const jobs = Array.isArray(doc) ? doc : (doc.jobs ?? doc.draft_jobs ?? []);
const errs = [];
const seen = new Set();
jobs.forEach((j, i) => {
  const bad = (f, m) => errs.push(`jobs[${i}] ${j.job_id ?? "?"} ${f}: ${m}`);
  const tryf = (f, fn) => { try { fn(); } catch (e) { bad(f, e?.message ?? String(e)); } };
  if (j.schema !== undefined && j.schema !== "rc-job/1") bad("schema", 'must be "rc-job/1"');
  if (typeof j.job_id !== "string" || !JOB_ID.test(j.job_id)) bad("job_id", "bad id");
  if (seen.has(j.job_id)) bad("job_id", "duplicate"); seen.add(j.job_id);
  if (typeof j.kind !== "string" || !SAFE_NAME.test(j.kind)) bad("kind", "must be a backend name");
  tryf("purpose", () => checkPurpose(j.purpose));
  tryf("est_seconds", () => checkEst(j.est_seconds));
  tryf("submitter", () => checkSubmitter(j.submitter));
  tryf("campaign", () => checkCampaign(j.campaign));
  tryf("needs", () => parseNeeds(j.needs, j.kind));
  if (j.priority !== undefined && !Number.isInteger(j.priority)) bad("priority", "must be an integer");
  if (j.seeds && (!Array.isArray(j.seeds) || j.seeds.length > 10000)) bad("seeds", "list of <= 10000");
  if (j.spec !== undefined && (typeof j.spec !== "object" || JSON.stringify(j.spec).length > 65536)) bad("spec", "object <= 64 KB");
  if (j.outputs && !["summary", "raw", "sealed"].includes(j.outputs.class)) bad("outputs.class", "summary|raw|sealed");
  if (j.max_attempts !== undefined && !(Number.isInteger(j.max_attempts) && j.max_attempts >= 1 && j.max_attempts <= 20)) bad("max_attempts", "1..20");
  for (const r of j.repos ?? []) if (!REPO_NAME.test(r.name ?? "") || !REF.test(r.ref ?? "")) bad("repos", `bad repo ${r.name}`);
  const computeH = (j.est_seconds * (j.seeds?.length ?? 1)) / 3600;
  if (computeH >= 2 && !(j.expect?.stop && (j.expect?.metrics || j.expect?.artifacts))) bad("expect", "an expensive job needs expect {owner, metrics|artifacts, stop}");
  if (j.expect?.metrics && !j.expect.metrics.every((m) => METRIC.test(m))) bad("expect.metrics", "names");
  if (j.expect?.stop && (j.expect.stop.trim().length < 10 || j.expect.stop.length > 600)) bad("expect.stop", "10-600 chars");
  if (j.kind === "gpu_command" && !(j.needs?.gpu)) bad("needs.gpu", "a gpu_command job needs needs.gpu");
  if ((j.kind === "gpu_command" || j.kind === "py_command") && typeof j.spec?.run !== "string") bad("spec.run", "required");
});
if (jobs.length > 500) errs.push(`${jobs.length} jobs > 500 per submit`);
console.log(JSON.stringify({ jobs: jobs.length, errors: errs }));
"""


def validate(path: Path) -> list[str]:
    """Runs the coordinator's purpose/estimate/submitter/campaign/needs checks (coordinator/src, bundled with its own
    esbuild) plus the rcjob.ts field rules on every job. Raises PlanError when node or esbuild is missing."""
    src = RC / "coordinator" / "src"
    esb = RC / "coordinator" / "node_modules" / ".bin" / "esbuild"
    if not (src.is_dir() and esb.exists() and shutil.which("node")):
        raise PlanError("validator needs ~/research-cluster/coordinator (src, node_modules/.bin/esbuild) and node")
    with tempfile.TemporaryDirectory(prefix="cfcap-validate-") as td:
        entry, out = Path(td) / "entry.ts", Path(td) / "v.mjs"
        entry.write_text(VALIDATOR % {"src": src.as_posix()})
        r = subprocess.run([str(esb), str(entry), "--bundle", "--platform=node", "--format=esm", f"--outfile={out}",
                            "--log-level=error"], capture_output=True, text=True)
        if r.returncode:
            raise PlanError(f"esbuild: {r.stderr.strip()[:500]}")
        r = subprocess.run(["node", str(out), str(path)], capture_output=True, text=True)
        if r.returncode:
            raise PlanError(f"node: {r.stderr.strip()[:500]}")
        res = json.loads(r.stdout)
    return res["errors"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--allow-dirty", action="store_true")
    e = sub.add_parser("emit")
    e.add_argument("plan")
    e.add_argument("-o", "--out", required=True)
    e.add_argument("--draft", action="store_true", help="allow groups without pilot evidence (key draft_jobs)")
    e.add_argument("--allow-dirty", action="store_true")
    e.add_argument("--no-restage", action="store_true", help="reuse the current stage")
    e.add_argument("--validate", action="store_true")
    e.add_argument("--no-canary", action="store_true", help="do not prepend the per-batch canary jobs")
    e.add_argument("--resubmit-as", metavar="CAMPAIGN", help="emit the plan under this campaign name (new batch)")
    e.add_argument("--cfcap-ref", help="pin this cfcap commit instead of HEAD")
    e.add_argument("--first-shard-only", action="store_true",
                   help="only shard 0 of each group (canary; same ids and bodies as the full emit)")
    v = sub.add_parser("validate")
    v.add_argument("jobs")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "stage":
            stage(a.allow_dirty)
            return 0
        if a.cmd == "emit":
            plan = json.loads(Path(a.plan).read_text())
            doc = emit(plan, a.draft, a.allow_dirty, restage=not a.no_restage, first_only=a.first_shard_only,
                       canary=not a.no_canary, campaign=a.resubmit_as, cfcap_ref=a.cfcap_ref)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
            print(f"wrote {a.out} ({'DRAFT: not submittable' if 'draft_jobs' in doc else 'submittable'})")
            if not a.validate:
                return 0
            a.jobs = a.out
        errs = validate(Path(a.jobs))
        for x in errs:
            print(f"invalid: {x}")
        print(f"validate {a.jobs}: {'ok' if not errs else f'{len(errs)} error(s)'} (coordinator purpose/needs checks)")
        return 1 if errs else 0
    except PlanError as ex:
        print(f"make_jobs: {ex}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
