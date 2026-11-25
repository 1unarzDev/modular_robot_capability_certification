#!/usr/bin/env bash
# Local end-to-end of one emitted job as a rented box runs it, with the box root moved under a temp dir:
#   1. push every spec.inputs_rsync path the way crane-vast does (`rsync -a SRC box:/workspace/in/`);
#   2. run spec.run through `bash -lc` in a work dir with the command backend's env (CRANE_JOB_ID, CRANE_RUN_DIR,
#      CRANE_OUT_ROOT, CRANE_INPUTS_DIR), a clean environment (no PYTHONPATH from env.sh) and the system python3;
#   3. SIGTERM the process group mid-run (the backend's cancel) -> expect 75; rerun -> resumes;
#      kill -9 mid-run -> rerun -> completes with every finished unit kept.
# The only change to spec.run is the /workspace prefix -> $BOX/workspace (no root here).
# Usage: scripts/box_dryrun.sh JOBS.json [JOB_INDEX] [BOX_DIR]
set -euo pipefail
jobs=$1; idx=${2:-0}; BOX=${3:-$(mktemp -d /tmp/cfcap-box.XXXX)}
W=$BOX/workspace; mkdir -p "$W/in" "$W/out"
eval "$(python3 - "$jobs" "$idx" "$W" <<'PY'
import json, shlex, sys
d = json.load(open(sys.argv[1])); j = (d.get("jobs") or d.get("draft_jobs"))[int(sys.argv[2])]
run = j["spec"]["run"].replace("/workspace/", sys.argv[3] + "/")
print(f"JOB_ID={shlex.quote('crane/' + j['job_id'])}")
print(f"RUN={shlex.quote(run)}")
print("INPUTS=(" + " ".join(shlex.quote(p) for p in j["spec"].get("inputs_rsync", [])) + ")")
print(f"EST={j['est_seconds']}")
PY
)"
echo "box $BOX  job $JOB_ID  est ${EST}s"
for src in "${INPUTS[@]}"; do rsync -a --partial "$src" "$W/in/"; done
touch "$W/in/.ready"
ls "$W/in"
WORK=$BOX/work; RUN_DIR=$WORK/run; mkdir -p "$RUN_DIR"
start() {   # the command backend: new session, cwd = work dir, its env on top of a clean one
  ( cd "$WORK" && exec setsid env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 \
      CRANE_JOB_ID="$JOB_ID" CRANE_RUN_DIR="$RUN_DIR" CRANE_SEED=0 CRANE_OUT_ROOT="$W/out" \
      CRANE_INPUTS_DIR="$W/in" PIP_BREAK_SYSTEM_PACKAGES=1 bash -lc "$RUN" ) >>"$BOX/command.log" 2>&1 &
  PID=$!
}
OUT=$W/out/$JOB_ID
units() { ls "$OUT/units/"*.json 2>/dev/null | wc -l; }
wait_units() { for _ in $(seq 600); do [ "$(units)" -ge "$1" ] && return 0; sleep 0.5; done; echo "timeout"; return 1; }

echo "== run 1: SIGTERM to the process group after 2 units"
start; wait_units 2; sleep 1; kill -TERM -- "-$PID"; set +e; wait "$PID"; rc=$?; set -e
echo "exit $rc (want 75); units kept $(units)"; [ "$rc" = 75 ]
python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print('manifest', m['status'], m['units_done'], '/', m['units_total'])" "$OUT/manifest.json"
echo "== run 2: resume, then kill -9 after one more unit"
n=$(units); start; wait_units $((n + 1)); kill -KILL -- "-$PID"; set +e; wait "$PID"; set -e
echo "killed; units kept $(units)"
echo "== run 3: resume to completion"
t=$(date +%s.%N); start; set +e; wait "$PID"; rc=$?; set -e
echo "exit $rc (want 0) in $(python3 -c "import time; print(round(time.time()-$t,1))") s"; [ "$rc" = 0 ]
python3 - "$OUT" "$RUN_DIR" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
out, run = Path(sys.argv[1]), Path(sys.argv[2])
m = json.load(open(out / "manifest.json")); met = json.load(open(run / "metrics.json"))
ok = all(hashlib.sha256((out / f["path"]).read_bytes()).hexdigest() == f["sha256"] for f in m["files"] if f["path"])
print(f"manifest {m['status']} {m['units_done']}/{m['units_total']} rows {m['rows']} code {m['code']['cfcap']}; hashes ok {ok}")
print(f"metrics {len(met)} keys, all finite {all(math.isfinite(v) for v in met.values())}: "
      + ", ".join(f"{k}={met[k]:g}" for k in ('units_done', 'units_resumed', 'rows', 'wall_s', 'selftest.resumed_mid_unit') if k in met))
p = json.load(open(run / "progress.json")); print("progress", {k: p[k] for k in ("done", "total", "phase", "updated")})
print("run dir:", sorted(str(x.relative_to(run)) for x in run.rglob("*") if x.is_file()))
assert m["status"] == "complete" and ok and len(met) <= 1000
PY
echo "RC-PROGRESS lines in the log: $(grep -c RC-PROGRESS "$BOX/command.log")"
echo "dry run ok: $BOX"
