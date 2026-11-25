#!/usr/bin/env bash
# Stage the crane files T2 imports (read-only, git archive at crane HEAD) as a box input:
#   $CFCAP_OUT/compute/stage/crane_ml  ->  /workspace/in/crane_ml on the box (cfcap.t2.deps finds it there).
set -euo pipefail
SRC=${CFCAP_CRANE_ML:-$HOME/crane_ml}
DST=${1:-${CFCAP_OUT:-$HOME/.cache/modular-runs/2026-11/cfcap}/compute/stage/crane_ml}
FILES="crane/__init__.py crane/resources.py engine/__init__.py engine/jit.py engine/nbexact.py engine/jaxconf.py
engine/kernels/__init__.py engine/kernels/mecanum.py engine/kernels/mecanum_run.py"
rm -rf "$DST"; mkdir -p "$DST"
git -C "$SRC" archive HEAD $FILES | tar -x -C "$DST"
git -C "$SRC" rev-parse HEAD > "$DST/CRANE_COMMIT"
echo "staged $(find "$DST" -type f | wc -l) files of crane@$(cut -c1-8 "$DST/CRANE_COMMIT") -> $DST"
