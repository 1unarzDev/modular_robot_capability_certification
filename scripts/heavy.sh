#!/usr/bin/env bash
# Run a heavy local command in one of 2 shared slots (COMMON.md: <=2 heavy processes per session).
# Usage: scripts/heavy.sh [--gpu] <cmd...>. --gpu also takes the single GPU lock. Blocks until a slot frees.
set -euo pipefail
gpu=0; [[ "${1:-}" == "--gpu" ]] && { gpu=1; shift; }
run() { if [[ $gpu == 1 ]]; then flock /tmp/cfcap-gpu.lock nice -n 15 "$@"; else nice -n 15 "$@"; fi; }
while true; do
  for s in 1 2; do
    exec {fd}>/tmp/cfcap-heavy-$s.lock
    if flock -n $fd; then OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 run "$@"; rc=$?; flock -u $fd; exit $rc; fi
    exec {fd}>&-
  done
  sleep 5
done
