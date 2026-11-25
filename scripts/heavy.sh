#!/usr/bin/env bash
# Run a heavy local command in one of 2 shared slots (COMMON.md: <=2 heavy processes per session).
# Usage: scripts/heavy.sh [--gpu] <cmd...>. --gpu first waits for the single GPU lock (without holding a slot),
# then takes a slot, so queued GPU jobs never block CPU jobs. Blocks until both are free.
set -uo pipefail
gpu=0; [[ "${1:-}" == "--gpu" ]] && { gpu=1; shift; }
if [[ $gpu == 1 ]]; then exec {g}>/tmp/cfcap-gpu.lock; flock $g; fi
while true; do
  for s in 1 2; do
    exec {fd}>/tmp/cfcap-heavy-$s.lock
    if flock -n $fd; then
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 15 "$@"; rc=$?
      exit $rc
    fi
    exec {fd}>&-
  done
  sleep 5
done
