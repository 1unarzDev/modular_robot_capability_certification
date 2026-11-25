#!/usr/bin/env bash
# Read-only spend monitor for the cfcap rental window: live boxes and burn (crane-vast status), ledger spend since
# SINCE (UTC date, all sessions: the owner's cap is on total spend), the cap left, and the campaign's queue.
# Usage: scripts/cost_watch.sh [CAP_USD=35] [SINCE=2026-11-25] [CAMPAIGN=cfcap-b1] [PROJECT=crane]
#        watch -n 300 scripts/cost_watch.sh       (every 5 min, like crane-vast tick)
cap=${1:-35}; since=${2:-2026-11-25}; camp=${3:-cfcap-b1}; proj=${4:-crane}
C=$HOME/crane-program/compute
timeout 90 "$C/bin/crane-vast" status 2>&1 | tail -n 12
awk -F, -v s="$since" -v cap="$cap" '{ sub(/\r$/, "") } NR > 1 && $1 >= s { h += $4; u += $5; if ($7 == "on-demand") od += $5 }
  END { printf "ledger since %s: %.1f box-h, $%.2f of $%.2f (left $%.2f); on-demand $%.2f%s\n", s, h, u, cap, cap - u, od,
        (od > 0 ? "  <- owner: interruptible only" : "") }' "$C/LEDGER.csv"
timeout 60 python3 "$HOME/research-cluster/bin/research" status -p "$proj" --campaign "$camp" 2>&1 | tail -n 8
