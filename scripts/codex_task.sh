#!/usr/bin/env bash
# Dispatch one codex (codex-lb) task in the background with the cfcap rules prepended.
# Usage: [CODEX_MODEL=gpt-6.1-terra] scripts/codex_task.sh NAME SANDBOX(read-only|workspace-write) PROMPT_FILE  (owner: use a lower-tier model)
# Final message -> docs/codex/NAME.md ; full log -> $CFCAP_OUT/codex/NAME.log
set -euo pipefail
name=$1; sb=$2; pf=$3
source "$HOME/cfcap/env.sh"
mkdir -p "$CFCAP_OUT/codex" "$HOME/cfcap/docs/codex"
rules="$HOME/cfcap/scripts/codex_prompts/_RULES.md"
prompt="$(cat "$rules")

$(cat "$pf")"
nohup nice -n 15 codex exec -m "${CODEX_MODEL:-gpt-6.1-terra}" -C "$HOME/cfcap" -s "$sb" --skip-git-repo-check \
  -o "$HOME/cfcap/docs/codex/$name.md" "$prompt" > "$CFCAP_OUT/codex/$name.log" 2>&1 &
echo "codex task $name started (pid $!), log $CFCAP_OUT/codex/$name.log"
