#!/usr/bin/env bash
# Wait for the codex-lb quota to reset, then launch the queued codex tasks (each once).
source "$HOME/cfcap/env.sh"
until [ "$(date -u +%H%M)" -ge 0055 ] && [ "$(date -u +%H%M)" -lt 1200 ]; do sleep 60; done
while ! timeout 120 codex exec -s read-only --skip-git-repo-check "Reply with exactly: OK" 2>/dev/null | grep -q OK; do
  echo "$(date -u +%H:%M) codex still unavailable"; sleep 600
done
echo "$(date -u +%H:%M) codex available; launching"
for t in re_models re_t2 re_baselines re_gate; do
  "$HOME/cfcap/scripts/codex_task.sh" $t workspace-write "$HOME/cfcap/scripts/codex_prompts/$t.md"
done
