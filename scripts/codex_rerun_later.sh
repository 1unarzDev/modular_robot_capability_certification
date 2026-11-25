#!/usr/bin/env bash
# Re-run the codex re-analyses of models and baselines once their outputs have landed (after ~00:30Z).
until [ "$(date -u +%H%M)" -ge 0030 ] && [ "$(date -u +%H%M)" -lt 1200 ]; do sleep 120; done
for t in re_models re_baselines; do
  CODEX_MODEL=gpt-6.1-terra "$HOME/cfcap/scripts/codex_task.sh" ${t}_2 workspace-write "$HOME/cfcap/scripts/codex_prompts/$t.md"
done
