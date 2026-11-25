TASK re_models: second-implementation re-analysis (exploratory). Find the models role's landed outputs (see
docs/board/models.md: dec_models.csv.gz, diversity.csv, scaling.csv, latency_models.csv under results/models/paper/,
plus pulled runs under ~/.cache/vast-out/*/crane*/cfcap-models*). NEVER open conf_* splits or anything under a
`confirmatory` path. From the raw per-row decision files, independently recompute:
- coverage, certified fraction and unsafe rate by split (cal, test, H3, H4, H4b, het) and by parent size N, for the
  direct and compositional routes;
- on the decision-relevant stratum (faulted OR |ρ_dec−τ|<0.1), with episode-bootstrap CIs.
Compare with the role's own CSVs and report every disagreement > 0.5 pt with its cause. Write
results/codex/re_models.csv and scripts/codex/re_models.py.
