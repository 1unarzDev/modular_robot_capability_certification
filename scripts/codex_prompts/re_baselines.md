TASK re_baselines: second-implementation re-analysis of the baselines results (exploratory). Inputs:
docs/board/baselines.md, results/baselines/** (dec_baselines.csv.gz, dec_trivial.csv, smc_particle_sweep.csv,
pilot metrics), and runs under ~/.cache/vast-out/*/crane*/cfcap-baselines*. Never open `confirmatory` paths or
conf_* data.
Independently recompute, per method × shape × α:
- certified fraction at the matched 1 % operating point (fitted on cal only);
- unsafe-commit rate and coverage;
- the decision-relevant stratum and the E3 bars (learned vs best trivial ≥ 10 pts; vs the model-free bound ≥ 5 pts);
- the SMC accuracy-vs-particles selection.
Report disagreements with the role's numbers. Write results/codex/re_baselines.csv and scripts/codex/re_baselines.py.
