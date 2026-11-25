TASK a5_flips. Analyse the simulator-mismatch flips. Read docs/board/stress.md (claim C-stress-1) and
results/stress/A5_*.csv. Single modules flip their capability decision between T1 and T2 in 4.4 % of cases (pairs
1.8 %, ≥3 modules ≤0.25 %). Characterise which singles flip: leg type, fault type/level, |ρ − τ| margin, payload,
friction scatter, which plant effect. Fit a small interpretable model (logistic or tree) of flip ~ features with
cross-validation, and say whether a cheap observable (e.g. the clamp-probe misfit, AUC 0.78) plus the margin
predicts flips well enough to gate a certificate (report the precision/recall of an "abstain if predicted flip"
rule). Use only existing CSVs; if a feature isn't in them, say so instead of re-simulating. Write
results/codex/a5_flips_analysis.csv and scripts/codex/a5_flips_analysis.py.
