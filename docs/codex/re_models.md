Exploratory re-analysis completed for the available training-run decision files.  
The requested paper CSVs are absent, so disagreements cannot be assessed.  
H4b and heterogeneous evaluations are unavailable; compositional results are limited to cal and test.

For the decision-relevant test stratum in run `s008`, direct coverage was **86.46%** (episode-bootstrap 95% CI 85.15–87.71%), certified fraction was **96.24%** (95.73–96.68%), and unsafe rate was **0.85%** (0.16–1.81%). Compositional estimates were **95.04%** (93.61–96.34%), **96.92%** (96.32–97.47%), and **0.84%** (0.20–1.76%), respectively. Source: [re_models.csv](/home/lunarz/cfcap/results/codex/re_models.csv).

The CSV includes available split-level and parent-size estimates for both strata. Certification and unsafe rates use the role’s pooled-threshold denominators; bootstrap intervals resample episodes with calibration fixed. Cal estimates use the same rows for calibration and evaluation.

Files written:

- [re_models.py](/home/lunarz/cfcap/scripts/codex/re_models.py)
- [re_models.csv](/home/lunarz/cfcap/results/codex/re_models.csv)