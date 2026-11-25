Exploratory re-analysis completed, CPU only.  
Fixed references increase undominated disagreement.  
The H12 bar fails overall because dominated flips exceed the limit.

All+motor results, pooled across thresholds; percentages with episode-bootstrap 95% CIs. Source: [results CSV](/home/lunarz/cfcap/results/codex/n3_fixedref.csv).

| Metric | Own reference | Fixed reference |
|---|---:|---:|
| Undominated flips | 1.09 [0.78, 1.41] | 2.76 [2.24, 3.34] |
| Dominated flips | 1.74 [1.29, 2.26] | 1.83 [1.34, 2.40] |
| Undominated \|d\| > 0.05 | 9.04 [7.92, 10.12] | 57.13 [55.05, 59.14] |
| Dominated \|d\| > 0.05 | 4.13 [3.25, 5.15] | 5.53 [4.48, 6.70] |

The CSV includes threshold-specific and shape/leg/dominance strata for all requested arms. Original-label summaries reproduce, and reference and archive-join checks pass.

A4 geometry keys are missing; scoring uses saved references, and its shape strata identify module count.

Files written:

- [scripts/codex/n3_fixedref.py](/home/lunarz/cfcap/scripts/codex/n3_fixedref.py)
- [results/codex/n3_fixedref.csv](/home/lunarz/cfcap/results/codex/n3_fixedref.csv)