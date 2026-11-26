Exploratory re-analysis is complete for the landed training and heterogeneous evaluation files.
The comparable black-box decision metrics agree with the role’s exported decision rows.
The diversity export contains duplicate results from different runs; the scaling CSV and H4b data were not available.

The recomputation in [re_models.csv](/home/lunarz/cfcap/results/codex/re_models.csv) includes split and parent-size results, the decision-relevant stratum, and episode-bootstrap intervals. Certified fraction uses capable decisions as its denominator; unsafe rate uses incapable decisions. Thresholds are pooled as in [models_scaling.py](/home/lunarz/cfcap/scripts/models_scaling.py). Calibration-split results are evaluated on the same data used to fit the bound.

The discrepancies below are all for `ds-n6c-2`, test split. They compare the selected completed rerun with the older duplicate rows in [diversity.csv](/home/lunarz/cfcap/results/models/paper/diversity.csv); differences are in percentage points, independent result minus older export.

| Shape | Coverage | Certified fraction | Unsafe rate |
|---|---:|---:|---:|
| single | −3.534 | −2.022 | |
| pair | | −0.808 | |
| path4 | −1.295 | | |
| square | +4.950 | | |
| cyclic6 | −1.733 | | |
| chain4 | −1.712 | −0.670 | |
| L3 | −0.882 | −0.811 | −1.709 |
| L6 | −0.879 | | |
| 2x3 | −6.522 | +0.969 | |
| path6 | −1.178 | | |
| L5 | +1.064 | | |
| chain3 | −1.414 | −0.647 | |
| path5 | −0.745 | | |

Blank cells did not exceed the requested disagreement threshold. The export also contains rows matching the selected rerun. The missing [scaling.csv](/home/lunarz/cfcap/results/models/paper/scaling.csv), absent H4b evaluations, and missing completed `st-n6-1` training output limit comparison; latency is outside these recomputed decision metrics.

Files written:

- [re_models.py](/home/lunarz/cfcap/scripts/codex/re_models.py)
- [re_models.csv](/home/lunarz/cfcap/results/codex/re_models.csv)