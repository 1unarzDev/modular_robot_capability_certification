Exploratory re-analysis is saved in [re_baselines.csv](/home/lunarz/cfcap/results/codex/re_baselines.csv).
The matched operating point and E3 bars cannot be evaluated from the available pilot data.
The SMC sweep is incomplete, so no particle setting can be selected.

The available baseline units contain pilot evaluations but no matching pilot calibration rows, learned-method results, or model-free results. The CSV reports raw-bound results by method, shape and stratum; these are not matched or calibrated results. The decision stratum uses **faulted OR within 0.1 of τ**, consistent with the later scorer audit but different from the earlier AND definition in [baselines.md](/home/lunarz/cfcap/docs/board/baselines.md).

The SMC units contain 9, 9 and 5 episodes at 256, 1024 and 2048 particles respectively, with 4 episodes common to all settings ([re_baselines.csv](/home/lunarz/cfcap/results/codex/re_baselines.csv)). The role’s accuracy curve remains marked TBD, so there is no completed numerical claim to compare against ([baselines.md](/home/lunarz/cfcap/docs/board/baselines.md)).

Data observation: raw q05 coverage on the common SMC episodes is 46.2%–54.3%, unusually low ([re_baselines.csv](/home/lunarz/cfcap/results/codex/re_baselines.csv)).

Files written:

- [re_baselines.csv](/home/lunarz/cfcap/results/codex/re_baselines.csv)
- [re_baselines.py](/home/lunarz/cfcap/scripts/codex/re_baselines.py)