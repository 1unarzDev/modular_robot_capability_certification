# Confirmatory results

Scored by `scripts/analysis/confirmatory.py` against `docs/CONFIRMATORY.md` @ `986730dc906a047d126141df4a53f783c857aef3` on 2026-11-26 03:18Z. Inputs: results/baselines/confirmatory/dec_conf_baselines.csv.gz, results/models/confirmatory/dec_conf_models.csv.gz, results/stress/confirmatory/dec_conf_gate.csv.gz, results/stress/confirmatory/dec_conf_merge.csv.gz, results/env/confirmatory/missions_conf_env.csv.gz, results/env/buildings_ks.csv.

Scored once at 03:17Z 2026-11-26 under the coordinator rule 'earlier of (a) all three role exports present or (b) 03:45Z': (a) was met, after waiting for every export to pass gzip -t with a stable size (a 03:14Z load attempt stopped on a still-being-written models file before any scoring). Exports present: models dec_conf_models.csv.gz (03:15Z), env_conf_lower_*.csv.gz (03:14-03:15Z), wheels_conf_models.npz (03:15Z); baselines dec_conf_baselines.csv.gz (03:14Z); env missions_conf_env.csv.gz (03:06Z, written BEFORE the models' env lower files) + buildings_ks.csv; stress dec_conf_gate.csv.gz (02:55Z), dec_conf_merge.csv.gz (01:47Z). Scorer commit fb51ccb; pre-scoring suite 139 passed, 1 skipped (retired codex test_screen_bug:91). Calibration note (models): st-n6-2 C-MAIN/C-SCALE margins use the train-run exploratory cal (300 episodes, 100 composed; recorded in variant); C-HET uses het_cal (1,352 episodes).

Calibration disclosure (D-CAL, 5fd2720): every CI conditions on the fixed calibration split (margins are fitted once on cal and frozen); calibration-sampling variability is not included in any interval.

## Exclusions (§0.1)

- cal rows: dropped 72900 role-exported rows of ['always_go', 'metadata_nf', 'never_go']; the scorer computes them
- decision rows: excluded 87790 rows of canary shards [0] (200 clusters), 0 rows of pre-touched conf_test 0-39 (0 clusters)
- A-POWER: decision rows C-SCALE conf_H4: dropped 302850 rows of 225 clusters beyond 0-599
- A-POWER: decision rows C-HET conf_het_test: dropped 238220 rows of 500 clusters beyond 0-499
- A-POWER: decision rows C-HET conf_H1: dropped 214320 rows of 250 clusters beyond 0-499
- A-POWER: decision rows C-HET conf_H2: dropped 20180 rows of 100 clusters beyond 0-499
- decision rows: dropped 117840 role-exported rows of ['always_go', 'metadata_nf', 'never_go']; the scorer computes them
- mission rows: excluded 0 rows of canary shards [0] (0 clusters), 0 rows of pre-touched conf_test 0-39 (0 clusters)
- C-MAIN: 1220 episodes after exclusions (conf_test 40-2439)
- C-MERGE: 975 pairs after exclusions
- C-GATE: 220 episodes after exclusions (conf_test 3040-3639), severities [np.float64(0.0), np.float64(1.0), np.float64(2.0), np.float64(3.0)]
- C-ENV: excluded 0 rows of conf-ID mission 1499 (A-ENV); ID = missions 0-1,498

## Power checks (§0.4), before any verdict

| claim | cell | method | plant | incapable stratum decisions | deff | upper bound at 1 % | powered |
|---|---|---|---|---|---|---|---|
| C-MAIN | stratum | amortized_comp | T1 | 8,927 | 3.11 | 1.36 % | **UNDERPOWERED** |
| C-MERGE | stratum merges | comp_enum_size | T1 | 389 | 2.38 | 2.52 % | **UNDERPOWERED** |
| C-MERGE | stratum merges | comp_enum_size | T2 | 446 | 1.89 | 2.27 % | **UNDERPOWERED** |
| C-GATE | s3 v2 ge3 undominated | raw | T2 | 815 | 7.04 | 2.81 % | **UNDERPOWERED** |
| C-GATE | s3 v2 ge3 dominated | raw | T2 | 0 | — | — | **UNDERPOWERED** |
| C-GATE | s3 v2 ge3 undominated | par | T2 | 815 | 2.14 | 2.00 % | **UNDERPOWERED** |
| C-GATE | s3 v2 ge3 dominated | par | T2 | 0 | — | — | **UNDERPOWERED** |
| C-GATE | s=3 >=3-module undominated | robust | T2 | 452 | 2.08 | 2.32 % | **UNDERPOWERED** |
| C-GATE | s=0 >=3-module stratum (v2+clamp) | robust | T2 | 2 | — | — | **UNDERPOWERED** |
| C-GATE | s=1 >=3-module stratum (v2+clamp) | robust | T2 | 16 | 1.12 | 6.17 % | **UNDERPOWERED** |
| C-GATE | s=2 >=3-module stratum (v2+clamp) | robust | T2 | 131 | 1.95 | 3.38 % | **UNDERPOWERED** |
| C-GATE | s=3 >=3-module stratum (v2+clamp) | robust | T2 | 815 | 2.18 | 2.01 % | **UNDERPOWERED** |

## Verdicts

| claim | bar | metric | method | cell | plant | estimate | 95 % CI | one-sided upper / lower | threshold | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C-MAIN | 1 validity | E_ep | amortized_comp | all decisions | T1 | 0.95 % | [0.39 %, 1.61 %] | 1.51 % / 0.48 % | <= 0.1; upper95 <= 0.12 | **PASS** | all block (§9.1); block-restricted E_ep is descriptive (cells.csv) |
| C-MAIN | 1 validity | E_dec@U* | amortized_comp | stratum | T1 | 3.21 % | [2.60 %, 3.93 %] | 3.78 % / 2.68 % | <= 0.01; upper95 <= 0.013 | **FAIL (underpowered; exceeds bar)** |  |
| C-MAIN | 1b per-kind validity | E_dec@U* (merge) | amortized_comp | merge | T1 | 4.17 % | [1.63 %, 7.63 %] | 7.00 % / 1.94 % | upper95 <= 0.03 | **FAIL (underpowered; exceeds bar)** | pooled matched point; power: >= 1000 incapable decisions (A-KIND); claim (iii) rests on C-MERGE alone |
| C-MAIN | 1b per-kind validity | E_dec@U* (child >=2 modules) | amortized_comp | child >=2 modules | T1 | 6.72 % | [3.63 %, 10.18 %] | 9.53 % / 4.01 % | upper95 <= 0.03 | **FAIL (underpowered; exceeds bar)** | pooled matched point; power: >= 1000 incapable decisions (A-KIND) |
| C-MAIN | 1b per-kind validity | E_dec@U* (lone faulted child) | amortized_comp | lone faulted child | T1 | 2.89 % | [2.28 %, 3.57 %] | 3.48 % / 2.38 % | upper95 <= 0.03 | **FAIL** | pooled matched point; power: >= 1000 incapable decisions (A-KIND) |
| C-MAIN | P1 | CF@U* (lone-faulted): amortized_comp - capxfer | amortized_comp |  | T1 | 38.50 % | [34.45 %, 42.65 %] | 42.00 % / 35.05 % | >= 0.1; lower95 > 0.0 | **PASS** | Holm rejects |
| C-MAIN | P2 | CF@U* (stratum): amortized_comp - ghat_physics | amortized_comp |  | T1 | 35.27 % | [34.56 %, 36.09 %] | 35.95 % / 34.67 % | >= 0.05; lower95 > 0.0 | **PASS** | best trivial on cal: ghat_physics; Holm rejects |
| C-MAIN | 1 validity | E_ep | amortized_comp |  | T2 | — | [—, —] | — / — |  | **MISSING** |  |
| C-MAIN | 1 validity | E_dec@U* | amortized_comp |  | T2 | — | [—, —] | — / — |  | **MISSING** |  |
| C-MAIN | P3 | E_dec own (raw) | raw |  | T2 | — | [—, —] | — / — |  | **MISSING** |  |
| C-MAIN (task-library tau) | exploratory | CF (stratum) | amortized_comp |  | T1 | 94.65 % | [94.44 %, 94.88 %] | 94.83 % / 94.47 % |  | **DESCRIPTIVE** | exploratory; tau from results/annotations/task_library.csv; own bound |
| C-MAIN (task-library tau) | exploratory | E_dec (stratum) | amortized_comp |  | T1 | 0.22 % | [0.13 %, 0.34 %] | 0.32 % / 0.14 % |  | **DESCRIPTIVE** | exploratory; tau from results/annotations/task_library.csv; own bound |
| C-MAIN (task-library tau) | exploratory | share of trivially capable decisions | amortized_comp |  | T1 | 40.83 % | [39.17 %, 42.50 %] | 42.21 % / 39.48 % |  | **DESCRIPTIVE** | capable and outside the stratum |
| C-MAIN | cal-size sensitivity | E_dec@U* with margin from cal 0-999 (k = 0.791) | amortized_comp |  | T1 | 3.21 % | [2.59 %, 3.83 %] | 3.75 % / 2.67 % |  | **DESCRIPTIVE** | 100 cal episodes; CF = 0.9804; descriptive (critic C); the scored margin is cal 0-999 (A-POWER) |
| C-MAIN | cal-size sensitivity | E_dec@U* with margin from cal 0-1999 (k = 0.791) | amortized_comp |  | T1 | 3.21 % | [2.59 %, 3.83 %] | 3.75 % / 2.67 % |  | **DESCRIPTIVE** | 100 cal episodes; CF = 0.9804; descriptive (critic C); the scored margin is cal 0-999 (A-POWER) |
| C-MERGE | 1 delta_comp | decision flips | comp_oracle |  | T1 | 0.07 % | [0.03 %, 0.13 %] | 0.12 % / 0.04 % | upper95 <= 0.009999999999 | **PASS** | bar: upper95 < 0.01 |
| C-MERGE | 2 validity | miscoverage per pair | comp_enum_size |  | T1 | 0.41 % | [0.10 %, 0.82 %] | 0.72 % / 0.10 % | <= 0.1; upper95 <= 0.13 | **PASS** |  |
| C-MERGE | 2 validity | E_ep per pair | comp_enum_size |  | T1 | 0.10 % | [0.00 %, 0.31 %] | 0.31 % / 0.00 % | upper95 <= 0.1 | **PASS** | all block: any unsafe merge decision of the pair (§9.1) |
| C-MERGE | P4 | realised E_dec@U* | comp_enum_size |  | T1 | 0.00 % | [0.00 %, 0.00 %] | 0.00 % / 0.00 % |  | **DESCRIPTIVE** | descriptive unless the power rule holds |
| C-MERGE | P4 | CF@U* comp_enum_size | comp_enum_size |  | T1 | 90.84 % | [89.69 %, 91.85 %] | 91.68 % / 89.94 % | >= 0.85 | **PASS** | point-judged (§9.2) |
| C-MERGE | P4 | CF@U* (stratum merges): comp_enum_size - copy_parent | comp_enum_size |  | T1 | 55.18 % | [53.45 %, 56.98 %] | 56.73 % / 53.73 % | >= 0.1; lower95 > 0.0 | **PASS** | Holm rejects |
| C-MERGE | 1 delta_comp | decision flips | comp_oracle |  | T2 | 0.13 % | [0.07 %, 0.20 %] | 0.19 % / 0.08 % | upper95 <= 0.019999999999 | **PASS** | bar: upper95 < 0.02 |
| C-MERGE | 2 validity | miscoverage per pair | comp_enum_size |  | T2 | 2.87 % | [1.95 %, 3.90 %] | 3.79 % / 2.05 % | <= 0.15; upper95 <= 0.18 | **PASS** |  |
| C-MERGE | 2 validity | E_ep per pair | comp_enum_size |  | T2 | 0.41 % | [0.10 %, 0.82 %] | 0.82 % / 0.10 % | upper95 <= 0.1 | **PASS** | all block: any unsafe merge decision of the pair (§9.1) |
| C-MERGE | P4 (T2, reported) | realised E_dec@U* | comp_enum_size |  | T2 | 0.67 % | [0.00 %, 2.04 %] | 1.79 % / 0.00 % |  | **DESCRIPTIVE** | descriptive unless the power rule holds |
| C-MERGE | P4 (T2, reported) | CF@U* comp_enum_size | comp_enum_size |  | T2 | 90.50 % | [89.39 %, 91.53 %] | 91.36 % / 89.58 % | >= 0.85 | **DESCRIPTIVE** | point-judged (§9.2) |
| C-GATE | P5 | E_dec undominated (raw) > 2 % | raw | s=3 v2 >=3-module | T2 | 97.06 % | [93.35 %, 99.47 %] | 99.24 % / 94.10 % | >= 0.020000000001; lower95 > 0.01 | **FAIL** | power: UNDERPOWERED; Holm does not reject |
| C-GATE | 1 dominated | E_dec dominated (raw) upper95 <= 2 % | raw | s=3 v2 >=3-module | T2 | — | [—, —] | — / — | upper95 <= 0.02 | **MISSING** | dominated: UNDERPOWERED, contrast-judged (§9.3) |
| C-GATE | P5 | E_dec undominated (par) > 2 % | par | s=3 v2 >=3-module | T2 | 0.49 % | [0.00 %, 1.37 %] | 1.17 % / 0.00 % | >= 0.020000000001; lower95 > 0.01 | **FAIL** | power: UNDERPOWERED; Holm does not reject |
| C-GATE | 1 dominated | E_dec dominated (par) upper95 <= 2 % | par | s=3 v2 >=3-module | T2 | — | [—, —] | — / — | upper95 <= 0.02 | **MISSING** | dominated: UNDERPOWERED, contrast-judged (§9.3) |
| C-GATE | P6' | E_ep (s=3, >=3-module): par - robust | par |  | T2 | -6.06 % | [-10.00 %, -2.63 %] | -3.12 % / -9.26 % | >= -inf; lower95 > 0.0 | **FAIL** | paired episode bootstrap; condition 2 in 'P6' robust undominated'; Holm does not reject |
| C-GATE | P6' robust undominated | E_dec(robust, undominated) <= 1 % | robust | s=3 >=3-module undominated | T2 | 2.21 % | [0.30 %, 4.23 %] | 3.83 % / 0.62 % | <= 0.01; upper95 <= 0.013 | **FAIL (underpowered; exceeds bar)** | E_ep decides (next row) |
| C-GATE | P6' robust undominated | E_ep(robust) <= 0.10 (E_dec underpowered) | robust | s=3 >=3-module undominated | T2 | 7.27 % | [3.53 %, 11.59 %] | 10.76 % / 4.12 % | upper95 <= 0.1 | **FAIL** |  |
| C-GATE | robust validity s=0 | E_dec(robust, stratum) <= 1 % | robust | s=0 >=3-module stratum (v2+clamp) | T2 | 0.00 % | [0.00 %, 0.00 %] | 0.00 % / 0.00 % | <= 0.01; upper95 <= 0.013 | **UNDERPOWERED** | E_ep decides (next row) |
| C-GATE | robust validity s=0 | E_ep(robust) <= 0.10 (E_dec underpowered) | robust | s=0 >=3-module stratum (v2+clamp) | T2 | 0.00 % | [0.00 %, 0.00 %] | 0.00 % / 0.00 % | upper95 <= 0.1 | **PASS** |  |
| C-GATE | robust validity s=1 | E_dec(robust, stratum) <= 1 % | robust | s=1 >=3-module stratum (v2+clamp) | T2 | 0.00 % | [0.00 %, 0.00 %] | 0.00 % / 0.00 % | <= 0.01; upper95 <= 0.013 | **UNDERPOWERED** | E_ep decides (next row) |
| C-GATE | robust validity s=1 | E_ep(robust) <= 0.10 (E_dec underpowered) | robust | s=1 >=3-module stratum (v2+clamp) | T2 | 0.00 % | [0.00 %, 0.00 %] | 0.00 % / 0.00 % | upper95 <= 0.1 | **PASS** |  |
| C-GATE | robust validity s=2 | E_dec(robust, stratum) <= 1 % | robust | s=2 >=3-module stratum (v2+clamp) | T2 | 0.00 % | [0.00 %, 0.00 %] | 0.00 % / 0.00 % | <= 0.01; upper95 <= 0.013 | **UNDERPOWERED** | E_ep decides (next row) |
| C-GATE | robust validity s=2 | E_ep(robust) <= 0.10 (E_dec underpowered) | robust | s=2 >=3-module stratum (v2+clamp) | T2 | 0.00 % | [0.00 %, 0.00 %] | 0.00 % / 0.00 % | upper95 <= 0.1 | **PASS** |  |
| C-GATE | robust validity s=3 | E_dec(robust, stratum) <= 1 % | robust | s=3 >=3-module stratum (v2+clamp) | T2 | 3.31 % | [1.58 %, 5.37 %] | 5.06 % / 1.79 % | <= 0.01; upper95 <= 0.013 | **FAIL (underpowered; exceeds bar)** | E_ep decides (next row) |
| C-GATE | robust validity s=3 | E_ep(robust) <= 0.10 (E_dec underpowered) | robust | s=3 >=3-module stratum (v2+clamp) | T2 | 7.27 % | [3.53 %, 11.59 %] | 10.76 % / 4.12 % | upper95 <= 0.1 | **FAIL** |  |
| C-GATE | price of robustness | CF(par, v2) - CF(robust, v2+clamp) at s=0 | robust | s=0 >=3-module stratum | T2 | -56.66 % | [-57.99 %, -55.34 %] | -55.52 % / -57.76 % | <= 0.1 | **PASS** | descriptive, pre-stated; point-judged |
| C-GATE | lawcal held-out s=3 | E_dec(lawcal) <= 2 % | lawcal | s=3 >=3-module stratum | T2 | 0.49 % | [0.00 %, 1.37 %] | 1.17 % / 0.00 % | upper95 <= 0.02 | **PASS** | descriptive, pre-stated |
| C-GATE | dose response | Spearman(E_dec par undominated, s) | par |  |  | 77.46 % | [77.46 %, 86.60 %] | 86.60 % / 77.46 % |  | **DESCRIPTIVE** | pre-stated direction: >= 0 |
| C-ENV | prerequisite | KS p > 0.05 or D < 0.1 for every statistic (>= 500 generated buildings) |  |  |  | 100.00 % | [—, —] | — / — | >= 1.0 | **PASS** | 8 statistics |
| C-ENV | 1 validity | E_mis | amortized_comp | ID |  | — | [—, —] | — / — |  | **MISSING** |  |
| C-ENV | 1 validity | E_mis | amortized_comp_envcal | ID |  | — | [—, —] | — / — |  | **MISSING** |  |
| C-ENV | P7 | success - best baseline |  |  |  | — | [—, —] | — / — |  | **MISSING** |  |
| C-ENV | 4 reconfiguration | share of oracle plans that merge/split | oracle | ID |  | 33.45 % | [—, —] | — / — | >= 0.2 | **DESCRIPTIVE** |  |
| C-SCALE | 1 validity | E_ep | amortized_comp | N 7-8 |  | 1.67 % | [0.00 %, 3.89 %] | 3.33 % / 0.00 % | <= 0.15; upper95 <= 0.18 | **PASS** |  |
| C-SCALE | 1 validity | stratum coverage | amortized_comp | N 7-8 |  | 85.80 % | [82.07 %, 89.17 %] | 88.65 % / 82.68 % | >= 0.85 | **PASS** | point-judged (§9.2) |
| C-SCALE | 1 validity | E_ep | amortized_comp | N 9-10 |  | 1.67 % | [0.00 %, 4.17 %] | 4.17 % / 0.00 % | <= 0.15; upper95 <= 0.18 | **PASS** |  |
| C-SCALE | 1 validity | stratum coverage | amortized_comp | N 9-10 |  | 81.16 % | [75.53 %, 86.68 %] | 85.78 % / 76.43 % | >= 0.85 | **FAIL** | point-judged (§9.2) |
| C-SCALE | 2 comp vs direct | coverage slope comp - direct |  |  |  | — | [—, —] | — / — |  | **MISSING** |  |
| C-SCALE | 3 extrapolated share | closed-form share vs N |  |  |  | — | [—, —] | — / — |  | **NOT SCORED HERE** | theory role, deterministic T0 (not a stochastic claim) |
| C-HET | sanity gate | conventional module |rho_sway| < 0.02 |  |  |  | — | [—, —] | — / — |  | **SEE TEST** | generator unit test test_conventional_module_cannot_sway, run before generation |
| C-HET | 1 validity | E_ep (faulted) | amortized_comp | conf_H1 |  | 0.44 % | [0.00 %, 1.44 %] | 1.33 % / 0.00 % | <= 0.15; upper95 <= 0.18 | **PASS** | faulted-candidate block (A-HET) |
| C-HET | 1 validity | E_ep (all) | amortized_comp | conf_H1 |  | 0.40 % | [0.00 %, 1.27 %] | 1.20 % / 0.00 % | <= 0.15; upper95 <= 0.18 | **DESCRIPTIVE** | all-decisions block: descriptive (A-HET) |
| C-HET | 1 validity | E_ep (all) | amortized_comp | conf_H2 |  | 3.20 % | [1.80 %, 4.80 %] | 4.60 % / 2.00 % | <= 0.15; upper95 <= 0.18 | **PASS** |  |
| C-HET | 1 validity | E_ep (faulted) | amortized_comp_blind | conf_H1 |  | 0.00 % | [0.00 %, 0.00 %] | 0.00 % / 0.00 % | <= 0.15; upper95 <= 0.18 | **DESCRIPTIVE** | type withheld (interpretation, §6.3); faulted-candidate block (A-HET) |
| C-HET | 1 validity | E_ep (all) | amortized_comp_blind | conf_H1 |  | 0.00 % | [0.00 %, 0.00 %] | 0.00 % / 0.00 % | <= 0.15; upper95 <= 0.18 | **DESCRIPTIVE** | type withheld (interpretation, §6.3); all-decisions block: descriptive (A-HET) |
| C-HET | 1 validity | E_ep (all) | amortized_comp_blind | conf_H2 |  | 0.00 % | [0.00 %, 0.00 %] | 0.00 % / 0.00 % | <= 0.15; upper95 <= 0.18 | **DESCRIPTIVE** | type withheld (interpretation, §6.3) |
| C-HET | 2 degradation | coverage drop ID -> conf_H1 (faulted) | amortized_comp | conf_H1 |  | -0.02 % | [-0.13 %, 0.11 %] | 0.09 % / -0.11 % | <= 0.05 | **PASS** | ID - split, point-judged (§9.2); faulted block (A-HET) |
| C-HET | 2 degradation | coverage drop: direct - ours (conf_H1, faulted) | amortized_comp | conf_H1 |  | 6.28 % | [5.17 %, 7.38 %] | 7.21 % / 5.39 % | lower95 > 0.0 | **PASS** | secondary |
| C-HET | 2 degradation | CF drop ID -> conf_H1 (faulted) | amortized_comp | conf_H1 |  | -0.93 % | [-3.32 %, 1.22 %] | 0.88 % / -2.90 % | <= 0.05 | **PASS** | ID - split, point-judged (§9.2); faulted block (A-HET) |
| C-HET | 2 degradation | CF drop: direct - ours (conf_H1, faulted) | amortized_comp | conf_H1 |  | -3.46 % | [-6.03 %, -0.90 %] | -1.42 % / -5.64 % | lower95 > 0.0 | **FAIL** | secondary |
| C-HET | 2 degradation | coverage drop ID -> conf_H2 (all) | amortized_comp | conf_H2 |  | 0.16 % | [0.05 %, 0.28 %] | 0.26 % / 0.06 % | <= 0.05 | **PASS** | ID - split, point-judged (§9.2) |
| C-HET | 2 degradation | coverage drop: direct - ours (conf_H2, all) | amortized_comp | conf_H2 |  | 1.61 % | [1.32 %, 1.89 %] | 1.85 % / 1.37 % | lower95 > 0.0 | **PASS** | secondary |
| C-HET | 2 degradation | CF drop ID -> conf_H2 (all) | amortized_comp | conf_H2 |  | 19.84 % | [16.97 %, 22.45 %] | 22.14 % / 17.43 % | <= 0.05 | **FAIL** | ID - split, point-judged (§9.2) |
| C-HET | 2 degradation | CF drop: direct - ours (conf_H2, all) | amortized_comp | conf_H2 |  | 29.97 % | [27.19 %, 32.49 %] | 32.11 % / 27.73 % | lower95 > 0.0 | **PASS** | secondary |
| C-EXPLAIN | wheel precision | wheel precision | amortized_comp | C-MAIN T1 refusals (stratum) | T1 | 95.32 % | [93.13 %, 97.20 %] | 96.98 % / 93.49 % | >= 0.95 | **PASS** | point-judged; secondary, not in Holm |
| C-EXPLAIN | deficit-claim precision | deficit-claim precision | amortized_comp | C-MAIN T1 refusals (stratum) | T1 | 77.11 % | [74.62 %, 79.42 %] | 79.04 % / 74.92 % | >= 0.9 | **FAIL** | point-judged; secondary, not in Holm |
| C-EXPLAIN | informative share (incapable refusals) | informative share (incapable refusals) | amortized_comp | C-MAIN T1 refusals (stratum) | T1 | 100.00 % | [100.00 %, 100.00 %] | 100.00 % / 100.00 % | >= 0.8 | **PASS** | point-judged; secondary, not in Holm |
| C-EXPLAIN | false-reason rate (capable refusals) | false-reason rate (capable refusals) | amortized_comp | C-MAIN T1 refusals (stratum) | T1 | 12.97 % | [11.68 %, 14.32 %] | 14.08 % / 11.87 % | <= 0.05 | **FAIL** | point-judged; secondary, not in Holm |
| C-EXPLAIN | margin share | share of refusals with a margin claim | amortized_comp |  | T1 | 63.44 % | [62.17 %, 64.70 %] | 64.52 % / 62.37 % |  | **DESCRIPTIVE** | margin claims make no physical assertion |
| C-GATE | 1 H12 overall | undominated excess (P5) and dominated condition | raw, par | s=3 v2 >=3-module | T2 | — | [—, —] | — / — | all component rows PASS | **FAIL** | dominated rows: raw MISSING, par MISSING |
| C-GATE | P6' overall | E_ep(par) - E_ep(robust) > 0 (Holm) AND robust undominated validity | robust | s=3 >=3-module | T2 | — | [—, —] | — / — | component rows PASS | **FAIL** | E_ep (s=3, >=3-module): par - robust: FAIL; E_dec(robust, undominated) <= 1 %: FAIL (underpowered; exceeds bar); E_ep(robust) <= 0.10 (E_dec underpowered): FAIL |
| C-MAIN | P3 overall | all pre-registered conditions |  |  |  | — | [—, —] | — / — | every component row PASS | **MISSING** | P3 / E_dec own (raw): MISSING |
| C-MERGE | P4 overall | all pre-registered conditions |  |  |  | — | [—, —] | — / — | every component row PASS | **PASS** | P4 / CF@U* comp_enum_size: PASS; P4 / CF@U* (stratum merges): comp_enum_size - copy_parent: PASS |

## Holm table, P1–P7 (family-wise 0.05, m = 7)

| P | one-sided p | Holm level | rejects |
|---|---|---|---|
| P1 | 0.0005 | 0.00714 | yes |
| P2 | 0.0005 | 0.00833 | yes |
| P3 | — | — | no |
| P4 | 0.0005 | 0.01 | yes |
| P5 | 0.903 | 0.0125 | no |
| P6' | 1 | 0.0167 | no |
| P7 | — | — | no |
