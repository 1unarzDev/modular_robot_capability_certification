# env (HM3D-calibrated buildings and missions; plan §2a)

EXPORT: (research lead, 01:12Z; owner decision) no rental extension. After 01:45Z run `scripts/env_score.py --confirmatory` on conf-ID (missions 0–1498; 1499 excluded) and conf-E-H1..4, with models' env_conf_lower files (amortized_comp primary, envcal secondary, raw descriptive). Report unfinished families as missing. Scoring runs ONCE at ~03:10Z.

OPS: (ops, 22:12Z) cfcap-env is complete (28/28). Outputs are at `~/.cache/vast-out/*/crane/cfcap-env.<group>.sNNN/`. Exploratory groups are ready for analysis; leave the env-conf-* dirs unopened until 01:45Z.


OPS: (ops, 21:33Z) cfcap-env is submitted at 17bd917: 28 jobs (make_jobs emitted 28, not 30), priority 70–84. Conf outputs stay unopened until 01:45Z.


OPS: (env, 21:15Z) `scripts/campaigns/env_missions.json` (lineage cfcap-env) is ready: 30 jobs, 3.5 box-h, ~$0.64, `make_jobs emit` + `validate` ok at cfcap 3ba34ca+. The auto canary runs an exploratory spare unit (no conf labels). Conf groups use `--spare-shard0`, so their shard 0 is an exploratory timing unit. Do not read conf-* labels before 01:45Z.

**Status (2026-11-25 ~21:15Z):**
- `cfcap/env/` is built, tested (9/9, including the GPU label test) and committed.
- The calibration bar passes for every match-HM3D statistic. The C-ENV prerequisite file `results/env/buildings_ks.csv`
  is written.
- Local pilots are done. The campaign is ready for ops.

## What is built
- `cfcap/env/hm3d_stats.py` → `results/env/hm3d_stats.json`.
  - Reads the 900 HM3D-derived building records (MRMN real_buildings, read-only): 2,210 missions in 737 buildings.
  - HM3D (Ramakrishnan et al., 2021) under the Matterport academic terms (`DATA_TERMS.md`). Building ids are dropped
    on load. The JSON holds only derived per-mission features (no ids) and the same terms apply to it.
  - Field meanings come from MRMN `real_buildings.py`:
    - route_steps counts lattice moves plus 45° rotation steps plus the transition, so m/step < 0.1 occurs;
    - the grid is 0.1 m (`building_mesh.RESOLUTION`);
    - there is exactly **one narrow passage** per mission, by construction;
    - sites are the `geometry_coupled` reconfigure segments.
  - HM3D (exploratory):

    | statistic | mean | median | q05–q95 |
    |---|---|---|---|
    | route_m | 4.42 | 3.36 | 1.74–10.4 |
    | route_steps | 45.8 | 36 | 20–103 |
    | tortuosity | 1.99 | 1.31 | 1.02–5.4 |
    | passage_pos | 0.80 | 0.86 | 0.40–1.0 |
    | station_m | 1.80 | 1.86 | 0.51–2.87 |

    - Sites: 1 per mission in 97.6 % of missions. Levels: 1/2/3/≥4 = 35/41/21/2.5 %.
    - The records carry **no door widths**. Widths are a design distribution: 0.45–1.4 m, narrow passage
      0.45 + 0.95·Beta(2,2).
- `cfcap/env/buildings.py`: procedural buildings, route first, as in the HM3D draw.
  - A level is a graph of spaces (rooms and corridors) joined by doors.
  - Route knobs are drawn from a Gaussian KDE (0.5 × Scott) of the joint HM3D features. A polyline on the 8-heading
    lattice realises them (48 proposals; 20–250 steps as in MRMN).
  - It has turns with radius 0.3–1.5 m, μ patches 0.5–1.1 and a payload station (0–5 kg).
  - **Calibration** (`results/env/calibration.csv`, 3,000 missions in train/test and 1,000 in the other families;
    statistics recomputed from the realised geometry):

    | statistic | max KS D over in-distribution families |
    |---|---|
    | route_m | 0.058 |
    | route_steps | 0.059 |
    | m_per_step | 0.036 |
    | tortuosity | 0.086 |
    | passage_pos | 0.077 |
    | station_m | 0.040 |
    | n_sites | 0.011 |

    - Every match-HM3D row passes. Cliff's δ ≤ 0.07.
    - KS p is often < 0.05 at n = 3,000 vs 2,210, so the pass rests on effect size.
    - The passage count is 1 on both sides by construction.
    - E-H3 deliberately fails route_m / steps / tortuosity, because it is the HM3D upper tail.
- `cfcap/env/missions.py`: segment → requirement on the 10 legs, in route order.
  - straight → surge+ (or surge− for a reversal, sway± for a strafed corner);
  - 45° leg → diag±±;
  - corner rotation → yaw± at ω = v/r;
  - every door → a sway± alignment plus the footprint check.
  - **Requirement in ρ_dec units:** req[c,s] = v_req / (ρ_ref[c,leg] · |cmd[c,leg]|), the fraction of the nominal
    healthy body's achieved speed. Large bodies get larger yaw requirements because the T1 yaw command is capped by
    0.45/Lc.
  - **Footprint:** (min cell extent − 1)·0.32 + 0.24 m + 0.10 m clearance ≤ door. Bodies can strafe, so the narrow
    side counts.
  - **Load:** payload ≤ 2.0 kg per module. This is an assumption (deck rating). Together with footprint it creates
    the tension: heavy loads need more modules, while narrow doors need narrow bodies.
  - **Labels:** generator `simrun.label_conditions` (μ factor, payload at the candidate centre), one run per
    distinct (μ rounded to 0.05, loaded) condition × candidate × 10 legs.
  - Bodies come from generator episodes (self / children / merges, the `test` stream for test and E-H, index
    3,000,000 + 100k·family + mission; disjoint from gen-ref and conf_*).
- `cfcap/env/task.py:TASK` (runner, `--task-spec`): one row per (mission, candidate, segment).
- `tests/test_env.py` (7 tests):
  - the calibration bar on a 600-mission sample;
  - footprint hand cases;
  - determinism and family-disjoint streams;
  - ranges per family;
  - record and interface logic on stub labels;
  - a GPU label range test.
- Mission structure (60 missions per family; CPU): 8–9 segments, 2.3 conditions and 16–21 candidates per mission, of
  which 12–16 are footprint- and load-feasible. E-H1 has 8.1 feasible candidates, E-H4 6.2, and E-H2 has 20
  segments and 6.6 conditions.

## Interface for decide (proposal; `cfcap/env/missions.py`)
- `recs = label_missions([make_mission(fam, i) ...])` gives the truth. Each record holds:
  - **candidates C:** cand_kind, cand_n_mod, cand_width, cand_mask (module bitmask, bit 16·role + idx);
  - **segments S:** seg_leg, seg_kind, seg_phase (0 before the reconfiguration site, 1 after), seg_cond, seg_door_w,
    seg_narrow;
  - **conditions K:** cond_mu, cond_loaded;
  - **per (C, S):** req, fp_ok, load_ok;
  - **truth:** rho_dec (C, K, L), seg_ok, mission_ok, phase_ok.
- Body observations come from `(plate, body_split, body_index)` through `cfcap.gen.episode.make_episode` +
  `simrun.probe_episodes`, which is deterministic. Decide's posterior draws must be pushed through T1 **under each
  condition** (μ factor, payload) to give lb (C, K, L).
- `certified(rec, lb)` returns the (C, S) mask (geometry is known, so a NaN bound certifies nothing).
- `plan_options(rec)` lists the plans (a, b): body a drives phase 0 and body b phase 1. b is a split of a or a merge
  containing a; a == b means no reconfiguration.
- `plan_ok(rec, mask, plan)` scores a plan.
- Unsafe = a committed plan with a truly failing segment (E_mis in CONFIRMATORY §4).
- Decision-relevant stratum (`relevant`): footprint- and load-feasible, with |truth − req| ≤ 0.15.
- Confirmatory C-ENV stream: families `conf-ID` (300 buildings) and `conf-E-H1..4` (150 each), 5 missions per
  building, seeds `[92599, 23, b]`. No conf building had been drawn when calibration was committed (28169a3).

## Proposal (campaign `scripts/campaigns/env_missions.json`, lineage cfcap-env)
- **Pilot** (exploratory; 4070 Ti S; `$CFCAP_OUT/env/pilot1`, `pilot2`; scored in `results/env/pilot_missions.csv`
  by `scripts/env_score.py`):
  - **Speed.** 0.07–0.09 s/mission for test/cal/E-H1/E-H3/E-H4, and 0.17 s for E-H2 (6.6 conditions per mission).
    Each mission has ~160 rows (mission × candidate × segment).
  - **Decision-relevant pairs** (feasible, |truth − req| ≤ 0.15): 17.5 per mission ID and 48.6 per E-H2 mission.
  - **Footprint- and load-feasible segments** fail in truth 2.8–3.0 % of the time, after requirement ranges were
    matched to the codex task library:
    - surge r median 0.55, q95 0.97;
    - sway 0.11 / 0.34;
    - diag 0.32 / 0.96.
  - **Mission plans** (pilot2, 300 missions each; exploratory; no certificate yet):

    | | test | cal | E-H2 |
    |---|---|---|---|
    | oracle success | 0.94 | 0.93 | 0.96 |
    | oracle plans that reconfigure (bar 4 needs ≥ 0.20) | 0.33 | 0.30 | 0.34 |
    | nominal plug-in E_mis | 0.13 | 0.14 | 0.10 |
    | biggest_body E_mis | 0.10 | 0.11 | 0.08 |
    | greedy_feasible E_mis | 0.20 | 0.19 | 0.16 |

    So nominal capability is unsafe at the mission level, and a certificate has something to fix.
- **Plan** (`scripts/campaigns/env_missions.json`, 500-mission units, `--compress`):

  | group | missions | jobs | min/job |
  |---|---|---|---|
  | train | 20,000 | 4 | 11.7 |
  | cal | 10,000 | 2 | 11.7 |
  | test | 6,000 | 2 | 8 |
  | E-H1 / E-H3 / E-H4 | 6,000 each | 2 each | 7.5–8.5 |
  | E-H2 | 6,000 | 3 | 10 |
  | conf-ID | 1,500 (300 buildings × 5) | 2 | |
  | conf-E-H1..4 | 750 each | 2 each | |

  - **Total:** 30 jobs, **3.5 box-h, $0.64**.
  - **Expected yield:** ~100k decision-relevant (mission, segment) decisions in test and in each E-H family (≥ 20k
    bar). The confirmatory counts are the pre-registered ones.
  - unit_s = pilot × 1.3 for a 3090, plus 150 s overhead.
  - **Bytes:** ~25 KB/mission raw, ~1.6 GB total before compression.
- **Scoring after data:**
  - `python scripts/env_score.py <pulled dirs> --claim C-ENV --lower amortized_comp=… --lower raw=… --out
    results/env/confirmatory/missions_conf_env.csv.gz` writes the §5b format.
  - Built in: oracle, nominal, biggest_body and greedy_feasible.
  - `amortized_comp` and `raw` need per-row lower bounds (columns family, mission, cand, seg, lower) from the models
    role, computed under each segment's condition.

## CONFIRMATORY C-ENV mission file (03:12Z, after FINAL PULL COMPLETE)
- I wrote `results/env/confirmatory/missions_conf_env.csv.gz`:
  - 7,200 rows: ID missions 0–599 and each E-H family 0–299 (A-POWER);
  - claim C-ENV, all rows from shard 1 (the shard-0 spare units are exploratory and filtered by `--claim C-ENV`);
  - mission 1499 is excluded (outside the A-POWER range anyway).
- Command: `scripts/env_score.py <10 cfcap-env.env-conf-* dirs> --claim C-ENV --keep conf-ID=0:600 --keep
  conf-E-H{1..4}=0:300`. All 10 conf manifests were complete, and no duplicate rows were found.
- **Methods present:** oracle, nominal, biggest_body, greedy_feasible.
- **Methods missing:** `amortized_comp`, `raw` and `amortized_comp_envcal`. No models conf env lower file exists
  (`results/models/confirmatory/` has no env_conf_lower_*; `results/models/env/` holds only exploratory test-family
  bounds). The scorer will therefore report P7 and the validity bars for ours as MISSING.
- Descriptive (baselines; E_mis = failed / committed):

  | family | oracle success | oracle plans that reconfigure | nominal E_mis | biggest_body E_mis | greedy_feasible E_mis |
  |---|---|---|---|---|---|
  | ID | 0.942 | 0.33 | 0.10 | 0.09 | 0.15 |
  | E-H1 | 0.937 | 0.82 | 0.10 | 0.09 | 0.17 |
  | E-H2 | 0.930 | 0.37 | 0.12 | 0.10 | 0.18 |
  | E-H3 | 0.950 | 0.36 | 0.13 | 0.10 | 0.19 |
  | E-H4 | 0.837 | 0.36 | 0.09 | 0.09 | 0.09 |

- Bar 4 (oracle plans reconfigure in ≥ 20 % of ID missions): 0.33, before the scorer's building-cluster CI.

## Payload placement (answer to the research lead)
- In labelling, the payload is a point mass at the **candidate's centre**: (mass, 0, 0) in the candidate frame,
  passed through `simrun.label_conditions`.
- Missions give only a station and a mass, with no deck position. So the labels **approximate the load as centred**:
  there is no CoM offset and no host module. This differs from the generator episodes, which put the payload on a
  host module's deck with an offset of ±0.08 m.
- An off-centre load would shift the normal loads and could lower yaw and diagonal capability. The env truth is
  therefore optimistic for off-centre loads. It is a mission-model assumption, not a measured property.
- The 2 kg/module load rating is a separate hard constraint and is also an assumption.

## Confirmatory exclusions (A-ENV)
- conf-ID mission 1499 is excluded: ID uses missions 0–1,498. `scripts/env_score.py` drops it (`EXCLUDE`).
- The secondary certifier `amortized_comp_envcal` is scored next to the primary through
  `--lower amortized_comp_envcal=PATH` (same row format).

## Needs
- **decision:** confirm the interface. Mission-level policies (ours / raw / nominal / biggest_body /
  greedy_feasible / oracle) go in `cfcap/decide/`. They need lb under conditions.
- **models:** per-row lower bounds for amortized_comp and raw on the env bodies (observations from
  `make_episode(plate, body_split, body_index)` + `probe_episodes`), under each condition (cond_mu, payload at the
  candidate centre). Without them, C-ENV P7 and the validity bars for ours are MISSING.
- **ops:** submit env_missions.json (OPS line above).

## Next step
- Pull the exploratory groups and score nominal / biggest / greedy / oracle on test and E-H, with building-clustered
  CIs.
- Hand cal rows to decide/models for the mission-level margin.
- Known limits:
  - the turn-rate requirement uses W_NOM-capped yaw;
  - the 2 kg/module load rating is an assumption;
  - door widths are not in HM3D;
  - μ and payload change T1 truth only mildly, so faults and geometry dominate the failures.

## decision: interface confirmed (decision role, 2026-11-25)
- Confirmed: `record` / `certified(rec, lb)` / `plan_options` / `plan_ok` as above. The decision role consumes them
  unchanged.
- `cfcap/decide/missions.py` (tested) turns a predictor's point/scale (C, K, L) into the `lower` rows that
  `scripts/env_score.py` reads.
  - The certificate is a mission-level simultaneous score: max over the geometry-feasible (c, s) of
    (point − truth)/scale.
  - It is cross-fitted **by building**, so that no building straddles the calibration and test sides.
  - (env, A-ENV 0dc1aec) For `amortized_comp` (the T1-cal compositional constant) the E_mis bars are **empirical
    transfer bars, not a guarantee**: that constant was calibrated at nominal floor and payload conditions, and
    missions evaluate segments under μ factors and payloads outside that distribution. Only the secondary
    `amortized_comp_envcal` (q_env fitted on exploratory ID env missions) carries a finite-sample guarantee, and only
    on ID missions, by exchangeability.
- One request: expose `T_RECONF` as a module constant in `missions.py`, so the decision layer charges the same
  split/merge time (decide uses T_SPLIT_S = 30 s).

**models → env (03:14Z):** confirmatory env bounds written, `results/models/confirmatory/env_conf_lower_{amortized_comp,raw,amortized_comp_envcal,credible_lower}.csv.gz` (model_tag st-n6-2; columns family, mission, cand, seg, lower). Coverage: **conf-ID missions 0–499 only** (the complete models-conf-env units in the final pull; A-POWER range 0–599, so 500–599 are missing); **no conf-E-H units completed**, so E-H model comparisons are MISSING tonight. q_comp = 1.618 (st-n6-2 train-run eval-cal fallback), q_env = 1.813 fitted on 299 exploratory ID missions (test 0–298 with model rows; < the 400 planned). `raw` = plug-in median (not a bound). Rerun `scripts/env_score.py ... --lower amortized_comp=… --lower raw=… --lower amortized_comp_envcal=…`.
