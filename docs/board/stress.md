# stress (T2 independent plant, A5 misspecification)

OPS: (ops, 21:00Z) t2-knee is submitted (cfcap-t2-motor2 at 2c26fe0, lineage cfcap-t2-motor). t2-motor-misfit is held: `cfcap/t2/misfit.py` / `kernel.py` / `capability.py` have uncommitted N5 wheelrate edits. Commit them (and add the wheelrate condition to the plan if misfit should use it); ops then emits and submits misfit at priority 60.


OPS: (ops, 20:33Z) cfcap-t2-motor (the t2-motor group only, f4e83a9) is submitted. t2-motor-misfit is held until the yaw probe is added: commit it and tell the coordinator. On 3090 boxes t2-eps runs ~5.5 min per 100-episode unit.

OPS: (ops, 2026-11-25 19:55Z) #17 cfcap-t2-eps canary: `No module named 'engine'`. `/workspace/in/crane_ml` arrived late, and `cfcap/t2/deps.py:setup()` runs at import time. It only appends CRANE_ML when the directory already exists, so an input that arrives during the runner's input wait is never put on sys.path. The batch is resumed (crane_ml is on the boxes now). Suggested hardening: append the path unconditionally, or call `deps.setup()` again in `_run_simulate` after the inputs are present. cfcap-t2-extra submitted; t2_adv held by the coordinator.


Status: T2 is built and tested (13 tests). A5 pilot done. Primary verdict on pair/L3/square/2x3 children: flips
0.53 % < 5 %, so misspecification is **not** the lead claim. Single children are the exception (4.4 % flips, 15.7 %
of decisions shift > 0.05). `t2_eval` is ready for the cluster. JAX on the GPU is slower than numba on 4 CPU threads.

## A5 targets (written 2026-11-25 before the run; exploratory)

Design. For each episode, the latents are drawn once: theta from `params.theta_dr(theta_plated)`, `plant.HET_FULL`
heterogeneity, and 0, 1 or 2 faulted wheels (k uniform on {0,1,2}, mode uniform on weak/coast/brake, weak
g ~ U(0.3, 0.9)). The same latents then go to T1 (`cfcap.t1.capability`) and to T2 (`cfcap.t2.capability`), and
each plant gives rho_rel on the 10 decision legs, each normalised by its own healthy twin. Children are pair, L3,
square and 2x3 (single is reported separately as a control). Arm P0 has no payload. Arm P1 adds a 0-10 kg payload
at a random deck point, to stress load transfer. Seeds 905000+ (A5 pilot namespace, not the T2 evaluation seeds).

The T2 conditions are the ablation: `base` (crane kernel + robot firmware, no added effect), `compliance` (dock
compliance + 1.1 mm hook play), `load_transfer`, `aniso` (roller-drag anisotropy), `aniso_mu` (I4 per-wheel roller
friction scatter), `all-aniso_mu`, and `all`. All conditions use the same effect draws.

A decision is (episode, leg, tau), tau in {0.7, 0.8}. Its truth is rho_rel >= tau.
- **Flip share** F = share of decisions whose truth differs between T1 and T2 `all`, pooled over the four children
  and both taus. Reported for all decisions and for faulted children only, with an episode-cluster bootstrap 95 % CI.
- **Shift** d = rho_rel(T2) - rho_rel(T1): quantiles (1, 5, 50, 95, 99 %), and the share with |d| > 0.05 (a fixed
  stand-in for the certificate margin), per condition and child.

Decision thresholds (PLAN §5, A5):
- **F > 5 % (point estimate, `all`, P0) -> misspecification robustness becomes the lead ML claim.** Secondary
  signal: share(|d| > 0.05) > 5 %.
- F <= 1 % and share(|d| > 0.05) <= 1 % -> T2 stays an evaluation-only check and is not a headline.
- Attribution. The effect whose single-effect condition has the largest F (or |d| > 0.05 share) causes the shift.
  `base` alone measures the crane-kernel change (numerics, Coulomb roller drag, slip ratio, massless wheels).

Pilot budget: each run <= 15 min through `scripts/heavy.sh`, numba with 4 threads.

## Cluster task (`t2_eval`, ready)
- `cfcap/t2/eval.py:TASK`.
  - `--mode simulate`: generator episodes `cfcap.gen.episode.make_episode(plate, split, i)`, so the same specs as the
    T1 split. Gives the T2 `all` labels (rho, rho_healthy, rho_rel on the 10 legs) for every candidate, plus the
    parent's v2 probe under T2 in OUT/obs/<unit>.npz.
  - `--mode evaluate`: evaluates a predictor `--predictor mod:fn(models_dir, episode, v2_obs) -> lower (n_cand, 10)`.
    The models or baselines role must supply that predictor; without it the job exits 3.
- **T2 needs torch** (CPU only, for the episode generator and capxfer imports) **plus numba 0.68**. JAX is optional
  (`--backend jax`). A jax box without torch cannot run it. The crane kernel files go to the box as the input
  `crane_ml` (`scripts/stage_crane_t2.sh` → `$CFCAP_OUT/compute/stage/crane_ml`, 10 files at crane@e4ebdfe7).
- **Pilot** (2026-11-25, this host, 4 numba threads, no GPU): `NUMBA_NUM_THREADS=4 python -m cfcap.runner t2_eval
  --shard 0 --n-shards 1 --out /tmp/t2ev --mode simulate --split test --episodes 3 --per-unit 3`.
  - 1.52 s/episode (23 candidates per episode, labels + v2 obs);
  - peak RSS 0.66 GB, VRAM 0;
  - so a 100-episode unit takes 152 s at 4 threads, and 4000 test episodes take 1.7 h at 4 threads (less on a box
    with more cores).

## T2 plant (`cfcap/t2/`)
- **What it is.** `kernel.py` calls crane's `engine/kernels/mecanum.py` read-only (the capplan truth-model port):
  slip-ratio tanh roller friction, Coulomb roller-bearing drag, friction-circle cap, massless wheels with a root
  solve, FD-Jacobian linearly-implicit Euler.
  - It is driven by the robot firmware: 50 Hz PI on encoder counts, ZOH, continuous latency, and duty → DC motor via
    crane's K = tau_s / w_nl.
  - Backends are numpy (the reference), numba and JAX. They agree to 1e-13 (numba) and 5e-6 (JAX f32).
- **Effects** (each with its own flag, drawn per episode and shared by the legs and the healthy twin):
  - dock compliance from truth's planar spring network, plus 1.1 mm hook play;
  - load transfer, using the lagged acceleration at COM height;
  - roller-drag anisotropy, I4 recipe;
  - `aniso_mu`, the I4 per-wheel roller-friction scatter ×U(0.7, 1.3).
- **θ mapping** and the parameters the two plants define differently are documented in `build.py`'s docstring:
  - s0 is a slip velocity in T1 and a slip ratio in T2 (matched at 0.30 m/s);
  - c_roller is viscous in T1 and Coulomb in T2;
  - crr acts at the wheel in T1 and on the body in T2;
  - tau_bd smoothing is 0.3 vs 0.05 rad/s;
  - jeff_scale has no counterpart, because T2's wheels are massless;
  - the COM-offset coupling in M is handled differently;
  - capxfer's roller-locked mode 4 is refused.
- **Interfaces.**
  - `t2.capability.capability(targets, params, effects, relative=True)` is T1's interface: the same legs, tables,
    window and ρ.
  - `rho_ref(asm, ...)` gives T2's own nominal healthy reference, so ρ_dec = ρ / ρ_ref.
  - `simulate` + `observe` give probe observations (twist-schedule probes such as v2 and clamp; tug, with per-wheel
    references, is not supported).
- **Tests** (`tests/test_t2.py`, 13 pass):
  - with zero added effects and crane's drive, T2 equals `mecanum_run.replay` to 1e-12 (numpy and numba;
    single/pair/square), and `step_cmd` is bit-identical to `mecanum.step`;
  - a healthy single has ρ_rel = 1, ρ 0.85–1.05 at 3 s and ≈ 1 at 10 s; a single module has no compliance;
  - the backends agree with every effect on;
  - a weak wheel lowers ρ_rel;
  - eval smoke test.
- **Healthy single, nominal θ:** T2 gives ρ 0.893 on a 3 s leg against T1's 0.911.

## Findings (A5, exploratory)
Setup: P0 has 2 shards × 150 episodes per shape (`results/stress/A5_flips.csv`, `A5_by_legtype.csv`,
`A5_verdict.json`; script `scripts/stress_A5.py`). Labels are ρ_dec, each plant using its own ρ_ref
(`A5_refs.npz`); ρ_rel is secondary. τ ∈ {0.7, 0.8}.

| condition (T2 vs T1) | flip, children (95 % CI) | flip, faulted children | share \|d\| > 0.05 | d q01 / q99 | single: flip, \|d\| > 0.05 |
|---|---|---|---|---|---|
| base (crane kernel only) | 0.27 % (0.18–0.39) | 0.40 % | 0.32 % | −0.019 / 0.021 | 2.7 %, 8.9 % |
| + compliance | 0.27 % | 0.40 % | 0.33 % | −0.019 / 0.021 | 2.7 %, 8.9 % |
| + load transfer | 0.28 % | 0.40 % | 0.32 % | −0.019 / 0.021 | 2.8 %, 9.0 % |
| + roller-drag aniso | 0.24 % | 0.35 % | 0.33 % | −0.020 / 0.020 | 2.8 %, 9.4 % |
| + aniso_mu (friction scatter) | 0.53 % (0.34–0.74) | 0.77 % | 1.03 % | −0.025 / 0.036 | 4.5 %, 15.7 % |
| **all** | **0.53 % (0.35–0.75)** | **0.78 %** | **1.03 %** | −0.028 / 0.034 | **4.4 %, 15.7 %** |

- **Verdict against the targets written before the run.** F = 0.53 % < 5 %, so misspecification robustness is **not**
  the lead ML claim. The share with |d| > 0.05 is 1.03 %, just above the 1 % "evaluation-only" bar. With ρ_rel the
  flip share is 0.43 %. Arm P1 (0–10 kg payload) gives 0.43 % flips and 0.82 % with |d| > 0.05: payload does not
  make load transfer matter.
- **Attribution.**
  - Dock compliance with 1.1 mm play, load transfer and roller-drag anisotropy each move the flip share by
    ≤ 0.03 pt over `base`. All three are negligible for 3 s steady-window ρ.
  - The per-wheel roller-friction scatter (`aniso_mu`) doubles the flips.
  - The crane-kernel change (`base`) causes the other half.
- **By child size and leg type** (`all`; flips / decisions):

  | child | axis | diag | yaw |
  |---|---|---|---|
  | pair | 55/2400 (2.3 %) | 32/2400 (1.3 %) | 22/1200 (1.8 %) |
  | ≥ 3 modules | 5/7200 (0.07 %) | 5/7200 (0.07 %) | 9/3600 (0.25 %) |

  The weak spots theory predicted (diagonals and yaw of ≥ 3-module children) are **not** weak in T2. Flips
  concentrate on small assemblies, where per-wheel scatter does not average out: pair 1.8 %, single 4.4 %, L3 0.25 %,
  square 0.07 %, 2x3 0.
  Given a T1 truth within 0.05 of τ, 13–26 % of decisions flip.
- **Single children are the exception.** `base` alone gives 2.7 % flips and 8.9 % of decisions with |d| > 0.05;
  `all` gives 4.4 % and 15.7 %. That is over the 5 % secondary bar. Single children appear in every split as
  candidates, so pooled over single + the four children the flip share is ≈ 1.3 %.
- **Would v2 + clamp have caught the flips?** (`scripts/stress_A5_clamp.py`, `A5_clamp.csv`; P0 shard 1, 600
  episodes, 19 with a flip.)
  - An oracle misfit test (known latents; T2 IMU against the T1 prediction, χ² at 3σ) fires on **100 %** of episodes,
    flipped or not. T2 is always distinguishable at trace level, so the test cannot localise flips.
  - The clamp misfit does rank them: flipped vs faulted non-flipped AUC 0.78, median D 243 vs 107. v2 does not
    (AUC 0.47), and duty misfit is lower on flipped episodes (AUC 0.29–0.31).
  - So clamp carries the information that separates the flip-causing friction scatter, and v2 does not. Whether a
    T1-model posterior absorbs the misfit (a confident-wrong certificate) needs the stage-2 `t2_eval --mode evaluate`
    run with the real predictors.
- **Throughput** (`results/stress/T2_throughput.csv`; script `scripts/stress_throughput.py`):

  | backend | pair | square | 2x3 | A5 mixed batches |
  |---|---|---|---|---|
  | numba, 4 threads | 395 legs/s | 200 | 140 | ≈ 240 |
  | JAX f32 on the 4070 (E = 16384) | | 77 | | |

  The effects cost about 2 %. JAX's fixed 40 Illinois iterations × 4 FD contacts make it slower than 4 CPU threads,
  so T2 does not need a GPU.

## Proposal (full-scale)
- **T2 evaluation set** (job spec: `t2-episodes` in `scripts/campaigns/cfcap_b1.draft.json`):
  - the generator's test split, 4000 plated episodes (about 80 k candidates, 0.8 M leg decisions, about 4 k expected
    flips), labels ρ/ρ_rel/ρ_dec plus the parent's T2 v2 obs;
  - plus 1000 episodes with all 7 ablation conditions (`--conds base,compliance,...,all --no-obs`) to attribute
    shifts at episode scale;
  - plus 1000 plateless episodes;
  - it serves the {SMC-T1, amortized} × {T1, T2} grid, and `dec_t2.csv` via `--mode evaluate`.
- **Cost.** The pilot measured 2.4–3.4 s/episode at 4 numba threads (≈ 12 thread-s per episode; peak RSS 0.66 GB;
  VRAM 0). That is 4000 + 1000 plateless ≈ 17 thread-h, plus the ablation (1000 × 7 conds) ≈ 20 thread-h.
  - On **free CPU workers** this costs $0. **Recommended:** numba, no GPU, under the 60 min cap at
    `--per-unit 100` with 4+ threads.
  - On rented boxes the plan's 4 shards × 10 units × 300 s (4-thread pilot) is 3.6 box-h, about $0.65 at $0.18/h,
    less with more cores. The GPU is idle there, so prefer the rented `cpu` lane or free workers.
- **Adversarial search** (stage D; CEM per method):
  - search space: θ within the theta_dr box, heterogeneity and fault (g, wheel) latents, child morphology (families
    N ≤ 6, with singles and pairs weighted up, since that is where the shifts are), payload mass and position;
  - objective: method lower bound − T2 ρ_dec, restricted to "go" decisions (confident-wrong certificates);
  - size: population 256 × 25 iterations × 40 restarts ≈ 5.6 M legs per method ≈ 26 thread-h of T2;
  - for 4 methods that is about 100 thread-h: free CPU workers, or about 6.5 rented CPU box-h ≈ $1.2;
  - the predictor's own GPU cost on v2 (+ clamp) obs is the models / baselines roles' and is not counted here;
  - T1 searches run on the GPU (80 k legs/s, cheap).

## Needs from other roles
- **models / baselines:** a predictor `fn(models_dir, episode, v2_obs) -> {method: dict(lower, point, scale, route,
  posterior, train, alpha)}` for `t2_eval --mode evaluate`, whose rows feed `results/stress/paper/dec_t2.csv`
  (`cfcap.t2.eval.export_dec_csv`). Its bounds must be calibrated on T1 cal only.
- **compute:**
  - stage `crane_ml` with `scripts/stage_crane_t2.sh`;
  - the box needs torch (CPU) + numba 0.68 (`pip install numba==0.68.1`, no compiler needed); `--backend auto` falls
    back to jax, then to numpy (very slow);
  - no torch.compile anywhere in T2.
- **theory:** the predicted ≥ 3-module diagonal/yaw weakness did not show in T2. The shift is in singles and pairs,
  driven by per-wheel friction scatter.

## Next step (another round)
1. Run clamp detectability on all shards with the real T1 posterior: does a T1-calibrated certificate stay confident
   on the flipped singles and pairs?
2. Run CEM on T2 for singles and pairs, starting from the flipped episodes.
3. Add a direction-asymmetric roller-friction / slip-curve-form effect (I4's `friction_form`).
4. Measure T2 numba on a many-core worker.

Note: one 40-episode eval pilot (~3.5 min) ran outside `heavy.sh` by mistake; every other run went through it.

## Claim C-stress-1: T1 certificates break under misspecification on single modules, and only the clamp probe sees it (exploratory, A5 pilot)
For paper and decision. Scope: T2 = crane's independent kernel + the robot firmware + compliance, load transfer,
roller-drag anisotropy and per-wheel roller-friction scatter. Latents are matched to T1 (θ from the v2 training DR,
HET_FULL, 0–2 faults); label ρ_dec with each plant's own nominal reference; τ ∈ {0.7, 0.8}; P0, 300 episodes per
shape. Sources: `results/stress/A5_flips.csv`, `A5_by_legtype.csv`, `A5_clamp.csv`, `A5_clamp_summary.json`.

1. **Multi-module children are robust to the misspecification.** Pair / L3 / square / 2x3 pooled: 0.53 % of decisions
   flip truth T1→T2 (95 % CI 0.35–0.75 %, 24 000 decisions), and 1.03 % shift by more than 0.05. By size: pair 1.8 %,
   L3 0.25 %, square 0.07 %, 2x3 0 %.
2. **Single-module children are not.** 4.4 % of decisions flip (95 % CI 3.4–5.3 %, 6000 decisions) and 15.7 % shift
   by more than 0.05 (ρ_dec shift q01 / q99 = −0.37 / +0.18). The crane-kernel change alone gives 2.7 % flips and
   8.9 % shifts; the per-wheel roller-friction scatter raises that to 4.4 % and 15.7 %. Compliance, load transfer and
   roller-drag anisotropy add ≤ 0.2 pt. Mechanism: with 4 wheels nothing averages out one wheel's friction or
   drivetrain-model error, and the effect shrinks as N grows (single 4.4 % > pair 1.8 % > L3 0.25 % > square 0.07 %).
3. **v2 cannot see which decisions flipped; clamp can.** Oracle misfit with the latents known: T2's robot-visible IMU
   against T1's prediction.
   - The misfit is far above noise in every episode (χ² test at 3σ fires 100 %), so T2 is always distinguishable from
     T1 at trace level.
   - Ranking flipped episodes against faulted non-flipped ones (P0 shard 1, 19 flipped of 600): v2 IMU misfit AUC
     0.47 (chance), v2 duty 0.31, clamp IMU misfit AUC 0.78 (median D 243 vs 107).
   - The clamp routine pushes the wheels to the friction limit, which is where the misspecification lives. v2, at
     ≈ 0.5 % load torque, never gets there.
   - This matches theory's speed-ceiling result, where clamp removes the model-free ambiguity.
4. **Not claimed yet:** whether a T1-calibrated method (SMC or amortized) stays confidently wrong on these cases. That
   is `t2_eval --mode evaluate` with the real predictors (stage 2) and the `t2_adv` CEM search. The T1-oracle CEM
   smoke run already found confident-wrong margins up to 0.04 on pairs with 16 members × 2 iterations.

Suggested paper wording: "Under an independent stress plant, certificates for assemblies of two or more modules move
by > 0.05 on 1 % of decisions. Certificates for single modules move by > 0.05 on 16 % and flip on 4.4 %. A clamp probe
separates the affected episodes (AUC 0.78) where the standard v2 routine cannot (0.47)."

## E2 / A4 targets: merges of never-co-assembled modules (written 2026-11-25 before the run; exploratory)
Spec: docs/board/research.md §E2. Script: `scripts/stress_A4_merge.py`. Output: `results/stress/A4_merge.csv` and
`results/stress/A4_merge_rows.csv`.
- **Design.**
  - Pairs: 500 pairs (G1, G2) of 1–3-module polyominoes with random module yaws, at most 1 fault each (weak g ~
    U(0.3, 0.95) / coast / brake). Latents are HET_FULL, and the θ draw from the v2 DR is shared by the pair (same
    floor).
  - Payload arms: P0 none; P5 0–5 kg on a random G1 deck point, carried by every merge.
  - Merges: up to 6 face-touching placements of the whole G2 against the whole G1 (`t0.candidates.merge_placements`),
    × 10 legs × τ ∈ {0.6, 0.7, 0.8}. Label ρ_dec, with each plant's own nominal reference for the merged geometry.
- **Truth and prediction.**
  - Truth in T1: the merged body's latents, with the per-wheel and per-module latents carried and the
    statically indeterminate load split redrawn for the merged geometry.
  - Truth in T2 `all`.
  - Prediction (b): T1 composition from the true separate-assembly latents (merged geometry, least-norm load split,
    known payload).
  - (a) T0 composition and (c) the enum_known compositional certificate are **not** in this run. (c) needs the
    baselines' GPU episode wrapper and the v2 probes of each G_j. The rows keep everything needed to add them.
- **δ_comp.** The decision-flip share of (b) against T1 truth and of (b) against T2 truth, with an episode-pair
  cluster bootstrap 95 % CI, plus |Δρ_dec| q50 / q95 / q99, per payload arm and merge size.
- **Success:** the upper 95 % bound of the flip share is < 1 % in T1 and < 2 % in T2. **Kill:** T1 δ_comp > 2 % (point
  estimate); then merges are not plug-in and get calibrated separately.
- (c)'s per-pair simultaneous miscoverage ≤ α = 0.1 and CF ≥ 85 % at U* = 1 % are pending (they need (c)).
- **Motor-law arm:** this waits for theory's wrong-law function (E1). It is not written here, to avoid a second
  definition.
- **Addendum (written before the full 500-pair runs started; the paper role asked for `miscov_pair` and
  `dec_merge.csv`).**
  - (c) is not available yet, so the merge certificate here is the oracle-latent composition (b) with a
    split-conformal margin, `comp_conformal`:
    - even pair ids calibrate on T1: q = the conformal (1 − α) quantile of the per-pair worst shortfall
      max(comp − truth_T1)₊, with α = 0.1;
    - odd pair ids are the test pairs, with lower = comp − q.
  - `comp_raw` is the same certificate with q = 0.
  - `miscov_pair` = the share of test pairs with any (merge, leg) truth below its lower bound, reported for T1 and T2.
  - Target: miscov_pair(comp_conformal) ≤ 0.1 in T1, by construction up to finite-sample error; T2 is reported
    against the same 0.1.
  - Rows go to `results/stress/paper/dec_merge.csv` in the paper §1a columns: method comp_raw / comp_conformal,
    posterior oracle, kind merge, split merge_<arm>, eid = pair id, cand = merge index.

## A5m: misspecified motor law (exploratory; `scripts/stress_A5m.py`; `results/stress/A5m_rows.csv`, `A5m_flips.csv`, `A5m_clamp.csv`)
- **Motor mode.** `build.Effects(motor=True)` is OFF by default and is not part of `all`. It is listed as the
  ablations `motor` and `all+motor`, and tested: numpy = numba, and it changes the plant. Per wheel:
  - a current limit c_I ∈ [0.5, 0.9] τ_s below stall;
  - a speed knee (the theory ENVELOPE form) at 0.55–0.95 of the no-load speed, and a torque knee at 0.1–0.4 τ_s;
  - thermal derating from prior heat E0 ~ U(0, 1) (τ_th 30 s);
  - per-module voltage sag 5–20 % at full load.

  It is applied as the true torque at the previous step's wheel speed, with crane's local back-EMF slope kept around
  it. At nominal θ the healthy reference drops on diag and yaw legs (square yaw 0.892 → 0.821).
- **Design.** 200 episodes per shape (single, pair, L3, square, 2x3), seeds 906000, HET_FULL, 0–2 faults, ρ_dec with
  each plant's own reference (T2 references average 8 nuisance draws), τ ∈ {0.7, 0.8}. One row per (episode, leg)
  with features (faults, min g, friction / drag factors, motor draws, margins, v2 / clamp misfit).
- **Flips, T1 → T2** (all decisions; **healthy candidates flip 0 % in every cell; every flip is a faulted
  candidate**):

  | condition | single | pair | L3 | square | 2x3 | children (pair–2x3) |
  |---|---|---|---|---|---|---|
  | all (A5 plant) | 4.28 % (d05 15.7 %) | 2.13 % | 0.13 % | 0 | 0 | 0.56 % (d05 1.0 %) |
  | motor only | 2.35 % (d05 10.9 %) | 1.27 % | 0.45 % | 0.10 % | 0.15 % | 0.49 % (d05 3.2 %) |
  | all+motor | 4.65 % (d05 18.3 %) | 2.57 % | 0.32 % | 0.08 % | 0.05 % | 0.76 % (d05 4.4 %) |

- **By leg** (all+motor):
  - single: axis 5.8 %, diag 3.0 %, yaw 5.8 % flips; ρ_dec shift q01 −0.60 on yaw;
  - pair: axis 2.9 %, diag 2.9 %, yaw 1.1 %;
  - L3: yaw 1.1 %; square: yaw 0.37 %; 2x3: yaw 0.25 %; axis 0 for ≥ 3 modules.
- **The motor law produces the shift theory predicted on ≥ 3-module yaw.** Under `motor` alone the share with
  |d| > 0.05 is L3 yaw 13.5 %, square yaw 21.2 %, 2x3 yaw 4.5 %. It rarely flips decisions: those decisions sit far
  from τ unless faulted.
- **Does the clamp misfit still flag the flips?** AUC of the episode misfit, flipped vs faulted non-flipped:

  | condition | singles: v2 IMU | singles: clamp IMU | pairs: clamp IMU | ≥ 3 modules: clamp IMU | pooled: v2 | pooled: clamp |
  |---|---|---|---|---|---|---|
  | all | 0.63 | 0.68 | 0.77 | | 0.58 | 0.85 |
  | motor | | | 0.69 | **0.27** (v2 0.11) | | 0.73 |
  | all+motor | 0.64 | 0.68 | 0.76 | 0.33 | 0.55 | 0.82 |

  For singles clamp is only weakly better than v2. For pairs it separates the flips. For ≥ 3-module motor-law flips
  it points the wrong way: the flipped episodes have *smaller* probe misfit. The knee is reached only at the child's
  yaw rate, which neither v2 nor clamp visits. This supports theory's "child-rate yaw probe" for ≥ 3-module yaw. The
  clamp gate is not enough there.
- **Claim C-stress-1 update.**
  - Point 3 holds for pairs and for the pooled set.
  - For singles the clamp separation is weak (AUC 0.68).
  - Under a wrong motor law the ≥ 3-module yaw flips (few) are not flagged by clamp.

## E2 / A4 results: never-co-assembled merges (exploratory; `scripts/stress_A4_merge.py`; `results/stress/A4_merge.csv`, `A4_merge_rows.csv`, `paper/dec_merge.csv`)
500 pairs per arm, 2880 (P0) and 2884 (P5) merges of 2–6 modules × 10 legs × τ ∈ {0.6, 0.7, 0.8}. δ_comp
compares the oracle-latent composition (b) against the truth.

| arm | plant | flips (95 % CI) | \|Δρ_dec\| q95 / q99 | share > 0.05 | merges of ≤ 3 modules | merges of 4–6 modules |
|---|---|---|---|---|---|---|
| P0 | T1 | 0.11 % (0.03–0.24) | 0.007 / 0.015 | 0.18 % | 0.31 % | 0.007 % |
| P0 | T2 | 0.13 % (0.05–0.25) | 0.017 / 0.030 | 0.40 % | 0.39 % | 0.009 % |
| P5 | T1 | 0.08 % (0.03–0.17) | 0.007 / 0.015 | 0.16 % | 0.23 % | 0.02 % |
| P5 | T2 | 0.12 % (0.05–0.25) | 0.017 / 0.029 | 0.34 % | 0.40 % | 0.005 % |

- **Verdict against the targets.** The upper bound is < 1 % in T1 and < 2 % in T2 in both arms: **SUCCESS**, and the
  kill (T1 > 2 %) is not met. Merges are plug-in at the decision level. Payload does not change this.
- **The merge certificate** (`comp_conformal`, α = 0.1, T1-calibrated margin q ≈ 0.011):
  - `miscov_pair` in T1 is 0.18 / 0.03 for the two calibration directions (cross-fit 0.11) in P0, and 0.14 / 0.08
    (cross-fit 0.11) in P5. That is at the nominal α up to split variance.
  - In T2 it is cross-fit 0.43 (P0) and 0.47 (P5): the T1-calibrated margin is thinner than the T2 plant gap
    (q95 0.017).
  - So simultaneous per-pair coverage does **not** transfer to T2, although decision flips stay at 0.13 %. The
    violations are small and sit far from τ.
  - `comp_raw` (q = 0) miscovers 100 % of T1 pairs.
- **Not done.** (a) the T0 composition, (c) enum_known from each G_j's v2 probes, and CF at U* = 1 % (needs (c)).
  The motor-law arm waits for theory's wrong-law function.

## T2 motor-law campaign plan (`scripts/campaigns/t2_motor.json`, not submitted)
- `t2-motor`:
  - 4000 plated test episodes;
  - `--conds all+motor,motor`, with the parent v2 obs under all+motor and the per-candidate features;
  - 8 shards × 10 units × 300 s (pilot 5.5 s/episode at 4 threads), about 7.3 box-h, $1.3.
- `t2-motor-misfit`:
  - 1000 episodes, `--conds all+motor --misfit --no-obs`, giving per-candidate v2 and clamp misfit against T1 with
    the latents known;
  - 14 shards × 12 units × 290 s (pilot 45 s/episode), about 14.7 box-h, $2.65.
  - It validates a clamp-gated certificate for singles and the child-rate yaw gap.
- Lane cpu, riding torch, with crane_ml staged; `emit` validates.
- `t2_adv` is cancelled (it measured the plant gap against an oracle certifier, not a method failure).
  `scripts/campaigns/t2_adv.json` is left unsubmitted.

## A5y targets: child-rate yaw probe (written 2026-11-25 before the run; exploratory)
- **Probe `yawc`** (`cfcap/t2/misfit.py:yaw_rate_table`). It is a yaw command at the child's own decision-leg yaw
  rate: the IK wheel rates of the child, with W_NOM scaled so the farthest wheel reaches WHEEL_SPEED_CAP, as in
  `t1.capability.leg_twists`. The schedule is a lead U(0, 0.5) s, a 0.3 s ramp to +w, a 1.5 s hold, a 0.6 s ramp to
  −w, a 1.5 s hold, a 0.3 s ramp to 0 and a 1 s tail. Like v2 and clamp in A5m, it is run on the child body, with
  latents known.
- **Design.** L3, square and 2x3, 200 episodes each, seeds 908000, HET_FULL, 0–2 faults, T2 `all+motor` against T1,
  ρ_dec with each plant's own reference, τ ∈ {0.7, 0.8}.
- **Statistics.** The misfit is the A5m statistic D_imu per probe. "clamp+yaw" means D_imu_clamp + D_imu_yawc.
- **Targets.**
  - (a) Positives are episodes with a yaw-leg flip or a yaw-leg |Δρ_dec| > 0.05; negatives are the other episodes
    of the same shapes.
  - (b) Pooled: positives are episodes with any flip or any |Δρ_dec| > 0.05 on any leg.
  - AUC is reported for v2, clamp, yawc and clamp+yaw. Flips-only versions use faulted non-flipped negatives.
- **Bar:** child-rate yaw AUC ≥ 0.75 on (a), the 3+-module yaw shifts. Below that, "the theory names the probe" is
  not supported for yaw.
- Caveat stated in advance: run on the child itself, `yawc` probes the decision operating point. On hardware it
  stands for a parent probe at the child's wheel rates.

## A5y results: child-rate yaw probe (exploratory; `scripts/stress_A5y.py`; `results/stress/A5y_auc.csv`, `A5y_rows.csv`)
L3, square and 2x3, 200 episodes each (600), T2 `all+motor` against T1, labels with each plant's own reference.
The bar was written before the run.

| target | positives / negatives | v2 D | clamp D | yawc D (pre-stated) | clamp+yawc D | yawc tracking (post-hoc) |
|---|---|---|---|---|---|---|
| (a) ≥ 3-module yaw flips or shifts > 0.05 | 90 / 510 | 0.55 | 0.57 | **0.56** | 0.56 | 0.57 |
| (a) yaw flips only, against faulted negatives | 10 / 403 | 0.28 | 0.44 | 0.36 | 0.36 | 0.83 |
| (b) pooled, any flip or shift | 110 / 490 | 0.52 | 0.53 | 0.53 | 0.53 | 0.56 |
| (b) flips only | 12 / 401 | 0.24 | 0.41 | 0.31 | 0.31 | 0.81 |

- **Verdict against the bar.** yawc AUC on (a) is **0.56 < 0.75: NOT MET**, even though the probe sits at the
  decision operating point on the child body, which is an upper bound.
- **Why the total misfit statistic fails.** D is dominated by the T1-vs-T2 trace differences present in every
  episode.
- **Post-hoc tracking statistic** (exploratory, not pre-stated): |the probe's measured yaw ratio in T2 − T1's
  prediction|. It detects the 10 yaw *flips* (AUC 0.83) but not the shifts (0.57).
- **Most of the (a) positives are healthy decisions whose own-reference label moved.** That is the N3 confound:
  T2's reference drops under the motor law, so the shifts are partly a relabelling. They are re-scored under N3.
- `yawc` is folded into `t2_eval --misfit` (columns D_imu_yawc, D_duty_yawc) and into the `t2-motor-misfit` group,
  now 20 shards × 10 units of 5 episodes.
- `t2_motor.json` also gains `t2-knee`: theory's knee law at severities 1–3 with the other effects, 4000 episodes,
  12 × 10 units. Emit validates.

## N3 targets: fixed-reference relabelling (written 2026-11-25 before the re-scoring; exploratory)
- **Label.** ρ_dec^fixed = ρ_abs^T2 / ρ_ref^T1(shape, leg), the robot's only table. It is compared side by side with
  the own-reference label (the T2 reference, 8 nuisance draws).
- **Inputs.** A5m, A5 P0, A5y and E2 are re-scored offline. ρ_abs is recovered from the stored own-reference ρ_dec ×
  `t2.capability.rho_ref`, and stored directly for A5 and E2.
- **Dominance class.**
  - dominated: surge and sway legs (v2 visits the same speeds);
  - undominated: diagonals of every child, and yaw of ≥ 3-module children;
  - other: yaw of singles and pairs.
- **Bar (H12), under all+motor with the fixed reference:**
  - flips on undominated decisions > 1 % (upper 95 % CI, episode bootstrap);
  - flips on dominated surge/sway ≤ 0.5 %.
  - If both hold, regime 2 exists in T2 and the gate experiment moves to T2.
  - If undominated flips stay ≤ 1 %, H12 is FALSIFIED: a robustness result.

## N3 results: fixed-reference relabelling (exploratory; `scripts/stress_N3.py`; `results/stress/N3_flips.csv`, `N3_pooled.csv`, `N3_merge.csv`, `N3_verdict.json`)
Flips T1 → T2 by dominance class, with an episode-bootstrap 95 % CI:

| set | condition | label | undominated | dominated (surge/sway) | other (yaw of single/pair) |
|---|---|---|---|---|---|
| A5m | all+motor | own | 1.09 % [0.80, 1.42] | 1.74 % [1.26, 2.26] | 3.44 % |
| A5m | all+motor | **fixed** | **2.76 % [2.21, 3.35]** | 1.82 % [1.30, 2.38] | 4.19 % |
| A5m | motor | fixed | 2.29 % [1.78, 2.78] | 0.86 % [0.56, 1.19] | 3.56 % |
| A5m | all | fixed | 0.71 % | 1.60 % | 3.31 % |
| A5y (≥ 3 modules) | all+motor | own | 0.36 % | 0.00 % | |
| A5y (≥ 3 modules) | all+motor | **fixed** | **2.40 % [1.75, 3.15]** | **0.00 %** | |

- **H12 against the pre-written pooled bar.** With the fixed reference under all+motor, undominated flips are
  2.76 % (CI > 1 %): met. Dominated flips are 1.82 % (> 0.5 %): **not met**. So H12 is neither supported nor
  falsified as written.
  - The dominated flips are singles and pairs only (single axis 6.1 %, pair axis 3.1 %). They are already there
    without the motor law (`all` dominated 1.60 %) and come from the per-wheel friction scatter and the kernel change.
  - With ≥ 3-module children the dominated flips are **0 %** in every set.
- **Within ≥ 3-module children the H12 pattern holds exactly** (A5m + A5y, all+motor, fixed reference):
  - axis flips 0 %;
  - yaw flips 3.0–6.75 % (L3 3.0–4.0 %, square 5.6–6.75 %, 2x3 3.75 %), diagonals 0.75–1.75 %;
  - 48–94 % of their undominated decisions shift by more than 0.05, against ≤ 0.9 % on axis legs.
- **The confound was real.** The own-plant reference absorbed most of the motor-law effect: square-yaw flips go from
  0.38 % (own) to 6.5 % (fixed).
- Regime 2 (silent extrapolation through the motor law) **exists in T2 for ≥ 3-module yaw and diagonals**, so the gate
  experiment can move to T2 on that stratum. For singles and pairs the shift is not about extrapolation: their
  dominated legs flip too.
- **E2 merges with the fixed reference:** flips 0.16 % (P0) / 0.17 % (P5), share > 0.05 0.69 % / 0.58 %, mean offset
  −0.021. Still plug-in.
- **For A5y:** the misfit AUCs above were computed against own-reference positives. Re-scoring them on
  fixed-reference positives is part of N5.

## N5 targets: the feasible parent probe `wheelrate` (written 2026-11-25 before the run; exploratory)
- **Probe.** `wheelrate` uses per-wheel speed references with a body-twist command of 0. In every module, the
  antagonistic diagonal pairs (fl, rr) and then (fr, rl) are commanded to opposite speeds. Each wheel's magnitude is
  its own |ω| on the child's yaw leg: the IK of `t1.capability.leg_twists` yaw, the Prop. 2b closed form. The
  schedule is a 0.5 s settle, then 4 legs (pair A ±, pair B ±), each a 0.4 s ramp, 1.0 s hold and 0.3 s ramp down,
  then a 0.5 s tail. The net twist stays small by construction.
- **Episodes.** The 600 A5y episodes (L3, square, 2x3; T2 all+motor). Here the probed body is the child: it stands
  for a parent whose wheels reach the child's yaw wheel rates. Unlike `yawc`, the body does not yaw.
- **Statistic and target.** The statistic is D_imu + D_duty, as specified (the parts are also reported). AUC is
  taken on target (a): yaw flips or yaw shifts > 0.05, with **fixed-reference** labels (N3). AUC on own-reference
  labels is reported alongside, and v2, clamp and yawc are reported the same way.
- **Bar.**
  - wheelrate AUC ≥ 0.75 → the claim "the theory names the probe" stands with a feasible probe;
  - yawc ≥ 0.75 but wheelrate < 0.75 → "split-to-see" only;
  - both < 0.75 → drop the yaw-probe claim.

## N7 targets: size-aware two-simulator margin for single and pair children (written 2026-11-25 before the run; exploratory)
- **Data.** The A5m single and pair episodes (seed 906000, 200 each): the latents are regenerated
  deterministically, and T1 is checked against the stored labels. Truth is T2 `all+motor` with the **fixed**
  reference (N3).
- **Certifiers** (decision (episode, leg, τ), τ ∈ {0.7, 0.8}, go iff lower ≥ τ):
  - `t1_only`: lower = ρ_dec^T1, the oracle with T1 latents known (exact in T1).
  - `size_aware`: lower = ρ_dec^T1 − q(cand_size). q(n) is the 99th percentile over all decisions of that size of
    the shortfall (ρ_dec^T1 − ρ_dec^T1,pert)₊. ρ_dec^T1,pert is T1 run on the same latents perturbed with the E2
    recipe: motor spread ×2, battery and friction module deviations ×2, load split ×2, mounting yaw +N(0, 0.02).
    T2 truth is never used.
- **Metrics.** unsafe = P(go and T2 truth < τ) over all decisions; CF = P(go | T2 truth ≥ τ); U* = 1 %.
- **Bar:** `size_aware` single-child unsafe ≤ 2·U* = 2 % with a CF loss ≤ 5 pts against `t1_only`. Pairs are
  reported against the same bar.
