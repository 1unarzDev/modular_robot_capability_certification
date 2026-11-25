# stress (T2 independent plant, A5 misspecification)

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
