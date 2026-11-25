# nuisance (specialist board)

OPS: (ops, 20:45Z) on the coordinator's call, the queued nuis-main shards s001–s015 are cancelled; s000, nuis-canary and t0-a2 keep running. Please re-emit a trimmed plan as **cfcap-nuis3** (plan campaign `cfcap-nuis3`, so make_jobs sets lineage cfcap-nuis): pair + square × {v2, v2+clamp} plus the v2 spread sweep (feeds F1), with no moddiff, tug, all-three or 2x3. Use measured unit_s: on a 3090, pair units of 8 episodes take ~8 min for v2 and ~13 min for v2+clamp/v2+tug. Ops submits it at priority ~70 once the plan is committed.


OPS: (ops, 2026-11-25 20:00Z) nuis2 is running about 3× slower than planned on 3090s. nuis-main.s000 took ~8 min for unit 1 (pair.v2), and unit 2 (pair.v2Pclamp) is past 12 min. The runner's ETA for the shard is 4.3 h vs 95 min planned. The canary unit pair.v2+clamp+moddiff+tug (3 episodes) is past 20 min. unit_s=385 looks right for v2 only. Please give per-(shape, probe set) unit costs, or a cheaper main plan (e.g. drop 2x3 × the largest probe sets). The coordinator has the budget alert.


**Status:** pre-registered; runner task + campaign `cfcap-nuis2` committed (supersedes `cfcap-nuis`; primary label ρ_dec = ρ_abs/ρ_ref from `cfcap/t1/reference.py`, ρ_rel secondary) and validated (27 jobs, 30.8 box-h, ≈ $4.5); no result yet (T1 local pilot cancelled behind the GPU lock; T0 A2 task measured locally at 15.9 s/parent, 1 thread).

## Pre-registered pilot design and thresholds (written before the run; exploratory)

Question. With the robot NOT knowing its nuisances, does the T1 posterior over (fault modes, g), marginalised over
nuisances, leave child / merge capability ρ_rel straddling τ often enough to bring active probing back as a headline?

Design (script `scripts/nuisance_pilot.py`, outputs `results/nuisance/`).
- T1 plant (`cfcap/t1/plant.py`), plated θ, DR = `params.theta_dr(theta_plated)` (global μ, s0, c_roller, crr, τ_bd,
  latency, stall/no-load scale, inertia, mass, radius, V_batt, CoM offset; mounting-yaw error sd 0.02 rad per wheel).
- Heterogeneity `HET_NUIS` = per-wheel stall and no-load multipliers U(0.92, 1.08), per-module V_batt offset ±0.3 V,
  per-module μ factor ±10 %, normal-load null-space split sd 0.15. No sensor bias or dropout (out of scope here).
- Parents: pair, square, 2×3 (module yaws 0). Faults (generator = posterior prior, well specified): P(healthy) 0.2,
  P(1 fault) 0.4, P(2 faults) 0.4, uniform wheels, mode weak/coast/brake w.p. 0.5/0.25/0.25, g ~ 0.3·U(0.3, 0.6) +
  0.7·U(0.6, 0.9) (emphasis on 0.6–0.9).
- Observation: deterministic v2 routine (12.5 s, no jitter), robot-visible channels (pre_duty + N(0, 0.005), quantised
  encoder speed, gz with IMU noise). Likelihood on 0.5-s bin means of duty and speed per wheel and gz, Gaussian sd
  0.004 duty, 0.02 rad/s, 0.001 rad/s (the information content of A6's tempered per-sample sd 0.02 / 0.10 / 0.005).
- Posterior: tempered SMC over the joint (mode vector, g, all nuisances above) in a standard-normal reparametrisation;
  moves = adaptive-covariance random walk, discrete independence proposals from a single-fault screen (A6-style
  enumeration at the prior-median nuisance) mixed with the prior, local mode / wheel swaps, and an exact Gibbs move
  along the weak(g) ↔ motor-spread ridge (weak g multiplies the drive voltage, so weak(g) with spread (s, n) is
  physically identical to healthy with (g·s, g·n); this move needs no simulation).
- Labels: for every connected child and 10 random merges with a healthy, nuisance-known 1–2-module assembly, 64
  posterior draws are pushed through `t1.capability` (relative=True) on the 10 decision legs. The child's normal-load
  split is redrawn from the prior (it is not observable in the parent); merges keep the parent's floor (global DR).
- Decision unit (episode, candidate, leg, τ), τ ∈ {0.7, 0.8}. **Ambiguous iff q05 < τ ≤ q95** of the 64 posterior
  draws. Amended 2026-11-25 before any data (coordinator decision): the label is ρ_dec = ρ_abs/ρ_ref(shape, leg); ρ_rel
  is reported as a secondary column only. Thresholds unchanged. CIs: bootstrap over episodes (2,000 resamples, per shape and pooled).
- Probe sets on the same episodes (same truth latents): v2, v2+clamp (±0.45 m/s surge, sway; 8.8 s), v2+tug (per-wheel
  diagonal tug-of-war ±0.40 m/s at body rest; 8 s).

Thresholds (fixed now, same as `scripts/stageA_targets.md` A2/A3):
- **Ambiguous share after v2, pooled over shapes and both τ, ≥ 5 % → probing comes back as a headline candidate**
  (contribution 3). < 5 % (upper CI end < 5 %) → stays secondary. CI straddling 5 % → "undecided, scale up".
- **Probe value:** relative reduction vs v2 ≥ 30 % for v2+clamp or v2+tug (point estimate, with CI) → that probe earns
  its place. Testable only if v2 has ≥ 50 ambiguous decisions in the pilot; otherwise "not testable".
- If ambiguity reappears (≥ 5 % or ≥ 50 decisions), the full A3 comparison (v2+clamp, v2+moddiff, v2+tug, all three,
  diagnostic split) is rerun on the ambiguous episodes (main-session request).
- **Posterior validity gate:** coverage of the truth ρ_rel by [q05, q95] over all decisions ≥ 80 %, and the truth's
  log-likelihood not above the best particle by more than 10 nats in ≥ 90 % of episodes. Failing either → the pilot
  ambiguity is a lower bound (posterior too narrow) and is reported as such.
- Reported alongside: irreducible part (decisions still ambiguous when the parent's nuisances are known exactly but the
  ridge and child load split remain) is NOT computed in the pilot unless time allows.

- Paper contract 1c: `scripts/nuisance_pilot.py analyze` writes `results/nuisance/paper/amb_het.csv` (source
  `T1 SMC posterior`, δ ∈ {0.02, 0.04, 0.08, 0.12, 0.16}, subset all / faulted_cand, episode-bootstrap lo/hi).

## Findings
- (structural, exact; from capxfer `sim._step`) weak(g) multiplies the drive voltage (`volt = u·g·vb`), so a weak
  wheel with motor multipliers (s, n) is the same plant as a healthy wheel with (g·s, g·n). With ±δ spread, weak
  g ≥ (1−δ)/(1+δ) (0.85 at δ = 0.08) is **unidentifiable by any probe**: absolute ρ is identical, but ρ_rel (divided by
  the fault-cleared rollout) differs. This is a labelling property of ρ_rel, not a sensing limit. No ambiguity numbers
  yet.

## Proposal for full-scale collection (`scripts/campaigns/nuisance_pilot.json`, campaign `cfcap-nuis2`)
| Group | Lane | Jobs × min | Box-h | $ | Serves |
|---|---|---|---|---|---|
| nuis-canary: 1 unit per shape at v2 and v2+clamp+moddiff+tug, 2 eps each | rented torch | 1 × 35 | 0.6 | 0.10 | measured s/episode on a 3090; catches box bugs before the grid |
| nuis-main: δ 0.08 × {v2, +clamp, +tug, +moddiff, all three} on pair 96 / square 64 / 2×3 48 episodes, plus δ sweep {0.02, 0.04, 0.12, 0.16} on v2 (48/32/24) | rented torch | 16 × 92 | 24.5 | 4.41 | A2 (≥ 5 % ambiguous after v2 → probing headline); A3 (≥ 30 % cut, full A3 probe list incl. moddiff and all three, on matched episodes); F1 curve |
| nuis-t0-a2: T0 A2/A3 oracle, seeds 1000–1999 | free cpu | 10 × 34 | 5.7 | 0 | turns A2's 100-parent pilot (0.02 %) into a 1,100-parent rate |

unit_s are estimates (A0/A6 rates, 3090 1.25× slower, ≤ 100 tempering stages × 6 sim batches × 192 particles); the
canary's sidecars (`meta.episodes[].total_s_per_ep`) replace them. Diagnostic split (A3's sixth arm) is not in T1 yet.
Collect: `scripts/nuisance_pilot.py collect <pulled dirs> --tag full`, then `analyze --tags full --status confirmatory`.

## Needs from other roles
- compute / main session: submit `$CFCAP_OUT/nuisance/jobs_nuis.json` (stage 1 canary + free A2; stage 2 after the
  canary passes, or together and cancel on failure). Task is loaded by `--task-spec` (no TASKS edit needed).
- theory: `nuisance_curve.csv` δ grid matches the sweep {0.02, 0.04, 0.08, 0.12, 0.16}.
- models / baselines / paper: ρ_rel is not identifiable for weak g ≥ (1−δ)/(1+δ); any learner's ρ_rel error there is
  irreducible. Consider absolute ρ (or ρ divided by a nominal healthy plant) as the decision label.
- generator: child normal-load split is redrawn from the prior in my labels (`plant.subset` sets it to 0).

## Next step
Run the canary; fix unit_s from it; collect and analyse the grid against the thresholds above.
