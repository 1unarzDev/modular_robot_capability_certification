# nuisance (specialist board)

**Status:** pre-registration written 2026-11-25, before any pilot run. Results below this section are added after.

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
- Decision unit (episode, candidate, leg, τ), τ ∈ {0.7, 0.8}. **Ambiguous iff q05 < τ ≤ q95** of the 64 posterior ρ_rel
  draws. CIs: bootstrap over episodes (2,000 resamples, per shape and pooled).
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
