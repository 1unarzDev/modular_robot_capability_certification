# theory (specialist board)


OPS: (ops, 2026-11-25 19:55Z) cfcap-theory-cpu: the cpu-profile box image (python:3.12-slim, numpy only) has no torch or pandas, and `theory_phase` needs both. That gave 4 broken attempts, and both cpu boxes are quarantined. The canary passed on a torch box's spare cores, so the batch now runs there. Future CPU plans should use `"ride": "torch"` if they import torch or pandas.


**Status (2026-11-25):** round 1 done. Four results stated and checked on T0 (`docs/theory/THEORY_cfcap.md`). All
numbers are exploratory, from simulation only.

**Headline candidate: probing matters exactly where the motor law could be wrong (v2 does not load the torque side); T2/A5 tests this.**

## Summary
- **Why A2 is 0.02 % ambiguous (Lemma 1 + Prop. 2).** In T0/T1 a driven wheel is fully described by two inverse motor
  constants (α torque, β speed). A weak fault and motor spread are the same kind of object. v2 runs at almost zero
  load torque (median 0.5 % of duty), so it pins β, which carries the speed ceiling, and leaves α and μ open. The
  oracle extrapolates β to undominated operating points along the known motor law. **The low ambiguity comes from
  that extrapolation, not from coverage.** Under a nonparametric speed-ceiling class, the same v2 data leaves 7 %
  of decisions ambiguous (20.7 % at τ = 0.8); v2+clamp brings this to 0.06 % (`coverage_speed_only.csv`).
- **Closed form for coverage** (`coverage_dominance*.csv`). Surge and sway legs of every child need exactly the v2
  speeds. Diagonal legs need 1.40× the v2 speed on half the wheels: undominated by v2 in 99.9 % of child legs,
  dominated by clamp in 100 %. Yaw legs of children with ≥ 3 modules are undominated even after clamp (96–98 %).
  All 24 of A2's ambiguous decisions are diagonal legs of a lone faulted module (`coverage_attrib.csv`).
- **Nuisance band (Prop. 3).** A(δ) ≈ 0.0056·δ when the parent's β evidence is used, ≈ 0.012·δ when it is ignored
  (the A2-prior analogue); duty-noise floor ≈ 0.07 %. The p99 band half-width in ρ is ≈ 0.13·δ. Linear prediction
  equals the exact box image to within 10 % (`nuisance_curve.csv`). Realistic δ cannot reach the 5 % ambiguity level.
- **Slack (Prop. 4).** Simultaneity costs q_sim/q_marg = 2.4× for Gaussian independent residuals (equal to the
  Bonferroni z_{1−α/M} with M ≈ 1,170 entries/episode), 1.6× at within-episode correlation 0.9, and 4–12× for t₃.
  **Cliff rule:** on T0 margins the certified fraction holds only while s·q̂ < 1 − τ (0.2 at τ = 0.8). capxfer's
  58–75 % on lone modules matches σ_r ≈ 0.10 (marginal) or ≈ 0.05 (simultaneous) (`slack_decomposition.csv`).
- **Merges (Thm 5′).** capxfer T5 extends to merges of modules never assembled together, with one budget per
  assembly and an explicit composition defect δ_comp. Stage 1 is family-free: one coverage event certifies every
  child and merge at once. MC check: simultaneous miscoverage 1.7 % ≤ bound 19 %, and 0 violations given coverage
  (`merge_coverage.csv`).
- **Paper-grade:** (1) Lemma 1 + Prop. 2, the identification regimes plus the minimal covering probe; (2) Thm 5′ +
  Prop. 4, family-free compositional simultaneity against a measured simultaneity price. Textbook: the Prop. 4
  algebra, Le Cam two-point, and Prop. 3's linearisation.

## E1 pre-registration (copied from docs/board/research.md §4 E1 before any run, 2026-11-25)
Trust-gated certificate on T0 under a misspecified motor law. Script `scripts/theory_gate.py`; shared law
`cfcap/t0/envelope.py:motor_envelope` (stress imports it for the T2 motor-law arm). Output `results/theory/gate_cert.csv`.
- Design: 1,200 T0 parents of 2–6 modules, children including singles and pairs, 1–2 faults (weak/coast/brake),
  δ = 0.08. Truth law ∈ {two-constant, knee severity 1, 2, 3}, 300 parents each. Probes {v2, v2+clamp}.
- Certifiers, α = 0.1 simultaneous per episode, cross-fitted 2-fold:
  - `par`: the two-constant fit at the probes plus SimultaneousLower, calibrated on two-constant-truth episodes. This
    is the physics-trusting one.
  - `par_pooled`: the same, calibrated pooled over laws.
  - `pi`: the speed-ceiling / torque-floor identified-set low end, no calibration. This is the model-free one.
  - `gate`: `par` if the decision is dominated or the probe misfit ≤ q_gate, else `pi`. q_gate and the margin are
    calibrated end to end on the cal fold, pooled over laws, with q_gate chosen on one half and the margin on the
    other.
  - `raw`: the `par` point with q = 0. This is the uncalibrated one.
  - Trivial: `nominal` (healthy-body prediction plus a pooled margin) and `detector_gated` (nominal on fault-free
    children only).
- Strata: all; decision (faulted candidate or |ρ − τ| < 0.10); undominated (some driven wheel's needed speed or
  torque exceeds the parent's probed maximum). unsafe_rate = unsafe commits / incapable decisions of the stratum;
  ep_any_unsafe per episode.
- **Success** (decision stratum):
  - under the two-constant truth, gate CF ≥ par CF − 2 pts;
  - under the knee truth (the worst severity at which par's unsafe-go > 2 %), gate unsafe-go ≤ 1 % and gate CF ≥
    pi CF + 5 pts with v2 only.
- **Kill:** par's unsafe-go on undominated decisions ≤ 1 % (U*) at every knee severity. Then extrapolation is safe for
  these envelope classes; report that as a robustness result and drop F1.
- **Added before the full run (coordinator N2, 2026-11-25):** the 16-episode smoke verdict ("KILL True") is void
  (4 episodes per law, non-finite margins). `theory_gate.py analyze` now prints INSUFFICIENT unless every
  (law, severity, probe) cell has ≥ 100 episodes and every margin is finite. **If the knee law does not bite in T0**
  (the kill is met, or raw/par unsafe stays ≈ 0 on undominated decisions at every severity), the regime-2 evidence
  moves to T2 all+motor (stress N3), not T0, and E1 is reported as "extrapolation through the two-constant fit is
  safe for T0 knee envelopes".
- Also reported: the share of certified decisions that are undominated, by leg type and child size, under the
  two-constant truth.

## E1 results (exploratory; `scripts/theory_gate.py`, `results/theory/gate_cert.csv`, `gate_verdict.txt`, `gate_extrapolated_share.csv`, `gate_thresholds.csv`)
**Setup.** 1,200 parents of 2–6 modules with 1–2 faults, 300 per law (two-constant law, knee severities 1–3 from
`cfcap/t0/envelope.py`). Every connected child × 10 legs × τ ∈ {0.6, 0.7, 0.8}: 57–58k decisions per law and probe
set. α = 0.1 per episode, 2-fold cross-fit; folds are independent of the law. Knee fixed points converged on
almost all legs.

**Decision stratum, v2 / v2+clamp.** Each cell is CF % · per-decision unsafe % · ep_any_unsafe %.

| certifier | two-constant | knee 1 | knee 2 | knee 3 |
|---|---|---|---|---|
| raw (uncalibrated) | 99.2·6.3·11 / 99.3·6.5·12 | 99.2·6.6·12 / 99.2·6.7·13 | 99.3·8.0·23 / 99.3·8.5·24 | 99.5·**48**·**89** / 99.5·**49**·**89** |
| par (physics-trusting, cal on two-constant truth) | 96.6·2.3·4 / 96.1·1.4·2 | 96.6·3.0·6 / 96.0·1.6·2 | 96.5·3.7·12 / 95.9·2.1·7 | 97.0·**29**·**74** / 96.4·**22**·**67** |
| par_pooled (cal over all laws) | 78.9·0.6·1 / 71.8·0.5·1 | 77.4·0.8·1 / 70.5·0.6·1 | 77.6·0.2·1 / 70.4·0.2·1 | 73.6·1.1·8 / 67.4·1.1·8 |
| pi (model-free corners, no cal) | 75.1·0.2·1 / 92.7·0.3·1 | 74.1·0.1·0 / 92.2·0.2·1 | 73.0·0.1·0 / 90.8·2.6·9 | 70.9·2.3·18 / 89.6·13·56 |
| pi_cal | 72.4·0.1·0 / 90.7·0·0 | 71.1·0.1·0 / 90.1·0·0 | 69.9·0·0 / 86.9·0.1·0 | 66.9·1.1·10 / 79.0·0.4·4 |
| **gate** | 81.8·0.5·1 / 94.1·1.0·3 | 79.5·0.7·1 / 92.9·1.7·3 | 79.4·0.2·1 / 87.9·0.6·2 | 72.1·1.7·12 / 77.4·0.4·4 |

(nominal 0 % CF; detector_gated 0–1.4 % CF.)

**Verdict against the pre-registered bars** (`gate_verdict.txt`):
- **Kill not met:** the knee law does bite. Its main effect is at severity 3. There the physics-trusting `par` has 29 %
  unsafe on undominated decisions with v2 (22 % with v2+clamp), and 74 % / 67 % of episodes contain an unsafe commit.
  - At severities 1–2 `par` is close to its two-constant level, but ep_any rises 4 → 12 % with v2 (α = 10 %).
  - Caveat: `par`'s per-decision unsafe rate is already 2.3 % under the *correct* law. Its guarantee is per episode
    (ep_any 4 % ≤ α), so the 1 % per-decision kill bar is partly confounded.
- **SUCCESS-1 FAILS.** Under the two-constant truth with v2, gate CF is 81.8 % against par's 96.6 % (−14.8 pts). With
  v2+clamp it is 94.1 vs 96.1 (−2.0, at the bar), but the bar was set on v2.
- **SUCCESS-2 FAILS.** At severity 3 with v2, gate has 1.7 % unsafe (> 1 %), and its CF of 72.1 % is only +1.2 pts over
  pi's 70.9 %. With v2+clamp it has 0.4 % unsafe and 4 % ep_any, but its CF of 77.4 % sits below pi_cal's 79.0 %.
- **What the data do say:**
  1. **Trusting the law fails silently.** raw and par commit 48 % / 29 % unsafe at knee 3, and raw is at 6 % even under
     the correct law.
  2. **Any end-to-end calibration that has seen the wrong law stays valid** (par_pooled, pi_cal and gate all keep
     ep_any ≤ 12 %). The question is which one keeps CF.
  3. **The clamp probe is what buys CF back.** Under v2 the valid certifiers sit at 67–82 % CF. Under v2+clamp,
     gate / pi_cal reach 94 / 91 % at the correct law and 77 / 79 % at knee 3.
  4. The misfit gate as specified adds little over pi_cal once clamp is run. Clamp's dominance coverage does the
     work, not the misfit statistic. F1 survives only as "gate = dominance after clamp".
- **Extrapolated share of certified decisions** (par, two-constant truth, all τ): **63 % with v2** (speed 56 %,
  torque 23 %) and **21 % with v2+clamp** (speed 10 %, torque 19 %).
  - With v2, every diagonal decision and every yaw decision of a ≥ 2-module child is extrapolated.
  - For ≥ 3-module children, yaw is 84–99 % torque-extrapolated as well (`gate_extrapolated_share.csv`).
- Per the pre-registered fallback: T0's knee law bites only at severity 3, so the regime-2 headline should also be
  scored on T2 all+motor (stress N3), now that the shared `knee_factors` exists.
- Caveat: `pi` is a corner approximation of the identified set and is not a valid bound at knee 3 with clamp (13 %
  unsafe), which is why only `pi_cal` is reported as valid.

## Extrapolated share vs parent size N ≤ 16 (exploratory; `scripts/theory_dominance_N.py`, `results/theory/dominance_vs_N.csv`, `dominance_vs_N_by_size.csv`)
Closed-form speed coordinate only; the torque side is in E1 for N ≤ 6. Setup: 150 grown random parents per N, and
children sampled stratified by size (≤ 6 per size). The share is uniform over child sizes 1..N−1, with bootstrap CIs
over parents.
- **v2:** the share of (child, leg) decisions that are extrapolated rises from 40 % at N = 2 to 53 % at N = 4,
  57 % at N = 8 and 59 % at N = 16. Diagonals are always extrapolated. Yaw goes 0 → 67 → 86 → 93 %.
- **v2+clamp:** 0 % (N ≤ 3), 6 % (N = 4), 14 % (N = 8), 17 % (N = 16). All of it is yaw: child yaw rises 31 → 70 →
  85 %.
- **v2+clamp+2× yaw:** 0 → 5 → 8 % at N = 4 / 8 / 16 (yaw 10–41 %).
- **Reading:** larger parents yaw slowly (yaw is scaled by the parent's L_c), so their children's yaw legs are reached
  only by extrapolation. A fixed probe set leaves a growing extrapolated share; only a yaw probe at the child's rate
  bounds it.

## Predictions other roles can test
- **nuisance** (T1 SMC pilot):
  - (P1) With the posterior using v2's duty evidence, the ambiguous share is ≪ 5 %. Predicted at δ = 0.08: ~0.05 %
    on T0. T1 adds load-split redraws and per-module V_batt. V_batt scales α and β together, so β identifies it. The
    load split is unidentified and acts like μ. Expect T1 ≤ 10× T0, so still < 1 %.
  - (P2) The posterior ridge weak(g) ↔ spread does not widen the child ρ band on legs where every child wheel tracks:
    S^α = 0 on 78 % of (candidate, leg) pairs.
  - (P3) The ambiguous decisions concentrate on diagonal legs of lone or 2-module children, and on legs where a child
    wheel saturates.
  - (P4) v2+clamp reduces the ambiguity by ≥ 30 % only through weak wheels that clamp saturates (it pins α) and
    through diagonal dominance. v2+tug adds little.
  - Curve to compare against: `results/theory/nuisance_curve.csv` (variants `ident_sensor`, `none`).
- **baselines:**
  - (B1) For any learned certificate, log the normalised residual r per entry. Report its tail (kurtosis) and its
    within-episode correlation c, then q̂_marg, q̂_sim and CF under both quantiles. Predicted: CF_sim ≈ mean_τ
    P(s(q̂ − r) ≤ 1 − τ) on healthy-dominant candidates, and the simultaneity price q_sim/q_marg ∈ [1.6, 2.4] for
    near-Gaussian residuals.
  - (B2) The capxfer certificate's 58–75 % implies σ_r ≈ 0.05–0.10. The gap to the oracle's 99.98 % should split
    into roughly ≤ 1 pt estimator + marginal, ~15–20 pt simultaneity price, and ~2 pt finite n at n_cal = 50.
  - (B3) Per-episode unsafe-go of a *marginal* certificate is 3–14 %, against ≤ 0.75 % for a simultaneous one, even
    though the per-decision rate is ≤ 0.02 %.
- **stress / misspecification (A5, T2):** (S1) T2-vs-T1 decision flips should concentrate on decisions that are
  undominated in the (torque, speed) plane: diagonal legs, and yaw legs of ≥ 3-module children. On surge and sway
  legs the operating points are identical to v2, so misspecification of the envelope shape cannot flip them. Only
  load and μ effects can.
- **paper:** A2's low ambiguity should be framed as "identified by the motor-law extrapolation". The coverage
  closed form then says which decisions rest on that assumption.

## Proposal for full-scale runs (CPU only; packaged, validated, not submitted)
Plans: `scripts/campaigns/theory_sweeps.json` (lane `free`, $0) and `scripts/campaigns/theory_sweeps.cpu.json` (lane
`cpu`, rented, ~$1.7). Both contain the same two groups. Tasks are in `cfcap/theory_task.py`, run with
`--task-spec`; outputs are runner npz units.
1. **phase** (`PHASE`, serves the Prop. 2 / Prop. 3 figure): 3,000 episodes (4–8-module parents, 1–2 faults
   weak/coast/brake).
   - Probe sets {v2, v2+clamp, v2+22.5° oblique, v2+2× yaw, v2+clamp+2× yaw}; δ ∈ {0.04, 0.08, 0.16}.
   - 15 jobs × 51 min = 12.8 CPU-h.
   - Pilot at 1 thread: 4.0 s for an N=6 parent, 14.5 s for an N=8 parent. A unit of 10 episodes is set to 150 s.
   - My earlier ~10 CPU-h figure was low.
2. **envelope** (`ENVELOPE`, serves S1): a knee-motor truth (speed and torque knees drawn independently of the
   probes) against the two-constant oracle fitted at the parent's v2 or v2+clamp probes, with per-decision dominance
   ratios.
   - 3,000 episodes, 10 jobs × 26 min = 4.3 CPU-h.
   - Pilot: 4.3 s per episode at 1 thread.
   - Predicted: unsafe-go and |ρ̂ − ρ| are near 0 on dominated decisions and concentrate where the speed or torque
     ratio exceeds 1. Clamp shrinks the speed-undominated part.
3. **Slack on real models:** no compute. Baselines logs r, and I fit the decomposition.

## Needs from other roles
- **baselines:** residual tables (pred, truth, scale per entry, episode id) for capxfer and every new model.
- **nuisance:** their T1 ambiguous-decision table with leg and child size, to test P1–P4.
- **stress:** δ_comp for merges from A4 (T1 truth vs T0 composition). It is the only merge-specific term in Thm 5′.

## Next step if given another round
Run proposal 1 at small scale locally, then write the envelope-misspecification variant (2) and test S1. Tighten
Thm 5′ with the per-candidate band union for stage 2 when θ is not known.

Files: `docs/theory/THEORY_cfcap.md`; scripts `scripts/theory_{coverage,nuisance,slack,merge}.py`; results
`results/theory/*.csv`; bulk `$CFCAP_OUT/theory/`.
