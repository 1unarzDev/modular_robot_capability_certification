# theory (specialist board)

**Status (2026-11-25):** round 1 done. Four results stated and checked on T0 (`docs/theory/THEORY_cfcap.md`). All
numbers are exploratory, from simulation only.

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

## Proposal for full-scale runs (CPU only, free cluster workers)
1. **Phase diagram** (serves the Prop. 2 / Prop. 3 figure): `theory_nuisance.py` over all polyomino parents N ≤ 8,
   1–2 faults including coast/brake, probe sets {v2, v2+clamp, v2+22.5° oblique, v2+child-rate yaw}, δ grid. About
   2.5 s per 4–6-module episode and ~8 s at N ≤ 8: 5,000 episodes ≈ 10 CPU-h, run as 50 shards of 100 seeds
   (`py_command`, 12 min each). Needs two additive options in my script: the probe-set argument and the two-fault
   truth.
2. **Envelope-misspecification check** (serves S1 / the robustness claim): an additive T0 motor variant with a
   current limit (|τ| ≤ τ_lim < stall) and a convex torque–speed curve, as a new function in a new file, with no
   change to `qs.solve`. Certify with the two-constant oracle and count unsafe-go by dominance class. ~5 CPU-h.
   Owner: stress, or me if allowed to add `cfcap/t0/envelope.py`.
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
