# cfcap coordinator status (updated 2026-11-25 ~19:55Z)

All results so far are exploratory and come from simulation.

## Rented campaigns (owner override: ≤16 interruptible boxes, ≤$35 total until 2026-11-26 03:00Z)
| Campaign | What | Box-h (est.) | $ (est.) |
|---|---|---|---|
| cfcap-gen-ref2 | Reference splits: cal/test 20k + H3/H4/H6/H7 5k per plate state, ρ_dec | 4.3 | 0.8 |
| cfcap-nuis2 | Nuisance-marginalized ambiguity × 5 probe sets + spread sweep; T0 A2 seeds on the free lane | 22.5 + 8.3 free | 4.4 |
| cfcap-baselines-pilot2 | Nominal, oracle, RLS, capxfer-Thm5, known-nuisance enumeration, nuisance-SMC; 294 episodes | 7.2 | 1.3 |
| cfcap-theory-cpu | T0 phase diagram (N 4–8, 5 probe sets, δ) + misspecified-envelope check | 17 CPU-h | 1.7 |
| cfcap-t2-eps | 4000 plated T2 test episodes (independent crane-kernel plant), CPU on torch spare cores | 4.0 | 0.7 |
| cfcap-decide | Probe policies × certifiers on T1 posterior draws | 30.5 | 5.5 |
| cfcap-models-b1 | Set transformer / direct / DeepSets × 5 diversity levels × 3 seeds | 43.2 | 7.8 |
Committed estimate ≈ $22 of ≈ $32 usable.

## Findings to date
- A2/A3 (T0 oracle, motor law known): 0.02 % of decisions are ambiguous after v2, and the oracle certifies 99.98 %
  vs capxfer's 58–75 %.
- Theory: that low ambiguity comes from trusting the two-constant motor law, because v2 never loads the torque side.
  A model-free speed-ceiling class leaves 7 % ambiguous (20.7 % at τ=0.8); clamp cuts it to 0.06 %. Merges extend
  capxfer Thm 5. The simultaneous-guarantee price is 1.6–2.4× the margin.
- Label: a weak wheel at g ≥ 0.85 is indistinguishable from motor spread, so the decision label is now
  ρ_dec = ρ_abs / nominal reference.
- A5 (T1 vs T2): 0.53 % of decisions flip pooled; singles flip 4.4 %. The clamp misfit separates flips
  (AUC 0.78) while v2 does not (0.47).
- Models pilot: amortized beats capxfer on mild faults (square 0.59 → 0.75 detection; |ĝ−g| 0.077 → 0.027). The
  compositional route degrades less than direct with size (2×3 coverage 0.90 vs 0.81). Diversity matters.
- Decision T0 pilot: the posterior + simultaneous certificate gives 0.54 % unsafe and 99.4 % useful commits. EVOI is
  the best probe per second, but no policy beats v2 on regret.

## Known gaps
- Outputs return when boxes drain; per-job pulls have been requested from the compute session.
- torch.compile fails on slim boxes (no C compiler); gcc has been requested. Code falls back to CUDA graphs, which
  are slower.
- Pending from stress: T2 plateless/ablation and adversarial-search plans.
- Pending from decision: model-free envelope predictor (needs theory), T2 re-scoring.
- Deferred: the paper refinement loop (`docs/PAPER_REFINEMENT_LOOP.md`), once all result CSVs exist.
