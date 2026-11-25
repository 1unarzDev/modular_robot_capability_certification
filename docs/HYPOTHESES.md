# cfcap hypothesis ledger

This file is the scientific state of the study. Update it whenever a result lands. Each entry gives the claim, its
status, the evidence (file), and the next test. Every result here is exploratory and simulation-only unless marked
otherwise.

**Status key:** SUPPORTED · FALSIFIED · OPEN · REFRAMED. The date is when the status last changed.

## Core thesis
A modular robot can certify the capability of configurations it is not in (children and merges) with a
finite-sample bound on unsafe commits. When it can't, a targeted probe or split settles the decision.

## Hypotheses

**H1. Weak wheels are hidden in assemblies (the capxfer premise).** REFRAMED, 11-25.
- With identical motors, a simple duty heuristic detects 92–99 %, so the hiding is not trivial but is weak.
- With realistic ±8 % motor spread, detection drops by about 15 pts. Mild faults (g 0.75–0.9) are caught only 25–40 %
  by the heuristic and 73–76 % by the capxfer self-model.
- At g ≥ 0.85 a weak wheel is the *same plant* as a slower healthy motor, so it is identifiable only up to the motor
  spread. Hence the switch to the ρ_dec label.
- Evidence: `results/stageA/A1_summary.csv`, `docs/board/nuisance.md`.
- Implication: "hidden" means hidden *within the nuisance class*. The masking rate is not new (capxfer Prop 2).

**H2. Counterfactual decisions are often ambiguous after the standard probe (v2).**
- FALSIFIED under a known motor law, 11-25: the T0 oracle leaves 0.02 % ambiguous; T0 decision pilot 1.1 %.
- REFRAMED under a model-free envelope: 7 % ambiguous, or 20.7 % at τ=0.8 (theory). Ambiguity exists exactly when the
  robot doesn't trust its motor model.
- OPEN in T1 with unknown nuisances: cfcap-nuis2 is running.
- Kill line (pre-written): < 5 % ambiguous → probing is not the headline.

**H3. A targeted probe (clamp, tug, module-differential) removes most of the remaining ambiguity.** OPEN.
- T0 oracle: not testable (24 ambiguous decisions).
- Model-free class: clamp takes 7 % → 0.06 % (theory).
- A5: the clamp misfit separates T1→T2 flips (AUC 0.78; v2 gives 0.47).
- T0 decision pilot: decision-EVOI resolves 23 % of the ambiguous cases, the best per probe-second. No policy beats v2
  on regret.
- Next: cfcap-decide (T1) and cfcap-nuis2 (A3 probe sets).

**H4. A learned amortized posterior beats physics Bayes (enumeration or SMC).** OPEN.
- Win conditions: robustness under misspecification (T2) and decision latency inside the probing loop.
- Pilot: amortized 3.4 ms/episode vs enumeration 2–4 s. On mild faults it beats capxfer (square detection
  0.59→0.75; |ĝ−g| 0.077→0.027).
- Next: cfcap-models-b1 + cfcap-baselines-pilot2, then the 2×2 {SMC, amortized} × {T1, T2}.

**H5. Composition (module latents → physics) generalizes to unseen, larger shapes better than direct
prediction.** SUPPORTED (pilot), 11-25.
- On 2×3 when trained on ≤4 modules: coverage 0.90 vs 0.81, pinball 0.00093 vs 0.00147.
- Diversity matters: training on single + pair only collapses on 2×3 (detection 0.03).
- Next: the models-b1 diversity sweep. Must replicate on H3/H4.

**H6. Simultaneous certification across the candidate family is affordable.** SUPPORTED (theory), 11-25.
- The price is 1.6–2.4× the per-decision margin (Gaussian or correlated) and 4–12× for heavy tails.
- The compositional per-wheel route (the Thm 5 merge extension) avoids it: one coverage event certifies all children
  and merges. Monte Carlo miscoverage 1.7 % vs a 19 % bound, so the bound is loose.
- Next: measure the actual price on learned residuals (models-b1, baselines).

**H7. The certificate is tighter than capxfer's (58–75 % certified) at equal unsafe-go.** OPEN; the main headline
candidate.
- Oracle 99.98 %. T0 decision pilot: posterior + simultaneous certificate gives 99.4 % useful commits at 0.54 %
  unsafe. All 6 unsafe cases came from the hypothesis screen dropping the true 2-fault state, which is a fixable bug
  class.
- Next: the baselines and models dec_*.csv on the decision-relevant stratum.

**H8. Misspecification (T1→T2) breaks T1-calibrated certificates.** MOSTLY FALSIFIED, 11-25.
- Pooled flips are 0.53 % (CI 0.35–0.75); ≥3-module children ≤0.25 %.
- EXCEPT single modules: 4.4 % flips, 15.7 % shifted by more than 0.05. Pairs: 1.8 %.
- The cause is per-wheel friction scatter plus differences in the kernel numerics. Compliance, load transfer and
  anisotropy are negligible.
- Theory predicted weak spots at the diagonals and yaw of ≥3-module children. NOT observed.
- Size gradient: single 4.4 % > pair 1.8 % > L3 0.25 % > square 0.07 % flips. The kernel change causes about 60 % of
  single-module flips; per-wheel roller-friction scatter causes the rest (`docs/board/stress.md` C-stress-1).
- Adversarial pilot: confident-wrong margins of 0.11–0.15 on pairs after 3 CEM iterations, with an oracle-latent T1
  certifier.
- Next: T2 evaluation of the trained certificates (stage 2), and adversarial search (`t2_adv`, held pending ranking).

**H9. The certificate stays honest when the posterior is overconfident.** SUPPORTED (T0 pilot), 11-25.
- Running all three probes makes the posterior overconfident. The end-to-end calibration catches it: q rises 1.0 → 5.2
  and useful commits fall to 87 %.
- Possible paper point: end-to-end calibration of the whole probe→posterior→certificate pipeline.

## Open questions and ideas queue (ranked by the research lead; see docs/board/research.md)
- (to be filled by the research lead)
