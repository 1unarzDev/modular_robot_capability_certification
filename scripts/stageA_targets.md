# Stage A targets (A2, A3), written 2026-11-25 before the A2/A3 runs

Exploratory. T0 (quasi-static) oracle analysis; engine `cfcap/t0/stagea.py`, runner `scripts/stageA2_ambiguity.py`,
analysis `scripts/stageA2_ambiguity.py analyze` and `scripts/stageA3_probes.py`. Decision unit: (episode, candidate,
decision leg, tau), tau in {0.6, 0.7, 0.8}, 10 decision legs (surge+-, sway+-, 4 diagonals, yaw+-), candidates =
every connected child of the parent plus 10 random merges with a healthy 1-2-module assembly.

Primary cell (fixed now): probe set v2, noise preset `sensor`, rho_model `known` (idealised oracle: uncertainty only
over the discrete <=2-fault hypothesis; continuous plant known; heterogeneity absorbed as sigma inflation), k = 3,
pooled over all decisions, heterogeneity on and off reported separately and pooled (50/50 draw).
Secondary cells (reported, not used for the kill decision unless stated): noise `floor`, rho_model `prior`
(nuisance-prior upper bound, heterogeneity-on episodes only).

## A2 decision ambiguity
- Ambiguous share = n_ambiguous / n_decisions after v2.
  **< 5 % -> probing is secondary** (contribution 3 is not a headline).
- Oracle certified fraction of capable decisions = n(go and capable) / n(capable), reported overall and for the
  capxfer-comparable cell (lone faulted module: child size 1 containing a fault).
  **Gain < 10 pts over capxfer's 58-75 % (i.e. oracle < 68-85 %) -> attack the conformal slack, not the learner.**
  The comparison uses the upper end (75 %): gain = oracle - 75 pts.
- Oracle unsafe-go must be 0 for rho_model `known` (scorer check; truth is in the set by construction).

## A3 probe value
- Relative reduction of the ambiguous share vs v2, per probe set (v2+clamp, v2+moddiff, v2+tug, all three combined,
  diagnostic split), same episodes, primary cell.
  **No probe set cuts the ambiguous share by >= 30 % relative -> drop active probing as a headline.**
- If the v2 ambiguous share is so small that the reduction is noise (fewer than 50 ambiguous decisions in the
  primary cell), A3 is reported as "not testable: nothing to reduce", which also means probing is not a headline.
- Time cost per probe set: probesets.set_time (capxfer leg durations); diagnostic split costs T_SPLIT_S = 30 s
  (ESTIMATE, B3 timing not measured) + the child's v2 (12 s).

## Sizing
Pilot 300 parents (seeds 0-299) to measure cost; scale up within local limits (one niced process, 4 threads) in
shards of 100 seeds; final N stated in A2_A3.md.
