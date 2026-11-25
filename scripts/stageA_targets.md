# Stage A targets (exploratory; written before each run)

## A1: is a weak wheel hidden at all? (written 2026-11-25, before the run; script `scripts/stageA1_hidden.py`)
Design. T1 plant (`cfcap/t1/plant.py`), plate state plated, DR = the v2 self-model's training DR around theta_plated
(`params.theta_dr`), v2 probe routine with gen_routine jitter (amplitude U(0.85,1.1), lead U(0,1) s). Shapes pair
(8 wheels), square (16), 2x3 (24). Heterogeneity OFF (`HET_OFF`) and ON (`HET_WHEEL8`: per-wheel stall and no-load
multipliers U(0.92,1.08), nothing else). 20,000 episodes = 3 shapes x 2 levels x 3,334; half healthy, half one weak
wheel (g ~ U[0.3,0.9], uniform wheel). Robot-visible channels only (pre_duty, quantised encoder speed, IMU + noise).

Detector 1 (heuristic). Cruise bins = translational command (wz = 0) unchanged for the previous 0.2 s; per cruise
window and wheel, mean |u| (u = pre_duty); g_hat_i = median over the wheel's module siblings of mean|u_j| / mean|u_i|,
median over windows. Episode score = max_i 1/g_hat_i.
Detector 2. capxfer self-model v2 (`rung_v2/models/selfmodel_plated_s{1,2,3}.pt`, probabilities averaged over seeds and
windows as capxfer `trial_probs`); score = max_i (1 - p_ok,i); g_hat from the g head. 2x3 is outside its training
range (<= 4 modules), reported but flagged.
Metrics per (shape, heterogeneity, g bin [0.3,0.45,0.6,0.75,0.9]): detection = share of weak episodes whose score
exceeds the 99th percentile of the healthy episodes of the same (shape, heterogeneity) cell (1 % false alarms);
localisation = argmin g_hat is the weak wheel; |g_hat - g| at the true weak wheel (median, share <= 0.05).

Kill / decision thresholds (heuristic, pair and square, all g pooled):
- **Trivial:** detection >= 95 % AND median |g_hat - g| <= 0.05 with heterogeneity OFF on both pair and square ->
  the weak-wheel story is trivial; pivot the lead to the truly hidden quantities (plan §10 risk 1).
- **Leak:** detection(OFF) - detection(ON) > 15 points on pair or square -> the homogeneous sim was leaking labels;
  heterogeneity stays on in every later dataset.

## A2, A3 (T0 oracle; written 2026-11-25 before the A2/A3 runs)

Exploratory. T0 (quasi-static) oracle analysis; engine `cfcap/t0/stagea.py`, runner `scripts/stageA2_ambiguity.py`,
analysis `scripts/stageA2_ambiguity.py analyze` and `scripts/stageA3_probes.py`. Decision unit: (episode, candidate,
decision leg, tau), tau in {0.6, 0.7, 0.8}, 10 decision legs (surge+-, sway+-, 4 diagonals, yaw+-), candidates =
every connected child of the parent plus 10 random merges with a healthy 1-2-module assembly.

Primary cell (fixed now): probe set v2, noise preset `sensor`, rho_model `known` (idealised oracle: uncertainty only
over the discrete <=2-fault hypothesis; continuous plant known; heterogeneity absorbed as sigma inflation), k = 3,
pooled over all decisions, heterogeneity on and off reported separately and pooled (50/50 draw).
Secondary cells (reported, not used for the kill decision unless stated): noise `floor`, rho_model `prior`
(nuisance-prior upper bound, heterogeneity-on episodes only).

### A2 decision ambiguity
- Ambiguous share = n_ambiguous / n_decisions after v2.
  **< 5 % -> probing is secondary** (contribution 3 is not a headline).
- Oracle certified fraction of capable decisions = n(go and capable) / n(capable), reported overall and for the
  capxfer-comparable cell (lone faulted module: child size 1 containing a fault).
  **Gain < 10 pts over capxfer's 58-75 % (i.e. oracle < 68-85 %) -> attack the conformal slack, not the learner.**
  The comparison uses the upper end (75 %): gain = oracle - 75 pts.
- Oracle unsafe-go must be 0 for rho_model `known` (scorer check; truth is in the set by construction).

### A3 probe value
- Relative reduction of the ambiguous share vs v2, per probe set (v2+clamp, v2+moddiff, v2+tug, all three combined,
  diagnostic split), same episodes, primary cell.
  **No probe set cuts the ambiguous share by >= 30 % relative -> drop active probing as a headline.**
- If the v2 ambiguous share is so small that the reduction is noise (fewer than 50 ambiguous decisions in the
  primary cell), A3 is reported as "not testable: nothing to reduce", which also means probing is not a headline.
- Time cost per probe set: probesets.set_time (capxfer leg durations); diagnostic split costs T_SPLIT_S = 30 s
  (ESTIMATE, B3 timing not measured) + the child's v2 (12 s).

### Sizing
Pilot 300 parents (seeds 0-299) to measure cost; scale up within local limits (one niced process, 4 threads) in
shards of 100 seeds; final N stated in A2_A3.md.
