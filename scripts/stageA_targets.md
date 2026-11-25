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
