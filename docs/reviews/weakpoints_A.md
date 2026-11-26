# Weak-points audit A (skeptical evidence auditor, 2026-11-25)

Scope: docs/HYPOTHESES.md, docs/board/research.md, docs/CONFIRMATORY.md, checked against non-confirmatory files under
results/. Nothing under results/confirmatory or conf_* was opened. Fifteen-minute pass, so each item cites the file that
shows the gap. Ranked by how much damage it does to the submission.

## 1. MAJOR (could become fatal): the held-out-severity "robustness to an unseen law" arm does not show that K mattered or was even applied
- Claim (ledger ~23:30Z; decision.md "Held-out-severity result: bars PASSED"): K fitted on severities 0–2 (26.9) vs pooled
  K (28.7) give the same empty-screen share, unsafe rate and usefulness, so "K's exact value doesn't matter".
- Evidence: `results/decision/hos_K02_rows.csv` and `results/decision/hos_Kpool_rows.csv` are **byte-identical**
  (`cmp`, 4,272 rows each), and so are the `_summary` and `_paired` files. Two different screen thresholds produced
  identical per-row probe times, regrets and certified pairs. That fits K never reaching the screen (a plumbing bug)
  better than it fits genuine insensitivity. The bar "useful ≥ pooled-K arm − 5 pts" then passes by construction.
- Also: the convergence filter drops 33/300 (11 %) of severity-3 test episodes, more than at lower severities, which
  selects away the hardest cases. 3/267 unsafe gives an exact upper 95 % bound of ≈3.2 %.
- Cheapest fix (≤30 min CPU): rerun about 50 test seeds with K = 26.9, 28.7 and an extreme K (e.g. 5 or 100), then
  assert that the screen sizes differ. Log K inside the rows file. Report the dropped-episode count and an ITT version
  that counts non-converged episodes as abstain or unsafe.

## 2. MAJOR: the H3 probing "win" is probe-time accounting, not better decisions
- Evidence: `results/decision/hos_Kpool_summary.csv` (env_cal). EVOI − clamp regret is −5.93 s and the probe-time gap is
  8.8 − 2.31 = 6.49 s. EVOI − dominance is −1.68 s against a probe gap of 2.24 s. So EVOI's decisions are slightly
  *worse* (by about 0.5 s) and the whole regret advantage is the probing it skips. Same pattern in
  `results/decision/n1bK_summary.csv`: EVOI − dom regret ≈ −1.9 s vs a probe gap of 2.8 s. task_rel_cert is lower for
  EVOI (0.79–0.83 vs 0.82–0.86). The ledger itself puts the effect at about 3 % of oracle utility.
- Fix: no run needed. Reframe it in the paper as "EVOI matches always-probe usefulness at ¼ of the probe time", and
  report decision regret without probe time as a separate column (computable from the rows files in minutes). Don't
  call it a regret improvement from better decisions.

## 3. MAJOR: every decision-level, probing, screen and H10/H11/H12 result is T0 (quasi-static) only
- Evidence: all of `results/decision/*` (envpost, n1bK, hos_*, t0_n1_*) and `results/theory/gate_cert.csv` are T0.
  H3 is marked "Not shown in T1/T2". The T2 results (`results/stress/*`) use oracle-latent or T1 certifiers, not the
  decision pipeline.
- Fix (≤30 min): run the n1bK EVOI/clamp/v2 policies on about 100 T1 episodes for one cell (≥3-module yaw), as a smoke
  replication. Otherwise add the sentence: "All probing and screen results are in the quasi-static T0 model; dynamic
  (T1) and misspecified-dynamics (T2) replication is limited to the certificate-level stress tests."

## 4. MAJOR: the baselines are pilot-sized, uncalibrated and not on the same footing (H7/H15 untested)
- Evidence: `results/codex/re_baselines.csv` has 98 pilot episodes per shape, and only `raw_lower` rows are filled.
  Every `matched` (equal-unsafe) row is empty. There is no identified-set/PI baseline, no conformal-selection
  (Jin & Candès) baseline, and capxfer was never run on the same episodes. `results/baselines/` holds only
  `smc_small_ids.json`, so no SMC accuracy exists (H4's accuracy arm is "unmeasured"). The "capxfer 58–75 % certified"
  comparison in H7 is cross-paper.
- Fix (≤30 min CPU): split-conformal-calibrate copy_parent, weakest_member, physics_gains and detector_gated on the
  pilot bank at α = 0.1 and fill the `matched` columns (CF at equal unsafe). Use the existing PI ("pi") certifier
  from gate_cert as the training-free rival. If it can't be done, state that H15 and the "beats capxfer" claim are
  untested.

## 5. MAJOR: H5 compositional generalization rests on one seed, n = 600, no CI, and undercovers faulted children
- Evidence: `results/models/cf.csv` (2x3). Compositional cov90 is 0.901 vs direct 0.805 (n = 600, seed s1 only). On
  **faulted** children, compositional is 0.817 and direct 0.802, so the coverage advantage comes almost entirely from
  healthy children, and both miss 0.90 where it matters. `direct_n2_s1` collapses (cov 0.25), and in
  `results/models/calib.csv` n2_s1 on 2x3 has ECE 0.16. That is one seed, so "diversity matters" may be a seed or
  training failure.
- Fix (≤30 min GPU): bootstrap CIs over the 600 children (CPU, minutes); add the faulted-stratum coverage to the
  headline; train one more seed of n2 if a small model fits in 30 min. Otherwise call it a single-seed pilot.

## 6. MAJOR: H4 "amortized beats physics Bayes" has a latency arm and no accuracy arm
- Evidence: `results/models/latency.csv` (3.4 ms) vs ops-measured SMC ≥ 500 s (log only, no results file). There is no
  SMC accuracy or CF file. "Beats capxfer on mild faults 0.59 → 0.75" (`results/models/detect.csv`, square,
  0.75–0.90 bin, n = 201) is against the capxfer_v2 detector, not against SMC or enumeration, and is single-seed.
- Fix: reword to "a 10⁵× latency advantage; accuracy versus SMC not established". Optionally run SMC on 20 of the
  `smc_small_ids.json` episodes (likely beyond 30 min at 500 s each, so the reword is the realistic fix).

## 7. MAJOR: H12 (silent extrapolation) depends on a post-hoc restriction and underpowered cells
- Evidence: the ledger says "post-hoc restriction" to ≥3-module children. The pooled N3 test was neither supported
  nor falsified (`results/stress/N3_verdict.json`, `N3_flips.csv`: undominated 2.76 % vs dominated 1.82 %). At s = 1–2
  the excess is 3.4–3.9 % vs 2.3 %, and "every cell is underpowered by the 20k rule". `results/theory/gate_cert.csv`
  uses 300 episodes per law, so the episode-level CIs are wide. N5 "theory names the yaw probe" FAILED
  (`results/stress/A5y_auc.csv`, 0.44–0.56).
- Fix: rely on the confirmatory §3 test for the primary cell, and present the exploratory version as
  hypothesis-generating. Add Wilson CIs to the s-level table (minutes). Drop all "theory names the probe" language.

## 8. MAJOR: per-pair coverage misses its own bar in several places, yet the claims read as "supported"
- Evidence: `results/stress/N4_verdict.json`: T1 miscov 0.108 / 0.110 > 0.10 (bar_miscov false). `A4_merge.csv`: the
  T1 margin miscovers 43–47 % of T2 pairs. `results/stress/N7_verdict.json`: the size-aware margin costs 7.1 / 6.7 CF
  pts against a ≤5 bar. H8 singles flip 4.3–4.65 %.
- Fix: no run. State the guarantee as "relative to the calibration plant". The decision-level flip rate is the
  supported claim and coverage transfer is not. For singles and pairs, present it as a safety/usefulness trade-off.
  A ±split-variance CI on 0.108 (repeat the cross-fit over 20 random splits, CPU minutes) would make the "within split
  variance" statement legitimate.

## 9. MINOR-MAJOR: the in-model calibration story undercuts the method contribution (H10)
- Evidence: raw posterior ep_any_unsafe 1.7 % < α in-model (ledger 21:00Z), and the certificate is "~30× more
  conservative than its level" (research.md critique). In `results/decision/n1bK_summary.csv`, par (uncalibrated
  parametric) is 97.5 % useful at 5 % ep-unsafe, and par_cal falls to 73 % useful. Calibration's value appears only
  under the wrong law, which is T0-only (#3) and has a K-plumbing question (#1).
- Fix: an honest-limitation sentence: "In-model, calibration buys a guarantee, not safety; its practical value is
  confined to the misspecified-law regime." Make raw posterior a reported baseline in every table.

## 10. MINOR: the H16 environment and H17 explanation results are pilots with LLM-in-the-loop validation
- Evidence: `results/env/pilot_missions.csv` (3,600 rows, all `exploratory`). The 13–14 % / 10 % / 20 % failure
  rates have no CI, and the oracle reconfigures in about 1/3 of plans. H17
  (`results/annotations/c_explain_summary.csv`): n = 200, faithful 0.895 [0.845, 0.935], validated by another LLM
  (codex). No human spot-check is recorded. The LLM's plant-attribution accuracy is 55 %.
- Fix (≤30 min): a human audit of 30 random explanations (Cohen's κ vs codex), and bootstrap CIs over missions in the
  env pilot. Otherwise label both as secondary or appendix and pilot-scale.

## Cross-cutting
- Almost every result is "exploratory" by the ledger's own label, with many single-run fresh-seed replications.
  Report the number of exploratory arms tried per claim (decision NEXT-1 failed, then NEXT-1b passed after a screen
  change) so reviewers can judge the forking paths.
- Single calibration split throughout the decision results. The paired bootstrap CIs resample episodes but not the
  calibration set, so the CIs are optimistic.
