# cfcap coordinator status (updated 2026-11-25 ~19:55Z)

All results so far are exploratory and come from simulation.

## Ops alerts
- **21:08Z BUDGET/UTILISATION.** 24 boxes at $3.83/h, $4.30 spent: projection to 03:00Z ~$27.2, at the ceiling. About 11 boxes are GPU-idle, because the scheduler reports "outranked" while models-b1 (31), b2r (36) and smc-small (31) wait. Sent to the compute session, asking them to hold autoscale. No cut yet: idle boxes leave after 15 min. If the projection passes $27.5, ops cancels decide (priority 90), then theory-ride.
- 21:05Z: submitted the generator v3 campaigns (e9a0cf3), each with its own lineage:
  - cfcap-gen-scaling16: 7 jobs, priority 65;
  - cfcap-gen-hetero: 5 jobs, priority 66–69;
  - cfcap-gen-confirm: 12 jobs, priority 75–80. Its auto-canary is cancelled; instead the spare last shards (conf-test/H4/H4b s002) lease first at priority 1. Confirm outputs are pulled and stored unread until 01:45Z.
- 21:05Z: spend is $4.02 with 23 boxes at $3.65/h. Projected to 03:00Z: ~$26; the cap is $27. The 24-box cap holds burn near $3.8/h, so no cut yet. Ops cuts from the bottom of §3 (decide → theory-ride → nuis3) if burn passes $3.9/h.
- 21:00Z: cfcap-nuis3 submitted (8 jobs, 10.8 box-h, priority 85, lineage cfcap-nuis, 5ccd33e). nuis2 nuis-main.s000 keeps running (its 7 units are excluded from nuis3); if it dies, ops reruns those units.
- **21:00Z, loop resumed (24-box cap until 03:00Z, keep the projection ≤ ~$27).**
  - Board edits committed (79227a9).
  - cfcap-decide: the 15 cancelled jobs are requeued in the same batch at priority 90.
  - Submitted cfcap-baselines-smc-small stage 1 (31 jobs, 14.1 box-h, priority 60–62); stage 2 stays held.
  - Submitted cfcap-t2-motor2 = t2-knee (12 jobs + canary, 11 box-h, priority 55; lineage cfcap-t2-motor).
  - t2-motor-misfit is HELD because stress has uncommitted N5 wheelrate changes in `cfcap/t2/misfit.py`. Commit them and ops submits.
  - b2 re-emit: not resubmitted. The cheap-method code is unchanged between 43a8dda and 2c26fe0 (smc.py only gained --ids and nuisance_smc_p<N>).
  - Priorities now: models-b1/gen-ref2 30 → b2r 40–53 → t2-motor/knee 55 → t2-extra 60, smc-small 60–62 → broadening 65–70 → confirmatory 75 → nuis3 85, theory-ride 88, decide 90.
  - Spend: $3.80 at 20:50Z, burn $2.76/h with 17 boxes, which projects ~$21 to 03:00Z. 24 cheaper boxes (~$0.15/h) would project ~$26.
- **20:45Z: reprioritised by scientific value (coordinator).**
  - cfcap-decide: 15 queued jobs cancelled (pre-fix code). The leased decide-canary and main.s001 finish. Decision resubmits after N1.
  - cfcap-nuis2: the 15 queued nuis-main shards are cancelled. Shards interleave every shape × probe set, so they cannot be trimmed in place. nuis-main.s000, nuis-canary and the free-lane t0-a2 jobs keep running. The nuisance role is asked to re-emit the trimmed plan as **cfcap-nuis3** (lineage cfcap-nuis): pair + square × {v2, v2+clamp} plus the v2 spread sweep.
  - Queue priorities are now: models-b1 and gen-ref2 at 30, baselines-b2r at 40–53, t2-motor at 55 (all already leased), t2-extra at 60, theory-ride at 65. Trimmed nuis3 goes in at about 70.
  - SMC stays held. Baselines is asked to resize it to ~100 test + 100 cal episodes on pair and square only, with fewer particles if accuracy holds.
- 20:33Z: **cfcap-t2-motor submitted**: the t2-motor group only, 8 jobs + canary, 7.3 box-h (~$1.3), priority 40, lineage set, at f4e83a9. t2-motor-misfit is held. The estimate is fine as is: t2-eps measures 5.5 min per 100-episode unit vs 5.0 planned. t2_extra was **already submitted at 19:52Z** (cfcap-t2-extra at 8c5dd30, 1/11 done); its plan did not change in f4e83a9, so it was not resubmitted.
- 20:40Z: theory_gate (E1) will not be submitted: theory runs it locally (73d5711, c81f9f6). Alert cleared.
- **20:18Z, for the coordinator: estimates and projection.**
  - Done: `make_jobs.py` now sets `spec.lineage` (the campaign base name: nuis2→cfcap-nuis, baselines-b2r→cfcap-baselines) on every job. The auto-canary already had `spec.canary: true`. Committed as 060ae9f; tests 28 pass. Future emits carry both.
  - The "overestimated" rows in perf/REPORT.md are canary-only (n=1–2 shrunk jobs). Real jobs are on or above their estimates. gen-ref2 cal.s001 took 23.5 min (est 17.6), and cal.s000 is past 40 min on a 12-core box (CPU-bound, GPU 0%). nuis-main runs ~13 min/unit (est 6.4). The SMC bl-canary is ≥ 500 s/episode (b2 plan: 80).
  - So no unit_s was lowered. The unsubmitted plans (models_eval2, b2 SMC, b2_matched) have no measured basis for a cut, and b2 SMC needs raising.
  - The spend projection does not drop. It is box-bound: 16 boxes now at $2.70/h, $2.07 spent, so ~$21 at 03:00Z if 16 boxes stay up.
  - Real budget headroom is ~$9 to the $30 line, but box-hours to 03:00Z (~108) are already oversubscribed by queued work (nuis-main alone ~64 box-h at the measured rate).
  - Ops recommendation: no new nuis2 spread or models seeds until the queue drains.
- 20:12Z: b2 was re-emitted at 43a8dda. The 36 queued cfcap-baselines-b2 jobs are cancelled (2 running jobs finish) and the campaign is resubmitted as **cfcap-baselines-b2r**: 39 jobs, priority 40–53. The SMC groups are still held, at `$CFCAP_OUT/compute/ops/b2r-smc.held.json`.
- 20:09Z: the guard's "stalled: no_progress" flags (nuis-main, gen-ref2 cal.s000, bl-canary) are false positives: the processes are at 100% and progress.json is current. Reported to the compute session.
- **20:06Z baselines.** The 15 queued pilot2 jobs are cancelled (pre-fix code); the 3 leased ones keep running. **cfcap-baselines-b2 submitted WITHOUT the nuisance-SMC groups**: 39 jobs (cheap + E3), ~16.6 box-h, priority 40–63 (file priorities −80). b2-smc-cal/test (46 jobs, 24.8 box-h estimated) are HELD at `$CFCAP_OUT/compute/ops/b2-smc.held.json`. Reason: on a 3090 the pilot2 bl-canary has not finished one 3-episode unit with the same SMC settings in ~25 min, i.e. ≥ 500 s/episode against the 80 s in the b2 plan. Those jobs would run 6× or more past their estimate and hit job limits. Ops will submit them when the bl-canary sidecar gives a measured nuisance_smc_s_mean and baselines resizes them (fewer particles or episodes), or the coordinator says to go.
- Spend view: real spend is bounded by boxes × hours, not by job estimates. At ≤16 boxes × ~$0.21/h to 03:00Z the ceiling is ~$25 total; the current burn is $1.9–2.7/h with $1.75 spent. Queued work above box capacity costs no money, but campaigns at higher priority numbers may not finish by 03:00Z.
- baselines_b2_matched.draft.json is held until models posts its median end-to-end time.
- 19:58Z: The compute session found a scheduler bug (it scans only the top-60 queued jobs, and gated jobs fill that window); a fix is deploying. As a workaround, ops set priority 50 on 70 queued jobs: gen-ref2, models-b1, nuis2 t0-a2 (free lane) and the decide-canary. nuis-main is left at its own priority while the budget decision is open.
- 19:58Z: cfcap-theory-cpu's 23 queued jobs are cancelled (the cpu image has no torch). The campaign is resubmitted as **cfcap-theory-ride** (riding torch, priority 50/51, 26 jobs) from the derived plan `$CFCAP_OUT/compute/plans/theory_sweeps.ride.json`. REQUESTS row logged.
- 20:00Z: t2-eps canary is valid after the resume. models_eval2 plan is queued in ops until the models-b1 weights are pulled back and exported (its own prerequisite).
- baselines_b2 plan (~41 box-h, ~$7.5, estimates only) is HELD until cfcap-baselines-pilot2 measures unit costs, as the plan itself says.
- t2-adv is cancelled for good (coordinator). Budget cut order if needed: models-b1 seeds 2–3 at capacity, then DeepSets.
- **19:57Z BUDGET RISK, nuis2.** Measured rates are about 3× the plan. nuis-main.s000 has done 1/14 units in 25 min, and its own ETA is 4.3 h (plan: 95 min). The nuis-canary's v2+clamp+moddiff+tug unit has run more than 20 min for 3 pair episodes (plan: 330 s/unit). Larger probe sets and 2x3 cost much more than the flat unit_s. At this rate nuis-main's 16 shards are about 70 box-h (~$14) instead of 22.5 (~$4.4), which pushes the total toward the $35 cap. Ops is letting the leased shards run. The coordinator should decide whether to cut nuis-main (fewer probe sets, or shapes ≤ square) or accept it. (Update 20:06Z: box-hours limit spend to ≤ ~$25, so this is a wall-time and box-share risk, not a money risk. No cut for now; the coordinator's cut order applies if the projection passes $30.)
- 19:55Z: 13 torch boxes are idle or starved while GPU jobs wait as "outranked: 8 matching workers busy". This is a scheduler or worker-count issue, sent to the compute session. Burn is $2.74/h; projection to 03:00Z is ~$21 (under $30).
- 19:52Z: #17 t2-eps tripped because the crane_ml input was pushed late (infra). The input is on the boxes now; resumed.
- 19:52Z: cfcap-t2-extra submitted ($0.81), REQUESTS row logged. t2_adv is held for the coordinator.
- theory-cpu: the cpu-profile image has no torch or pandas, so 2 cpu boxes are quarantined. The canary passed on torch spare cores.

## Rented campaigns (owner override: ≤16 interruptible boxes, ≤$35 total until 2026-11-26 03:00Z)
| Campaign | What | Box-h (est.) | $ (est.) | Jobs done/total (leased) |
|---|---|---|---|---|
| cfcap-gen-ref2 | Reference splits: cal/test 20k + H3/H4/H6/H7 5k per plate state, ρ_dec | 4.3 | 0.8 | 15/19 (4) |
| cfcap-nuis2 (nuis-main queued cancelled; nuis3 pending) | Nuisance-marginalized ambiguity × 5 probe sets + spread sweep; T0 A2 seeds on the free lane | 22.5 + 8.3 free | 4.4 | 6/29 (8) |
| cfcap-baselines-pilot2 (queued part cancelled) | Nominal, oracle, RLS, capxfer-Thm5, known-nuisance enumeration, nuisance-SMC; 294 episodes | 7.2 | 1.3 | 0/18 (3) |
| cfcap-theory-ride (was theory-cpu) | T0 phase diagram (N 4–8, 5 probe sets, δ) + misspecified-envelope check | 17 CPU-h | 1.7 | 1/26 (0) |
| cfcap-t2-eps | 4000 plated T2 test episodes (independent crane-kernel plant), CPU on torch spare cores | 4.0 | 0.7 | 3/9 (6) |
| cfcap-decide (queued cancelled 20:45Z) | Probe policies × certifiers on T1 posterior draws | 30.5 | 5.5 | 1/18 (2) |
| cfcap-models-b1 | Set transformer / direct / DeepSets × 5 diversity levels × 3 seeds | 43.2 | 7.8 | 1/49 (14) |
| cfcap-t2-extra | 1000 plateless T2 episodes + 7-condition ablation (1000), CPU on torch spare cores | 4.5 | 0.8 | 1/11 (5) |
| cfcap-baselines-b2r | Cheap baselines + E3 on generator cal/test/H* ids (SMC groups held; replaces b2) | 16.6 | 3.0 | 1/39 (2) |
| cfcap-t2-motor | T2 motor-law arm, 4000 episodes × {all+motor, motor} (misfit group held) | 7.3 | 1.3 | 1/9 (8) |
| cfcap-t2-adv | Adversarial CEM on T2: cancelled by the coordinator | | 0 | not submitted |

Spend so far (ledger, UTC day): $3.58 at 20:43Z.
Committed estimate ≈ $23 of ≈ $32 usable (excluding t2-adv). Idle boxes cost more than these estimates: at the current burn, ~$21 is projected to 03:00Z.

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
