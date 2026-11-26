# cfcap coordinator status (updated 2026-11-25 ~19:55Z)

All results so far are exploratory and come from simulation.

## Ops alerts
- **01:00Z (box clock): #24 conf4 and #25 eval2 tripped on CUDA OOM from co-location.**
  - On 16–20 GB cards, eval2 conf-env jobs held ~9.4 GB each against a declared 6 GB, so conf4 jobs (~3.5 GB, declared 6) and other eval2 jobs ran out of memory beside them.
  - Fix: the 46 queued eval2 jobs are resubmitted unchanged except **vram 12**, as **cfcap-models-eval2b** (priority 1, same lineage, so it opens at once).
  - #24 resumed: conf4's own peak is fine, so its canary re-validates first.
  - The 3 eval2 jobs still running finish under the old batch.
- **00:54Z (box clock): conf4 is LEASED** (14 running, 71 queued). eval2 main has 16 running, 33 queued, 3 done; eval2h is 11/26 done. Left to run: ~35 box-h of GPU work over 20 boxes, so it finishes around 02:40Z if leasing keeps up. Credit $17.22, burn $3.1/h. (Earlier alerts ran ~20 min ahead of the box clock.)
- **01:15Z: capacity triage (coordinator).**
  - Cut for the window, to rerun tomorrow on free workers. These are queued jobs only, cancelled; leased jobs finish:
    - cfcap-nuis3 (2), cfcap-baselines-smc-small (4), cfcap-baselines-b2r2 (30), cfcap-theory-ride (1), cfcap-decide2 (9);
    - cfcap-t2-extra had nothing queued.
  - Also cancelled: the st-n6-1 retrain cfcap-models-b1r.s000, no longer needed because eval2 uses st-n6-2.
  - Running first: eval2 main, conf4 (conf-cal + conf-main first), eval2h, stress-conf-gate.
  - Future resubmits reuse the replaced batch's lineage string so they skip the canary.
- **00:55Z: conf3 → cfcap-baselines-conf4 (vram 6).**
  - conf3 at vram 12 never fit: the scheduler reserves declared VRAM per host, and boxes carrying eval2/decide/nuis jobs had no 12 GB left.
  - The baselines board shows a 1.3 GB peak for a conf-cal unit, so the 85 queued jobs were resubmitted unchanged except vram 6, as **cfcap-baselines-conf4**: conf-cal priority 0, conf-main 2, het 3, scale 4, same a8dba51 and lineage.
  - It is now leasing (1 so far). Slots free as the eval2 jobs finish: 25 boxes, eval2 has 25 running.
  - Queued nuis3, b2r2 and theory-ride are set to 90, so priority aging cannot lift them above conf4.
- 00:55Z: #23, C-GATE (cfcap-stress-conf-gate): 3 jobs were aborted by the stall watchdog ("hung: no progress", R2), so 22 were paused; 2 done, 3 dead. Shards progress on some boxes and stall on others, probably CPU oversubscription on the spare cores of torch boxes. Resumed so healthy shards continue. Stress is asked to check for a hang (numba threads) and to report progress inside units.
- 00:50Z: conf3's canary is valid but 0 of 85 jobs had leased ("outranked", GPUs busy). To meet the ~01:00Z lease deadline, ops cancelled the 5 **leased** b2r2 jobs. Each held a whole 24 GB GPU on pre-fix, OOM-prone code (43a8dda), and one had shown no progress for 108 min. The 31 queued b2r2 jobs drop to priority 60, below everything in the coordinator's order except nuis3. Baselines can re-emit b2 cheap-mode at a8dba51 later if capacity frees.
- **00:45Z: cfcap-baselines-conf3 submitted** (86 jobs, ~27 box-h, vram 12, a8dba51). Priorities: conf-cal 1, conf-main 2, conf-het 3, conf-scale 4. It is rented-only because it needs the capxfer_v2 weights input. The leased alert follows.
- **00:42Z: eval2 main is LEASED.** Canary valid, 4 jobs running, 46 queued at priority 1. eval2h: 24 leased, 2 done. Credit $18.25, burn $3.0/h: about $11 left at 03:00Z.
- 00:40Z: the 85 queued cfcap-baselines-conf2 jobs are cancelled, per the coordinator. cfcap-baselines-conf3 (cheap methods, no CUDA graph, any-worker) will be submitted at top priority as soon as baselines posts its OPS line.
- **00:36Z: eval2 main SUBMITTED** as cfcap-models-eval2: 51 jobs + canary, priority 1, 1117327, using **st-n6-2 instead of st-n6-1** (coordinator decision). The models-eval2 group selects st-n6-2,st-n6-3,bb-n6-1,ds-n6-1,stl-n6-1, with st-n6-2 duplicates removed. The b1r2 replica is cancelled; b1r.s000 (st-n6-1) keeps running on its one GPU.
- **00:28Z: the st-n6-1 retrain was preempted at 8/12** and restarted from 0/12 on another box, because its checkpoint does not move between boxes.
  - The new ETA is ~01:30Z, which leaves eval2 main (51 jobs) about 1.5 h before 03:00Z.
  - For redundancy, a duplicate is submitted as cfcap-models-b1r2.s000 (priority 1); whichever finishes first is used.
  - **The coordinator/models decision on swapping in seed 2 (st-n6-2, already staged) for eval2 main is now urgent.** It would let eval2 main start at once.
- 00:03Z: the guard's UTC day rolled over ("today $0.00"), so ops now tracks spend by credit. Credit was $33.36 at 19:50Z and is $19.74 now: **$13.6 spent this window.** At $3.0/h to 03:00Z the window total comes to ~$22.6, and the credit ends at ~$10.7 (floor $2).
- **23:48Z OUTPUTS READY: generator.**
  - gen-ref complete: gen-ref2 (14 shards present) + gen-ref3 (plated H3/H4 s000) + gen-ref4 (plated cal.s001, test.s000, plateless test.s000).
  - gen-hetero2 complete (5/5).
  - gen-confirm2 complete (12/12), **stored unread until 01:45Z**.
  - Paths: `~/.cache/vast-out/*/crane/cfcap-gen-{ref2,ref3,ref4,hetero2,confirm2,scaling16r}.*`. For duplicated shard names, use the gen-ref3/4 copies.
- **23:45Z BLOCKER, C-MAIN baselines: #22 cfcap-baselines-conf2 tripped on CUDA OOM, even at vram_gb 22.**
  - The conf-cal jobs run out of memory during CUDA-graph instantiation (`Failed to allocate resources for the kernel nodes`) on 24 GB cards. This is a code issue; ops has no knob for it (no engine or batch option in `smc.py`, `posterior.py` forces `engine="graph"`).
  - b2r2 has the same exposure.
  - **Baselines must commit a memory cap**: chunk the batch in `simulate_bucket`, or an `--engine eager` / `--max-batch` option. Ops re-emits conf2/b2r2 within minutes of the sha.
  - The batch is paused with 85 queued.
- 23:40Z: #21 is the old cfcap-baselines-b2 batch: the same CUDA OOM as #18, on its one leftover cheap-cal job, which is now dead. b2r2 (vram 22) has superseded it, so no action.
- 23:33Z: #20 eval2h canary tripped (infra). The new 170 MB models-b1-weights input reached live boxes after the runner's 6-min input wait. It is present on the boxes now, so the batch is resumed.
- **23:52Z: eval2 is LEASED.** cfcap-models-eval2h has its canary and 3 eval2-het shards running. cfcap-models-b1r.s000 (the st-n6-1 retrain) is leased too.
- 23:50Z: the runner now mirrors complete outputs (files ≤ 95 MB, ≤ 400 MB per job) to `$CRANE_RUN_DIR/artifacts/out`, so the coordinator keeps them if a box dies (dbc0a28, tests 30 pass). It applies to jobs emitted from now on, starting with eval2 main.
- **23:40Z: models_eval2, part 1 submitted.**
  - The compute session recovered 4 of the 5 missing weight sets, so their reruns are cancelled. Re-export: 74 train runs; bb-n6-1, sth-het-1 and bbh-het-1 are now staged.
  - Submitted **cfcap-models-eval2h** (eval2-het + conf-het: 26 jobs, priority 2, a3d17a9, lineage cfcap-models-eval).
  - Only st-n6-1 (b1 s000) is truly lost. Its rerun, cfcap-models-b1r.s000, is leased now. The remaining eval2 groups (main/scale/env/conf-main/conf-env, 51 jobs) are submitted when it lands, at ~00:40Z.
  - Note for models: the re-export doubled dec rows (327k → 655k). There may be duplicate run dirs, from requeued shards pulled from two boxes; check dedup before using `dec_models.csv.gz`.
- 23:33Z: queue priorities now follow the coordinator's order: b1r/hetr 1 (eval2 0–16 once submitted), baselines-conf2 10, conf-gate/conf-merge/confirm2 20, decide2 25, b1x 30, b2r2 31, smc-small 35, nuis3 40, theory-ride 88. Spend $12.9, projection ~$24.
- **23:30Z URGENT, models_eval2: the key weights are lost.**
  - Export is done: 70 train runs, `results/models/paper`. 43 weights are staged in `$CFCAP_OUT/compute/stage2/models-b1-weights` (156 MB).
  - **st-n6-1 (b1 s000), bb-n6-1 (b1w s001), b1x s006 and hetx s000/s003 finished, but their outputs are on no live box and not in vast-out.** The boxes were destroyed unpulled, or free workers kept them as artifacts. This is reported to the compute session.
  - Every eval2 group selects st-n6-1, bb-n6-1 or sth-het-1/bbh-het-1.
  - Rerunning at priority 1 on rented (pullable) boxes as cfcap-models-b1r (s000, s001, s006) and cfcap-models-hetr (s000, s003). A b1 shard takes ~70 min, so the ETA is ~00:45Z.
  - eval2 (77 jobs, 17.8 box-h, priority 0–16, 68c18bd) is emitted and validated, and is submitted the moment those weights are staged.
  - **Coordinator decision needed:** to start now, eval2 could select seed 2 (st-n6-2, bb-n6-2, both staged) instead of seed 1. Check CONFIRMATORY.md first: if it names seed 1, wait.
- **23:08Z OUTPUTS READY: cfcap-t2-eps** (9/9, 4000 plated T2 test episodes) at `~/.cache/vast-out/*/crane/cfcap-t2-eps.*`. Stress (and models-t2, which needs stage2/b1-t2eps) can merge them. Spend is $12.26 with 20 boxes at $3.03/h, so the projection is ~$24.
- 22:40Z: box 62567770 is unreachable and holds 3 done gen-ref2 shards (plated-cal.s001, plated-test.s000, plateless-test.s000). They are rerun as cfcap-gen-ref4 (same refs, priority 20). The manual pull brought back 20 other stranded outputs (t2-eps, models-b1, theory-ride, nuis2 …).
- **22:37Z OUTPUTS READY: cfcap-gen-scaling16r** (7/7, H4b 11–16 modules) at `~/.cache/vast-out/*/crane/cfcap-gen-scaling16r.*`. gen-hetero2 is 4/5. Ops is also pulling done outputs that never left the old boxes (t2-eps, models-b1, theory-ride), using a read-only rsync into vast-out.
- 22:37Z: GPU slots are underused again (11/20 boxes at 1/3 jobs, 8 at 0% GPU) while 8 GB jobs wait as "outranked". Sent to the compute session. Burn is $3.04/h and spend $10.24, so the projection is ~$24.
- **22:30Z gen-ref2: 19/19 done, but 2 outputs were lost.** plated-H3.s000 and H4.s000 sat on box 62570123, which was destroyed before its outputs were pulled (sent to the compute session). They are rerun as cfcap-gen-ref3 (same f70d265 refs, priority 20). Three more shards (plated-cal.s001, plated-test.s000, plateless-test.s000) are being rsynced from a live old box. Every other shard is in `~/.cache/vast-out/*/crane/cfcap-gen-ref2.*`. Outputs-ready note for the generator once all are present.
- **22:25Z OUTPUTS READY: cfcap-stress-n4** (20/20 done, all pulled to `~/.cache/vast-out/*/crane/cfcap-stress-n4.*`). Stress is told, so the N4 margins can be committed before 01:45Z.
- **22:18Z RISK: b2r2 / conf2 (123 jobs, ~50 box-h) have leased nothing in 25 min.** At vram 22 each job needs a whole 24 GB GPU, and the boxes keep packing 8 GB jobs, so a whole GPU rarely frees. Best fix: baselines caps the sim batch (capxfer `_integrate` B) so peak ≤ 10 GB, and ops re-emits at vram 10. Otherwise these campaigns (b2 cheap table and the conf baselines) may not finish by 03:00Z.
- **22:12Z OUTPUTS READY: cfcap-env** (28/28 done). They are pulled to `~/.cache/vast-out/*/crane/cfcap-env.*` (31 dirs, including canaries). The env board documents no collect script, so env runs its own analysis. Conf groups (env-conf-*) stay unopened until 01:45Z.
- 22:08Z: paxton (WSL, a Windows path with spaces) broke the b1w/hetw jobs: the unquoted `$S` in spec.run gave "binary operator expected", and paxton was quarantined for those batches. make_jobs now quotes all paths (148eb2d, tests 29 pass). The queued b1w/hetw jobs were re-emitted quoted as **cfcap-models-b1x** (10 + canary) and **cfcap-models-hetx** (4 + canary). b1w s001/s005 keep running on rented boxes.
- 22:00Z: **free workers enabled.**
  - θ JSONs committed to `data/theta/`, and `make_jobs.py --any-worker` added (2c3b88c, tests 29 pass). It drops "rented" and the θ push, and takes θ from `$S/cfcap/data/theta`, else the capxfer stage.
  - The queued models-b1 (12) and models-het (4) jobs are cancelled and resubmitted as **cfcap-models-b1w** and **cfcap-models-hetw**: same refs and lineages, priority 30 and 67.
  - b2r2, conf2, T2, N4 and smc-small need staged inputs (capxfer_v2, crane_ml, a4_e2), so they stay rented. models_eval2 is emitted with --any-worker only if its weights input can be avoided.
- **21:55Z: #19 cfcap-nuis2 t0-a2 (free lane) tripped: a code bug.** `cfcap/t0/qs.py:147 solve` does `np.concatenate` on an empty list when no hypothesis is alive (`stagea.run_episode` line 147). It is deterministic for shards s005 and s009, and 8 of 10 shards are done. The nuisance role is asked to fix it. Ops resubmits the two shards afterwards with `--resubmit-as`. GPU nuis-main.s000 is unaffected.
- **21:50Z: #18 b2r tripped on CUDA OOM.** This is a spec bug: the cheap-cal jobs peak at 18.8 GB but declared 10 GB, so they were packed beside other jobs on 24/16 GB cards. Fix: queued b2r (36) and baselines-conf (85) are cancelled and resubmitted unchanged except for `vram_gb` 22, as **cfcap-baselines-b2r2** and **cfcap-baselines-conf2** (same commits and lineages). smc-small (vram 16) is untouched; ops watches it.
- 21:40Z: **cfcap-stress-n4 submitted at top priority (5)**: 19 jobs + canary, 4.8 box-h, b08ac71, inputs include stage/a4_e2. Ops tells stress as soon as its outputs land.
- 21:40Z: misfit is resubmitted as **cfcap-t2-motor4** at b08ac71, which saves v2+clamp observations, fault modes and fixed-reference labels. All 21 cfcap-t2-motor3 jobs are cancelled (all leased, none done); **ignore any cfcap-t2-motor3.* outputs**.
- 21:31Z: submitted cfcap-stress-conf-gate (C-GATE: gate-cal at priority 75 first, conf-gate at 80; 28 jobs, 17.2 box-h of spare cores, 0dc1aec, lineage cfcap-stress).
- 21:25Z (clock): submitted cfcap-t2-motor3 (misfit: 20 jobs + canary, 17.8 box-h of spare cores, priority 60, edfbb9f) and cfcap-stress-conf-merge (C-MERGE: 19 jobs, 7.8 box-h, priority 75, lineage cfcap-stress; unopened until 01:45Z). Spend $5.56 at 21:21Z, 24 boxes, $3.80/h: projection ~$27.0.
- 21:33Z: generator v3 resubmitted at **1591e17**: cfcap-gen-scaling16r (7, priority 65), cfcap-gen-hetero2 (5, priority 66–69), cfcap-gen-confirm2 (12, priority 75–80, no auto-canary; spare shards at priority 1; stored unread until 01:45Z).
- 21:33Z: cfcap-env submitted (28 jobs, 3.5 box-h, priority 70–84, 17bd917). Its conf groups are pulled and left unopened until 01:45Z.
- 21:25Z: submitted cfcap-baselines-conf: 86 jobs, 33.3 box-h, priority 75–81, commit 6e1ebd3 (which includes the 7efaec7 ρ fix). Its labels stay unopened until 01:45Z.
- 21:25Z: submitted cfcap-decide2: 18 jobs, priority 90–100, lineage cfcap-decide, commit 77c774e. The requeued cfcap-decide jobs are cancelled again because decide2 replaces them. decide2's queue is the first cut if the projection passes ~$29.
- Spend is box-bound: 24 boxes until 03:00Z ≈ $27 total, whatever the queue holds. Queued work above capacity only competes by priority.
- 21:17Z: cfcap-models-het submitted (4 jobs + canary, 3.7 box-h, priority 67, a8b81a0). models_eval2 waits for the b1 + het weights to be exported and staged.
- **21:15Z: generator v3 label bug (ρ_dec blow-up for heterogeneous types, fixed at 7efaec7).**
  - Cancelled: queued + leased of cfcap-gen-hetero and cfcap-gen-confirm, and queued of cfcap-gen-scaling16.
  - Old confirm outputs (9 dirs, 706 MB) were deleted from ~/.cache/vast-out **unopened**; the list is in /tmp/cfops/confirm_discard.log. Any late pull of `cfcap-gen-confirm.*` is auto-deleted.
  - Old hetero outputs (canary plus 1 shard) are kept for the generator's debugging. **Do not consume them.**
  - Resubmission from the 7efaec7 plans (same lineages, --resubmit-as *2) waits until the generator board confirms the GPU test passed.
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

## Rented campaigns (owner: ≤24 interruptible boxes until 2026-11-26 03:00Z, ≤$35, floor $2; ops target ≤ ~$27)
| Campaign | What | Box-h (est.) | Priority | Done/total (leased, queued) |
|---|---|---|---|---|
| cfcap-models-b1 | Training grid: arch × diversity × seeds | 43.2 | 30 | 32/49 (5, 0) |
| cfcap-models-b1w | models-b1 remainder (2 running) | — | 30 | 1/13 (1, 0) |
| cfcap-models-b1x | models-b1 remainder, any worker (quoted) | ~13 | 30 | 1/11 (9, 1) |
| cfcap-gen-ref2 | Reference splits cal/test/H*, ρ_dec | 4.3 | 30 | 19/19 (0, 0) |
| cfcap-gen-ref3 | Rerun of 2 lost gen-ref2 shards (H3/H4 plated s000) | 0.4 | 20 | 0/2 (2, 0) |
| cfcap-baselines-b2r2 | Cheap baselines on generator ids (vram fix of b2r) | 16.6 | 40–53 | 1/37 (1, 35) |
| cfcap-t2-motor | T2 motor-law arm, 4000 episodes | 7.3 | 55 | 3/9 (6, 0) |
| cfcap-t2-motor2 | T2 knee severities 1–3, 4000 episodes | 11.0 | 55 | 4/13 (9, 0) |
| cfcap-stress-n4 | N4 merge (margins before 01:45Z) | 4.8 | 5 | 20/20 (0, 0) |
| cfcap-t2-motor4 | T2 motor misfit (v2+clamp+yawc, obs + true faults) | 17.8 | 60 | 2/21 (19, 0) |
| cfcap-t2-eps | 4000 plated T2 test episodes | 4.0 | — | 8/9 (1, 0) |
| cfcap-t2-extra | Plateless T2 + 7-condition ablation | 4.5 | 60 | 6/11 (5, 0) |
| cfcap-baselines-smc-small | SMC particle sweep (stage 1) | 14.1 | 60–62 | 4/31 (1, 26) |
| cfcap-gen-scaling16r | H4b 11–16 modules | 0.7 | 65 | 7/7 (0, 0) |
| cfcap-gen-hetero2 | Module types het/H1/H2 | 0.8 | 66–69 | 4/5 (1, 0) |
| cfcap-models-hetx | Training on mixed module types (any worker) | 3.7 | 67 | 1/5 (1, 3) |
| cfcap-env | HM3D-calibrated env missions | 3.5 | 70–84 | 28/28 (0, 0) |
| cfcap-gen-confirm2 | Confirmatory conf_* splits (unopened until 01:45Z) | 1.4 | 75–80 | 7/12 (2, 3) |
| cfcap-baselines-conf2 | Baselines on conf ids (vram fix; unopened until 01:45Z) | 33.3 | 75–81 | 1/86 (0, 85) |
| cfcap-stress-conf-merge | C-MERGE confirmatory (unopened until 01:45Z) | 7.8 | 75 | 15/19 (3, 1) |
| cfcap-stress-conf-gate | C-GATE: gate-cal + conf-gate (unopened until 01:45Z) | 17.2 | 75/80 | 2/28 (21, 5) |
| cfcap-nuis3 | Trimmed nuisance (pair/square × v2/v2+clamp + sweep) | 10.8 | 85 | 1/8 (7, 0) |
| cfcap-nuis2 | Running remainder (nuis-main.s000, canary, t0-a2) | — | — | 10/29 (2, 1) |
| cfcap-theory-ride | T0 phase diagram + envelope (spare cores) | 17 CPU-h | 88 | 12/26 (14, 0) |
| cfcap-decide2 | Decide T1 grid (first to cut) | 30.5 | 90–100 | 1/18 (0, 17) |

Spend so far (ledger, UTC day): $10.24 at 22:35Z. Superseded/cancelled: gen-pilot*, canary*, baselines-pilot2, b2, theory-cpu, decide, gen-scaling16/hetero/confirm (ebba94b), t2-adv.

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
