# baselines (specialist board)

OPS: (ops, 23:45Z) **BLOCKER, #22.** conf2 (vram 22, alone on 24 GB cards) still OOMs in CUDA-graph instantiation in capxfer `sim._integrate` (from `t1/plant.simulate_bucket`). Please add a memory cap now: either a `--max-batch N` that chunks the jobs/candidates in each simulate_bucket call, or `--engine eager`. Set the plan's vram_gb to the measured peak (`torch.cuda.max_memory_allocated` on the largest 2x3/H4 episode), commit, and post the sha here. C-MAIN depends on it, and 3 h are left.


OPS: (ops, 22:18Z) **urgent.** At vram_gb 22, b2r2 and conf2 cannot lease, because no 24 GB GPU is ever free of other jobs. Please make the smc task cap its simulation batch (for example, chunk B in `t1.plant.simulate_bucket` / capxfer `_integrate`) so peak memory stays ≤ 10 GB, verify the peak locally (`torch.cuda.max_memory_allocated`), commit, and post the sha here. Ops then re-emits b2r2/conf2 at vram 10 at that commit.


OPS: (ops, 21:50Z) **#18 b2r: CUDA OOM.** b2-cheap-cal peaks at ~18.8 GB of PyTorch memory (capxfer sim `_integrate`, B×K staging plus CUDA graphs), but the plan says vram_gb 10, so the scheduler packed it next to other jobs. Ops resubmitted b2r and baselines-conf with vram_gb 22 (cfcap-baselines-b2r2 / -conf2). For future plans, set vram_gb to the measured peak, or cap the sim batch so it fits in 10 GB. Check whether smc-small (vram 16) peaks above that at 1024/2048 particles.


OPS: (ops, 20:45Z, coordinator) keep b2 SMC held. Resize it to the measured ≥ 500 s/episode: ~100 test + 100 cal episodes on pair and square only, and fewer particles if accuracy holds (this is enough for the T3 grid). Commit the plan, and ops submits it as a new campaign (lineage cfcap-baselines).


OPS: (ops, 20:12Z) b2 is resubmitted at 43a8dda as cfcap-baselines-b2r, without b2-smc-cal/test (39 jobs). The SMC groups are held because pilot2's bl-canary, with the same nuisance_smc settings (256 particles, 150 stages), has done 0 of 2 three-episode units after ~25 min on a 3090. That is ≥ 500 s/episode against the plan's 80 s. Please read the bl-canary sidecar once it lands (pulled to ~/.cache/vast-out/vast-torch-62569107/crane/cfcap-baselines-pilot2.bl-canary.s000/) and resize the SMC groups. Ops will then submit `$CFCAP_OUT/compute/ops/b2r-smc.held.json`, or a re-emit.


OPS: (ops, 2026-11-25 19:59Z) `baselines_b2.json` is held, as its own pilot field asks, until cfcap-baselines-pilot2's bl-canary sidecars give measured per-method seconds. Its canary is still running on 62569107 (9+ min). Once the numbers land, put them in unit_s and ops will emit and submit at priority ≤50. Note: nuisance SMC on 3090s is running about 3× slower than planned in nuis2 (see STATUS ops alerts).


**Status (2026-11-25, updated after review round 0 and the codex audits):** the six baselines are built and committed behind one interface, together with the runner task
`smc` and the rented pilot campaign `scripts/campaigns/baselines_pilot.json`, which validates and is ready to submit.
**The pilot table is not measured yet.** The local GPU queue never freed, and the coordinator moved the GPU pilot to
rented 3090s. All numbers below are smoke tests or estimates, and all are exploratory.

## What exists
- **Interface** (`cfcap/baselines/base.py`): `Baseline.__call__(Episode) -> Prediction(point, scale, lower)` per
  (candidate, leg).
  - An empty candidate set, or a non-finite bound, refuses.
  - The conformal layer is `conformal.core.SimultaneousLower` on (point, scale). It is fitted 2-fold cross-fit, with
    α = 0.1 simultaneous over every (candidate, leg) of an episode.
- **Label** is ρ_dec = ρ_abs / ρ_ref(shape, leg), per the coordinator's decision. ρ_ref comes from `cfcap/t1/reference.py`
  (new and additive): nominal θ as a point domain, healthy, no spread, cached per canonical shape.
- **Episodes** (`cfcap/baselines/episode.py`) come from two sources:
  - the pilot bank (`pilot_episode(split, j)`): pair, square and 2×3, HET_FULL, 0–2 faults, children only;
  - the generator's `make_episode(plate, split, index)` (`from_gen`), with its own v2 observation and labels. Here
    eid is the generator index; candidates are self, children and merges. Merge partners count as known.
- **The baselines:**

| name | what | privileged |
|---|---|---|
| `nominal` | ρ_dec = 1, scale 1 | no |
| `rls_interval` | per-wheel LS (= RLS with λ = 1) of ω = k·u + b·sign(ω_ref); ĝ = k/median k ± (2 SE + 0.20); passive flag at ĝ < 0.5; propagated through the T0 QS plant (point at ĝ, lower bound at the worst case of g_lo and coast/brake) | no |
| `capxfer` | capxfer v2 TCN ensemble → Mondrian sets at the Theorem 5 levels (t_OK at α₁/2n_G, faults at α₁/2K) → projection S\|_B with the global K = 2 budget → per-mode bands (2nd smallest of 39 fresh draws, ρ_dec per leg, T1, cached per child shape) → min over S\|_B. Reported both raw and re-calibrated by SimultaneousLower. | no |
| `enum_known` | A6 exact enumeration over ≤2 faults (g grid 0.3:0.1:0.9, coast, brake), every nuisance known, generator prior; top hypotheses pushed through T1 | yes |
| `nuisance_smc` | role nuisance's sampler `cfcap/nuisance_post.py`, used read-only: faults × every nuisance, 256 particles, ≤150 stages. Wrapped for any Episode: binned observation, gyro bias from the leading rest, known payload. 32 draws go to the candidates. | payload only |
| `oracle` | the T1 label | yes |

- **Interim SMC.** `posterior.py` has my own nuisance IS/SMC (`NuisanceSMCIS`). It is kept for comparison only; the
  shared sampler is nuisance's.
- **Tests.** `tests/test_baselines.py` has 10 tests, all passing:
  - the oracle has zero unsafe commits and certifies 100 %;
  - nominal is exact on a healthy nominal body (ρ_dec = 1);
  - an empty candidate set refuses (nominal, oracle, RLS);
  - capxfer's projection returns empty when over budget or when a set is empty;
  - the fault prior sums to 1;
  - the SMC task's units cover the split.

## Findings (exploratory; smoke runs on the shared 4070, 3 episodes per shape, `/tmp/bl_try*.log`)
- `enum_known`:
  - the posterior on the true mode vector is 1.0 in 9/9 episodes;
  - 2×3 takes 1.8–4.4 s for the posterior plus 2.6 s for propagation (22,573 hypotheses, compile engine). With the
    graph engine it took 11 s.
- **My interim IS-SMC:**
  - it reached β = 1 in 5–11 stages on 2×3, but stalled at β ≤ 0.04 on 2 of 3 pair episodes;
  - its q05 covered the truth on only 33–92 % of entries;
  - it took 15–34 s per episode.

  This is why the shared nuisance sampler replaces it.
- `rls_interval`: 1.4 s per 2×3 episode on the CPU. Its lower bounds are loose (≈0.7–0.8 on healthy children), because
  ĝ spreads 0.77–1.28 on healthy wheels with heterogeneity on.
- `capxfer`: about 3 s per pair episode on the CPU, using the per-shape band cache.

## Pilot (rented; submit `scripts/campaigns/baselines_pilot.json`)
- **bl-canary**: 1 job, 6 episodes, all six methods.
- **bl-pilot**: 16 jobs, 294 episodes (pilot split, eids 6–299).
- Estimate: 7.2 box-h, $1.30. Inputs: theta plus `capxfer_v2` (5.7 MB of TCN weights and capxfer's sim calibration
  probabilities).
- When the outputs are back, run:
  `$PY scripts/baselines_pilot.py from-runs --runs ~/.cache/vast-out --hardware "RTX 3090"`.
  It writes `results/baselines/pilot_metrics.csv` (cert_frac, cert_frac_faulted, unsafe_commit_rate, ep_any_unsafe
  per τ, shape and stratum), `pilot_timing.csv` (s per decision, peak VRAM, sims per episode), and the paper contract files
  `results/baselines/paper/dec_baselines.csv.gz` and `latency_baselines.csv`.

- **Strata.** Only 2.3 % of labels are below 0.8 (coordinator). Every baseline is therefore reported on two strata:
  - `all`;
  - `decision`: entries of candidates that contain a faulted module, with |ρ_dec − τ| < 0.1. This is the paper
    headline. It is per τ, and the bounds are still calibrated on all entries.

  The dec rows carry `has_fault` (= `contains_fault`, also written) and the ρ_dec `truth`, so the paper can form the
  stratum itself.

- **Incident #16.** On the slim torch image, torch.compile failed with no C compiler. `posterior.loglik` now falls back
  from compile to graph automatically, as the generator does (da8a065).
  - Re-emitted as `cfcap-baselines-pilot2`: `$CFCAP_OUT/baselines/jobs_baselines_pilot2.json`, 18 jobs, validated,
    not submitted.

## Review round 0, codex audits, E3 (2026-11-25; code done, nothing scored yet)

**Screen truncation** (`docs/codex/screen_bug.md`):
- `enum_known` has no screen: every hypothesis' likelihood is computed. Propagation now covers 1 − 10⁻³ of the
  posterior mass (cap 512). If more mass is left out, the episode abstains: lower = −inf, and the simultaneous layer
  refuses too.
- The legacy IS-SMC (`NuisanceSMCIS`, top-96 screen) is **not reported**. It is not in the runner's method list.
- `nuisance_smc` (nuisance's sampler) mixes its screen with the prior, and its discrete moves reach every mode vector.
- Per-episode truth retention is recorded for both reported Bayes baselines, and `from-runs` reports it per method
  and shape in `*timing.csv`. The columns are `truth_in_support` (≤ 2 faults), `truth_retained` (the true mode vector
  is among the propagated hypotheses or draws), `mass_true` and `dropped_mass`.

**Scorer audit** (`tests/codex/test_audit_baselines.py`, now passing):
- capxfer refuses when any parent wheel's set is empty;
- the decision stratum is faulted **OR** |ρ_dec − τ| < 0.1;
- a ±inf or NaN lower bound is a refusal, never a commit;
- entries with a non-finite truth are not decisions.

No baselines-pilot2 output had come back when these fixes landed, so nothing was scored with the old code.

**Review round 0** (a–d). These are offline certifiers in `from-runs`, applied to the same rows:
- (a) The uncalibrated posterior 5 % quantile (`raw`, written to dec as `enum_known_raw` / `nuisance_smc_raw`).
- (b) Conformal selection (Jin & Candès) at FDR 0.05 and 0.1 (`cselect_q05/q10`). Entries within an episode are
  dependent, so this is an empirical certifier.
- (c) Each method at its own guarantee: SimultaneousLower at α = 0.05 and 0.1 (`sim_a05/a10`), plus Mondrian per
  parent shape. Strata per true fault class (`class0/1/2`). A shifted test that importance-weights episodes by 2^faults,
  i.e. doubled odds of each parent fault (`shift = faults2x`), with calibration unchanged.
- (d) `baselines_b2.json` runs capxfer and every other method on the generator's own ids:
  - cal 0–1999 and test 0–1999 for the cheap methods (enum, capxfer, RLS, nominal, oracle, trivial);
  - H3, H4, H6 and H7 0–499; H4 skips the enumeration (40-wheel parents);
  - SMC on cal and test 0–499 of the same ids.

**Matched compute** (models):
- `--wall-budget-s` caps nuisance-SMC's stages per episode to fit a wall budget per decision (screen + SMC +
  propagation). Seconds per simulated row are measured on the episode's screen, then refined per shape.
- The group is in `baselines_b2_matched.draft.json`. It stays a draft until models' median end-to-end time is in
  `docs/board/models.md`.

**E3 trivial baselines** (`cfcap/baselines/trivial.py`; `model_free` is offline):
- `detector_gated`, `physics_gains`, `copy_parent`, `weakest_member`, `model_free`;
- they are scored by body size (`size<k>` strata) and written to `results/baselines/paper/dec_trivial.csv`;
- capxfer is also scored on **lone faulted modules** (`lone_faulted` stratum: single-module candidates that hold a
  fault), the population behind its published 58–75 %.

**E3 success criteria, written before any scoring.** At α = 0.1 simultaneous, decision stratum, τ ∈ {0.6, 0.7, 0.8},
generator test split (pilot bank first, exploratory):
1. The learned method (models' amortized posterior) certifies ≥ 10 points more of the capable decisions than the best
   trivial baseline (max over detector_gated, physics_gains, copy_parent, weakest_member, model_free), at
   ep_any_unsafe ≤ α.
2. It certifies ≥ 5 points more than `model_free`.

Either failing is reported as a failure of the learned method's claim, not tuned away.

## N4 / N6 and the label reference (2026-11-25)
- **N4: merge baselines** (`cfcap/baselines/compose.py`, one implementation, for role stress and the baselines
  runner):
  - `merge_enum_known(probed, cands)`: the exact known-nuisance enumeration per separately probed assembly. The
    posteriors are independent, so a merge's posterior is their product. Each assembly is kept to 1 − 10⁻³ mass, and
    the product to 1 − 10⁻³ of its mass. Anything more left out makes the candidate abstain.
  - `merge_copy_parent`: the weakest constituent's own measured v2 ρ.
  - `merge_detector_gated`: 0 if any constituent wheel is flagged, else 1.
  - `gen_merge_inputs(plate, split, index)`: builds the probed assemblies, the merge candidates and their T1 ρ_dec
    labels from a generator episode.
  - Unit-tested (module composition order, empty → refuse). Not yet run on a GPU episode.
- **N6.** No stratum numbers from baselines were ever produced (pilot2 had not returned when the AND→OR fix landed),
  so nothing is void. E3 is checked once the b2 generator-eid rows exist.
- **Label reference.** Every baselines certificate predicts, and is scored against, ρ_dec = ρ_abs / ρ_ref with ρ_ref
  from the fixed T1 nominal healthy table (`cfcap.t1.reference`, `episode.ref_table`).
  - T2 truth for the same candidates must use that same table, never T2's own healthy reference.
  - `ref_table(cands, plate)` is the function to call.

## SMC cost and the accuracy-vs-particles curve (2026-11-25; for T3)
- **Measured.** The cfcap-baselines-pilot2 canary measured nuisance-SMC at **≥ 500 s per episode** on a 3090, with
  256 particles and ≤ 150 stages. The plan assumed 80 s. Amortized takes 3.4 ms per decision (models), a gap of about
  10⁵×.
- **Redesign** (`scripts/campaigns/baselines_smc_small.json`, campaign cfcap-baselines-smc-small, lineage
  cfcap-baselines). The SMC groups are removed from b2.
- **Episode ids.** Only generator ids in 0–1999 (the models role's ids) whose parent is a pair (2-module chain) or a
  square (4-module grid). They are listed in `results/baselines/smc_small_ids.json`:
  - test: 79 pairs + 21 squares;
  - cal: 72 pairs + 28 squares.

  Squares are rare in 0–1999, so all of them are used.
- **Stage 1, particle sweep.** 256, 1024 and 2048 particles ("full"), each on the same 10 test ids (5 pair, 5
  square), one episode per job: 30 jobs, ~14 box-h, ~$2.5.
- **Selection rule, fixed before the sweep.** Take the cheapest setting within 1 pt of 2048 on all three of:
  - `truth_retained` (the true mode vector is among the propagated draws);
  - raw q05 coverage of ρ_dec over all entries;
  - raw q05 coverage over faulted-or-near-0.8 entries.

  Run it with `scripts/baselines_pilot.py smc-sweep --runs ~/.cache/vast-out`, which writes
  `results/baselines/smc_particle_sweep.csv` (s/episode, sims, retention, mass on the truth, coverage, |point −
  point_full|).
- **Stage 2.** 100 test + 100 cal episodes at the chosen setting: 68 jobs, ~32 box-h, ~$5.7 at 256 particles. The plan
  is written at 256; if the sweep picks more, update `--particles` and `unit_s`, then re-emit.
- **Accuracy curve:** TBD (filled from smc_particle_sweep.csv when stage 1 returns).

## Confirmatory (docs/CONFIRMATORY.md; plans emitted 2026-11-25; no confirmatory label opened)
- **Output.** `scripts/baselines_pilot.py conf-export --runs ~/.cache/vast-out` writes
  `results/baselines/confirmatory/dec_conf_baselines.csv.gz`, with the columns of paper §5a.
  - Columns are §1a plus `claim`, `shard` and `parent_n`.
  - `truth` is ρ_dec from the generator's fixed T1 reference.
  - It also carries exploratory `cal` rows (point, scale and a cross-fit lower) for every method, for the matched
    point.
  - It **refuses to run before 2026-11-26 01:45Z** (except with `--mock`). An optional `--t2-truth` adds T2 rows with
    the same T1-calibrated bounds.
- **Methods by claim:**
  - **C-MAIN** (conf_test 40–3,039):
    - `always_go`, `never_go`;
    - `metadata_nf`: calibration (candidate size, leg) quantiles, no observation;
    - `detector_gated`, `ghat_physics`, `copy_parent`, `min_over_members`;
    - `capxfer`: Theorem 5, re-calibrated;
    - `credible_lower`: enum_known's Bayes q05, raw;
    - `cfBH`: conformal selection at FDR 0.1 on enum_known. Its lower is the largest τ at which the entry was
      selected, or −inf;
    - `oracle`;
  - **C-SCALE:** `capxfer`, `metadata_nf`.
  - **C-HET:** `metadata_nf`.
  - **Not produced here:** `raw` (amortized posterior, q = 0, which is models') and `dominance_pi` (theory).
- **Single reference.** `cfcap/t1/reference.py` now delegates to the generator's `cfcap.gen.rho_ref`: the mecanum
  nominal reference of the shape, the same numbers as the labels.
  - Candidate latents carry the module wheel types (`wtype`) for het episodes.
- **Shard 0 is unscored by the scorer**, so every test group's shard 0 is a sacrificial block:
  - conf-main: exactly conf_test 0–39 (76 shards × 40);
  - other groups: ids ≥ 5000, outside the claim ranges.
- **Plan** (lineage cfcap-baselines), trimmed by the coordinator's ruling: `baselines_conf.json`, 86 jobs,
  33 box-h, ~$6.0.
  - conf-cal: exploratory cal 0–1999;
  - conf-main: conf_test 40–3,039, 21 shards × 150; shard 0 = 0–39 + 5000–5109;
  - conf-scale-H4 / H4b: 0–599 each. About 5.4k incapable stratum decisions per split, below the 20k power rule, so
    E_dec is expected UNDERPOWERED and E_ep is reported;
  - conf-het ×3: 0–999 each.

  The optional SMC conf plan is dropped (coordinator: descriptive; smc_small covers T3). `raw` is models';
  `dominance_pi` is theory's.
- **Incident #22 (00:31Z).** CUDA-graph OOM on conf-cal. Fixed in a8dba51:
  - every T1 call (likelihood, candidate capability, generator probes and labels, capxfer bands) runs through a
    fallback chain compile → graph → eager, and any failure, OOM included, moves to the next engine;
  - batches are smaller: 2¹⁴ wheels per likelihood batch, 2¹⁵ per capability or label run;
  - `--engine`, `--chunk-wheels` and `--cap-max-wheels` are task flags;
  - one local conf-cal unit (2 episodes, the 8 conf3 methods, graph engine) ran in 22 s at 1.3 GB peak VRAM with no
    fallback triggered.
- **conf3** (research-lead rescue): `jobs_baselines_conf.json` = cfcap-baselines-conf3, 86 jobs, 27 box-h, ~$4.9.
  - Methods: capxfer, detector_gated, ghat_physics, copy_parent, min_over_members, rls_interval (run, not exported),
    nominal, oracle. enum_known and nuisance_smc are out of the confirmatory path; credible_lower comes from models.
  - Ranges: conf-cal 0–1999 and conf-main 40–3,039 first, unchanged; conf-scale cut to 0–299 per split.
- **Unit times are estimates.** The H4 / H4b estimate (~35 s per episode) is the least certain.

## Full-scale proposal (estimates until the canary lands)
Generator episodes (`--source gen`). Conformal calibration needs the cal split, so every method runs on cal and on
test.

| group | episodes | s/episode (3090) | box-h | $ @ 0.18 |
|---|---|---|---|---|
| SMC-T1 test (`smc-t1` in cfcap_b1.draft.json, all six methods) | 500 | ~80 | 11 | 2.0 |
| SMC-T1 cal (same, `--split cal`) | 500 | ~80 | 11 | 2.0 |
| cheap methods only (`--methods enum_known,capxfer,rls_interval,nominal,oracle`), cal + test + H3/H4/H6/H7 | 2000 + 2000 + 4 × 500 | ~10 | 17 | 3.0 |

- Cheap enough for the full test set: nominal, oracle, rls_interval (CPU), capxfer and enum_known.
- SMC is the costly one, roughly 10⁵ sims per episode. 500 + 500 episodes is the proposal.
- **Caveat on `enum_known`.** Generator episodes can carry 3 faults (10 %), which fall outside the ≤2-fault
  enumeration. The truth is then outside its support, and this is reported, not hidden.

## Needs from other roles
- **compute / coordinator:** submit `baselines_pilot.json` (canary first). Then paste the canary's measured
  `*_s_mean` into the `pilot` fields here and in `cfcap_b1.draft.json` `smc-t1`.
- **nuisance:** `NuisanceSMC` calls `nuisance_post.Space / Sim / screen / Q / smc` as they are. Please keep those
  signatures stable, or tell me when they change.
- **generator:** none. `from_gen` uses `make_episode`, `simrun.probe_episodes` and `label_episodes` read-only.
- **paper:** rows follow §1a. ρ_rel is not in the CSV; the truth column is ρ_dec.

## Next step
Ingest the rented pilot and fill the table. Then size `smc-t1` and the cal and cheap groups from the measured
s/episode.
- OPS/baselines (from generator, 7efaec7; coordinator agreed): on het/H1/H2, `cand_rho_dec` divides by the **mecanum** reference of the same shape (`cand_rho_ref`), whatever the module type. The per-type reference is only `cand_rho_ref_type` (secondary). Any per-type normalisation in baselines should use `cand_rho_ref`.
