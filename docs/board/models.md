# models (amortized morphology-conditioned posterior, direct head, training sweep)

**Status:** the pilot is done (own T1 data, ρ_rel label, local 4070).
- Stage B training, `scripts/campaigns/models_b1.json`: 48 jobs plus a canary, 43.2 box-h, $7.78. Emitted and
  validated; not submitted yet.
- Stage 2 eval, extension and T2, `scripts/campaigns/models_eval2.json`: 32 jobs plus canaries, 7.9 box-h, $1.42.
  Validated; it needs the b1 outputs before it can run, and the extension also needs the generator's new splits.

## Model (`cfcap/models/posterior.py`)
- **Architecture.** A set transformer over wheel tokens. Defaults: h = 96, 3 pre-norm layers, 0.97 M parameters,
  3.9 MB weights.
  - **Wheel token.** A strided dilated TCN over (IK ref/5, pre_duty, speed/5) for the whole v2 trial (735 bins),
    plus an MLP of the wheel's geometry: position, w_hat, r_hat, module centre, and offset from the module centre.
  - **No index embeddings.** There are no wheel- or module-index embeddings, so the model is equivariant to
    relabelling wheels and modules (tested).
  - **Assembly token.** One token from a TCN over (command, gz, ax, ay).
- **Heads.**
  - Per wheel: mode logits (ok / weak / coast / brake) and a Beta(a, b) over the weak gain.
  - Per module: Gaussian over log μ, log stall scale and log no-load scale, from wheel tokens mean-pooled by module.
  - Direct counterfactual head (the black-box comparator). The candidate's tokens are re-embedded with their geometry
    in the candidate's frame, passed through one attention layer, and read by leg queries. It outputs monotone
    quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of ρ per leg. A merge uses the same call on the two parents' token sets
    concatenated (`gendata.forward`).
- **Compositional route** (`cfcap/models/compose.py`).
  1. Sample per-wheel modes mean-field from the posterior, g from the Beta head, and module μ, stall and no-load from
     the nuisance head. Draw everything else from the T1 prior.
  2. Run `t1.capability` rollouts of the candidate.
  3. Report the sample quantiles.
- **Ablations in the grid.** `ds` is DeepSets (pooled context, no attention, close to capxfer). `bb` is the black box:
  same encoder, direct-head loss only.

## Findings (exploratory; local RTX 4070 Ti SUPER, shared GPU)
Pilot data are my own T1 episodes (`scripts/models_pilot_gen.py`): HET_FULL, jittered v2, 2 children per episode, label
ρ_rel. Training used single/pair/L3/square (30k episodes). Test sets have 2,000 episodes each of pair, square and 2×3;
2×3 is a size never trained on. The baseline is the capxfer v2 self-model, 3 seeds, scored exactly as Stage A1 D2.
Detection uses each detector's own threshold at 1 % false alarm on healthy episodes. Sources are
`results/models/{detect,calib,cf,latency}.csv` from `scripts/models_pilot_eval.py`. The model is `n4_s1` (10 min,
21k steps).

**Mild faults, g in [0.75, 0.9]:**

| shape | detection, capxfer → amortized | median \|ĝ−g\|, capxfer → amortized |
|---|---|---|
| pair | 0.75 → **0.80** | 0.055 → **0.026** |
| square | 0.59 → **0.75** | 0.077 → **0.027** |
| 2×3 (unseen size) | 0.49 → 0.49 | 0.095 → **0.034** |

- **Calibration (ECE).** Top-label ECE of the mode posterior is 0.002 / 0.002 / 0.002 for the amortized model (pair /
  square / 2×3), against 0.014 / 0.003 / 0.002 for capxfer. The faulted-vs-ok ECE on 2×3 is 0.013 against 0.004 for
  capxfer: calibration degrades with size. On faulted wheels, mode accuracy is 0.85 / 0.85 / 0.84 against
  0.81 / 0.75 / 0.68.
- **Direct head, size generalisation** (trained on ≤ 4 modules). On 2×3, pinball is 0.0014 against 0.0068 for the
  marginal quantiles. Its 90 % coverage is **0.81** overall and 0.80 on faulted children, against 0.92 on square, which
  is in distribution. The intervals are too narrow off-distribution: conformal calibration is needed, and a
  size-diverse training set.
- **Compositional vs direct** (300 episodes, 600 children per shape).
  - On 2×3, compositional pinball is **0.00093** against 0.00147 for the direct head, and coverage 0.90 against 0.81.
  - On faulted 2×3 children: 0.0018 against 0.0024, coverage 0.82 against 0.80.
  - On square, the direct head is slightly better on faulted children (0.0033 against 0.0031).
  - So the compositional route degrades less with size.
  - Oracle-mode composition (true faults, prior nuisances) has pinball 0.0017 and coverage 0.76 on faulted 2×3
    children. That is the irreducible nuisance part under ρ_rel.
- **Diversity.** Training on single + pair only (`n2`, 4 min) collapses on larger bodies. Mild-fault detection is
  0.24 on square and 0.03 on 2×3. Direct-head coverage on 2×3 is 0.25 and ECE is 0.16. Training-size diversity is
  necessary.
- **Latency.**
  - Amortized inference: 3.4 ms per episode at batch 1, every shape.
  - Compositional route: 18 / 23 / 31 ms per child (S = 16, batched).
  - For reference, A6 exact enumeration costs 3.9 s per 2×3 episode.
- **Overfitting.** With 30k fixed episodes, the validation Beta-NLL was best near 8k steps and worse by 21k (−1.93 vs
  −1.11). Stage B therefore uses fresh pools from the generator.
- **Training throughput.** Uncontended, on own data: 35 steps/s, 6.6k episodes/s.
  - On generator episodes (local, GPU at 99 % from other roles): 5.8 steps/s, generation 0.80 s per full-candidate
    episode, peak VRAM 1.3 GB.
  - The generator board measures 17.9 episodes/s steady when uncontended.
- **Caveats.** The pilot is ρ_rel on my own episodes. The coordinator's ρ_dec label and generator episodes apply from
  Stage B on. With ±8 % spread, weak g ≥ 0.85 is the same plant as a slower healthy motor, so the mild-fault misses
  there are partly irreducible.

## Stage B training sweep: `scripts/campaigns/models_b1.json` (emitted and validated, not yet submitted)
- **Grid b1** (shard = entry, seed 1 first):
  - `st`, `bb` and `ds` × diversity {n6, n4, n2, n6c, n4c} × seeds 1–3;
  - plus `stl` (h = 128, 4 layers) × n6 × 3 seeds;
  - 48 jobs in all.
- **Diversity levels** are parent filters on the generator's train split (cheap pre-filter on the parent shape):

  | level | max N | train families |
  |---|---|---|
  | n2 | 2 | 2 |
  | n4c | 4 (single/chain/grid) | 5 |
  | n4 | 4 | 8 |
  | n6c | 6 (single/chain/grid) | 8 |
  | n6 | 6 | 37 |

  n6c and n4 hold family count fixed while size changes, and n6c and n6 hold size fixed while families change.
- **Data.** Generator train episodes made on the fly from their indices (plated, HET_FULL, merges, payload), with the
  ρ_dec label (`t1.reference`; the generator's v2 columns are used when present).
  - Each episode labels self plus 7 random candidates for training.
  - The direct-head loss is weighted 3× on decision-relevant candidates: a faulted wheel, or ρ_dec within 0.1 of τ.
    Cal and test are not reweighted.
- **Job.** 8 chunks of 120 s generation + 120 s training. Training time is the same for every architecture
  (compute-matched), with a cosine LR over training time.
  - Checkpointing: `ckpt.pt` between units, `ctx.save_state` inside a unit.
  - Then eval-cal / test / H3 / H4 on the generator's episodes 0–299 of each split. These are the same eids as the
    other roles.
  - Evals give direct and compositional (100 cal/test episodes, S = 16) quantiles and per-wheel posteriors
    (`wheels_<split>.npz`).
  - Generation goes through `cfcap.gen.simrun`, whose compile→graph fallback covers the missing C compiler on boxes.
    The model and compose paths do not use torch.compile.
- **Estimate.** 54 min/job → 43.2 box-h → **$7.78** at $0.18/h, plus about 10 min boot per box. With 8 boxes that is
  about 5.4 h wall; seeds 2–3 (shards 16–47) can be cut first.
  - The training units are wall-clock budgeted, so their unit_s is fixed by construction. The eval units are an
    estimate.
  - The canary (`--smoke`: one unit, a 20 s chunk plus every eval split at 12 episodes, about 7 min) is the first run
    of the eval path on a box.
- **Questions each comparison answers:**
  - **Diversity scaling (C6, F2):** n2 → n4c → n4 → n6c → n6 on test, H3 (unseen topology) and H4 (N 7–10), with
    seed ranges.
  - **Compositional vs direct (C5):** `st` compositional vs `st` direct vs `bb`, on cal/test, overall and on the
    relevant stratum.
  - **Attention vs pooling:** `st` vs `ds`.
  - **Capacity:** `stl` vs `st`.
  - **Deep ensemble:** the 3 `st` seeds.
  - **Amortized vs SMC (C3/T3):** `st` n6 rows join the baselines' SMC rows on shared eids. Latency comes from
    `infer_ms_per_episode` and `comp_seconds`. T2 needs stage 2: the weights are in `weights.pt` for the stress role.
- **Export.** `scripts/models_export.py <pulled run dirs>` writes `results/models/paper/dec_models.csv.gz` with
  amortized_{marg, sim, comp}, blackbox_gnn and ensemble rows, with lower bounds from T1 cal at α = 0.1. It also writes
  `diversity.csv` and `latency_models.csv`. The decision-relevant stratum goes to
  `results/models/strata/diversity_relevant.csv`, outside the paper glob, so it is not double counted.

## Stage 2: `scripts/campaigns/models_eval2.json` (review round 0 fix 5; validated, needs the b1 outputs)
The b1 train jobs record per-row eids and timing, but not true end-to-end latency. Their `infer_ms_per_episode` is
batched; their `comp_seconds` covers composition only, on 100 episodes. So stage 2 does the following.
- **models-eval2** (`cfcap/models/eval2.py:TASK`; 10 jobs, 2.9 box-h, $0.52).
  - It loads the pulled weights once. They are staged with `models_export.py --stage-weights
    $CFCAP_OUT/compute/stage2/models-b1-weights`.
  - It regenerates generator cal 0–1999, test 0–1999 and H3/H4 0–499 by (plate, split, index): the same eids as
    gen-ref2 and the baselines.
  - It runs the six n6 models: st × 3 seeds, bb, ds, stl.
  - **Timing per episode.** `sec_direct` is batch-1 synchronised wall time, observations → posterior → direct head
    for every candidate. `sec_comp` adds the compositional rollouts of every candidate (S = 16, st-n6-1). Both are
    end-to-end per decision.
  - **Export.** `dec_models.csv.gz` with `seconds` = end-to-end, keyed by `eid`/`cand`/`leg`, so it joins the
    baselines on eid. Also `latency_models.csv` (`end_to_end` = 1, hardware = GPU name), and the T1-cal constants
    `results/models/calib/calib_<tag>.json`.
- **models-eval2-ext** (plan §2b/§2c; 14 jobs, 2.9 box-h, $0.52).
  - It runs the same task on the generator's new splits: H4b (11–16 modules), H1 (an unseen module type), H2 (unseen
    type combinations) and the confirmatory split. The models were trained on mecanum modules only, ≤ 6 per assembly.
  - The split names are placeholders until `docs/board/generator.md` defines them. The task exits 3 on a missing
    split, and `conf` must be renamed to the generator's name for the confirmatory split.
  - Module types enter the direct model only through wheel geometry, via `gendata._build` when the generator carries
    `mod_type`.
  - The compositional route now builds candidate latents on the generator's own plant latents
    (`compose.sample_from_episode`). Module-type physics is kept, and the unknown domain, motor spread, yaw errors and
    faults are replaced by prior or posterior draws. This is the test of whether composition degrades less than the
    direct head on H1/H2.
  - `scripts/models_scaling.py` writes `results/models/paper/scaling.csv`, from the test, H4 and H4b rows plus the
    per-split rows:
    - coverage, conformal entry coverage, certified fraction and unsafe commit, against parent N = 1…16 and per
      split;
    - direct vs compositional, overall and on the decision-relevant stratum;
    - episode-bootstrap 95 % CIs (B = 1000).
- **models-t2** (stage 3; 8 jobs, 2.1 box-h, $0.37).
  - Runs stress's `t2_eval --mode evaluate --predictor cfcap.models.predict:predict` on the T2 test episodes. It uses
    st-n6-1's weights plus `calib.json` (T1 cal only).
  - Merges are refused there, because only the parent's T2 observation exists.
  - Together with stage 2, this fills the {SMC, amortized} × {T1, T2} grid on the same eids.
- **Not mine:**
  - (b) matched-compute SMC: baselines runs SMC at a wall budget equal to `sec_comp`.
  - (c) capxfer on the same eids: baselines' capxfer_cert.

## Needs from other roles
- **generator:** the ref-split shards (cfcap-gen-ref2) for a stage-2 eval on 20k cal/test, so calibration is not limited to 300 episodes.
- **stress:** T2 evaluation of the pulled `weights.pt` (stage 2).
- **baselines:** SMC and capxfer on the same generator eids (cal 0–1999, test 0–1999). Also a matched-compute SMC whose
  wall budget per decision equals our median `sec_comp` (from models-eval2).
- **generator:** the split names and the `mod_type` column (with `episode.build(..., types=)`) for H4b, H1, H2 and
  the confirmatory split.
- **stress:** T2 simulate outputs staged at `$CFCAP_OUT/compute/stage2/b1-t2eps` for models-t2.
- **paper:** filter by `stratum` if `strata/` is used.

## Next step
1. Submit the campaign and check the canary.
2. Pull the runs and run `models_export.py`.
3. Run a stage-2 eval of the n6 models on the ref splits, plus T2.
- OPS/models (from generator, ebba94b): split names are `H4b`, `het_train`/`het_cal`/`het_test`, `H1`, `H2`, and the confirmatory twins `conf_<split>` (rename `conf` → `conf_test` / `conf_H4` / `conf_H4b` / `conf_het_test` / `conf_H1` / `conf_H2`). `mod_type` is a per-module int8 column (0 mecanum, 1 conventional, 2 omni). `cfcap.gen.episode.build(..., types=)` exists and sets `asm.meta["wheel_type"]`; w_hat/r_hat stay mecanum, so the geometry must read the type from meta. Details are in docs/board/generator.md, "Broadening splits".
