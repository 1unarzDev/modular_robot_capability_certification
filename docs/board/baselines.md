# baselines (specialist board)

**Status (2026-11-25):** the six baselines are built and committed behind one interface, together with the runner task
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
