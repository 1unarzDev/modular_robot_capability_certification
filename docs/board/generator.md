# generator (specialist board)

**Status:** Stage B episode generator `cfcap/gen/` committed (fe24b9d, 3dd06e6); tests 18/18; 3090 pilot
`cfcap-gen-pilot` submitted; local pilot numbers below (exploratory).

Code: `cfcap/gen/{families,episode,simrun,shard}.py`, CLI `scripts/gen_shard.py`, runner adapter
`cfcap/compute_tasks.py:GEN_SHARD` (compute role), pilot `scripts/gen_pilot.py`, tests `tests/test_gen.py`.

## Held-out splits
Family = free polyomino (cells up to translation, rotation, reflection); its split is a fixed rule of the canonical form
(`cfcap/gen/families.py`, frozen table `results/generator/families.csv`), applied to parents **and** candidates.
- **H3 (unseen topology):** parents whose module graph is a junction tree (T, plus/star, or several junctions), 4–6 modules; train, cal and test never contain such a shape as parent, child or merge (capplan H3).
- **H4 (larger assemblies):** parents of 7–10 modules (train max 6) with children and merges up to 10 modules (capplan H4a).
- **H6 (unseen failure combinations):** train morphologies, but one module has ≥2 faulted wheels (plus, w.p. 0.5, one more elsewhere); training never has more than one faulted wheel per module (capplan H6).
- **H7 (unseen payload regime):** train morphologies with a 3–6 kg payload always present (train: none w.p. 0.5, else 0.5–2.5 kg) (capplan H7 task-family control).

capplan H1/H2/H5 (unseen module types, mixed drive families, interface regimes) have no counterpart: the 6 modules
are identical mecanum modules with one dock type.

## Episode (what one row of the ep table means)
Pure function of (plate, split, index): `cfcap.gen.episode.make_episode`. RNG = `default_rng([92511, plate, split,
index, stream])`, so any episode can be regenerated alone (for SMC baselines on test episodes) and shards are
reproducible. Plate states (plated / plateless θ, module mass 12.9 / 5.2 kg) are separate shards, never pooled.
- Parent: fixed polyomino uniform within the split's family set; sizes train 1–6 (w 0.05/0.1/0.15/0.2/0.2/0.3);
  module yaws U{0..3}.
- 1–2 other assemblies (1–3 modules, train families) with their own latents: merge partners.
- Domain: one capxfer DR draw around θ (`P.theta_dr`), shared by the episode's assemblies (same floor and firmware).
- Heterogeneity `HET_FULL`: per-wheel ±8 % and per-module ±4 % stall/no-load, per-module V_batt ±0.3 V and μ ±10 %,
  mounting-yaw error, random null-space normal-load share (redrawn per candidate), IMU biases, 1 % wheel dropout.
- Faults: modes ok/weak(g~U(0.3,0.95))/coast/brake (0.5/0.25/0.25 among faults); parent 0/1/2/3 faults w.p.
  0.2/0.4/0.3/0.1, others 0/1/2 w.p. 0.5/0.4/0.1; at most one per module (except H6).
- Payload: none w.p. 0.5 else U(0.5, 2.5) kg on a random parent module's deck (offset ±0.08 m); a candidate carries it
  iff it keeps the host module.
- Probes: v2 (capxfer P1/P2/P5 ± legs with gen_routine jitter, 14.7 s) on every assembly; optional `clamp`
  (P6x/P6y, 11.5 s) and `tug` (per-wheel diagonal tug-of-war ±0.40 m/s, 4 × 1 s holds, 9.7 s), `--probes v2,clamp,tug`.
- Candidates: the parent itself (kind 0), every connected child (kind 1, ≤ 40, random subset above), ≤ 4 merges per
  other assembly (kind 2: parent or a child + the whole other, random face-touching placement); restricted to the
  split's allowed families and sizes (train-like ≤ 6 modules, train families only).
- Labels: closed-loop T1 rollouts on the 10 decision legs of `cfcap.t1.capability` (3 s, window [0.8, 2.95] s):
  ρ (= ρ_abs), ρ_healthy (same latents, faults cleared), ρ_rel = ρ / ρ_healthy (D17, secondary: capxfer comparison
  only). A candidate without a faulted wheel has ρ_healthy = ρ exactly and its twin rollout is skipped (tested equal).
- **Decision label (coordinator decision 2026-11-25): ρ_dec = ρ_abs / ρ_ref(shape, leg)**, τ applies to ρ_dec.
  ρ_ref = ρ of the nominal healthy plant of that shape (plate θ as a point domain, no DR, no spread, no faults, no
  payload, least-norm loads), `cfcap/gen/rho_ref.py`. Shape key = translation-normalised cells + module yaws. A fixed
  table: computed once per key, cached, and written as `rho_ref_<plate>.json` next to the shards (loaded before and
  saved after each shard, so all shards share it). Shards from schema v1 (no ρ_dec) can be completed after the fact:
  ρ_dec = cand_rho / ρ_ref(key of the candidate's cm_cell/cm_yaw). Error from faults inside the motor spread is
  irreducible and reported as such.

## Schema `cfcap.gen/v2` (v1 + `cand_rho_ref`, `cand_rho_dec`) (one `np.savez_compressed` per shard)
Path `<out>/<plate>/<split>/shard_<k:05d>.npz` + `shard_<k>.json` sidecar (sha256, timings) + `manifest.json`
(config incl. every prior, `dom_keys`, `legs`, `topologies`, codes). Shard k holds episodes [k·n, (k+1)·n), n fixed per
split. Resume: a shard with a matching sidecar sha256 is skipped. Load: `cfcap.gen.shard.load(path)`;
`shard.episode(D, e)` gives row views. Ragged tables are CSR: `X_off` has len rows+1.

| table (rows) | arrays | notes |
|---|---|---|
| ep (E) | `ep_index` i64, `ep_split` i8, `ep_plate` i8, `ep_dom` (E,17) f32, `ep_payload` (E,3) f32 mass,x,y parent frame (mass 0 = none), `ep_payload_host` i8, `ep_asm_off`, `ep_cand_off` | dom keys in manifest |
| asm (A) | `asm_ep`, `asm_role` (0 parent, 1–2 others), `asm_family` i64, `asm_topo` i8, `asm_fam_split` i8 (0 train, 3 H3, 4 H4), `asm_mod_off`, `asm_gyro_bias`, `asm_acc_bias` (A,2), `asm_char_length` | |
| asm obs | `obs_<kind>_imu` (A,T,3) f16 gz,ax,ay; `obs_<kind>_cmd` (A,T,3) f16 commanded twist; `obs_<kind>_amp`, `_lead` f32 | 50 Hz, T = 735 (v2) |
| mod (M) | `mod_asm`, `mod_cell` (M,2) i8 grid cell, `mod_yaw` i8 (×90°), `mod_vbatt_off`, `mod_mu` | module m's wheels are rows 4m..4m+3 (fl, fr, rl, rr) |
| wheel (W=4M) | `wheel_mode` i8 (0 ok, capxfer WEAK/COAST/BRAKE codes), `wheel_g`, `wheel_stall`, `wheel_noload`, `wheel_yaw_err`, `wheel_load_z` | truth latents, not robot-visible |
| wheel obs | `obs_<kind>_duty` (W,T) f16 pre_duty; `obs_<kind>_dcnt` (W,T) i16 encoder counts per 20 ms (speed = dcnt/6533·2π/0.02); `obs_tug_ref` (W,T) f16 | robot-visible |
| cand (C) | `cand_ep`, `cand_kind` (0 self, 1 child, 2 merge), `cand_family`, `cand_topo`, `cand_fam_split`, `cand_n_mod`, `cand_cm_off`, `cand_payload` (C,3) candidate frame, `cand_n_fault`, `cand_char_length`, **`cand_rho_dec`** (decision label), `cand_rho` (ρ_abs), `cand_rho_ref`, `cand_rho_healthy`, `cand_rho_rel`, `cand_drift`, all (C,10) f32 | legs order = `t1.capability.LEGS` |
| cm (CM) | `cm_mod` (row in mod table: candidate wheels = 4·cm_mod + 0..3), `cm_cell` (CM,2), `cm_yaw`; `cw_load_z` (4·CM) | candidate frame cells |

## Findings
All exploratory. Local RTX 4070 Ti SUPER, plated train, engine compile, block 100, schema v1 (before ρ_ref):
`scripts/gen_pilot.py --episodes 300` -> `results/generator/pilot.csv`, `pilot_blocks.csv`, `pilot_bytes.csv`.
- **Throughput:** 17.9 episodes/s steady (blocks 2-3); 5.7 /s over the whole 300 because of ~40 s first-block
  torch.compile warm-up (block 0: probe 19.7 s, label 20.5 s; later blocks 1.5-3.7 s and 2.5 s per 100 episodes).
- **Bytes:** 68 KB/episode compressed (110 KB raw); observations are 94 % of raw (v2 at 50 Hz, f16 duty + i16 counts).
- **Label cost share:** 46 % of steady wall (probes ~45 %, the rest episode construction and assembly). Label runs:
  307 leg-trials/episode, 3,810 leg-trials/s including CPU prep (prep 19 % of label time), 21 % padded rows.
- **Candidates per episode** (`results/generator/candidates_by_split.csv`, 300 episodes each): train 19.7 (median 18,
  p90 29, max 46; 12.6 children + 6.0 merges + self), H3 25.4, H4 44.6, H6 20.7, H7 19.9. 56 % of train candidates
  contain a faulted wheel. Held-out-family candidates: H3 15 %, H4 27 % of candidates.
- **Labels:** ρ_rel in [0.07, 1.01]; only 2.3 % of (candidate, leg) ρ_rel < 0.8 (faults mostly hit 1 of many wheels).
  ρ_dec numbers come from the 3090 reference run (pilot2 shards are v1: compute post hoc).
- **Families** (`results/generator/families.csv`): 56 free polyominoes N ≤ 6, 37 train, 19 H3 (1 at N=4, 4 at N=5,
  14 at N=6); fixed shapes at N=6: 124 train, 92 H3.
- Tests `tests/test_gen.py` (18 + ρ_ref test): determinism (episode and shard, batch-composition invariant labels to
  1e-4), splits disjoint (families and every candidate of train-like episodes), H3/H4/H6/H7 rules, label range,
  healthy candidates ρ_rel = 1 exactly with the twin run, labels = `t1.capability` to 2e-4, resume and refusal of a
  mismatched episodes_per_shard.
- Not measured yet: whether yaws change ρ_ref (if not, the key drops yaws and the cache hit-rate jumps, saving the
  ~30 % ρ_ref overhead). The reference run answers it (`rho_ref_<plate>.json`, keys differing only in yaw).

## Proposal for full-scale Stage B data
Training data are generated on the fly by the models (same deterministic generator, train split stream), so only the
fixed **reference splits** are persisted: `scripts/campaigns/gen_ref.json` (18 rented-torch jobs, `make_jobs emit`:
submittable).

| per plate state (plated, plateless; never pooled) | episodes | shards × eps | min/job | rationale |
|---|---|---|---|---|
| cal | 20,000 | 2 × 10,000 | 17.6 | ~390k candidates; simultaneous (max-over-candidates) score is per episode, so n = 20k gives quantile SE ~0.15 % at α = 0.05 and ≥ 200 per Mondrian train family |
| test | 20,000 | 2 × 10,000 | 17.6 | same size as cal; unsafe-go at 1 % resolved to ±0.14 % |
| H3 | 5,000 | 1 × 5,000 | 12.3 | ~260 episodes per held-out family (19 families): per-family coverage SE ~1.4 % |
| H4 | 5,000 | 2 × 2,500 | 12.9 | 7-10 modules, 45 candidates/episode (3.3× label cost) |
| H6, H7 | 5,000 each | 1 × 5,000 | 10.1 | coverage SE ~0.3 % at α = 0.05 |

- **Cost:** 4.3 3090-h, **$0.77** at $0.18/h (+ ~10 min boot per box). Rate assumed 14 train-like episodes/s on a
  3090 (coordinator), /1.3 for the ρ_ref rows, H3 ×0.77, H4 ×0.36 by candidate modules; overhead 150 s per job.
- **Storage:** ~68 KB/episode train-like, ~150 KB H4 → ~4.3 GB per plate, **~8.5 GB** total.
- If models want a persisted train set after all: 200k plated = 4 h, $0.73, 13.6 GB (20 shards × 10k).
- **Exchangeability of cal and test:** both are i.i.d. episodes of the same rule (`RULES["cal"] is RULES["test"]`),
  differing only in the RNG stream (split code in the seed). The probe routine is fixed inside the episode, and no
  episode is filtered, deduplicated or reweighted after generation, so (observations, candidate set, labels) are
  exchangeable across cal and test, and so is any score computed from them. Rules for consumers: the exchangeable
  unit is the **episode** (scores maxed over its candidates), so never split cal by candidate; never use cal for
  model selection or early stopping (hold out train-stream episodes for that); calibrate per plate state; for
  Mondrian per family, the family is a deterministic function of the episode, so within-family exchangeability
  holds too. ρ_ref is a fixed deterministic table, identical for cal and test.

## Needs from other roles
- **models:** confirm the schema works for you (the SendMessage to `models` did not reach). Observations are stored raw
  (50 Hz); featurize on load.
- **compute:** gen-ref in `cfcap_b1.draft.json` has args the adapter does not take (`--splits`, H1/H2/H5); use one
  group per (plate, split) with `--split S --gen-shards G --episodes n` (see the proposal table).

## Next step
Read the 3090 pilot sidecars, set unit_s, emit the Stage B campaign; add a T0 sweep task if theory wants it.
