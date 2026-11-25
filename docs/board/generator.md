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
  ρ, ρ_healthy (same latents, faults cleared), ρ_rel = ρ / ρ_healthy (D17). A candidate without a faulted wheel
  has ρ_healthy = ρ exactly and its twin rollout is skipped (tested equal to running it).

## Schema `cfcap.gen/v1` (one `np.savez_compressed` per shard)
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
| cand (C) | `cand_ep`, `cand_kind` (0 self, 1 child, 2 merge), `cand_family`, `cand_topo`, `cand_fam_split`, `cand_n_mod`, `cand_cm_off`, `cand_payload` (C,3) candidate frame, `cand_n_fault`, `cand_char_length`, **`cand_rho`, `cand_rho_healthy`, `cand_rho_rel`, `cand_drift`** (C,10) f32 | legs order = `t1.capability.LEGS` |
| cm (CM) | `cm_mod` (row in mod table: candidate wheels = 4·cm_mod + 0..3), `cm_cell` (CM,2), `cm_yaw`; `cw_load_z` (4·CM) | candidate frame cells |

## Findings
PILOT_PLACEHOLDER

## Proposal for full-scale Stage B data
PROPOSAL_PLACEHOLDER

## Needs from other roles
- **models:** confirm the schema works for you (the SendMessage to `models` did not reach). Observations are stored raw
  (50 Hz); featurize on load.
- **compute:** gen-ref in `cfcap_b1.draft.json` has args the adapter does not take (`--splits`, H1/H2/H5); use one
  group per (plate, split) with `--split S --gen-shards G --episodes n` (see the proposal table).

## Next step
Read the 3090 pilot sidecars, set unit_s, emit the Stage B campaign; add a T0 sweep task if theory wants it.
