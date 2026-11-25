# Study 09 "cfcap": Can I Trust My Next Body? Certified counterfactual self-models for reconfigurable robots

## Context
capxfer (study 05, *Hidden by Morphology*) certifies, from assembly probes, a **lone** module's capability after a
wheel fault. It is safe (unsafe-go ≤1.3 % in sim) but conservative: it certifies only 58–75 % of capable plated modules.
It never predicts a different *multi-module* configuration, never predicts merges, and never decides whether to
probe further. The owner wants a more ambitious study that does all three:
- a **simulation-only ML paper** that can stand alone, using today's compute (owner, 2026-11-25: use the ~$19.5 credit
  plus a $15 top-up; raise the rented-box cap to 8 for today; 6 modules exist; firmware accepts per-wheel speed/duty
  commands);
- **new field blocks** for the next modular session. The 11-24 collection already ran, so these are additions, not
  amendments.

It traces to the owner's own ideas (`inputs/ORIGINAL_IDEAS_transcripts.md` #24: "understand what it can do, recognize
what it cannot do, determine whether changing its configuration is worthwhile"; history §3: "GNN capability learning,
active capability identification, held-out configuration families").

**Plain application.** A team of docked robots must split or merge for its next jobs. Before it does, it knows which
new shapes it can trust with which job. When it can't tell, it runs the one short test that settles that decision.

## 1. What already exists (reuse; do not redo)
- **capxfer theory** (`~/worktrees/capxfer-cluster/capxfer/docs/THEORY.md`, not yet in the RSS paper):
  - Prop 2: masking at rate 1/(n−1);
  - Prop 3: reveal-on-split, and diagnoses that don't transfer;
  - Prop 4: a single 22.5° probe covers the P1+P5 blind spot;
  - Thm 5: a compositional sequential certificate for any child, with no child-family calibration.

  **So "information about each module shrinks with N" and generic Fisher-range identifiability are NOT new
  contributions.** Cite them as foundations.
- **capxfer sim** (`capxfer/sim.py`, `assembly.py`):
  - batched PyTorch planar plant: tanh slip, roller drag, PI firmware, latency, sensors;
  - θ identified from the corpus, per plate state (`~/.cache/modular-runs/2026-11/capxfer/sysid/theta_*.json`);
  - arbitrary polyominoes via `assembly.build` / `random_assembly`;
  - throughput about 1–2.5 M 29-s trials/min on the 4070 Ti S;
  - limits: rigid docks, least-norm normal loads, motor/μ/battery shared across wheels, `simulate_multi` needs equal
    wheel counts.
- **capxfer self-model and certificate** (`corpus/selfmodel.py`, `capability.py`, `scoring.py`): the TCN + Mondrian
  conformal + lookup-table certificate. **It is our main baseline.**
- **crane kernel** `~/crane_ml/engine/kernels/mecanum.py` + `domains/modular/`: an independently implemented array-API
  plant (numpy/numba/JAX; root-finding wheel speeds; padded mixed assemblies). capplan truth
  (`platform-modular/Tools/Fleet/capplan/truth/model.py`) adds latch compliance, load transfer and planar compliance.
  capplan `configs.py` provides held-out morphology families H1–H7.
- **Compute.**
  - Free cluster CPU workers (innovation 8 fastsim + 2 py_command slots, paxton, harmony, tranquility) and the local
    4070 Ti S.
  - vast RTX 3090s via `compute/REQUESTS.md`, `torch`/`jax` profiles, ~$0.15–0.20/h. Jobs of 5–30 min that checkpoint
    themselves; outputs come back to `~/.cache/vast-out/`.
  - The queue is idle (785 held ALD jobs only).

## 2. Scientific direction (decided)
**Thesis.** A modular robot can decide, with a finite-sample bound on unsafe commits, which configurations it is
*not* in can take which task, or whether it must probe or split first. A configuration's capability depends on each
module's operating envelope at that configuration's own operating points. The current assembly can be driven so that
it reveals exactly the envelope regions the decision needs.

**The corrected mechanism** (from the design critique; Stage A verifies it):
- Per-wheel `pre_duty` under PI is a torque readout, which is why the real weak pair was caught 43/43.
- A weak wheel is a **speed ceiling** (g·ω0), not a torque deficit on this floor. It is detectable when probes reach
  its duty clamp, or through its 1/g duty ratio against its siblings, but only if per-wheel motor spread is smaller
  than the fault.
- Truly hidden quantities:
  - faults inside the motor spread;
  - coast vs brake below the back-drive threshold;
  - roller channels the parent's feasible twists never excite;
  - μ and normal load at the child's operating points (normal loads change on split or merge);
  - internal integral "fights" from wheel yaw and radius errors in large rigid assemblies.

**Contributions to aim for** (ranked; Stage A decides which survive):
1. **Operating-point coverage theory (partial identification under shape constraints).** A child's capability on a
   decision depends on envelopes at operating points O(child, d). The parent can reach only O(parent, u): wheel speeds
   lie in the 3-D twist image, torques are friction-bounded.
   - Sharp identified interval for c(m′); ambiguity is zero iff the probed points dominate the child's points
     (monotone envelopes).
   - Two-point lower bound: any certificate's unsafe-go rate is at least the prior mass of undominated,
     decision-critical envelopes.
   - Minimum-time covering probe as an LP over per-module twists and **per-wheel antagonistic commands** (now
     confirmed available).
   - Corollaries: when "reconfigure to see" (a diagnostic split) is worth its time cost; when merges are plug-in.

   This is partial identification with a decision cost. It goes beyond Cristofaro & Johansen's fixed-actuator FDI and
   beyond capxfer Props 2–4.
2. **Simultaneous counterfactual certificates over the candidate family M(m)**: every connected child, plus merges with
   other assemblies.
   - Max-normalized conformal score, so picking the best-looking candidate keeps the guarantee.
   - Conformal risk control on unsafe-commit loss, with costed abstain/probe actions.
   - A compositional per-module route (capxfer Thm 5 extended to merges) gives the only guarantee that survives
     held-out morphologies.
3. **Decision-aware probing.** Probes are chosen by decision-EVOI (expected reduction in decision risk minus time
   cost) from a library: v2 legs, clamp ramps, module-differential twists, per-wheel tug-of-war, diagnostic split.
   Baselines are fixed, EIG and EPIG. It becomes **central only if A2/A3 pass** (≥5 % of decisions ambiguous after
   v2; a targeted probe removes ≥30 % of them). Otherwise it is secondary.
4. **Learning where it earns its place** (the ML claim is tested, not assumed):
   - an amortized, morphology-conditioned posterior over module latents: a set transformer over wheel tokens (TCN
     encodings + module/position embeddings) with a mixture head (discrete mode × continuous gain), plus per-module
     μ/heterogeneity latents;
   - composed through physics: a quasi-static LP for speed, a short sim rollout for labels;
   - it must beat GPU-SMC with the simulator as likelihood on **robustness to misspecification (T2)** and on
     **decision latency inside the probing loop**. If SMC wins on both, the paper becomes "physics Bayes + conformal
     decisions" and the word "amortized" is dropped.

**Not claimed:** a coverage guarantee for held-out morphologies or for sim→real (reported empirically); "scaling laws"
of masking (already Prop 2).

**Paper strategy.**
- One sim-only paper: *"Can I Trust My Next Body? Certified Counterfactual Capability for Reconfigurable Robots."*
  Target **ICML 2027 (abstract Jan 14, paper Jan 22)**, with an arXiv preprint as soon as Stage C holds. Fallback:
  TMLR or RA-L.
- Active probing becomes its own follow-up only if it outgrows a section.
- The physical version (new field blocks + corpus cross-layout check) targets CoRL 2027 (~May, unverified) or
  IROS/RA-L (Mar 1).
- capxfer (RSS Stage 1, Dec 4) is untouched. It is cited as the hardware precursor. The cfcap certificate later
  becomes the capability layer program 04's planner consumes, but this paper is not a fleet-optimization paper.

## 3. Formulation (for the paper and code)
- **Modules and latents.**
  - Module k has latent θ_k: per-wheel mode ∈ {ok, weak(g), coast, brake}, motor spread, μ_k, battery voltage V_k.
  - A morphology m is a connected polyomino with yaws; normal loads follow from mass, payload and CoM.
  - Observations y ~ p(· | θ, m, u) for probe u: per-wheel encoders and pre_duty, plus one IMU per assembly.
- **Capability.** c(θ, m′) ∈ R^K: closed-loop achievable twist ratio ρ on K decision legs (surge/sway/yaw ±, diagonals,
  payload-laden legs). These quantities can be measured on hardware from TRUTH, so they match the real `ratio_*`
  columns.
- **Tasks and decisions.**
  - A task is a requirement vector r.
  - A decision chooses a partition/merge of the current assemblies, assigns pieces to tasks, or abstains, or probes
    (cost t_u), or does a diagnostic split (cost t_split, measured by 04's split/combine timing).
  - Unsafe commit: a piece assigned with c < r.
- **Guarantees reported.** Marginal over generator episodes (end-to-end over the probe policy); simultaneous over
  M(m); decision-level risk ≤ α; Mondrian per family only where n_cal ≥ 200; held-out H1–H7 and T2 empirical only.

## 4. Simulator architecture (multi-fidelity, reuse first)
- **T0, quasi-static (new, numpy/JAX, CPU, millions of assemblies per minute).**
  - Traction-limited allocation LP gives capability polytopes; noise-thresholded consistent sets; Fisher and
    operating-point coverage.
  - Uses: theory checks, phase diagrams over all polyominoes N ≤ 10, the probe-cover LP, fast EVOI.
  - Start from the QS solver used for the THEORY.md checks (locate it in capxfer `ideas/`).
- **T1, dynamic (wrap capxfer `sim.py` as a read-only dependency; don't edit the capxfer repo).**
  - Size-bucketed batches (no padding needed).
  - Fix the shortcuts the critique found: per-wheel and per-module motor/stall/no-load spread (±8 %), per-module V_batt
    and μ, randomized load split (instead of least-norm), slip-family DR (tanh / box / ellipse, s0 per wheel), sensor
    bias and dropout.
  - Payload and CoM offset; per-wheel command probes.
  - Labels come from closed-loop rollouts of each candidate on ~3 s decision legs, never from the T0 formula.
- **T2, independent stress plant.** crane `mecanum.py` (different numerics, JAX on GPU) plus capplan-truth effects:
  dock compliance and hook play, dynamic load transfer, roller-drag anisotropy (it broke every method in the retired
  I4 study), roller-locked wheels. Used only for evaluation and adversarial search, never for training.
- **Not used:** Unity, MRMN Gazebo (no relevant physics).
- **Kernel improvements** (analytic wheel Jacobians, JAX closed loop) are proposed to `crane-ml-tooling-cleanup` for
  crane main; the study does not depend on them.

**Code location.** A new study repo `~/cfcap/` with its own `.venv` (Python 3.12, torch CUDA). capxfer is imported from
`~/worktrees/capxfer-cluster/capxfer` on `PYTHONPATH`. Layout: `cfcap/{t0,t1,t2,gen,models,conformal,decide,probe,baselines}/`,
`scripts/`, `results/` (CSVs + analysis scripts), `paper/`. Outputs go under `~/.cache/modular-runs/2026-11/cfcap/`.

## 5. Stage A: cheapest kill tests (today, first ~4 h, local GPU + free CPU)
Targets are written in `cfcap/scripts/stageA_targets.md` before each run. Everything here is exploratory.

| Test | What | Kill / decision threshold |
|---|---|---|
| A0 throughput | T1 bucketed sim on the 4070 and one 3090; T0 per-core rate | sets Stage B/C sizing |
| A1 hidden at all? | Heuristic ĝ = sibling-median(u)/u_i vs the capxfer TCN on pair/square, 20k eps, with and without per-wheel heterogeneity | heuristic ≥95 % detection without heterogeneity → the weak-wheel story is trivial, so pivot to the truly hidden quantities; a drop >15 pts with heterogeneity → the sim leaked labels, so keep heterogeneity on |
| A2 decision ambiguity | T0: random 4–8-module parents, 1–2 faults; consistent set under v2; share of (episode, candidate, decision) that straddle τ, over children and merges | <5 % → probing is secondary; oracle consistent-set certificate gains <10 pts over capxfer's 58–75 % → attack the conformal slack, not the learner |
| A3 probe value | A2 rerun with v2 + clamp ramp, + module-differential twists, + per-wheel tug-of-war | no probe cuts the ambiguous share by ≥30 % → drop active probing as a headline |
| A4 merge composition | T1 truth for merged assemblies vs T0 composition from the true latents | <1 % decision error → merges are plug-in (state it so) |
| A5 misspecification | T2 vs T1 child ρ at matched θ | shift beyond the certificate margin on >5 % of decisions → misspecification robustness becomes the lead ML claim |
| A6 exact-posterior cost | sparse enumeration/SMC for N = 6, K ≤ 2 on the 4070 | sets whether SMC is a full baseline or a subsample |

**Go/no-go after Stage A** (one short note in `cfcap/docs/DIRECTION.md`): choose the lead among contributions 1–4.
Record the redirect in platform-modular `Docs/Research/DIRECTION.md`.

## 6. Stages B–D (adaptive; run concurrently once A0 is known)
- **B, data and models (today evening → day 2).**
  - Generator: parents of 1–6 modules (and N ≤ 10 for scale), ~30 candidates each (children + merges), heterogeneity,
    payload; per plate state (never pooled).
  - Sizes: 200k training episodes (~36 M short sims) plus calibration/test splits and H1–H7 held-out families.
  - Models: set-transformer posterior (3 seeds), black-box direct counterfactual GNN, deep ensemble, RLS/interval
    baseline.
  - Sweeps in parallel: architecture × training diversity (number of morphology families, max N) × heterogeneity.
    This is the "does diversity buy generalization to larger N" scaling curve.
- **C, baselines and certification (day 2–3).**
  - Baselines: nominal; RLS + intervals; GPU-SMC (200–500 test episodes); black-box GNN; ensembles; capxfer
    certificate; full-state oracle.
  - Conformal layers: marginal, simultaneous, compositional, weighted-shift.
  - The 2×2 grid {SMC-T1, amortized-wideDR} × {test T1, test T2}.
  - Metrics: certified fraction at matched unsafe-go, unsafe-commit rate, coverage (per family, H1–H7), set size,
    wall-time per decision.
- **D, decision layer and stress (day 3–5).**
  - Greedy decision-EVOI vs EIG / EPIG / random / fixed v2 / always-split.
  - DAD/TNDP-style amortized probing policy only if A3 passed.
  - Adversarial search (CEM over θ, morphology and payload) for confident-wrong certificates on T1 and T2.
  - Real cross-layout corpus check: predict single and pair ρ from square/L3 sessions on held-out sessions, plus the
    11-24 B1/B2 data once it lands (square→single counterfactual with the same injected fault).

**Compute plan** (filled in with A0 numbers before any rental):
- Rented, up to 8 × 3090 today, ≤ $35 total. Main uses: SMC baselines and decision-EVOI oracles (the costliest
  workloads, around 10⁶ sims per episode), the architecture/diversity sweeps (dozens of 30–60 min training jobs),
  adversarial search, and T2 JAX evaluation.
- Free CPU workers take T0 sweeps and the theory phase diagrams (`py_command`, `cloud-ok`).
- The local 4070 runs pilots (≤2 heavy processes, `nice -n 15`).
- Every job: 5–30 min, checkpoint and resume, columnar outputs + `manifest.json` under
  `/workspace/out/$CRANE_JOB_ID/`, `progress.json`. Analysis scripts exist before submission (REQUESTS rule).

## 7. Coordination actions (first hour of execution, in parallel)
1. **Compute session** (`vastai-gpu-cluster-integration`, owner of `compute/`): SendMessage with the owner's
   decisions:
   - `max_instances` 8 for 2026-11-25 only, then revert;
   - total ≤ $35 (the owner adds the $15 credit on vast; the credit floor stays $2);
   - interruptible only.

   Also send our needs: a `torch` profile with capxfer's deps; `inputs_rsync` of `~/cfcap` + capxfer source +
   θ JSONs (no corpus telemetry needed for sim-only work); a `gpu_command` lane on innovation if cheap. Add a
   `compute/REQUESTS.md` row after A0.
2. **Field-collection subagent** (general-purpose, background), brief below (§8).
3. **capxfer owner session**: notify it that cfcap imports capxfer read-only and cites THEORY.md. Ask it to flag any
   API changes. No edits to capxfer.
4. **crane-ml-tooling-cleanup**: one message proposing the kernel upgrades (analytic Jacobians, JAX closed loop,
   per-wheel heterogeneity). Not a dependency.
5. **Registry.** Add a "09 cfcap" section (status, deliverables, compute ask, data asks, owner decisions) and a
   shared-asset row for the T0/T1 generator. Record the owner decisions in platform-modular `Docs/Research/DIRECTION.md`.
6. A **critical-review subagent** after Stage A and again after Stage C. It gets the draft paper and results, not our
   conclusions.

## 8. Physical data collection (subagent brief; next modular session, 6 modules)
The subagent works from `datacollection/PLAN.md`, `05_capxfer.md` + runsheet, `02_03_04_planning.md`, the corpus
docs and `08_nlc.md`. It writes `datacollection/09_cfcap.md` (+ runsheet) in the README template and adds a
"next modular session" section to `PLAN.md`. Before editing, it messages the planning/capxfer sessions to avoid
conflicting changes.

**Reuse first.** The 11-24 B1 (lone faulted) and B2 (pair/L3/square, same faults) already give square→single and
pair→single counterfactual pairs, and split/combine timing gives the cost of a diagnostic split. Ask the owner where
the 11-24 data is.

**Candidate new blocks** (the subagent sizes, ranks and cuts them):
- **C1 split ladder** (must): 6-module parents (2×3, and an L/T shape) with 1–2 injected faults, g near the decision
  threshold (0.55–0.8), plus a coast/brake below back-drive. Probe the parent → we write predictions → run the same
  faults on children (pairs, L3, singles), with real physical splits on a subset to check that the split itself
  changes nothing.
- **C2 merge prediction** (must): modules probed alone or in pairs → predict a merged 4–6-module shape → build it and
  run decision legs.
- **C3 diagnostic probe library** (must): v2 vs clamp ramp vs module-differential twists vs per-wheel tug-of-war
  (firmware supports it), with the same faults. Gives the information value per second of each probe.
- **C4 payload / operating point** (should): lifter payload (2 kg) on the child vs the parent, which shows the
  operating-point shift.
- **C5 prospective decisions** (R3, should): the model picks probes and a partition/assignment; the owner executes
  the decision legs. Held-out confirmation.

**Log:** a morphology graph (module ids, grid positions, yaws), `chain_id` linking pre/post trials, which modules were
previously docked together, per-wheel command channel, fault spec, payload/plate, per-module battery voltage if
available, and decision-leg `ratio_*` columns (05's definition).

**Constraints:** no force or current sensing; one IMU per assembly; module-to-module docking not logged; plate state
fixed per day; shares the platform-time pool with 04 and 08. The sim-only paper never waits on this data.

## 9. Roadmap and dependencies
| When | Work | Depends on | Parallel with |
|---|---|---|---|
| Today, h0–1 | repo scaffold; coordination messages; data subagent launched | — | all |
| h1–4 | A0–A6 (T0 on CPU workers, T1 on local GPU) | scaffold | data subagent |
| h4–5 | go/no-go note; REQUESTS row; first rental | A0, A2/A3 | — |
| h5–24 | Stage B generation + training sweeps on ≤8 3090s; T2 port | go/no-go | theory write-up (contribution 1) |
| Day 2–3 | Stage C baselines, SMC, conformal layers, 2×2 grid | B models | adversarial search |
| Day 3–5 | Stage D probing, real cross-layout check, reviewer pass | C | paper draft |
| ≤ Dec 11 | arXiv preprint if C holds | | |
| Jan 22 | ICML submission | | field blocks C1–C5 |

**Earliest workloads to ship** (once the code exists, about 2 h): T0 phase-diagram sweep (CPU `py_command`, all
polyominoes N ≤ 10 × fault draws × probe sets); T1 dataset shards (`gpu_command`, 20-min shards, 1 M episodes each);
SMC oracle shards for calibration/test episodes.

## 10. Risks and fallbacks
1. **Weak faults aren't hidden** (A1 is trivial). Lead with truly hidden quantities (μ/normal load at child operating
   points, passive torque laws, unexcited rollers) and the operating-point theory.
2. **Reads as incremental over capxfer.** Lead with merges + simultaneous decision risk control + certificate tightness
   (58–75 % → ≥90 % certified at equal unsafe-go). Check against `capxfer/docs/NOVELTY_CHECK.md`.
3. **Physics Bayes beats the amortized model everywhere.** Becomes an SBI + conformal decision paper; amortization
   kept only for probing latency.
4. **Held-out morphology coverage collapses** (as in the retired I4). Lead with the compositional per-module guarantee.
5. **Sim-only rejected at robotics venues.** Target ICML/TMLR; add the real cross-layout corpus check; field blocks
   C1–C5 feed the physical paper.
6. **Simulator shortcuts.** Mitigations: heterogeneity, slip-family DR, T2-only evaluation, no wheel-index embeddings,
   permutation tests on module identity.

## 11. Verification
- Every scorer and conformal routine has unit tests (`pytest cfcap/tests`), including:
  - empty-set → refuse (the capxfer scorer-bug lesson);
  - simultaneous coverage on synthetic exchangeable data;
  - T0 LP vs closed-form polytopes for a single module;
  - T1 wrapper bit-identical to the capxfer `simulate` on shared cases.
- A small local pilot of each job class is required before its REQUESTS row; a 2-job canary per batch.
- Results are CSVs plus analysis scripts. Every number cites its file and is labelled exploratory or confirmatory.
- The H1–H7, T2 and confirmation seeds stay untouched until their targets are written.
- Data subagent output: `09_cfcap.md` follows the README template, with must-have hours stated and no column left
  undefined. Its ingest is tested on mock rows.
