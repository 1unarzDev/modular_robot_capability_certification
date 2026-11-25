# research lead (science state, frontier ideas, next experiments)

## NEXT (ranked; the coordinator dispatches from here). Updated 2026-11-25 20:58Z
Hypothesis numbers follow the ledger. My proposals H11–H14 of 20:15Z are renumbered **H12–H15**, because the
ledger's H11 is the conformalised screen.

**N1. Fix the inference-difficulty abstention (decides H11; owner: decision; CPU, offline, $0, < 1 h).**
- **Evidence** (`results/decision/t0_k5_meta.csv`, `t0_k5_summary.csv`, my analysis): after the screen fix,
  35 % of episodes abstain.
  - `unknown=budget` fires on 74/240 episodes, `empty_screen` on 11.
  - 85 % of *healthy* episodes abstain. A healthy body is consistent with many near-healthy hypotheses (weak
    g ≥ 0.85 ≈ motor spread), so the 2048-particle cap truncates.
  - Useful commits fall to 65 % (raw/sim). The guard is sound but throws away the easiest bodies.
- **Spec.** Drop budget-abstention as a hard rule. Keep `empty_screen` abstention. Then:
  - (a) end-to-end calibration of the full screened/truncated procedure, which is marginally valid by
    exchangeability whatever the truncation;
  - (b) a **Mondrian quantile by observable inference difficulty**: bins on truncated posterior mass {0, (0, 1e-3],
    > 1e-3} × n_screen tercile, with a ≥ 60-episode-per-bin floor (merge bins otherwise).

  Rerun `scripts/decide_pilot.py` on the same 240 + 360 T0 episodes; certifiers `sim0.1`, `simM0.1` (Mondrian) and
  `raw`.
- **Bar.**
  - `simM0.1`: per-bin ep_any_unsafe ≤ 0.1 (episode bootstrap upper CI ≤ 0.15);
  - healthy-episode abstention ≤ 5 %;
  - useful commits ≥ 95 %.
  - If (b) fails its bins but (a) holds marginally, report the marginal claim only. H11 then becomes "end-to-end
    calibration absorbs approximate inference", and the screen threshold is a compute knob, not a safety knob.

**N0. The confirmatory pre-registration is written** (`docs/CONFIRMATORY.md`, 8ffc33a + amendment 8ff2954). The
generator emits `conf_*` only after this file. The paper role writes `scripts/analysis/confirmatory.py` and tests it
on mock data before 01:45Z.

**N2b. Gate statistic choice (owner: theory + stress; decide by 22:15Z, CPU ≤ 30 min, decides H13).** Trace misfit D
vs the capability-tracking residual r_track = |measured probe ratio − T1-predicted ratio|.
- Use exploratory t2-knee / A5m episodes, fixed-reference positives, ≥ 3-module children.
- AUC with an episode bootstrap CI.
- Commit the one with the higher lower-CI bound. CONFIRMATORY §3 allows exactly one.

**N2. Rerun E1 at full size and refuse verdicts on smoke data (decides H12/H13; owner: theory; ops submits).**
- **Evidence.** `results/theory/gate_cert.csv` has **4 episodes per law** (n_episodes = 4), with q_gate = −1 and an
  infinite margin. Every calibrated certifier (`par`, `pi_cal`, `gate`, `nominal`) certifies 0 %, so
  `gate_verdict.txt`'s "KILL True" is an artefact and is **void**.
- **Spec.**
  - (1) Commit `cfcap/theory_task.py` (GATE) + `cfcap/t0/envelope.py` (the ops alert says they are uncommitted, so
    the run is HELD).
  - (2) `theory_gate.py analyze` must write "INSUFFICIENT" unless every (law, severity, probe) cell has
    ≥ 100 episodes and finite q.
  - (3) Run the full 1,200 parents on the ride-along plan (6 jobs).
- **Hint from the smoke, to pre-register now.** Under knee severity 2 the raw certificate had 0 unsafe in 28
  incapable decisions. If the full run confirms that T0's knee law does not bite, the regime-2 evidence must come
  from T2 `all+motor` (N3), not from T0.
- **Bar.** As pre-registered (theory board §E1).

**N3 (DONE 20:50Z; see critique log). Fixed-reference relabelling of the misspecification tests (decides H12, silent extrapolation; owner: stress;
CPU, offline if ρ_abs is stored, else ≈ 1.5 CPU-h).**
- **Confound.** A5m reports 0 % flips on healthy candidates under every condition. But each plant's ρ_dec is divided
  by *that plant's own* nominal healthy reference.
  - Under the motor law the T2 reference itself drops (square yaw 0.892 → 0.821, −8 %), so the label absorbs the
    misspecification for every body that looks nominal.
  - The robot only has its T1 reference table. A task requirement is absolute: it is τ·ρ_ref^T1, not τ·ρ_ref^T2.
- **Spec.**
  - Re-score `A5m_rows.csv` (and A5, A5y, E2) with ρ_dec^fixed = ρ_abs^T2 / ρ_ref^T1(shape, leg).
  - If ρ_abs is not stored, recompute ρ_ref^T2 per (shape, leg) with `t2.capability.rho_ref` (8 nuisance draws, as
    now) and multiply back.
  - Report flips and |d| > 0.05 by shape × leg type × dominance class, both labels side by side.
- **Bar (H12).** Under all+motor with the fixed reference:
  - flips on *undominated* decisions (diagonals; yaw of ≥ 3-module children) > 1 % (upper CI);
  - flips on dominated surge/sway decisions ≤ 0.5 %.
  - If both hold, regime 2 exists in a plant we did not tune, and the gate experiment moves to T2.
  - If undominated flips stay ≤ 1 %, H12 is FALSIFIED. The paper then says "extrapolation through the motor law is
    safe for these misspecifications" (a robustness result) and drops the gate as a method.

**N4. Merge certificate with inferred latents: the actual headline (decides H14; owner: baselines + stress; local
GPU ≤ 1 h or 1 box-h ≈ $0.2).**
- **Evidence** (`results/stress/A4_merge.csv`).
  - δ_comp flips are 0.11 % (CI 0.03–0.24) in T1 and 0.13 % (0.05–0.25) in T2: merges are plug-in at the decision
    level (H14 part 1 SUPPORTED).
  - The T1-calibrated per-pair margin miscovers 43–47 % of pairs in T2.
  - **My check** (A4_merge_rows): the mean signed T2 − composition error is only −0.002 per leg, and de-meaning by leg
    leaves 46 % miscoverage. So it is *spread*, not a reference offset. The spread sits in 2-module merges (sd 0.047
    vs 0.008 for ≥ 3).
  - The size-aware margin (`q_pert_by_nmod`, from T1 perturbations) gives T2 miscov 0.02 with CF 98 %.
  - All of this uses **oracle latents**, so it measures only δ_comp.
- **Spec.**
  - Route (c): enum_known (or nuisance-SMC) posterior from each G_j's *own* v2 probe (the assemblies probed
    separately), composed to every merge.
  - Margins: (i) the T1 cross-fit conformal margin and (ii) the size-aware margin.
  - Same 500 pairs × 2 arms; truth T1 and T2 (fixed reference per N3). Also add `copy_parent` (min of the two
    parents' ρ) and `detector_gated` as merge baselines.
  - Rows go to `results/stress/paper/dec_merge.csv` (method `comp_enum`, `comp_enum_size`).
- **Bar.**
  - CF on capable decision-stratum merges ≥ 85 % at realised unsafe ≤ 1 %;
  - per-pair miscov ≤ 0.1 in T1 (cross-fit) and ≤ 0.15 in T2 with the size-aware margin;
  - CF beats `copy_parent` by ≥ 10 pts.
- **Note.** State the size-aware margin as an *empirical* robustness margin (it comes from T1 perturbations), not a
  guarantee.

**N5. Child-rate yaw probe, made feasible on the parent (decides "the theory names the probe" for yaw; owner:
stress; ≈ 1 CPU-h).**
- **Evidence.** Clamp misfit AUC is 0.82 pooled, but 0.27–0.33 on ≥ 3-module yaw: inverted, and the knee is never
  visited. A5y (`yawc`) is running *on the child body*, so it probes the decision point itself and is an **upper
  bound**, not a parent probe.
- **Spec.** Add `wheelrate`: a parent probe built from per-wheel speed commands (the firmware supports them).
  - Each wheel's reference ramps to the maximum |ω| that any ≥ 3-module child of this parent needs on its yaw legs
    (closed form, Prop. 2b).
  - The commands are split into antagonistic pairs, so the parent's net twist stays small (least squares on the
    twist).
  - Misfit as D_imu + D_duty. Same 600 A5y episodes, AUC on target (a).
- **Bar.**
  - `wheelrate` AUC ≥ 0.75 → the claim stands with a feasible probe;
  - `yawc` ≥ 0.75 but `wheelrate` < 0.75 → claim only "the child's own yaw test" (i.e. after a split), which is the
    split-to-see case;
  - both < 0.75 → drop the yaw-probe claim.

**N6. E3 trivial / identified-set baselines (decides H15; owner: baselines).** Unchanged spec (§4 E3). Waiting for
the generator-eid b2 rows. Re-check after the scorer-audit fixes. The `base.py` AND-vs-OR stratum bug means every
earlier stratum number from baselines is void.

**N7. Size-aware two-simulator margin for singles and pairs (F4; owner: stress; offline).** It now has a template:
the E2 `q_pert_by_nmod` approach. Apply it to children: q by cand_size from T1 perturbations, scored on T2 with the
fixed reference. **Bar:** T2 single-child unsafe ≤ 2·U* at CF loss ≤ 5 pts against T1-only.

**Dropped or deprioritised.**
- E4 real corpus: not feasible. 0 scorable faulted lone targets; `achieved_gain` is ODOM, not TRUTH
  (`docs/codex/e4_pairs.md`). Use the 11-24 B1/B2 data when it lands.
- t2-adv: cancelled (it attacked the oracle certifier).
- EVOI as a headline. In `t0_k5`, EVOI and EPIG run 0 probes and EIG 0.09/episode; `all` costs 16 s for no
  safety gain.

### Broadening directions (plan §2a–2c): what makes each a generalisable claim, not a demo
These are the bars, in `docs/CONFIRMATORY.md` §4–6. The critique points behind them:
- **Env missions (HM3D-calibrated buildings).** These are a demo unless all of the following hold:
  - (i) the calibration is tested: KS per statistic, committed before drawing;
  - (ii) held-out *environment families* fixed by building seed, scored with a **building-clustered** bootstrap.
    Missions in one building are not independent;
  - (iii) the mission layer is the *same certifier* composed over segments, with no new tuning. The claim is then
    "per-decision certificates compose to mission-level risk": the episode-level simultaneous event already covers
    every segment of a candidate;
  - (iv) footprint baselines that are not straw men: `greedy_feasible` and `biggest_body`;
  - (v) a pre-stated check that reconfiguration is actually *needed* (≥ 20 % of oracle plans differ from the parent
    as-is). Otherwise the environment is scenery.

  Risk: the segment → leg mapping (turn → yaw at the turn rate) is our own design. Report a sensitivity row with the
  turn rate ±30 %.
- **N ≤ 16.** It generalises only if the *margin is not refit per size*, so that the test is honest extrapolation,
  and the comparison is composition vs direct on the same episodes.
  - The theory curve "extrapolated share vs N" makes this a mechanism.
  - Prop. 2b already has yaw of ≥ 3-module children undominated in 96–100 %, so the share should saturate early.
    Say so, rather than claiming growth.
- **Module types H1/H2.**
  - The honest claim is conditional: *composition transfers to a new module type whose wheel-channel model is known*.
    The type is observable metadata.
  - A learned encoder that never saw omni wheels is not expected to infer them. Run `amortized_comp_blind` to
    measure exactly that.
  - Without the blind arm, a reviewer reads H1 as "we told it the answer".
  - The paper sentence it earns: "the certificate is a property of the wheel channels, not of mecanum".

### Critique log (newest first)
- **20:55Z, N3 fixed reference** (`results/stress/N3_flips.csv`, `N3_verdict.json`).
  - **H12 as written (pooled): neither supported nor falsified.** Undominated flips are 2.76 % [2.21, 3.35] > 1 %,
    but dominated flips are 1.82 % > 0.5 %, all of them in singles and pairs (H8 scatter, present without the motor
    law).
  - For ≥ 3-module children the pattern is exact: axis 0 %, yaw 3.0–6.75 %, diagonals 0.75–1.75 %.
  - That restriction is **post-hoc**, but it is the Prop. 2b mechanism, so it is now the pre-registered primary cell
    of C-GATE (CONFIRMATORY amendment, before data).
  - **The confound I flagged was real.** Square-yaw flips go from 0.38 % (own reference) to 6.5 % (fixed).
  - Consequence: every own-reference T2 number on the boards understates misspecification. The fixed reference is
    now the T2 default label (CONFIRMATORY §0.3).
- **20:55Z, A5y.**
  - The child-rate yaw probe missed its bar (AUC 0.56 on own-reference positives): the total-trace misfit is swamped
    by T1-vs-T2 differences present in every episode.
  - The post-hoc tracking residual reaches AUC 0.83, but on **10** flips: the CI is roughly ±0.15. Do not cite it.
  - Lesson for the gate: the gate statistic should be *decision-relevant* (a predicted-vs-measured capability
    residual at the probe), not a whole-trace likelihood misfit. That is a genuine method refinement (N2b).
- **20:50Z, decision t0_k5 after the scorer fixes.**
  - The old T0 pilot headline (99.4 % useful at 0.54 % unsafe) is **superseded**: it came from the leaky cross-fit,
    a missing Gaussian normaliser and the healthy fallback on an empty screen.
  - New (240 episodes): sim0.1 gives 0.36 % unsafe per assignment and 0.42 % ep_any_unsafe, but only **65 % useful
    commits** and 35 % abstention. The oracle serves 93 %.
  - Regret 66–100 s against oracle U 186 s.
  - n = 240 gives a per-episode unsafe CI of about ±0.8 pts. That is enough for "≤ α = 0.1", not for comparisons at
    0.5 %.
  - H7 is now at risk on T0 until N1 lands.
- **20:45Z, E1 smoke.** Void (see N2). Also: `raw` certifies 99.8 % under knee 1–2 with 0 unsafe, which hints that
  T0's knee is too mild or binds on non-decision legs. Request per-leg knee-binding diagnostics in the full run.
- **20:40Z, E2/A4.** Decision-level success. Per-pair coverage fails in T2 because of 2-module-merge spread, not a
  reference offset (checked). The size-aware margin works empirically. Every result so far uses oracle latents (N4).
- **20:35Z, A5m.**
  - Healthy flips 0 % are confounded by own-plant references (N3).
  - Motor-only |d| > 0.05 on ≥ 3-module yaw is 13–21 %, yet flips are ≤ 0.45 %. Those decisions sit far from τ.
  - Clamp's inverted AUC (0.27–0.33) on ≥ 3-module yaw matches the Prop. 2b prediction (an undominated operating
    point). Positive for C1.
- **20:30Z, codex scorer audit** (19 failing tests; most were fixed in 02a70d9, b08d5d5, 9008146, 2f6c236).
  - Every number produced before those commits is void: the decide cross-fit, the baselines stratum and the analysis
    matched margin.
  - Re-derive before citing.
  - The `capxfer_cert.py` empty-support refusal is fixed in b08d5d5. Any capxfer row from before it is void.


**Status (2026-11-25 ~19:55Z, about 7 h of rented window left, about $10 headroom):** this note re-reads every board,
THEORY_cfcap, the ledger, capxfer THEORY/NOVELTY_CHECK and LIT_CRITIC. It ranks what to run next. The research lead
runs no compute. Every number cited is exploratory and simulation-only. Literature marks: [S] = seen in search
snippets this session; [M] = from memory, not re-checked.

---

## 0. Response to round-0 review (4/10): the reframing for an accept-level paper

**The review's claim.** "The paper's own data removes the problem": with the law known, 0.02 % of decisions are
ambiguous, probing loses to no probing, and the raw posterior is near-oracle (H10).

**Our answer: the data removes the *ambiguity*, not the *problem*. It turns visible ambiguity into invisible
extrapolation.**
- The 99.98 % that the oracle certifies is mostly certified by *extrapolating through the motor law*, not by
  observation.
- Prop. 2b: v2 leaves 99.9 % of child diagonal legs undominated, and 99–100 % of yaw legs of ≥ 3-module children
  (96–98 % even after clamp). Only the surge/sway legs are covered by observation.
- So roughly half of all certified counterfactual decisions (diagonals + yaw = 6 of 10 legs; to be measured exactly,
  §3.8) are confident because the robot trusts a two-constant DC law at operating points it has never visited.
- Every current test plant (T0, T1 and T2) shares that law (review weakness 4). So the experiments so far *cannot*
  show where it fails, and "the raw posterior is near-oracle" is in-model by construction.
- The real problem is therefore: **certify bodies the robot has never been, without silently trusting a model at
  operating points it has never visited.**

**Reframed thesis.** *A reconfigurable robot's counterfactual capability splits into decisions it has observed
(operating points dominated by its own probes) and decisions it is extrapolating. A certificate should spend physics
trust only where the probe evidence vouches for it.*

The paper's empirical spine becomes three regimes, one experiment each:
1. **In-model** (law correct). Nearly everything is certifiable from v2, and a raw posterior is near-oracle.
   - We *say so*: the H10 raw baseline sits in every table.
   - Learned amortization earns latency and the size generalisation of composition. It does not earn safety here.
2. **Law misspecified** (knee or current-limit motor in T0 = E1, and a T2 motor-law arm, assigned to stress).
   - Plug-in, raw-posterior and parametric certificates fail silently on exactly the undominated legs Prop. 2b
     predicts.
   - The identified-set (model-free) certificate stays valid but conservative.
   - The **trust-gated certificate** (F1) recovers most of the parametric certified share at the model-free safety
     level.
   - This is where the guarantee "earns its place" (the review's own strongest version, sentence 4).
3. **Never-existed bodies** (merges of modules never docked, E2). Composition is the only route with a guarantee;
   δ_comp is measured in T1 and T2.

**What this does to probing (review fix 1).** Probing is no longer "settle ambiguity"; it is **"buy back trust"**.
- A probe matters when the gate rejects extrapolation, and the closed form names the probe: clamp for diagonals, a
  child-rate yaw probe for yaw of ≥ 3-module children.
- Compare fixed v2+clamp (and +yaw-at-child-rate) against EVOI *in regime 2*. Report the plain result if fixed
  v2+clamp ties EVOI: a closed-form minimal probe that matches the learned policy is a clean finding.
- EVOI itself goes to the appendix unless it wins in regime 2.
- The plain-language sentence keeps "when it can't tell, it runs the one short test", now defined as "can't tell
  without trusting an unvisited model".

**What it does to the theory critique ("textbook").**
- Present one characterization, *where counterfactual decisions are identified* (Lemma 1 as a paragraph,
  Prop. 2a/b/c as the main result, Prop. 3 and 4 as tools), instead of five theorems.
- Reword Thm 5′ honestly. The simultaneity cost moves into stage 1 at per-wheel levels. It then scales with
  log(n_wheels), linear in N, not with log|M(m)|, and the number of connected children and merges grows exponentially
  in N.
  - That is the precise and defensible version of "no simultaneity price".
  - Measure both prices on real learned residuals (review fix 3, last bullet).

**What it does to the statistical critique (review fix 2).**
- Lead every table with each method's own guarantee at a stated α (realised episode-level unsafe rate + CI), then
  the certified share. The matched-U* comparison becomes secondary.
- Justify the per-episode FWER notion: one unsafe commit in a reconfiguration is the failure the operator pays for.
  Report the Jin & Candès cfBH FDR selection as the alternative error notion (assigned to baselines).
- Add a fault-prior shift test (0.1×, 0.5×, 2× fault rate) as one appendix table. That needs no new simulation:
  importance reweighting of the existing test episodes by fault count, scored with the `weighted_quantile` already
  in `core.py`.

**What it does to H9 (now FALSIFIED: the certificate is non-monotone in information).** Fix by construction (to the
decision role, which owns it):
- calibrate the *running-max* lower bound ℓ̃_t = max_{s ≤ t} ℓ_s end to end, i.e. conformalise the score of ℓ̃ along
  the policy's own probe sequence. Validity holds because the whole sequence is one per-episode procedure.
  Monotonicity holds by definition;
- add a scale floor at the label resolution (≈ 0.01 in ρ_dec);
- use a Mondrian quantile per family-size bin (|M(m)| ∈ {≤ 10, 11–30, > 30}).

**Hardware (review weakness 4c).** Out of reach in 7 h. But capxfer's real corpus already holds assembly→lone-module
counterfactual pairs with the same injected faults: square/pair probes → the lone module's real ρ. Re-scoring them
with the cfcap certifier is a real-data counterfactual check that needs no new collection (E4, gated on
feasibility). C1/C2 merges on hardware stay in the next field session (PLAN §8).

**Expected score movement** (the review's own rubric):
- fix 1 (regime 2 + probing as buying back trust), fix 3 (monotone running-max certificate) and fix 4 (a misspecified
  motor law) are exactly E1 + the stress motor-law arm + the decision fix;
- fix 2 is a reporting change;
- fix 5 is models' 2×2.

With 1–4 landed, the review projects 6–7.

---

## 1. The current scientific story

### What the evidence says, in one paragraph
- If the robot trusts its two-constant motor law, the current body tells it almost everything about the bodies it
  could become. The T0 oracle leaves 0.02 % of decisions ambiguous and certifies 99.98 %.
- The ~30 points capxfer loses (58–75 % certified) are certificate slack: estimator spread, the simultaneity price
  (1.6–2.4× the margin, Prop. 4) and finite n. They are not missing information.
- The information is only "almost everything" because the law is used to *extrapolate*: v2 never loads the torque
  side, and it never reaches the diagonal and child-yaw speeds (Prop. 2b).
- Drop the law (the model-free monotone envelope) and 7 % of decisions are ambiguous (20.7 % at τ = 0.8). Clamp, the
  one probe that dominates the diagonal operating points, takes that back to 0.06 %.
- When the plant really is different (T2), multi-module bodies barely move (0.53 % flips). Singles and pairs do move
  (4.4 % / 1.8 %), where per-wheel scatter does not average out. Clamp misfit sees this (AUC 0.78) and v2 does not
  (0.47).
- The learned posterior is stronger than capxfer on mild faults and composes to larger unseen bodies better than a
  direct head does (2×3 coverage 0.90 vs 0.81). It is about 1000× faster than enumeration.
- An end-to-end conformal certificate stays honest when the posterior becomes overconfident: q goes 1.0 → 5.2.

### Strongest honest thesis
The scarce resource is not more fault detection. It is knowing **which counterfactual decisions rest on
extrapolation through a motor model, and which ones the robot's own probes have already covered**. A certificate
built on that split is nearly as tight as trusting physics, and stays safe when physics is wrong. One short test buys
back trust exactly where it is missing.

**Headline (one sentence, plain; updated for §0).** *Before a team of docked robots splits or merges, it can certify which of the
bodies it has never been (including merges of modules that were never docked together) can do which job. It knows
which of those answers lean on an untested motor model, and one short clamp test settles them.*

### Three contributions (paper wording targets)
1. **Identification by dominance vs by extrapolation (theory, C1).**
   - Present it as one characterization of where decisions are identified, not as five theorems (§0).
  - Lemma 1: two motor constants per driven wheel.
   - Prop. 2: a decision is identified model-free iff its operating points are dominated by probed ones. Otherwise it
     is identified only through the parametric law.
   - Closed form for which legs no translation probe covers: diagonals, and yaw of children with ≥ 3 modules.
   - The minimal covering probe: clamp ≥ 2d.
   - Measured gap: 0.02 % vs 7 % ambiguous, back to 0.06 % after clamp.
   - This is partial identification with a decision target, not FDI rank.
2. **Family-free, compositional certificates over every child and merge (method, C2).**
   - Thm 5′ certifies all children and cross-parent merges from one per-wheel coverage event at no simultaneity
     price.
   - The direct route pays a measured 1.6–2.4× price (Prop. 4 attribution).
   - New in this note: a **trust-gated** lower bound (idea F1). It uses the parametric/learned extrapolation where
     the probe evidence vouches for the law, and the model-free dominance bound elsewhere.
3. **A learned, morphology-conditioned posterior inside an end-to-end conformal pipeline (ML, C3).**
   - It composes to larger unseen bodies better than a direct counterfactual head (H5).
   - It decides in milliseconds inside the probe loop (H4, latency arm).
   - It is calibrated over the whole probe→posterior→certificate pipeline, which catches probe-induced
     overconfidence (H9).
   - Misspecification is localised: singles and pairs only (H8 exception). A size-aware margin handles it.

### Which hypotheses carry the paper, and which are at risk
| H | Role | State | Risk |
|---|---|---|---|
| H2 (reframed) | carries C1 | parametric 0.02 % vs model-free 7 % | low; needs the ENVELOPE (knee-truth) check to show the parametric certificate *can* fail |
| H6 | carries C2 | theory + Monte Carlo | low; still needs the price measured on learned residuals |
| H7 | the headline number | oracle 99.98 %, T0 pilot 99.4 % | **medium**: it must beat strong *non-learned* baselines on the stratum, not only capxfer (see §3) |
| H5 | carries C3 | pilot only | medium; b1 must replicate it on H3/H4 |
| H9 | was "supports C3"; now FALSIFIED (non-monotone certificate) | round-0 critic | a defect to fix (running-max certificate, §0); the end-to-end calibration point survives only once that fix is in |
| H10 | raw posterior near-oracle | T0 pilot: raw 1.1 % unsafe | **true in-model and expected**; it is the regime-1 baseline. The paper's safety claim must come from regimes 2–3 |
| H4 | C3, amortized vs SMC | latency wins; T2 robustness unknown | **high** on the robustness arm. Expect "wins on latency only", which is enough under the T3 rule |
| H3 | probing as a headline | T0: no regret gain | **high** under a trusted law. It is alive only in the model-free/gated regime, where F1 and F5 tie probing to trust |
| H8 | stress result | mostly falsified | it becomes a positive, targeted result (singles/pairs), not a lead claim |
| H1 | background | reframed | not a contribution (capxfer Prop. 2) |

---

## 2. Frontier ideas (ranked)

### F1. Trust-gated certificate: extrapolate only where the probe vouches for the law (rank 1)
- **Idea.** Each (candidate, leg) gets one of two lower bounds:
  - the parametric/learned bound ℓ_par, which extrapolates through the two-constant law;
  - the sharp model-free bound ℓ_PI, the step-envelope low end E^low from Prop. 2a.

  Use ℓ_par iff (i) the needed operating points are dominated (then ℓ_par = ℓ_PI anyway), or (ii) a
  motor-law specification statistic passes. The statistic is the clamp/probe misfit of the fitted (α, β) line:
  residual of the saturated duty against a·α̂ + b·β̂. Otherwise use ℓ_PI.
- **Validity.** Calibrate the gate's threshold conformally on episodes drawn from a family of mixed truths
  (two-constant and knee-envelope). The end-to-end score is then max_j over the gated residual, so the guarantee is
  over the whole gated pipeline. Selecting between two scores this way is covered by conformal after data-dependent
  selection (Liang, Zhu & Barber, arXiv 2408.07066, JASA 2026 [S]). The simpler route is end-to-end calibration of
  the composite rule.
- **Novelty.** Partial-identification specification tests have been studied (Molinari review, arXiv 2004.11751 [S];
  test-by-empty-set [S]). So has conformal validity under misspecification (distributional CP, PNAS [S];
  CP4SBI 2508.17077 [S]; FMCPE 2509.23385 [S]). None of these switches between a model-based extrapolating bound and a
  model-free identified-set bound per decision, gated by probe evidence. None ties that switch to *which bodies the
  robot has not yet been*. This is the method form of the owner's "recognize what it cannot do".
- **Expected gain.** It turns H2's reframing (a negative "probing is secondary") into the method contribution.
  - Near-parametric certified fraction when the law holds.
  - Bounded unsafe-go when it does not (knee truth).
  - A natural home for clamp: run it only when the gate fails.
- **Cheapest test.** T0 on CPU, local, $0. The theory role already wrote the `ENVELOPE` task in
  `cfcap/theory_task.py` (knee-motor truth vs the two-constant oracle fitted at the parent's probes, per-decision
  dominance ratios). Add three certifiers offline. Full spec: E1 in §4.
- **Kill.** Under the knee truth, the parametric certificate's unsafe-go on undominated decisions is ≤ U* = 1 %. That
  means law misspecification is harmless, so there is nothing to gate. Then report "extrapolation is safe for these
  envelope classes" as a robustness result and drop F1.
- **Risk.** The knee family is our own choice ("you built the failure"). Mitigations:
  - draw knees independently of the probes;
  - report a sweep over knee severity;
  - also gate on T2, which is a misspecification we did not design.

### F2. Merges of never-co-assembled modules as the unique capability (rank 2)
- **Idea.** Make "certify a body that has never existed" the experimental centrepiece.
  - Probe two or three assemblies *separately*, then certify every merge layout.
  - capxfer cannot do this at all.
  - A direct counterfactual model has no calibration data for those merges unless merges are in its cal set. Run it
    honestly with merge rows held out of calibration.
- **Thm 5′'s only merge-specific term is δ_comp, and nobody has measured it.** A4 is missing from `results/stageA/`;
  the theory board still lists it as a need. This is a hole in a headline theorem.
- **Novelty.** Coalition-game reconfiguration under uncertainty (AAAI workshop [S]) and "Recompositional Robotics"
  (arXiv 2609.37734, 2026 [S]) ask whether reconfiguring is worth it, but give no certified capability. Shen & Lipson
  2403.10496 [S] identify configurations after the fact. Nothing certifies an unformed merged body.
- **Expected gain.** The single most distinctive capability claim, plus closure of Thm 5′. High reviewer value
  against "incremental over capxfer".
- **Cheapest test.** E2 in §4: oracle-latent composition first (T1 and T2), learned-posterior composition when the b1
  weights land.
- **Kill.** δ_comp decision error > 2 % in T1. Then merges are not plug-in; report δ_comp as the binding term and
  calibrate merges separately.
- **Risk.** In T1, rigid docks make composition near-exact, which a reviewer may read as "by construction". The T2
  arm (compliance, hook play, load transfer) answers that.

### F3. Partial-identification bound as a training-free baseline that is hard to beat (rank 3, may reshape C3)
- **Idea.** ℓ_PI from observed tracking speeds and duties alone (speed-ceiling class), propagated through T1, with
  no learning. On T0 with v2+clamp it certifies 99.9 % of capable decisions (`coverage_speed_only.csv`).
- **Why it matters.** It is the strongest trivial-in-spirit baseline (COMMON §6). If it matches the learned posterior
  after clamp, the ML claim must move to:
  - "v2 only, no extra probe": the learned posterior extrapolates, ℓ_PI cannot;
  - latency;
  - the gate (F1).

  We need to know this before reviewers tell us.
- **Novelty.** Not novel as a baseline. Its value is in making the comparison honest.
- **Cheapest test.** Part of E3: a `dominance_pi` baseline in `cfcap/baselines/`, CPU only.
- **Kill (for C3 as currently worded).** ℓ_PI after v2+clamp within 3 pts CF@U* of the best learned method on the
  stratum. Then reword C3 as "learning buys the extra probe back".
- **Risk.** In T1 the noisy duties make E^low looser than in T0. That is fine, because it is also what makes the
  comparison fair.

### F4. Size-aware misspecification margin from simulator disagreement (rank 4)
- **Idea.** The T1→T2 shift is a size law: flips 4.4 % / 1.8 % / 0.25 % / 0.07 % for single / pair / L3 / square.
  So calibrate a Mondrian-by-size margin, plus a misspecification budget ε(n_wheels) set from the T1-vs-T2
  disagreement on a held-out *simulator* split. This is two-simulator conformal: the second plant acts as a stand-in
  for the shift.
- **Prediction to pre-register.** ε scales like the per-wheel scatter's averaged tail, about σ/√n_w or faster.
- **Novelty.** Domain-aware CP calibrates on translated sim data (2611.26065 [S], via LIT_CRITIC); SCAPE uses
  paired sim–real samples [S]. Using an *independent second simulator* as the shift proxy, size-stratified, is a
  modest but clean twist. It makes the singles/pairs exception a positive targeted result.
- **Expected gain.** It turns H8's exception into a short subsection with a fix, and answers reviewer risk 2.
- **Cheapest test.** Offline, once `t2-eps` labels and any method's T1 lower bounds exist on shared eids (baselines
  `enum_known` or models). CPU, < 30 min.
  - Fit ε(size) on half of the T2 eids.
  - Score the other half: unsafe-go per size before and after.
- **Kill.** The size-stratified plain T1 Mondrian margin already gives T2 single-module unsafe-go ≤ 2·U*. Then no
  budget is needed; state that and move on.
- **Risk.** It uses T2 for calibration, so T2 is no longer evaluation-only. Keep a disjoint T2 test half, and name the
  split explicitly.

### F5. Decision-theoretic "split-to-see" and "probe-to-see" break-even (rank 5)
- **Idea.** Closed-form break-even: probe u (or a diagnostic split) pays iff
  P(amb) · E[ΔU | amb] > t_u, with E[ΔU | amb] ≤ min(V, C_unsafe·p_bad).
- **Prediction.**
  - Under a trusted law (P(amb) ≈ 1 %), nothing pays unless C_unsafe/V ≳ 100.
  - Under the model-free class at τ = 0.8 (P(amb) ≈ 21 %), clamp pays at C_unsafe/V ≈ 5. The current constants
    are 600/120.
  - A diagnostic split essentially never pays, because clamp dominates it in time.

  This puts the owner's "determine whether changing its configuration is worthwhile" into the paper honestly.
- **Novelty.** Andrews & Chen "Certified Decisions" (arXiv 2502.17830 [S]) and HC-ROCP (2611.12662 [S]) certify
  actions; Lekeufack et al. conformal decision theory [M]. None prices an information-gathering
  *reconfiguration*.
- **Cheapest test.** Offline re-scoring of `results/decision/pilot_t0_episodes.csv` and the T1 decide pickles:
  - C_unsafe ∈ {600, 3000, 12000};
  - add the `fixed_clamp` policy;
  - add the gated predictor from E1.

  CPU, < 1 h of implementation.
- **Kill.** Regret of `fixed_clamp` and `evoi` stays within 1 se of v2 even at C_unsafe/V = 100 in the model-free
  class. Then probing goes to the appendix (C4 fallback).
- **Risk.** The utility constants are assumptions. Present a phase diagram, not one number.

### F6. End-to-end, monotone calibration of the adaptive pipeline (H9 fix) (rank 6, cheap, defensive)
- **Update after round 0.** The q 1.0 → 5.2 jump is a score defect (scale collapse), not a feature. The
  contribution is now the running-max certificate of §0: monotone by construction and valid end to end. The text
  below still applies to the validity argument.
- **Idea.** Treat the probe policy, posterior and certifier as one procedure, applied i.i.d. per episode. Calibrate
  the per-episode max score on episodes run through the *same* adaptive policy. Exchangeability then holds at episode
  level, with no feedback-shift reweighting (contrast Fannjiang et al. 2022, feedback covariate shift [M]).
- **Expected gain.** It pre-empts "adaptive probing voids your guarantee". The T0 pilot already shows that a fixed q
  calibrated on v2 is unsafe after `all`.
- **Cheapest test.** Offline on the T0 pilot. Report the per-episode unsafe rate of (a) q calibrated on v2 and reused
  for each policy, vs (b) end-to-end q, at α ∈ {0.05, 0.1}, with an episode bootstrap. 15 min.
- **Kill.** (a) is already ≤ α for every policy. Then it is a remark, not a contribution. Separately, the
  running-max certificate must certify ≥ the v2 count on the `all` policy (7341, vs 5339 now) at realised episode
  unsafe ≤ α; otherwise the fix failed.
- **Risk.** Low novelty on its own; it is a supporting point.

### F7. Learned residual correction T1→T2 (rank 7, NOT recommended now)
- It would train on T2, so the independent-plant evaluation is lost. The shift is small except for singles and pairs,
  and F4 covers those more cheaply and more honestly.
- Revisit only with real cross-layout corpus data (PLAN Stage D).

---

## 3. Adjustments to running and queued work (exact fixes)
1. **The H7 headline compares against the wrong population.** capxfer's 58–75 % is on *capable plated faulted lone
   modules*. Singles are also where T2 shifts most (H8).
   - Fix (paper role, `scripts/analysis/t1_certificate.py`): add a row block, stratum = `kind==child & cand_size==1
     & has_fault==1 & truth>=tau`, CF@U*, for every method on T1 *and* T2.
   - The capxfer comparison sentence must quote this block, not the pooled stratum.
2. **Missing strong trivial and non-learned baselines** (baselines role, `cfcap/baselines/simple.py`):
   - `detector_gated`: the A1 sibling-median duty heuristic ĝ_i = median_sib(u)/u_i, flag at the 1 % false-alarm
     threshold. Go (lower = 1) iff the candidate holds no flagged wheel, else refuse. This is the "rank by gain"
     analogue.
   - `ghat_physics`: the same ĝ point, propagated by the T0 QS plant, then SimultaneousLower. RLS is close, but its
     +0.20 slack makes it a straw man.
   - `copy_parent`: the child's ρ on each leg = the parent's measured v2 ratio on that leg (constant answer per
     body).
   - `dominance_pi` (F3).
   - Keep `metadata_nf` in the main table; mark `metadata` (true fault flag) as privileged.
   - From the review's baseline list (items 2–4, 8, 9), deduplicated against the above:
     - `credible_lower`: the posterior 5 % quantile, uncalibrated;
     - `min_over_members`: the weakest member's lone-module capxfer certificate × a static per-(size, leg) factor
       fitted on cal;
     - `ghat_physics`, which doubles as the detect-then-nominal / MARS-style quasi-static margin;
     - `raw` (q = 0).

     The coordinator gave baselines the α-guarantee reporting and cfBH; keep those, and these are additions.
3. **cfcap-decide (T1).** It is mis-specified against the theory.
   - (a) The `all` policy bundles three probes, while theory says clamp alone is minimal. Add `fixed_clamp` to
     `cfcap/decide/policies.py`. This is offline: the per-draw clamp predictions are already in the pickles.
   - (b) Regret is dominated by healthy commits. Report regret and the unsafe rate on the stratum.
   - (c) Add the C_unsafe sweep {600, 3000, 12000} in `decide_t1_eval.py` (F5).
   - (d) Add `detector_gated` as a decision policy. No rented-job change is needed.
4. **cfcap-t2-adv** (held). CEM against a *T1-oracle* certifier measures the T1–T2 gap, not any method's failure.
   - Do not submit as is.
   - If budget remains after the top 3, retarget `--certifier` at `enum_known` or the amortized+sim bound, restricted
     to singles and pairs. Cap at 100 restarts (~25 thread-h, ≈ $0.5).
5. **cfcap-models-b1** ($7.8, the largest item).
   - If any top-3 item needs money, cut seeds 2–3 of `stl` and of `ds` first (saves ≈ $2). Keep the three `st`
     seeds, because they are the ensemble and carry the H5 replication.
   - Make sure eval-test includes the merge candidates with `kind`, so E2 can reuse them.
6. **cfcap-nuis2.** It is expected to confirm < 5 % ambiguity under the parametric T1 posterior, which is fine: it
   gives the F1 curve. Add (offline, nuisance role) the speed-only identified-set ambiguity on the *same* episodes'
   v2 and v2+clamp traces. That gives the T1 version of the 7 % → 0.06 % result, which C1 currently has only on T0.
7. **cfcap-theory-cpu ENVELOPE group.** Keep it; it feeds E1. Make sure it saves, per decision: the dominance flag,
   ℓ_par, ℓ_PI, the clamp misfit statistic and the truth. If it saves only aggregates, E1 runs it locally instead
   (spec below).
8. **The stratum definition** (has_fault OR |ρ−τ| < 0.1) is fine for the headline. Also report two sub-strata,
   so the reader sees where certified mass comes from:
   - "undominated" (needs extrapolation);
   - "dominated".

   This is the empirical face of C1.

---

## 4. Ranked next experiments (top 3 to launch now)

### E1. Trust-gated certificate on T0 under a misspecified motor law (owner: theory; consumer: decision)
- **Files.**
  - `cfcap/theory_task.py:ENVELOPE` (existing; extend additively to emit per-decision rows).
  - New `scripts/theory_gate.py` (run + analyze).
  - Reuse `scripts/theory_coverage.py` (dominance) and `scripts/theory_nuisance.py:run_episode` (speed-only bound).
- **Design.**
  - 1,200 T0 parents of 2–6 modules (include singles and pairs as children), 1–2 faults (weak/coast/brake), δ = 0.08.
  - Truth law ∈ {two-constant, knee-envelope at 3 severities}, 300 parents each.
  - Probes {v2, v2+clamp}.
  - Certifiers, all at α = 0.1 simultaneous and cross-fitted 2-fold:
    - `par`: two-constant fit at the probes, plus SimultaneousLower;
    - `pi`: step-envelope E^low, sharp, no calibration;
    - `gate`: par if dominated or the clamp misfit ≤ q_gate, else pi; q_gate and the margin calibrated end to end on
      the cal fold, pooled over truth laws;
    - trivial: `nominal`, `detector_gated`;
    - `raw`: the par point with q = 0 (the H10 baseline, which is expected to fail silently under the knee truth).
  - **Shared definition with stress.** Use the same knee/current-limit law that stress puts into T2 for review fix 4.
    Write it once as an additive function in `cfcap/t0/qs.py`, e.g. `motor_envelope(kind, severity)`, and have stress
    import it. Then E1 (T0) and the T2 motor-law arm are one experiment at two fidelities, and the gated certificate
    is scored on both.
  - Also report the **share of certified decisions that are undominated**, by leg type and child size, under the
    two-constant truth. That is the number behind §0's "half of certified decisions are extrapolated".
- **Cost.** About 4.3 s/episode at 1 thread × 2 probe sets ≈ 3 CPU-h. Local 4 threads ≈ 45 min, or the free lane.
  $0.
- **Output.** `results/theory/gate_cert.csv`, with columns law, severity, probe, certifier, stratum ∈ {all,
  decision, undominated}, cert_frac_capable, unsafe_rate, ep_any_unsafe, n.
- **Feeds.** A new Fig. F4, "certified fraction vs unsafe-go as the motor law degrades": one panel per probe set,
  lines par/pi/gate. Plus C1/C2 text and Sec. theory.
- **Success** (pre-written; scored on the decision stratum):
  - under the two-constant truth, gate CF ≥ par CF − 2 pts;
  - under the knee truth (the worst severity at which par's unsafe-go > 2 %), gate unsafe-go ≤ 1 % and gate CF ≥
    pi CF + 5 pts with v2 only.

  **Kill:** par's unsafe-go ≤ 1 % at every knee severity (see F1).

### E2. A4: merges of never-co-assembled modules, composition defect δ_comp in T1 and T2 (owner: stress; generator supplies episodes)
- **Files.**
  - New `scripts/stress_A4_merge.py`.
  - Uses `cfcap/t1/capability.py`, `cfcap/t2/capability.py`, `cfcap/t0/qs.py` and `cfcap/t0/candidates.py` (merge
    layouts).
  - Reuse `scripts/theory_merge.py` for the stage-1 sets.
- **Design.**
  - 500 pairs (G1, G2) of 1–3-module assemblies with random yaws, ≤ 1 fault each, HET_FULL, payload 0 / 0–5 kg arms.
  - Every merge layout (≤ 6 per pair, ≈ 3,000 merges) × 10 legs × τ ∈ {0.6, 0.7, 0.8}.
  - Truth in T1 and in T2 `all`.
  - Predictions:
    - (a) T0 composition from the true latents;
    - (b) T1 composition from the true latents (separate assemblies' latents, merged geometry);
    - (c) T1 composition from per-assembly enum_known posteriors (the Thm 5′ certificate), its v2 probes run on each
      G_j alone.
  - δ_comp = the decision-flip share and the |Δρ| quantiles, (b) vs T1 truth and (b) vs T2 truth.
- **Cost.**
  - T1 on the local GPU (one heavy slot): ≈ 3,000 merges × 10 legs ≈ 30k 3-s legs, < 5 min at A0's 80k legs/s.
  - enum_known: 1,000 assemblies of ≤ 3 modules at ~1 s ≈ 20 min.
  - T2 numba: ≈ 1.5 s per 23 candidates → ≈ 5 min at 4 threads.
  - Total < 1 h wall locally, $0. If the GPU lock is busy: one 3090 box ≈ 1 box-h, ≈ $0.2.
- **Output.** `results/stress/A4_merge.csv` (per plant, payload arm, merge size: flip share with an episode-cluster
  CI, |Δρ| q50/q95/q99, simultaneous miscoverage per pair for (c), CF on capable merges at U*). Plus
  `results/stress/paper/dec_merge.csv` in the §1a contract (kind = merge, method = compositional_enum, oracle).
- **Feeds.** A new Table "never-co-assembled merges" (CF and unsafe for compositional vs capxfer "n/a" vs direct
  with merges held out once b1 lands); the δ_comp term in Thm 5′; the abstract's merge sentence.
- **Success** (pre-written):
  - δ_comp flip share < 1 % in T1 and < 2 % in T2 (95 % CI upper bound);
  - (c) simultaneous miscoverage per pair ≤ α = 0.1;
  - CF on capable decision-stratum merges ≥ 85 % at U* = 1 %.

  **Kill:** T1 δ_comp > 2 % (see F2).

### E3. Strong trivial and partial-identification baselines on the decision stratum, size-stratified (owner: baselines; paper role adds the analysis rows)
- **Files.**
  - `cfcap/baselines/simple.py`: add `detector_gated`, `ghat_physics`, `copy_parent`, `credible_lower`,
    `min_over_members` (§3.2).
  - New `cfcap/baselines/dominance.py`: `dominance_pi`. Per wheel, the speed ceiling is the max |ω| observed while
    tracking, or the observed saturated speed. The torque side is the healthy envelope. Set g_lo = ceiling/ω̄_0
    (clipped to [0.3, 1]) and propagate through `t1.capability`. No calibration is needed (sharp bound), but also
    report it with SimultaneousLower for a matched U*.
  - Register them in `scripts/baselines_pilot.py --methods`.
  - Analysis: `scripts/analysis/t1_certificate.py` adds the lone-faulted block (§3.1) and a by-`cand_size` block
    (1 / 2 / ≥ 3).
- **Episodes.**
  - The same 294 pilot eids as `cfcap-baselines-pilot2` (cal = even, test = odd, per the 2-fold cross-fit);
  - plus the `cfcap-t2-eps` eids for T2 truth, once pulled;
  - plus v2+clamp traces for `dominance_pi` (the nuisance/decide probe library; if unavailable on those eids, run it
    on v2 only and state that).
- **Cost.** All CPU (QS and T1 capability batched on CPU or one short GPU slot): ≈ 1–2 CPU-h, $0. If run on a box
  next to pilot2: < 0.5 box-h, ≈ $0.1.
- **Output.** `results/baselines/paper/dec_trivial.csv` (§1a contract; route = raw/simultaneous;
  posterior = none). Plus `results/baselines/size_strata.csv` (method × cand_size × plant: CF@U*, unsafe, ep_any_unsafe).
- **Feeds.** Table T1 (the "trivial / non-learned" rows); an appendix table "by body size" (it carries F4/H8); claims
  C2 and C3.
- **Success** (pre-written; decision stratum, T1 test, CF@U* = 1 %):
  - the best posterior-based method beats max(detector_gated, ghat_physics, copy_parent, metadata_nf) by ≥ 10 pts;
  - it beats `dominance_pi` (v2 only) by ≥ 5 pts.

  **If it fails:** C3 is reworded to the latency / v2-only / gate framing (F3). The paper says so, and C2 rests on
  the compositional and merge results.

### E4 (gated on a 30-min feasibility check, launch when b1 weights land). Real-corpus counterfactual check
- **Owner and inputs.** Models, with baselines for capxfer. capxfer's real modular corpus
  (`~/field_corpus_real_evaluation/modular/`, read via capxfer's adapters, read-only; no telemetry into git).
- **Feasibility check.** Count the (assembly session, lone session) pairs that share the same module and injected
  fault (pair/square → single).
- **If ≥ 30 pairs exist.** Predict the lone module's real ρ_dec on the decision legs from the assembly's real v2
  trial, using:
  - the amortized posterior with the compositional route;
  - capxfer's certificate;
  - `raw`;
  - `detector_gated`.

  Score unsafe commits and certified share with a session-grouped bootstrap.
- **Output.** `results/models/paper/dec_real.csv` (test_plant = `real`). It feeds a small real-data table in §exp,
  answering review weakness 4c without new collection.
- **Success.** The cfcap certificate's realised unsafe rate ≤ α + CI, and its certified share ≥ capxfer's on the
  same pairs.
- **Kill.** Fewer than 30 matched pairs. Then say "field blocks C1/C2 planned" and drop it.

### Next in line (if budget and time remain)
4. F4: size-aware two-simulator margin. Offline, as soon as T2 labels and any T1 lower bounds share eids.
5. F5 + F6: decide re-scoring (fixed_clamp, fixed v2+clamp+yaw-at-child-rate, C_unsafe sweep, the running-max
   monotone certificate). Offline on the T0 pilot now, on
   T1 when cfcap-decide lands.
6. The retargeted t2-adv (§3.4).

**Budget check.** E1–E3 run on CPU or the local GPU, at about $0–0.3 in total. They leave the ≈ $10 headroom for
overruns of models-b1 and decide, which carry H4 and H5.
