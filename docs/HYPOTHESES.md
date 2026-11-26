# cfcap hypothesis ledger

This file is the scientific state of the study. Update it whenever a result lands. Each entry gives the claim, its
status, the evidence (file), and the next test. Every result here is exploratory and simulation-only unless marked
otherwise.

**Status key:** SUPPORTED · FALSIFIED · OPEN · REFRAMED. The date is when the status last changed.

## Core thesis
A modular robot can certify the capability of configurations it is not in (children and merges) with a
finite-sample bound on unsafe commits. When it can't, a targeted probe or split settles the decision.

## Hypotheses

**H1. Weak wheels are hidden in assemblies (the capxfer premise).** REFRAMED, 11-25.
- With identical motors, a simple duty heuristic detects 92–99 %, so the hiding is not trivial but is weak.
- With realistic ±8 % motor spread, detection drops by about 15 pts. Mild faults (g 0.75–0.9) are caught only 25–40 %
  by the heuristic and 73–76 % by the capxfer self-model.
- At g ≥ 0.85 a weak wheel is the *same plant* as a slower healthy motor, so it is identifiable only up to the motor
  spread. Hence the switch to the ρ_dec label.
- Evidence: `results/stageA/A1_summary.csv`, `docs/board/nuisance.md`.
- Implication: "hidden" means hidden *within the nuisance class*. The masking rate is not new (capxfer Prop 2).

**H2. Counterfactual decisions are often ambiguous after the standard probe (v2).**
- FALSIFIED under a known motor law, 11-25: the T0 oracle leaves 0.02 % ambiguous; T0 decision pilot 1.1 %.
- REFRAMED under a model-free envelope: 7 % ambiguous, or 20.7 % at τ=0.8 (theory). Ambiguity exists exactly when the
  robot doesn't trust its motor model.
- OPEN in T1 with unknown nuisances: cfcap-nuis2 is running.
- Kill line (pre-written): < 5 % ambiguous → probing is not the headline.

**H3. A targeted probe (clamp, tug, module-differential) removes most of the remaining ambiguity.** OPEN.
- T0 oracle: not testable (24 ambiguous decisions).
- Model-free class: clamp takes 7 % → 0.06 % (theory).
- A5: the clamp misfit separates T1→T2 flips (AUC 0.78; v2 gives 0.47).
- T0 decision pilot: decision-EVOI resolves 23 % of the ambiguous cases, the best per probe-second. No policy beats v2
  on regret.
- Next: cfcap-decide (T1) and cfcap-nuis2 (A3 probe sets).

**H4. A learned amortized posterior beats physics Bayes (enumeration or SMC).** OPEN.
- Win conditions: robustness under misspecification (T2) and decision latency inside the probing loop.
- Pilot: amortized 3.4 ms/episode vs enumeration 2–4 s. On mild faults it beats capxfer (square detection
  0.59→0.75; |ĝ−g| 0.077→0.027).
- Next: cfcap-models-b1 + cfcap-baselines-pilot2, then the 2×2 {SMC, amortized} × {T1, T2}.

**H5. Composition (module latents → physics) generalizes to unseen, larger shapes better than direct
prediction.** SUPPORTED (pilot), 11-25.
- On 2×3 when trained on ≤4 modules: coverage 0.90 vs 0.81, pinball 0.00093 vs 0.00147.
- Diversity matters: training on single + pair only collapses on 2×3 (detection 0.03).
- Next: the models-b1 diversity sweep. Must replicate on H3/H4.

**H6. Simultaneous certification across the candidate family is affordable.** SUPPORTED (theory), 11-25.
- The price is 1.6–2.4× the per-decision margin (Gaussian or correlated) and 4–12× for heavy tails.
- The compositional per-wheel route (the Thm 5 merge extension) avoids it: one coverage event certifies all children
  and merges. Monte Carlo miscoverage 1.7 % vs a 19 % bound, so the bound is loose.
- Next: measure the actual price on learned residuals (models-b1, baselines).

**H7. The certificate is tighter than capxfer's (58–75 % certified) at equal unsafe-go.** OPEN; the main headline
candidate.
- Oracle 99.98 %. T0 decision pilot: posterior + simultaneous certificate gives 99.4 % useful commits at 0.54 %
  unsafe. All 6 unsafe cases came from the hypothesis screen dropping the true 2-fault state, which is a fixable bug
  class.
- Next: the baselines and models dec_*.csv on the decision-relevant stratum.

**H8. Misspecification (T1→T2) breaks T1-calibrated certificates.** MOSTLY FALSIFIED, 11-25.
- Pooled flips are 0.53 % (CI 0.35–0.75); ≥3-module children ≤0.25 %.
- EXCEPT single modules: 4.4 % flips, 15.7 % shifted by more than 0.05. Pairs: 1.8 %.
- The cause is per-wheel friction scatter plus differences in the kernel numerics. Compliance, load transfer and
  anisotropy are negligible.
- Theory predicted weak spots at the diagonals and yaw of ≥3-module children. NOT observed.
- Size gradient: single 4.4 % > pair 1.8 % > L3 0.25 % > square 0.07 % flips. The kernel change causes about 60 % of
  single-module flips; per-wheel roller-friction scatter causes the rest (`docs/board/stress.md` C-stress-1).
- Adversarial pilot: confident-wrong margins of 0.11–0.15 on pairs after 3 CEM iterations, with an oracle-latent T1
  certifier.
- Next: T2 evaluation of the trained certificates (stage 2), and adversarial search (`t2_adv`, held pending ranking).

**H9. More probing never weakens the certificate (monotonicity).** FALSIFIED (round-0 critic,
docs/reviews/review_round0.md), 11-25.
- Running the full probe library resolves 137/138 ambiguous decisions, yet certified pairs fall 7341 → 5339 because the
  normalised-score quantile jumps from about 1.1 to 5.23 (the score scale collapses).
- This is a defect of the score design, not a safety feature.
- Fix under way: a scale floor, a per-family/size-aware score, and a test that certification is monotone along probe
  sequences.

**H11. A conformalized hypothesis screen gives end-to-end validity including approximate inference.** OPEN, 11-25.
- Root cause of every unsafe commit (codex, docs/codex/screen_bug.md): the hard screen at nominal nuisances drops the
  truth in 4.5 % of episodes (double faults 24/250), then silently substitutes "healthy".
- Fix being built: calibrate the screen threshold on the true state's residual (α_s) and fold it into the union bound.
- Candidate method contribution: validity that survives inference truncation.

**H10. The raw, uncalibrated posterior is already near-oracle (critic).** OPEN. If true, calibration and amortization
must show their value elsewhere (misspecification, latency, held-out shapes). It becomes a required baseline.

**Round-0 critic (4/10 weak reject):** top fixes in docs/reviews/review_round0.md.
1. Probing only where ambiguity is real (model-free envelope class; yaw legs of ≥3-module children), with EVOI vs fixed
   v2+clamp.
2. Report each method's own guarantee at a stated α; per-fault-class calibration; shifted fault rates; a conformal
   selection (Jin & Candès) baseline.
3. Fix H9 (score monotonicity).
4. A T2 motor-law misspecification (current limits, derating); the adversarial search; the merge error term.
5. A matched-compute amortized-vs-SMC grid with end-to-end timing, and capxfer on the same episodes.

## Research-lead verdicts (newest first; detail in docs/board/research.md "Critique log")
- **02:40Z, H11 held-out severity: REINSTATED (exploratory, T0)** (decision board "Critic-A audit", 3b80d98;
  `results/decision/hos_audit_K.csv`, `hos_audit_worstcase.csv`).
  - **K is applied.** K = 15 gives a median screened set of 348 and 17.8 % empty screens; K = 60 gives 0 % empty.
    Between K = 26.9 and 28.7 only near-zero-weight hypotheses enter, and they never reach the ≤ 6 top-mass set. That
    explains the byte-identical arms (a property, not a bug).
  - **Worst case**, with the 33 non-converged severity-3 episodes counted as abstain at oracle regret: EVOI 78.2 % vs
    clamp 78.4 % useful, both 1.0 % E_ep. The pre-registered bars still hold.
  - **Caveat.** On the 45-episode audit subset, EVOI E_ep is 4.4 % (2/45) at every K, against 1.1 % on the full
    267-episode run. That is consistent with noise, since 2/45 has an exact upper 95 % of ≈ 13 %. But it shows that
    the severity-3 validity rests on few events.
- **02:40Z, H3 final wording: "the same usefulness at ~¼ of the probe time".**
  - Decision regret excluding probe time: EVOI is worse by +1.13 s [0.21, 2.15] vs clamp, and +0.92 [0.10, 1.85] vs
    dom.
  - Its total-regret win (−6.0 s) is the probing it saves (6.5 s).
  - Not "better decisions". T0 only.
- **02:40Z, RNG stream fix** (critic C; `envelope_post`).
  - Only NEXT-1's robust-likelihood arm was affected. Its clamp-leg sd draws collided with the v2 stream of other
    seeds, and it feeds only an auxiliary healthy-draw sd.
  - The NEXT-1 verdict (64.8 % useful vs a 75 % bar; the robust arm equalled the v2 arm) cannot flip.
  - No other result used that path.
- **01:50Z, status corrections for the paper (after blind critics A and B; `docs/reviews/weakpoints_{A,B}.md`).**
  - **H12 → HYPOTHESIS-GENERATING.** The ≥ 3-module restriction was chosen after seeing N3 (pooled bar neither
    met nor failed). It is tested only by C-GATE P5. If C-GATE is missing or partial, the paper may state H12 only
    as an exploratory, mechanism-motivated finding.
  - **H4 → "latency advantage established (3.4 ms vs ≥ 500 s/episode SMC); accuracy against SMC not established"**,
    unless smc-small lands before scoring. The T3 rule cannot be evaluated without SMC rows.
  - **H3 (probing) wording.** EVOI matches always-clamp usefulness at ~¼ of the probe time. Its regret gain is
    probe-time accounting: decision regret without probe time is ~0.5 s *worse* than clamp (critic A §2). It is
    not "better decisions". T0 only.
  - H11 held-out severity: suspended at 01:50Z, then REINSTATED at 02:40Z after the K-sweep audit (see the entry
    above).
  - **Everything at decision level is T0** (decision, probing, screen, gate). T1/T2 evidence is certificate-level
    only (critic A §3). This goes in the limitations.
  - **Critic B points carried to the final ledger:**
    - the trust gate is a negative result (dominated by `par_pooled` and `pi_cal`);
    - held-out-family coverage is 85–88.9 % against the nominal 90 %, so C5 is weak;
    - the diversity effect (C6 / F2) is null;
    - the theory claims are Prop. 2(b)/(c), plus the measured Prop. 4 slack. Lemma 1 and Thm 5′ are tools.
  - **Pooled matched point hides unsafe merges** (critic B §2) → new confirmatory bar 1b (A-KIND).
- **01:50Z, forking paths (for the limitations section).** Exploratory arms tried and abandoned, in order:
  - trust gate (E1: falsified; replaced by robust / lawcal before any conf data, A-GATE);
  - child-rate yaw probe (A5y: missed 0.56 < 0.75) and its parent version (N5: failed);
  - NEXT-1, the robust clamp likelihood (failed its useful bar). Then NEXT-1b, a knee-tolerant conformal screen
    (passed after the screen change);
  - N7, the size-aware single/pair margin (narrow miss);
  - the own-reference T2 label (replaced by the fixed reference after N3 showed it hid the shift);
  - budget abstention (replaced by Mondrian-by-difficulty, N1).

  Each switch was pre-registered on a board before its own run, on fresh seeds. But the *sequence* is adaptive.
  Only CONFIRMATORY.md cells, scored once, carry confirmatory weight.
- **21:58Z H3 replication: SUPPORTED on 3 disjoint seed sets** (decision envpost / n1robust / n1bK).
  - T4 must headline task-level metrics. EVOI's lower candidate certified share (73.8 vs 84.0 %) lies entirely in
    episodes where it skipped a probe that did not matter for the task (342/342 pairs). It misses value in 13/585
    episodes.
- **21:58Z H11 extended: SUPPORTED** (decision NEXT-1b). A conformal screen calibrated on the knee family removes the
  empty-screen failure, which was 30 % of episodes (75 % at severity 3). It is in-family; held-out-severity
  calibration is still to do.
- **21:35Z H3 (probing) in regime 2, T0: SUPPORTED** (decision board pre-registration edfbb9f, seeds 2100–2699, 583
  episodes; `results/decision/envpost_*`).
  - EVOI matches always-clamp usefulness at 18 % of the probe time, and beats the dominance trigger by 3.3 s
    [2.5, 4.1].
  - Effect size ~3 % of oracle utility, with 0 unsafe for all policies. Not shown in T1/T2.
- **21:35Z, N7 (size-aware single/pair margin): NARROW MISS** (`results/stress/N7_verdict.json`). Unsafe 4.3 → 0.85 %
  (single), but CF −7 pts against the ≤ 5 bar.
- **21:40Z H13 (trust gate): FALSIFIED** (exploratory E1, `results/theory/gate_cert.csv`). Replaced in CONFIRMATORY
  A-GATE by the `robust` certificate (dominance + clamp, calibrated) and by `lawcal` with a held-out severity.
- **21:40Z H12 (silent extrapolation): SUPPORTED in T0 at s = 3** (par undominated unsafe 31 %, raw 51 %). At s = 1–2
  the excess over the in-law rate is small (3.4–3.9 % vs 2.3 %). Every cell is underpowered by the 20k rule.
- **21:40Z H3 (probing) in regime 2: REVIVED, pending N8** (`results/decision/envelope_summary.csv`). EVOI regret
  14.3 vs v2 31.4, but with true fault modes and no dominance-triggered baseline.
- **21:40Z, scaling mechanism: SUPPORTED** (deterministic, `results/theory/dominance_vs_N.csv`). Undominated yaw share
  is 0.93 at N = 16 with v2, and 0.85 even after clamp.
- **21:00Z H11 (screen): SUPPORTED, exploratory T0** (`results/decision/t0_n1_verdict.json`).
  - With end-to-end + Mondrian-by-difficulty and budget-abstention off: ep_any_unsafe 0.33 % (upper 0.8 %), useful
    95.4 %, healthy abstention 0.
- **21:00Z H10 (raw posterior near-oracle): SUPPORTED in-model, T0.**
  - raw ep_any_unsafe 1.7 % < α; Bayes q05 0.17 % with regret 9.5 s ≤ simM's 11.1 s.
  - Calibration buys the guarantee, not safety, in regime 1.
- **21:00Z H3 (probing): FALSIFIED in regime 1** (T0 N1). EVOI = v2. Every time-matched probe policy has higher
  regret. Open only in regime 2 (≥ 3-module yaw, N5).
- **20:55Z H12 (silent extrapolation). PARTIAL; SUPPORTED for ≥ 3-module children (exploratory, post-hoc
  restriction).** Sources `results/stress/N3_flips.csv`, `N3_verdict.json`.
  - With the fixed reference: undominated flips 2.76 % [2.21, 3.35]; dominated 1.82 %, all of it in singles and
    pairs (H8 scatter).
  - ≥ 3-module children: axis 0 %, yaw 3.0–6.75 %.
  - Confirmatory test: CONFIRMATORY §3, primary cell ≥ 3-module children.
- **20:55Z, "the theory names the yaw probe". NOT MET** (`results/stress/A5y_auc.csv`: yawc AUC 0.56 < 0.75, even
  on the child). Pending N5 `wheelrate` and the fixed-reference re-score.
- **20:55Z H14 (merges) with the fixed reference: still SUPPORTED** (flips 0.16–0.17 %, `results/stress/N3_merge.csv`).
- **20:50Z H7. AT RISK on T0.** The old pilot (99.4 % useful at 0.54 % unsafe) is superseded by the scorer-audit
  fixes: leaky cross-fit, Gaussian normaliser, empty-screen healthy fallback.
  - `results/decision/t0_k5_summary.csv`, 240 episodes: sim0.1 gives 0.42 % ep_any_unsafe but only 65 % useful,
    with 35 % abstention.
- **20:50Z H11 (screen). PARTIAL.** The k = 5 screen equals the 0.95 quantile of the truth's z
  (`screen_calibration.csv`).
  - The budget-abstention guard fires on 85 % of healthy episodes (`t0_k5_meta.csv`). Sound but wasteful.
  - Fix: end-to-end + Mondrian-by-difficulty (research NEXT N1).
- **20:45Z H12 / H13 (silent extrapolation / gate). NO EVIDENCE YET.** `results/theory/gate_verdict.txt` is void: 4
  episodes per law, degenerate calibration (q_gate −1, margin ∞), and every calibrated certifier at 0 %. It must be
  rerun at full size (N2).
- **20:40Z H14 (merges plug-in). SUPPORTED at decision level; coverage transfer FAILED; size-aware margin WORKS
  (empirical).** Source `results/stress/A4_merge.csv`, oracle latents only.
  - δ_comp flips: 0.11 % (CI 0.03–0.24) in T1, 0.13 % (0.05–0.25) in T2.
  - The T1 margin miscovers 43–47 % of T2 pairs. The cause is 2-module-merge spread, not a reference offset (mean
    signed error −0.002; 46 % after de-meaning).
  - The size-aware margin gives T2 miscov 0.02 with CF 98 %.
  - Inferred-latent route pending (N4).
- **20:35Z H12 mechanism (A5m). CONSISTENT, CONFOUNDED.** Sources `results/stress/A5m_flips.csv`, `A5m_clamp.csv`.
  - Motor-only |d| > 0.05 on ≥ 3-module yaw is 13–21 %, and clamp AUC there is 0.27–0.33 (inverted), as Prop. 2b
    predicts.
  - But labels use each plant's own reference, which absorbs the law shift for healthy bodies (0 % healthy flips).
    Fixed-reference re-score: N3.
- **20:30Z, all stratum and decide numbers before the codex fixes are VOID** (`docs/codex/scorer_audit.md`, fixes
  in 02a70d9 / b08d5d5 / 9008146 / 2f6c236).

## Open questions and ideas queue (ranked by the research lead; see docs/board/research.md)
Updated 2026-11-25 ~20:15Z, after round-0 review. Full specs, novelty notes and kill lines are in research.md.

**Reframed thesis (research lead).**
- The data removes the ambiguity, not the problem.
- About half of all certified counterfactual decisions extrapolate through the motor law at operating points the
  robot never visited: diagonals, and yaw of ≥ 3-module children (Prop. 2b).
- Every plant so far shares that law, so the paper must show three regimes:
  - in-model: the raw posterior is near-oracle (H10 expected true);
  - law-misspecified: plug-in fails silently, and the trust-gated certificate holds;
  - never-existed merges.

**Proposed new hypotheses** (pre-registered; renumbered H12–H15 at 20:50Z because the ledger's H11 is the screen):
- **H12. Silent extrapolation failure.** Under a knee/current-limit motor truth, parametric and raw certificates
  exceed U* = 1 % unsafe on undominated decisions, while dominated decisions stay ≤ U*. (E1; stress T2 motor-law
  arm.)
- **H13. Trust-gated certificate.**
  - Under the two-constant truth: CF ≥ parametric − 2 pts.
  - Under the knee truth: unsafe ≤ 1 % and CF ≥ model-free + 5 pts (v2 only). (E1.)
- **H14. Merges are plug-in.** δ_comp decision flips < 1 % (T1) and < 2 % (T2). The compositional certificate's
  simultaneous miscoverage per pair is ≤ α. (E2.)
- **H15. Learning beats strong non-learned baselines.**
  - The best posterior method is ≥ 10 pts CF@U* over detector-gated, ĝ-physics, copy-parent, metadata_nf and
    min-over-members.
  - It is ≥ 5 pts over the identified-set bound with v2 only. (E3.)
  - If H15 fails, C3 is reworded to latency, v2-only and gating.

**Ranked queue.**
1. E1: trust-gated certificate on T0 under a misspecified motor law. Theory role, CPU, $0. Uses the ENVELOPE task.
   It shares the law definition with the stress T2 arm.
2. E2: A4 merge composition defect plus a certificate for never-co-assembled merges, T1 + T2. Stress role, local,
   about $0–0.2.
3. E3: trivial + identified-set + review baselines, by body size, on the pilot eids. Baselines role, CPU, $0.
4. E4: real-corpus assembly→lone counterfactual re-scoring, behind a 30-min feasibility check (≥ 30 matched pairs).
5. F4: size-aware two-simulator margin for singles and pairs (H8 exception, turned into a positive result).
6. F5/F6: decide re-scoring.
   - fixed v2+clamp (+ child-rate yaw);
   - a C_unsafe/V sweep giving the split-to-see break-even;
   - the running-max monotone certificate (the H9 fix).
7. t2-adv retargeted to a real certifier on singles and pairs (not the T1 oracle), ≤ 100 restarts.
- Not recommended now: a learned T1→T2 residual (it would spend T2's independence).

**Open questions.**
- Exact share of certified decisions that are undominated (E1 reports it).
- Whether ĝ-physics (MARS-style) is within 5 pts of the learned posterior in-model.
- Whether FDR selection (cfBH) is the better error notion for fleet assignment.

**E4 (real-data counterfactual pairs), 11-25:** the corpus can't supply faulted pairs (codex, docs/codex/e4_pairs.md).
- 329 metadata matches (114 held-out, 21 with unloaded parents), all healthy.
- Full-axis TRUTH capability labels are not verifiable from summaries; the probe gain is an ODOM peak.
- The 43 weak-wheel coupling trials have no faulted lone counterpart.
- **Decision:** real faulted square→single pairs come from the 11-24 field collection (capxfer B1/B2, cfcap's derived
  view) once it lands. The corpus allows at most a healthy cross-layout no-false-alarm check, which is low priority.

**Update 11-25 ~20:30Z (stress, docs/board/stress.md):**
- **E2 merges, SUPPORTED at the decision level.** Flips: T1 0.08–0.11 %, T2 0.12–0.13 % (500 pairs/arm), under the
  bars. 4–6-module merges flip ~0.01 %.
  - BUT per-pair coverage does not transfer: T2 miscov_pair 0.43–0.47 vs α=0.1. The T1 margin (~0.011) is thinner
    than the T2 gap (p95 0.017).
  - Honest framing: the guarantee is relative to the calibration plant. Next: a size-aware misspecification margin
    (Thm 5′ δ_M) estimated from T1 perturbations.
- **Motor-law misspecification (T2 `all+motor`):** flips are singles 4.65 %, pairs 2.57 %, ≥3 modules ≤0.32 %, and
  only faulted candidates flip.
  - Square yaw shifts > 0.05 in 21 % of decisions (L3 13.5 %): theory's predicted extrapolation risk is real, but it
    rarely flips.
  - Clamp misfit AUC: 0.82 pooled vs v2 0.55. On ≥3-module yaw it is 0.27–0.33, below chance: neither probe visits
    the child's yaw rate.
  - Prediction under test now: a child-rate yaw probe ≥ 0.75 AUC on those, which would confirm "the theory names the
    probe".

**H4 evidence, 11-25 ~20:55Z (ops measurements, exploratory):** nuisance-SMC costs ≥ 500 s per episode on an RTX 3090
(the pilot2 canary), against 3.4 ms per episode for the amortized encoder (models pilot). Composition adds
18–31 ms per candidate. If SMC's accuracy is comparable, the latency side of H4 is decisive. Accuracy is still
unmeasured: the small SMC campaign plus models eval2.

**Update 11-25 ~21:30Z: E1 (theory) and decision envelope-class probing (exploratory, T0).**
- **E1 trust gate: both pre-registered SUCCESS bars FAIL; KILL not met.** (results/theory/gate_cert.csv, gate_verdict.txt;
  1,200 parents, ~57k decisions per law × probe; 2-fold cross-fit after a fold/law correlation was fixed before scoring.)
  - **H12, silent failure, SUPPORTED.** Trusting the motor law fails silently: physics-trusting episode-unsafe rises
    4 % → 74 % at knee 3 (v2); uncalibrated 11 % → 89 %.
  - **Calibrating over episodes that include wrong laws restores ≈α validity**, but conservatively: par_pooled CF
    78.9 / 73.6.
  - **The misfit gate as a method: FALSIFIED (F1 killed).** It adds little over the calibrated model-free bound once
    clamp has run. What survives is "use dominance after clamp".
  - **Clamp buys certified share back:** model-free calibrated CF 72.4 → 90.7 % at the correct law, 66.9 → 79.0 % at
    knee 3, at ≤ 0.4 % unsafe.
  - **Extrapolated share of decisions the physics-trusting certificate approves:** 63 % with v2, 21 % with v2+clamp.
- **Extrapolated share grows with size (dominance_vs_N.csv, speed side, N up to 16).**
  - v2: 53 % → 57 % → 59 % (N = 4 / 8 / 16).
  - v2+clamp: 6 % → 14 % → 17 %, all yaw: child yaw legs 31 → 70 → 85 % extrapolated, because larger parents yaw
    slower.
  - v2+clamp+2×yaw: 0 → 5 → 8 %.
  - A generalizable, size-dependent statement.
- **H3 probing: SUPPORTED in regime 2** (decision, results/decision/envelope_*.csv; 463 usable E1 episodes, true fault
  modes known, so this isolates motor-law uncertainty).

  | policy | useful | regret | probe time |
  |---|---|---|---|
  | v2 | 83 % | 31.4 s | 0 |
  | always clamp | 95 % | 19.4 s | 8.8 s |
  | decision-EVOI | 95 % | 14.3 s | 2.05 s (probes 23 % of episodes) |

  - EVOI matches always-clamp on usefulness, with 27 % less regret and 4× less probe time. Probing becomes a headline
    in regime 2 only.
- **N1 PASS (decision):** simM0.1 per-bin episode-unsafe ≤ 1.0 % (upper ≤ 2.4 %), healthy abstention 0 %, useful
  95.4 %.
  - Monotonicity is fixed: certified pairs are flat along nested probe sets (10,636 → 10,631). The cause was the probe
    likelihood's per-particle nuisance.
  - The conformalized screen is a compute knob (misses 5.2 → 2.5 %, 3× cost); end-to-end calibration carries the
    safety.

**H16. Realistic missions require certified reconfiguration (env, exploratory pilot, 11-25 ~21:40Z).**
- Procedural buildings are calibrated to 2,210 HM3D missions (KS D < 0.1 on every matched stat; worst is tortuosity
  at 0.086; door widths are a design choice, since HM3D records lack them).
- Pilot: about 3 % of physically fitting segments still fail in closed-loop truth.
  - Assuming every body is nominal fails 13–14 % of missions; biggest-body 10 %; greedy smallest 20 %.
  - The oracle splits or merges in about 1/3 of plans (bar ≥ 20 %).
- So split/merge decisions matter in building-like missions, and a certificate has room to help.
- Confirmatory C-ENV is pending models' per-segment bounds.

**Update 11-25 ~21:40Z (stress; pre-written bars):**
- **N5 parent yaw probe: FAILED.** The wheelrate AUC is 0.44 (yawc 0.44) against a 0.75 bar. Under the fixed
  reference, 79 % of ≥3-module episodes shift > 0.05, so no trace statistic ranks them. **Drop the "theory names the
  yaw probe" claim.** Clamp's value stays (E1; envelope EVOI).
- **N7 size-aware single/pair margin: narrowly NOT met.** Unsafe falls 4.3 → 0.85 % (singles) and 3.0 → 0.28 % (pairs),
  but the CF loss is 7.1 / 6.7 pts against a ≤ 5 bar. Report it as a safety/usefulness trade-off.
- **N3: neither supported nor falsified as pooled.** Undominated 2.76 %, dominated 1.82 % (all dominated flips are in
  singles and pairs, from friction scatter and the kernel, even without the motor law). Within ≥3-module children the
  predicted pattern is exact: surge/sway 0 %, yaw 3–6.75 %.
- **N4 (merge with inferred latents): running.** Moved to rented boxes.

**H17. Faithful explanations of capability limits (annotation, codex-validated, exploratory T0).**
- Explanations generated from the robot's own posterior are faithful 89.5 % of the time (CI 84.5–93.5 %), and 99.2 %
  when the refusal was correct. A blamed wheel was truly faulted 193/193 times.
  - Failure mode: when the robot wrongly refuses a capable body, it blames a leg that isn't short (20/20).
  - Codex reported the belief correctly 100 % of the time, so every error comes from the posterior, not from the LLM.
- **Failure taxonomy:** 1,181 cases, 95.7 % of 4,491 checkable claims true. Six categories validated (near_threshold,
  probe_overconfidence, conformal_margin_slack, passive_wheel_hidden, mode_confusion, spread_masked_weak).
  - Of the unsafe certificates, 23/43 are screen misses (codex labels them correctly 19/21) and 9 are a passive wheel
    in the child that v2 never excites.
  - The LLM's attribution of A5 flips to plant effects is unreliable (55 % true), so cite the stress ablation instead
    (95 % of flips also occur with friction scatter alone).
- **Task library:** 558 realistic tasks; the median requirement is 0.37, well below the synthetic τ 0.6–0.9. Realistic
  jobs are easier than our decision thresholds, so report results at a realistic-requirement τ as well.

**H3 probing: SUPPORTED on fresh episodes (decision, 11-25 ~22:00Z; exploratory T0; pre-written bars).**
- 583 fresh episodes (seeds 2100–2699); fault modes from the posterior AND the motor law uncertain; calibrated envelope
  bound.

  | policy | useful | regret | probe time |
  |---|---|---|---|
  | v2 | 52.8 % | 79.2 s | 0 |
  | always clamp | 63.8 % | 69.2 s | 8.8 s |
  | dominance trigger | 63.7 % | 66.1 s | 5.5 s |
  | EVOI | 63.6 % | 62.8 s | 1.6 s (probes 18 % of episodes) |

  All policies 0 % unsafe.
- Paired episode bootstrap on regret: EVOI − clamp −6.4 s [−7.0, −5.6]; EVOI − dominance trigger −3.3 s [−4.1, −2.5].
- Caveats:
  - Usefulness is only 64 % (vs 95 % with fault modes known), the cost of a conservative bound over plausible faults.
    Probing recovers 11 pts of that.
  - Clamp legs are not used in the fault likelihood (they empty the screen on knee episodes).
  - ≥3-module yaw: 0 % unsafe here but only 30–50 % certified; with true modes it stays 2.4–3.6 % unsafe under every
    policy.
- This makes decision-aware probing a supported contribution in regime 2: EVOI beats the cheap closed-form rival with
  ~¼ of the probe time.

**H3 replication + NEXT-1 (decision, 11-25 ~22:10Z; pre-written bar; fresh seeds 2700–3299, 588 episodes):**
- **NEXT-1 FAILED** (clamp legs in the fault likelihood: useful 64.8 % vs a 75 % bar; episode-unsafe 0.17 %).
- **H3 REPLICATES** in both arms: EVOI − always-clamp −6.6 s [−7.2, −5.8], EVOI − dominance −3.7 s [−4.4, −3.0];
  EVOI probes 1.45 s vs 8.8 s.
- **Root cause of the usefulness gap: the v2 hypothesis screen**, not the likelihood.
  - 30 % of episodes have an empty screen: 2 / 7 / 40 / 75 % at knee severity 0 / 1 / 2 / 3.
  - Where the screen is non-empty, EVOI is 94.6 % useful.
- Next: a knee-tolerant screen (threshold calibrated on exploratory knee episodes), on fresh seeds 3900+.

**H3 + calibrated knee-tolerant screen: SUPPORTED (decision, 11-25 ~22:45Z; pre-written bars; calibration seeds
5000–5599, evaluation fresh 3900–4499, 585 episodes; exploratory T0).**
- Screen threshold K = 28.7 (split-conformal on exploratory knee episodes). Empty screens fall 30.1 % → 0.5 % overall
  (severity 3: 80 % → 1.5 %).
- EVOI: useful 64.9 % → 93.1 %, episode-unsafe 0.34 %, regret 16.9 s at 1.71 s of probing.
- vs always-clamp: useful 93.5 %, regret 22.9 s, 8.8 s of probing; vs the dominance trigger: regret 18.8 s, 4.5 s.
- Paired regret: EVOI − clamp −6.0 [−6.9, −4.9]; EVOI − dominance −1.9 [−2.8, −0.95].
- Caveats:
  - truth is in the kept hypothesis set only 52 % of the time, so end-to-end calibration carries the guarantee;
  - the uncalibrated bound is 12.5 % unsafe on large-child yaw legs, and the calibrated one 0–4 %;
  - the convergence filter still under-represents severe knees on large bodies.
- Contribution statement: *a screen calibrated on misspecified-law episodes, plus end-to-end calibration, gives a
  posterior that stays useful when the motor law is wrong; decision-aware probing then saves 3–4× probe time over
  fixed or heuristic probing at equal usefulness.*

**Held-out severity + task-relevant share (decision, 11-25 ~23:30Z; pre-written bars; exploratory T0):**
- **Robustness to an unseen motor law: SUPPORTED.**
  - Screen K fitted on severities 0–2 (26.9) and the certificate calibrated on 0–2 only; tested on 267 fresh
    severity-3 episodes.
  - Empty screens 2.6 %, EVOI episode-unsafe 1.1 % (α 0.10), useful 88.5 % (pooled 93.1 %). Paired regret: EVOI −5.9
    vs clamp, −1.7 vs dominance.
  - K's exact value doesn't matter once it doesn't empty the screen.
- **Task-relevant certified share is not equal:** EVOI 81.4 %, clamp 85.8 %, dominance 85.5 %, v2 72.0 %. EVOI
  serves tasks through other certified plans (useful 93.1 vs 93.5 %). Clamp bought net utility in 13/585 episodes
  (2.2 %); EVOI did better in 458.
  - Honest framing: EVOI certifies slightly fewer of the oracle's own pairs, but loses ≤ 0.4 pt of usefulness while
    saving 7 s of probing.

**N4, merge certificate with INFERRED latents (stress, 11-26 ~01:30Z; pre-written bars; exploratory, 500 pairs/arm):**
- comp_enum, T1: decision-stratum CF 99.9 % at 0.03 % unsafe (bar ≥ 85 % at ≤ 1 %): MET.
- Beats copy_parent by +33 pts (bar ≥ 10): MET.
- T1 per-pair miscoverage, cross-fit: 0.108 / 0.110 vs bar ≤ 0.10: narrowly MISSED (within split variance 0.10–0.12;
  reported as missed).
- T2 per-pair miscoverage with the size-aware margin: 0.034 / 0.036 (bar ≤ 0.15): MET. Margins by merge size
  0.17 / 0.05 / 0.05 / 0.04 / 0.03 for 2–6 modules; frozen in N4_margins.json (460999c) before any conf read.
- **Merges of modules never assembled together are certifiable with inferred latents (H14 supported).**
