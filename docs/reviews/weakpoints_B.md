# Weak points, reviewer B (fresh skeptical ICML 2027 read, 2026-11-25)

Inputs read: paper/main.tex + sections, numbers.tex, tables/*.csv/.tex, reviews/OUTLINE.md, docs/theory/THEORY_cfcap.md.
Not read: results/confirmatory, HYPOTHESES, board.

## Likely scores (current evidence + planned argument)
| Novelty | Significance | Soundness | Clarity | Overall |
|---|---|---|---|---|
| 5/10 | 4/10 | 4/10 | 4/10 | **4/10 (weak reject)**, 5 if fixes 1-3 land |

The core problem: the evidence that exists does not back the paper's method. The headline certificate is a
**baseline** (the black-box GNN). The amortized posterior, SMC, capxfer and compositional rows are empty. The trust-gate
experiment fails its own rule and is beaten by a simpler certifier. The oracle shows the problem is ~99.98 % identified.

## The 8 weakest points (with the cheapest fix)

1. **The headline number belongs to a baseline.** `cfTOneBestMethod = Black-box counterfactual GNN` (97.8 %). The
   abstract calls it "our certificate". The Baselines paragraph lists "a black-box counterfactual network" as a baseline.
   The method section's amortized posterior (set transformer + physics rollout) has no T1 row. The T3 "Amortized (wide
   DR)" row (93.9 % / 0.39 %) is identical to the "Deep ensemble" row in t1_nominal. So it is either a mislabel or the
   same model under two names. A reviewer will conclude that physics adds nothing over a black box.
   *Fix (writing + ≤30 min compute):* decide what "ours" is. If it is the GNN plus a simultaneous conformal wrapper,
   say so and reframe the contribution as the certificate, not the posterior. Otherwise run the physics-pushed
   posterior on the existing dec_models test set. Use one display name per model in every table.

2. **The matched-1 % operating point hides unsafe merges.** In t1_certificate.csv (T1, rel), the GNN at the matched
   pooled 1.47 % has **12.1 % unsafe on merges** (206 incapable), 6.9 % on size 2 and 17.4 % on size ≥3. The ensemble
   has 8.7 % on merges. The low pooled rate comes only from the 2,267 lone-faulted incapable decisions (88 % of all
   incapable). This directly undercuts claim (iii), "composition certifies merges".
   *Fix (≤30 min compute, CSV already exists):* report unsafe per group next to certified share in Table 1. Match the
   operating point per candidate kind (Mondrian by kind), or say that the guarantee is pooled. Note also that the
   GNN's "all" unsafe of 1.47 % is above the 1.0 % it is said to be matched at; the caption should say "matched on
   calibration".

3. **The trust gate (E1) is dominated by simpler certifiers, and the caption contradicts the CSV.** f4_gate.csv, knee
   severity 3, decision stratum:
   - v2 probe: gate 72.1 % cert / 1.72 % unsafe; `par_pooled` 73.6 % / 1.09 %, which is better on both.
   - v2clamp probe: gate 77.4 % / 0.43 %; `pi_cal` 79.0 % / 0.43 %.

   The caption says the model-free set "stays safe". But `pi` reaches 2.27 % unsafe (v2) and **13.3 %** (v2clamp).
   numbers.tex already says "E1 fails its pre-written success rule".
   *Fix (writing):* drop the gate as a contribution. Report it as a negative result: pooled calibration of the
   parametric certificate repairs misspecification. Rewrite the caption to match the CSV.

4. **There is little left to certify, so the theory explains why the problem is easy.** Oracle ambiguity is 0.02 % (24
   decisions). Prop. 3 predicts 0.09 % at 16 % spread. Composition flips 0.11 % / 0.13 % of merge decisions. The capable
   share in the decision-relevant stratum is 97.5 %. A reviewer will read "near-oracle certification" as "the task is
   nearly deterministic given a 12-s probe". The probing motivation also collapses: the intro's own TODO says C4
   needs ≥5 % ambiguous.
   *Fix (writing):* lead with the misspecified/knee regime, where the ambiguity is real. Shrink the T0-identified story
   to a single "why it is easy" paragraph.

5. **The theory looks thin.** Lemma 1 is linear DC-motor algebra (u = aα + bβ). Prop. 2(a) is a monotone
   partial-identification bound (Manski-style). Thm. 5′ is a union bound over stage-1 events plus an unmeasured
   δ_comp. The simultaneous certificate is the standard max-score split conformal. Risk control is cited. The
   appendix proofs are a TODO.
   *Fix (writing):* present Lemma 1 and Thm. 5′ as tools, not contributions. Make Prop. 2(b)/(c) (which legs and sizes
   are undominated for polyominoes) and Prop. 4 (measured slack attribution) the theory claims. Fill in Prop. 4's
   numbers, which are still a TODO. Add short proofs.

6. **Key baselines are missing, and the strongest comparison is not like for like.** These rows are empty: capxfer
   (TOneCapxferCertTOne, GainPts), GPU-SMC (t3_grid), the compositional route (t5_merge_cert has no rows), the
   dominance PI (E3 says "dominance_pi missing") and every T2 column of Table 1.
   - The intro compares an oracle at 99.98 % over all child/merge decisions with capxfer's 58-75 % on lone modules,
     taken from capxfer's paper. Those are different populations and different runs.
   - The latency claim (0.0034 s vs 3.5 s) compares against envpost K5 max, not SMC. Ktol takes up to 12.3 s, and
     enumeration takes 3.9 s.

   *Fix (≤30 min compute):* run the capxfer certificate on the lone-faulted cell of the existing test set; that cell
   is directly comparable. Label the latency baseline exactly. Drop the SMC row from the table if it will not run.

7. **Everything is in simulation, with a single platform and a single actuator class.** Mecanum X polyominoes on a
   hardware-identified T1 plus a T2 written by the same team. T2 has no certificate results yet. "Generality" rests on
   one kinematic family.
   *Fix (writing):* say clearly that this is a sim-only, single-platform study. If any real split from the capxfer
   fleet logs exists, score it as a sanity check. Otherwise move "robotics" out of the selling points.

8. **The statistics and splits are thin, and held-out coverage misses its target.**
   - Test groups have pair n_ep = 31, square n_ep = 5 and 2x3 **n_ep = 1**. The per-shape columns of Table 1 are
     effectively single episodes.
   - Episode-simultaneous coverage at the 90 % target is 88.9 % on H4 for both models. On all decisions it is
     86.7 % (GNN) and 85.0 % (ensemble), about 3 SE under the target. Even the in-distribution test split is at
     89.0 % (ensemble, all), which argues against a guarantee.
   - The diversity result is null: 98.9 % vs 99.1 %.
   - The probing gain over the dominance trigger is 1.9 s of regret, measured on a T0 plant only.

   *Fix (≤30 min compute + writing):* drop the per-shape columns or merge them into "single assembly". Report
   binomial CIs on ep_cov. State the H4 under-coverage. Present diversity as a null result or move it to the appendix.

## Numbers in the draft that disagree with the CSVs, or look suspicious
- **Abstract/intro "our certificate 97.8 %":** this is the blackbox_gnn row (t1_certificate.csv, all, cf = 0.9781), a baseline.
- **"Matched 1.0 % unsafe":** the GNN's test unsafe is 1.47 %, and on merges it is 12.1 %.
- **T3 "Amortized (wide DR)":** 93.9 / 0.39, identical to the "Deep ensemble" numbers in t1_nominal. Mislabel or duplicate?
- **Fig. 4 caption:** "model-free set stays safe" and "gate keeps parametric share at model-free safety". Both are
  contradicted (pi unsafe 2.27 % and 13.3 %; par_pooled and pi_cal dominate the gate).
- **`cfFTwoGainPts = -0.1`:** the displayed values give 98.9 − 99.1 = −0.2. This is a rounding artefact, but it is visible.
- **Abstract "simulator-likelihood inference 3.5 s":** the source is envpost K5 (max 3.49). Envpost Ktol is 1.5-12.3 s,
  and "GPU-SMC" has no timing.
- **Placeholders that render red:** 28 `\cfPH` remain, among them TOneCapxferCertTOne, TOneGainPts, TOneBestCertTTwo,
  all of TFiveCertComp, TThreeVerdict, FFourUndomShare and TFourResolved*.
- **Intro "oracle certifies 99.98 % ... capxfer 58-75 %":** the populations differ (all decisions vs lone modules).
  On this repo's own data, lone faulted is 81 % for the GNN and 72 % for the ensemble.
