# Review, round 0: "Can I Trust My Next Body? Certified Counterfactual Capability for Reconfigurable Robots"

Reviewer stance: independent and skeptical (robotics and conformal prediction). Read: `paper/` (built with tectonic
from `main.tex`, all sections, tables, `numbers.tex`), `docs/theory/THEORY_cfcap.md`, code under `cfcap/` (`conformal/core.py`,
`decide/{policies,evaluate}.py`, `t2/kernel.py`, `t2/adv.py` header) and the CSV/JSON files under `results/`. I did not read
`docs/board/` or `docs/HYPOTHESES.md`. Most headline numbers are still placeholders, so this review judges the design and
claims against the exploratory evidence that exists (Stage A, theory checks, the T0 decision pilot, the T1→T2 flip test,
model pilots).

## Scores

| | Score |
|---|---|
| Novelty | 3 / 5 |
| Significance | 2 / 5 |
| Soundness | 2 / 5 |
| Clarity | 3 / 5 |
| **Overall** | **4 / 10** (weak reject in its current form; it could reach 6 to 7 if fixes 1 to 3 land) |
| Confidence | 4 / 5 (I checked code and CSVs; the tables I scored against are placeholders) |

## Summary of what the evidence currently says

- **With the plant known, the problem is nearly solved by the probe the robot already runs.** The oracle certifies 99.98 % of
  capable decisions, and 0.02 % of decisions are ambiguous (24 in total; `results/stageA/A2_verdict.json`). All 24 are
  diagonal legs of a lone faulted module (theory doc section 2).
- **Nuisance does not change that.** Prop. 3's band gives 0.09 % ambiguous decisions at 16 % motor spread
  (`results/theory/nuisance_curve.csv`). The theory doc itself concludes that nuisance "cannot by itself produce the ≥ 5 %
  ambiguity that would make probing a headline."
- **Probing does not help in the T0 decision pilot** (`results/decision/pilot_t0_summary.csv`, 600 episodes, 2 to 4 modules):
  - No probing (`posterior,v2,sim0.1`) has regret 7.65 ± 2.9. Decision-EVOI has 9.83 ± 3.3, EPIG 9.52, EIG 11.5.
  - EVOI runs 0.02 probes per episode (0.18 s). In practice it is the fixed v2 policy.
  - The "7846 % per extra minute" in Table 4 comes from dividing by about 0.2 s of probe time. It is not evidence.
- **The full probe library behaves pathologically.** It resolves 137 of 138 ambiguous decisions but lowers certified
  decision-relevant pairs from 7341 to 5339 (`results/decision/paper/probe_pilot_t0.csv`). Its simultaneous quantile is
  q = 5.23, against q ≈ 1.1 for v2. The certificate gets worse as information grows (weakness 3).
- **The uncalibrated posterior is already near-oracle.** With the raw posterior (q = 0), the unsafe-commit rate is 1.08 % and
  92.7 % of capable decisions are served (oracle 100 %). CRC at α = 0.05 and 0.1 returns λ = 0 (`q_median` 0): the risk-control
  layer does nothing at the levels the paper uses. It only binds at α = 0.01, where it costs about 22 points of commits.
- **T2 is too close to T1 to count as a stress test.** Only 0.5 % of decisions flip from T1 to T2 (`results/stress/A5_verdict.json`).
  T2 is said to be independently written, but it uses the same linear DC torque-speed law (`cfcap/t2/kernel.py` docstring:
  "the linear torque-speed curve"). That is the exact two-constant model of Lemma 1. The theory doc names the
  failure mode that matters, "a misspecified motor law (current limit, thermal derating, nonlinear torque–speed curve, T2)",
  and T2 does not contain it.
- **The latency headline compares against the wrong baseline.** The abstract says 0.0027 s versus 3.9 s "for
  simulator-likelihood inference". The 3.9 s is exact enumeration on a 2×3 episode (`results/stageA/A6_summary.csv`), not the GPU-SMC
  baseline. It is also unclear whether the amortized time includes the per-candidate T1 rollouts that turn posterior draws
  into ρ samples. `results/models/latency.csv` reports 3.4 ms "per episode" for the amortized model, against 17 to 31 ms
  for the compositional route.

## The 5 biggest weaknesses, ranked by how much fixing them would move my score

### 1. The paper's own evidence removes the problem it motivates ("which one short test settles it")
**Problem.** The headline story, from the title through the abstract to contribution 3, is that the robot should know which
short test settles an ambiguous decision. But:
- ambiguity is 0.02 % with the plant known and 0.09 % with ±16 % nuisance;
- probing loses to not probing in the only decision experiment that exists;
- the abstract's quantitative support for Prop. 3 is "0.09 % of decisions at 16 % spread", a number that argues against
  the motivation.

Both "the problem is hard" and "here is the probe that solves it" cannot hold. At ICML this reads as a solution looking for
a problem, or as a benchmark built so the problem is easy.

**Fix (largest score mover).** Do one of the following:
- **Reframe:** the contribution becomes "when the motor law is right, one routine probe identifies almost every
  counterfactual decision; the risk is misspecification, and here is a certificate that stays valid under it". Then
  move EVOI to the appendix, as the authors' own todo already plans.
- **Find the regime where ambiguity is real:**
  - the nonparametric envelope class of Prop. 2(a), where 20.7 % of decisions are ambiguous at τ = 0.8 with v2
    (`coverage_speed_only.csv`);
  - unidentified normal-load splits, μ at high slip, and yaw legs of children with 3 or more modules (undominated in
    96 to 98 % of cases even after the clamp probe);
  - realistic fault base rates.

  In that regime, show EVOI against fixed v2+clamp, the minimal probe the closed form predicts, with episode-bootstrap CIs
  on regret and certified share. If v2+clamp matches EVOI, report that: it is a cleaner and more useful result than a
  learned policy.

### 2. The guarantee is stated against a synthetic prior, and the headline compares methods at a tuned operating point instead of by their guarantees
**Problems.**
- Coverage is "marginal over generator episodes". The authors choose that generator: 1 to 2 faulted wheels in most
  parents (3 of 4 parents in the theory checks), a fixed fault mix and fixed g ranges. A marginal guarantee under that
  prior tells a fleet operator nothing about a fleet with a different fault base rate. Performance on the
  decision-relevant stratum is not guaranteed at all, because the stratum is defined by labels.
- The headline comparison (Table 1) puts every method at a "matched unsafe-commit rate of 1 %", with the margin tuned on
  calibration data. That is an ROC-style comparison of point predictors. A well-ranked uncertainty score from any baseline,
  given a tuned margin, would compete. The conformal machinery that gives the paper its title plays no part in the
  headline.
- The levels do not line up. Nominal α = 0.1 is per-episode simultaneous, the matched rate is 1 % per decision, and
  Prop. 4's numerical check uses α = 0.05.

**Fix.**
- **Lead with each method's own guaranteed bound** at a stated α. Report realized episode-level miscoverage with CIs, then
  certified share. Use the matched operating point only as a secondary diagnostic.
- **Add conditional validity:** Mondrian calibration by fault class, candidate kind and size, or a stated
  covariate-shift reweighting (the `weighted_quantile` in `core.py` is already written and unused).
- **Run a prior-sensitivity experiment:** calibrate under the training fault prior, then test under shifted fault rates
  (0.1×, 0.5×, 2×), a different g distribution and unseen fault types.
- **Cite and compare against conformal selection** (Jin and Candès, below), which is exactly the "commit iff Y > c" problem
  with FDR control.

### 3. The simultaneous certificate is not monotone in information, and normalized max-scores are fragile
**Problem.** The score is `max_j (pred_j − truth_j)/scale_j` with `min_scale = 1e-6` (`conformal/core.py`). When probing
collapses the posterior, scales shrink, a single residual dominates the max, and q goes from about 1.1 to 5.23. The
certified share then drops by 27 % even though ambiguity was almost eliminated (fixed-all row). A certificate that
punishes the robot for running more probes undermines both the probing contribution and the simultaneity contribution.
There are two further issues:
- family sizes are ragged (7 to 51 decisions per episode), so the max-score distribution mixes very different
  multiplicities;
- the simultaneity price measured in Prop. 4 (2.4× to 12× in q, up to 18 points of certified share) is the paper's
  strongest argument for composition, but no real model has been used to measure it yet.

**Fix.**
- Use a scale floor tied to the label resolution, an unnormalized or hybrid score, or a per-family-size Mondrian quantile.
- Show that the certified share is non-decreasing along a probe sequence, by construction or empirically.
- Report the Prop. 4 attribution (estimator, simultaneity, finite-n) for the real amortized model and for SMC, not only
  for synthetic Gaussian or t₃ residuals.

### 4. The robustness claims rest on a plant that shares the critical modelling assumption, with no hardware decisions
**Problem.**
- T1 and T2 both use the linear DC motor law of Lemma 1, so the "parametric regime" (identification by extrapolation
  along a known line) is in-model by construction in every test. The theory doc says this outright.
- Only 0.5 % of decisions flip from T1 to T2, so a sim-calibrated certificate is barely stressed.
- No real split or merge decision is scored, even though T1 is said to be identified from real fleet logs and the
  hardware exists.
- δ_comp, the only merge-specific term in Thm. 5′, is "measured, not bounded", and has not been measured in T1 or T2
  yet (A4 pending).

**Fix.**
- **Strengthen T2's motor model:** add a current limit, thermal derating, a nonlinear torque-speed curve with friction
  breakaway, and backlash or deadband. Report how far the certificate degrades as misspecification grows.
- **Finish and report the adversarial CEM search** (`cfcap/t2/adv.py`): the rate and size of confident-wrong cases for the
  real certifier, not only for the T1 oracle.
- **Run a small hardware block:** even 20 to 40 real splits and merges on the six-module fleet, with predicted lower
  bounds against measured ρ. A handful of real counterfactual decisions, including merges of modules never docked
  together, would change the paper's standing more than any amount of simulation.
- **Bound δ_comp empirically** across dock stiffness and play, so Thm. 5′ produces a usable number.

### 5. The ML claim (amortization) is under-tested, and the delta over capxfer2026 is unclear
**Problem.**
- The 2×2 test (posterior × plant) is empty. The latency comparison uses enumeration rather than SMC and may leave out
  the per-candidate rollouts.
- The pilots show the uncalibrated net under-covers on faulted wheels (`results/models/cf.csv`: direct_n4 cov90 = 0.79 on
  faulted pair children, median width 0.017, versus 0.87 / 0.32 for the marginal baseline). That gap is what conformal
  has to repair.
- The paper builds on an anonymous "concurrent submission" (capxfer2026) for masking, reveal-on-split, the v2 probe, the
  compositional certificate T5 and the plant identification. The only stated delta is "decisions about other bodies".
  A reviewer cannot judge that increment without the other paper. The numerical gain over capxfer is also reported on a
  different population: capxfer's 58 to 75 % is for lone modules, while the comparison here uses the decision-relevant
  stratum of children and merges.

**Fix.**
- Fill the 2×2 with matched compute and end-to-end wall time per decision, including candidate rollouts. Report the
  amortization gap against SMC on T2 and on held-out families H3 and H4.
- Run capxfer and the new method on identical episodes and decisions, broken down by candidate type: lone child,
  multi-module child, merge.
- Add a supplementary note or diff against capxfer2026 that states which theorems and code are new.

## Novelty check against the closest prior work (2024–2026)

Snippets below are marked **[snippet]**. They are search-result excerpts that I did not verify against the full text.

**Conformal capability and failure certification for robots.**
- Conformal failure prediction for robot policies calibrates on successful rollouts, e.g. FIPER and FAIL-Detect:
  - **[snippet]** "Both failure prediction scores are calibrated using a small set of successful rollouts via conformal
    prediction" ([FIPER, arXiv 2510.09459](https://www.alphaxiv.org/abs/2510.09459)).
  - [FAIL-Detect, RSS 2025, arXiv 2503.08558](https://arxiv.org/pdf/2503.08558).
- Performance guarantees from simulator rollouts: [arXiv 2309.10874](https://arxiv.org/pdf/2309.10874).
- Schubert et al. (cited) certify one fixed vehicle.
- None of these certifies capability of *other* bodies (children and merges). On this axis the problem formulation is
  new, as far as I can tell.
- The certificate machinery is not new: split conformal, a max-score simultaneous quantile and CRC.

**Conformal selection, the closest statistical prior work, missing from the paper.** "Commit iff the lower bound ≥ τ over a
family of candidates" is a selection problem.
- **[snippet]** Jin and Candès study "screening procedures that aim to select candidates whose unobserved outcomes exceed
  user-specified values ... while controlling the proportion of falsely selected units"
  ([JMLR 24, 2023](https://jmlr.org/papers/volume24/22-1176/22-1176.pdf)).
- Selection-conditional coverage ([Bao et al., arXiv 2403.03868](https://arxiv.org/pdf/2403.03868)),
  [Multivariate Conformal Selection, arXiv 2505.00917](https://arxiv.org/pdf/2505.00917) and
  [Batch Predictive Inference, arXiv 2409.13990](https://arxiv.org/pdf/2409.13990) cover the simultaneous and batch case.
  **[snippet]** for batch inference: "coverage for each set separately says nothing about how the sets behave jointly".
- The paper should position its FWER-style per-episode max against FDR-style selection. FDR control would likely certify
  far more decisions and may be the better-matched error notion for fleet task assignment.

**Robot self-models under morphology change.**
- **[snippet]** Hu et al.: "a meta-self-model that can deduce robot morphology through proprioception ... predict unseen
  robot configurations" ([arXiv 2403.10496](https://arxiv.org/pdf/2403.10496)). This work is cited, but it infers
  configuration, not capability.
- Missing and closer: modular aerial robots under faults. **[snippet]** Huang et al., ICRA 2025, give "an efficient
  controllability analysis for MARS using a quasi-static model under control constraints and a novel controllability
  margin (CM)–based method for calculating optimal self-reconfiguration sequences"
  ([arXiv 2503.09376 / code](https://github.com/ruihuangnus/mars-reconfig)); see also
  [TransforMARS, arXiv 2509.14025](https://arxiv.org/pdf/2509.14025).
- This is a quasi-static capability margin for reconfigured bodies after actuator faults. It is the obvious
  non-learned, non-certified baseline (weakness 5 and the baselines section).
- Also relevant: fault-tolerant self-reconfiguration of chain modules
  ([Qi et al., 2024](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202300774)) and Mathews et al.'s
  mergeable nervous systems for split and merge.

**Decision-aware experimental design.**
- TNDP is cited. **[snippet]** "decision utility gain (DUG) to guide the next design choice toward downstream decision
  tasks" ([arXiv 2411.02064](https://arxiv.org/abs/2411.02064)).
- Not cited:
  - GoBOED ([arXiv 2605.26093](https://arxiv.org/pdf/2605.26093)). **[snippet]** "reducing parameter uncertainty doesn't
    necessarily improve downstream decisions".
  - Action-BED ([arXiv 2606.23662](https://arxiv.org/pdf/2606.23662)), which notes the expected-future-loss criterion
    goes back to Lindley.
  - Loss-driven Bayesian active learning (AISTATS 2026).
  - Robust decision-aware BED under misspecification ([arXiv 2607.08590](https://arxiv.org/pdf/2607.08590)).
- Decision-EVOI over a finite physical probe library with time costs is classical value-of-information analysis
  (Raiffa and Schlaifer, Howard). On novelty this contribution is weak, and the evidence does not support it either.

**Partial identification for control.**
- Prop. 2(a) is monotone-class partial identification (Manski-style bounds), applied to torque envelopes.
- The closest control work is set-membership identification of uncertainty sets
  ([arXiv 2309.14648](https://arxiv.org/pdf/2309.14648), ICML 2024) and reachset-conformant identification
  ([arXiv 2407.11692](https://arxiv.org/pdf/2407.11692)).
- Decision rules under partial identification come from econometrics
  ([Robust Bayes treatment choice, arXiv 2408.11621](https://ideas.repec.org/p/arx/papers/2408.11621.html); "Certified
  decisions", arXiv 2502.17830).
- None is cited. Applying partial identification to *counterfactual morphologies* through operating-point dominance is a
  modest but real novelty.

**Sim-to-real conformal.**
- Cited: Chenni et al. 2026 ([arXiv 2611.26065](https://arxiv.org/pdf/2611.26065) per search).
- Not cited: robust CP over a shift ambiguity set ([arXiv 2602.12616](https://arxiv.org/pdf/2602.12616)) and
  SCAPE's paired sim/real bias correction ([arXiv 2608.19425](https://www.alphaxiv.org/abs/2608.19425)). These are the
  obvious tools for the T1→T2 and sim→hardware gap.

**Verdict on novelty.**
- New: the problem (certifying children and merges of reconfigurable robots) and the operating-point dominance
  characterization.
- Not new: the statistical tools and the probe-selection rule.

## Are the theory results non-trivial or textbook?

| Result | Assessment |
|---|---|
| **Lemma 1** (two-constant reduction) | Textbook DC motor algebra (duty = torque/stall + speed/no-load). The modelling insight is useful: a weak fault and motor spread are the same object, and the healthy probe has a ≈ 0 so it is blind to α. It is one paragraph, not a lemma worth a contribution bullet. |
| **Prop. 2** (operating-point coverage) | (a) is standard monotone-class partial identification, plus a Le Cam two-point bound the theory doc itself calls textbook. (b) is an exact kinematic closed form for X-mecanum polyominoes. It is simple, but it gives the most actionable statement in the paper: diagonals need 2d/v = 1.4× the probe speed, the clamp probe dominates them, and yaw legs of children with 3 or more modules are covered by no translation probe. (c) is linear algebra. **The most interesting result, mainly as engineering insight rather than theory.** |
| **Prop. 3** (nuisance band) | A first-order Taylor expansion of a box image. Linearity in δ follows directly. The structural fact (S^α = 0 on tracking legs) is model-specific and nice, but the result yields 0.09 % ambiguity, which undercuts its relevance. Textbook. |
| **Prop. 4** (slack decomposition) | An accounting inequality, (u+v+w)₊ ≤ u₊+v₊+w₊, plus Bonferroni. The authors call it "elementary", and it is vacuous beyond the "cliff". Textbook. Its value depends on measuring the terms for real models, which has not been done. |
| **Thm. 5′** (compositional merges) | A union bound over per-assembly coverage events, extending the concurrent submission's T5. "Family-free" is true by construction because the stage-1 event does not depend on B. The simultaneity cost moves into stage 1 (per-wheel levels α₁/(2n_G)); it is not removed, so "at no simultaneity cost" overstates. δ_comp is an unbounded assumption term. Mildly non-trivial at best, and incremental over capxfer T5. |

Overall the theory is correct as far as I checked, but a statistics reviewer will see it as standard. It should be
presented as a characterization of *where counterfactual decisions are identified* (Prop. 2b and the dominance
corollaries), not as five theorems. Proofs are still missing from the appendix.

## Missing baselines (including trivial ones)

**Trivial and structural:**
1. **Raw posterior plug-in** (q = 0, no conformal). The pilot shows it is already near-oracle: unsafe 1.08 %, served
   90.5 % versus oracle 91.5 %. It must appear in every main table so the reader can see what conformal adds.
2. **Bayesian credible lower bound** (posterior 5 % quantile, no calibration). This is the natural non-conformal
   competitor.
3. **Detect-then-nominal:** run capxfer's fault detector, then commit iff the nominal model of the child or merge with the
   detected wheel masked or derated clears τ.
4. **Min-over-members rule:** a child's capability is bounded by its weakest module's lone-module capxfer certificate,
   scaled by a static geometric factor. This checks whether "certifying other bodies" needs anything beyond per-module
   certificates.
5. **Shape and leg prior with a conformal margin.** The size/kind/leg metadata baseline is listed; it also needs
   calibration so it is comparable at its own α.

**Statistical:**

6. **Conformal selection** (Jin and Candès cfBH) with FDR control, and Learn-then-Test / Bonferroni marginal sets, as
   alternatives to the per-episode max.
7. **Mondrian per family size** (simultaneous quantile per |M(m)| bin).
8. **The identified-set (Prop. 2a envelope) certificate with no learning**: commit iff the step-envelope lower end ≥ τ. It
   is model-light and robust to motor-law misspecification. It is the natural robust competitor and would test the
   paper's own thesis.

**Robotics:**

9. **MARS-style quasi-static controllability or capability margin** computed from nominal parameters with the
   detected faults.
10. **Physically test the candidate:** split, merge and drive. This is the time-cost upper bound and an oracle.
    `always_split` covers splits; merges are missing.

**Probing:**

11. **Fixed v2+clamp**, the minimal probe the closed form predicts. It is absent from Table 4: "fixed full library" is
    a different policy.
12. **The cheapest yaw probe at the child's own yaw rate**, for the one undominated class.

## The strongest version of this paper, in 5 sentences

Before a modular robot team splits or merges, it can certify the capability of every body it could become. The paper
shows which counterfactual decisions the current body's routine probe identifies and which it cannot: diagonals beyond
the probe speed and yaw legs of children with 3 or more modules are the closed-form blind spots, and one clamp probe
plus one yaw probe at the child's own rate covers them. A compositional per-wheel certificate then gives finite-sample,
family-wide guarantees for children and for merges of modules that never docked, while a direct simultaneous certificate
pays a measured multiplicity price, shown for real learned posteriors and not only for synthetic residuals. Under a
plant whose motor law is deliberately misspecified, and on a small set of real hardware splits and merges, the
compositional and identified-set certificates stay valid while plug-in and parametric-extrapolation certificates fail
silently, which is where the guarantee earns its place. Amortized inference makes this run at fleet-planning speed (end
to end, against matched-compute SMC), and the paper says plainly that adaptive probing matters only in the misspecified
regime, not in the nominal one.
