# cfcap confirmatory set: pre-registered metrics, cells, sample sizes and bars

Written by the research lead on 2026-11-25 at ~21:10Z, **before** the generator emits any confirmatory episode (plan
§2d, `~/.claude/plans/floating-questing-quiche.md`). Every number elsewhere in the study is exploratory. A result
is **confirmatory** only if:
- it comes from the seed streams defined in §0;
- it is scored by a script committed before those streams were generated;
- it is judged against the bars below, unchanged.

A bar that is missed is reported as missed. Cells that fail the power check (§0.4) are reported as underpowered and
do not count as passing.

## 0. Common rules

### 0.1 Seed streams (fresh; no role has touched them)
- **Generator streams.** They use the generator's confirmatory twins as already coded in `cfcap/gen/episode.py`:
  `SPLITS["conf_<split>"] = code + 64` with seed root `SEED_CONF = 92599` (≠ 92511).
- Confirmatory episodes come only from these splits:

  | claim | split(s) | content |
  |---|---|---|
  | C-MAIN | `conf_test` (plated) | train-like families, the main certificate table |
  | C-SCALE | `conf_H4` (7–10), `conf_H4b` (11–16) | larger bodies, test only |
  | C-HET | `conf_het_test` (ID), `conf_H1` (unseen omni type), `conf_H2` (mixed mecanum + conventional) | module types |

- **Streams outside the generator** must each take a fresh numpy seed root of the form `[92599, claim_code, index]`.
  The owning role adds it, and it must not reuse E2's / A5's / E1's seeds:
  - C-MERGE (stress A4 script), claim_code 21;
  - C-GATE knee episodes (t2 eval / gate), 22;
  - C-ENV buildings and missions, 23. Building b uses `[92599, 23, b]`; family membership is fixed by building
    seed (§2a).
- **Index ranges start at 0.** No confirmatory episode may be *scored* by any role before the scoring window
  (01:45Z).
  - Exception: ops' 1-shard canary checks only timing and sha256, never labels.
  - The canary shard (shard 0) is excluded from scoring.
- **Amendment (21:05Z, before any confirmatory label exists).** `conf_test` indices 0–39 were constructed (latents
  and candidates only) before this file was committed, so they are **excluded** from every claim (coordinator
  ruling). C-MAIN therefore uses indices 40–3,039, and C-GATE's block moves to 3,040–5,039.
- **Any `conf_*` episode loaded before this file's commit** (a test or a pilot) must be listed by the generator role
  in `docs/board/generator.md`. Those indices are then excluded.

### 0.2 Calibration
- Every margin, gate threshold, matched operating point and Mondrian quantile is fitted **only** on the existing
  exploratory `cal` split (gen-ref2, plated), or on the claim's own cal stream named below.
- Confirmatory streams are test-only. Nothing is refitted after seeing them: no model selection, no early stopping,
  no threshold change.

### 0.3 Statistics
- **Unit of resampling.**
  - The **episode** for C-MAIN, C-GATE, C-SCALE and C-HET.
  - The **assembly pair** for C-MERGE.
  - The **building** for C-ENV (missions in one building are correlated).
- 95 % percentile cluster bootstrap, B = 2000.
- **Superiority claims** ("A beats B by ≥ x pts") use a *paired* cluster bootstrap on the same clusters. Holm
  correction over the primary comparisons is marked **P1–P7** below (family-wise 0.05): the smallest p is tested at
  0.05/7, and so on. Report both the Holm verdict and the raw CI.
- **Validity bars** (unsafe ≤ x) use the one-sided 95 % upper bound.
- **Levels.** Nominal α = 0.10 for per-episode simultaneous certificates (the guarantee). U* = 1 % for the matched
  operating point. α = 0.05 is a secondary row only.
- **Headline stratum.** The decision-relevant stratum: the candidate holds a faulted module, or |ρ_dec − τ| < 0.10;
  τ ∈ {0.6, 0.7, 0.8}. The all-decisions rows are secondary.
- **Labels.**
  - ρ_dec with the **fixed T1 reference table** (ρ_abs / ρ_ref^T1(shape, leg)) for *every* plant, T2 included. This
    is the research-lead N3 correction: a task requirement is absolute, and the robot only has its T1 table.
  - Own-plant-reference labels are a secondary column.

### 0.4 Two error rates and the power check
- **E_ep:** the share of episodes (pairs, missions) with ≥ 1 unsafe commit. This is the quantity the certificate
  guarantees at α.
- **E_dec:** unsafe commits / *incapable* stratum decisions. This is the matched-point quantity at U* = 1 %.
- **Power rule** (plan §2d). A cell's E_dec bar can count only if the cell has ≥ 20,000 incapable stratum decisions
  **and** the realised cluster-bootstrap upper bound at the observed rate would be ≤ 1.3 % if the observed rate were
  1 % (i.e. the design effect is ≤ ~4.5).
  - The scorer computes and prints this check per cell before it prints any verdict.
  - If the cell fails, the E_dec bar is marked UNDERPOWERED and only E_ep (and the descriptive E_dec) is reported.
- **E_ep power.** ≥ 1,000 clusters gives a half-width of ≤ 1.9 pts at 0.10.
- **Sizing basis** (exploratory gen-ref2 plated test shard, 10,000 episodes, measured by the research lead):
  - 589 decisions per episode;
  - 341 stratum decisions per episode;
  - **9.0 incapable stratum decisions per episode**: lone faulted children 7.3, other children 0.6, merges 0.8,
    self 0.3.

## 1. Claim C-MAIN: the counterfactual certificate (main table)
- **Stream and plants.** `conf_test`, plated, **3,000 episodes** (indices 40–3,039). This gives about 27k incapable stratum decisions
  (lone-faulted block about 22k), which passes the power rule if deff ≤ 4.5. The same episodes are scored in T1 and
  in T2 `all` (T2 `all+motor` as a secondary column).
- **Methods.** All are calibrated on the exploratory cal split, with the same eids for every method:
  - ours: `amortized_comp` + simultaneous α = 0.1 (size-aware Mondrian if the models/stress roles have committed it
    by 22:15Z; otherwise the plain version; whichever is committed is the one scored);
  - `raw` (posterior, q = 0);
  - `credible_lower` (Bayes q05);
  - trivial: `always_go`, `never_go`, `metadata_nf`, `detector_gated`, `ghat_physics`, `copy_parent`,
    `min_over_members`;
  - `dominance_pi`;
  - `capxfer` (Thm 5, re-calibrated, refusal fix b08d5d5);
  - conformal selection `cfBH` at FDR 0.1;
  - `oracle`.
  - SMC runs on a **200-episode subsample**, the first 200 indices. It is descriptive only, never held to the power
    rule, and reported with its s/decision.
- **Cells.** method × plant {T1, T2} × block:
  - `stratum` (headline);
  - `lone-faulted` = child, 1 module, faulted, the capxfer-comparable population;
  - `multi-child` = child, ≥ 2 modules;
  - `self`;
  - `all`.
- **Bars.**
  1. **Validity (ours).** T1: E_ep ≤ 0.10 (upper 95 % ≤ 0.12) and E_dec at the matched point ≤ 1 % (upper ≤ 1.3 %,
     power rule). T2: E_ep ≤ 0.15 and E_dec ≤ 2 %. *T2 is empirical; its looser bar is stated now.*
  2. **P1, vs capxfer** (lone-faulted, T1). Certified fraction of capable decisions at matched U*: ours − capxfer
     ≥ 10 pts, paired CI lower bound > 0 (Holm).
  3. **P2, vs the best trivial / non-learned method** (stratum, T1). The best is the max over
     {metadata_nf, detector_gated, ghat_physics, copy_parent, min_over_members, dominance_pi}, picked on cal. CF@U*:
     ours − best ≥ 5 pts, paired CI lower bound > 0 (Holm).
  4. **P3, what calibration adds over the raw posterior** (stratum, T2). Pre-stated expectation: in T1, raw is
     near-oracle (H10). The test is in T2:
     - raw's E_dec > 2 % with lower 95 % > 1 %;
     - ours' E_dec ≤ 2 %;
     - the paired difference in E_ep (raw − ours) has CI lower bound > 0 (Holm).

     If raw is also ≤ 2 % in T2, the paper states "calibration buys no safety in these regimes; it buys the
     guarantee". The P3 claim is then dropped.
  5. **Conformal selection (descriptive; no bar).** cfBH's realised FDR, E_ep and CF are reported beside ours, with
     the pre-stated expectation that cfBH certifies more and has E_ep > α.

## 2. Claim C-MERGE: certificates for bodies that never existed
- **Stream.** C-MERGE seed root `[92599, 21, ·]`, **1,000 assembly pairs** per payload arm (P0, P5).
  - 1–3-module assemblies, ≤ 1 fault each, as in E2.
  - Every merge placement (≤ 6) × 10 legs × 3 τ ≈ 180k decisions per arm. Incapable merge decisions are few (E2
    level), so the E_dec bar applies only if the power rule holds. Otherwise it is descriptive.
- **Truth.** T1 and T2 `all`, fixed reference.
- **Methods.**
  - `comp_enum` (inferred latents: enum_known or nuisance-SMC on each G_j's own v2, composed), with the T1
    cross-fit margin;
  - `comp_enum_size` (size-aware margin from T1 perturbations, fitted on E2's exploratory pairs);
  - `copy_parent` (min of the parents' ρ);
  - `detector_gated`;
  - `comp_oracle` (true latents, the δ_comp reference).
- **Bars.**
  1. **δ_comp.** Decision flips of `comp_oracle` vs truth: upper 95 % < 1 % (T1) and < 2 % (T2).
  2. **Validity.** `comp_enum_size` per-pair miscoverage ≤ 0.10 in T1 (upper ≤ 0.13) and ≤ 0.15 in T2 (upper
     ≤ 0.18). E_ep (any unsafe merge commit per pair) ≤ 0.10 in both.
  3. **P4.** CF on capable stratum merges at realised E_dec ≤ 1 %: `comp_enum_size` ≥ 85 %, and beats `copy_parent`
     by ≥ 10 pts (paired CI lower bound > 0, Holm), in T1. T2 is reported against the same numbers.

## 3. Claim C-GATE: trust-gated extrapolation under a wrong motor law (T2 knee)
- **Stream.** `conf_test` indices 3,040–5,039 (disjoint from C-MAIN and from the excluded 0–39), with the knee seed root `[92599, 22, ·]` for the
  knee draws, **the same 2,000 episodes at every severity (a paired design: latents shared, only the knee draws vary) s ∈ {0, 1, 2, 3}**. s = 0 is the T2 `all` plant without the
  knee. Plated, train-like families. Truth from T2 with the knee law (`cfcap/t0/envelope.py` / `Effects(motor_kind=
  'knee', knee_severity=s)`), fixed reference.
- **Methods.**
  - `par`: the parametric/learned posterior with the simultaneous margin;
  - `raw`;
  - `pi`: the model-free identified-set bound;
  - `gate`: q_gate and the margin fitted on exploratory knee episodes (E1 / t2-knee exploratory seeds), pooled over
    severities. The gate statistic is whichever single statistic the theory/stress roles commit by 22:15Z: the
    trace misfit D, or the A5y capability-tracking residual |probe ratio_T2 − T1 prediction|. The latter is
    post-hoc, with AUC 0.83 on only 10 flips. Only one statistic may be used; no choosing after the data;
  - `detector_gated`.
  - Probes: v2, and v2+clamp.
- **Cells.** certifier × s × probe × size class {≥ 3-module children (primary), singles + pairs (secondary)} ×
  dominance class {dominated, undominated}. The dominance class is the closed-form Prop. 2b flag per decision.
- **Amendment (20:55Z, before any confirmatory data).** Exploratory N3 (`results/stress/N3_flips.csv`) showed that
  singles and pairs flip on *dominated* legs as well (axis 3–6 %). That is the H8 friction/kernel scatter, which
  exists without any motor-law change, not extrapolation. So the extrapolation bar is stated on ≥ 3-module
  children, where the exploratory dominated flips are 0 %. Singles and pairs are reported in their own cell against
  the size-aware margin (N7). This restriction comes from a mechanism stated in advance (Prop. 2b: for ≥ 3 modules,
  surge/sway are dominated and diagonals/yaw are not). It is named here so that it is not read as a post-hoc
  subgroup at scoring time.
- **Bars.**
  1. **H12, silent extrapolation (P5).** ≥ 3-module children, at s = 3, v2:
     - E_dec(raw) and E_dec(par) on *undominated* stratum decisions > 2 %, with lower 95 % > 1 %;
     - on *dominated* decisions, E_dec ≤ 1 % (upper ≤ 1.3 % if powered).

     Holm on the undominated part. Singles + pairs: descriptive.
  2. **H13, the gate.**
     - At s = 0: CF(gate) ≥ CF(par) − 2 pts.
     - At every s: E_dec(gate) ≤ 1 % (upper ≤ 1.3 %, power rule; otherwise E_ep(gate) ≤ 0.10).
     - At s = 3 with v2: CF(gate) − CF(pi) ≥ 5 pts (paired CI lower bound > 0; **P6**, Holm).
  3. **Dose response (descriptive, pre-stated direction).** E_dec(par) on undominated decisions is non-decreasing in
     s. Report Spearman ρ over s with a bootstrap CI.
- **Kill (as pre-registered for E1).** If bar 1 fails (par and raw stay ≤ 1 % on undominated decisions at s = 3),
  the gate is not a method contribution. The paper reports "extrapolation is safe under these misspecifications".
- **Power.** Undominated incapable decisions per episode are unknown before the run. If a severity cell has fewer
  than 20k, its E_dec verdict is UNDERPOWERED and E_ep decides.

### 3b. Amendment A-GATE (21:40Z, before any confirmatory label exists): the primary certifier replaces the falsified gate
**Reason.** Exploratory E1 (`results/theory/gate_cert.csv`, 300 episodes per law; `gate_verdict.txt`) falsified the
trust gate: SUCCESS-1 and SUCCESS-2 both fail. It also showed three things:
- the physics-trusting `par` fails silently at s = 3 (undominated unsafe 31 % with v2, 24 % with clamp);
- `pi_cal` with v2+clamp (dominance bound + clamp probe, conformally calibrated) stays at 0–0.4 % unsafe and
  certifies 79–91 % at every severity;
- `par_pooled` (calibrated over the law family) stays at 0.2–1.2 %.

Bars 2 (H13) and P6 above are **withdrawn**. Bar 1 (H12/P5) and bar 3 are unchanged. The new bars:
- **Certifiers in C-GATE.**
  - `robust` = **dominance + clamp, conformally calibrated** (`pi_cal` with v2+clamp). Its margin is fitted on
    exploratory knee episodes pooled over s ∈ {0, 1, 2, 3}. This is the **primary** certifier.
  - `lawcal` = the physics posterior calibrated over a law family (`par_pooled`, v2 only). Its margin is fitted on
    **s ∈ {0, 1, 2} only**, so that s = 3 is a held-out severity. This is the secondary certifier.
  - `par` (physics-trusting, calibrated on the correct law only), `raw` and `pi` are kept as comparators. `gate` is
    reported descriptively only.
- **P6′ (Holm, replaces P6). Robustness against physics-trusting**, ≥ 3-module children, s = 3:
  - E_ep(par) − E_ep(robust) > 0 (paired episode bootstrap, Holm);
  - E_dec(robust) on undominated stratum decisions ≤ 1 % (upper ≤ 1.3 % if powered; else E_ep(robust) ≤ 0.10).
- **Validity at every severity** (not Holm): E_dec(robust) ≤ 1 % on the stratum (power rule; else E_ep ≤ 0.10) for
  s = 0, 1, 2, 3.
- **Price of robustness** (descriptive, pre-stated): CF(par, v2) − CF(robust, v2+clamp) at s = 0 on the stratum is
  ≤ 10 pts. The exploratory E1 value is ~6 pts.
- **Held-out misspecification for `lawcal`** (descriptive, but a pre-stated bar): E_dec(lawcal) at the held-out
  s = 3 is ≤ 2 %. If it fails, the paper states "calibrating over a law family protects only inside that family".
  This is the honest limit of "calibration over wrong laws restores validity".
- **The plant is T2**, with the knee law and the fixed reference, on the C-GATE stream. The stress role produces
  the `robust` / `lawcal` / `par` / `raw` / `pi` rows. The paper role's scorer reads `robust` as P6′.

## 4. Claim C-ENV: missions in HM3D-calibrated buildings
- **Prerequisite (gating; no confirmatory scoring without it).** `cfcap/env/buildings.py` reproduces the HM3D
  statistics: two-sample KS p > 0.05, *or* effect size (KS D) < 0.1, for route length, steps per route, passage
  count/width and reconfiguration sites. This is computed on ≥ 500 generated buildings against the 900 records and
  committed before any C-ENV building is drawn.
- **Stream.** C-ENV (`[92599, 23, ·]`):
  - **300 buildings ID**, plus 150 per held-out family E-H1 (narrow doors), E-H2 (friction variance), E-H3 (long
    multi-turn), E-H4 (heavy payload);
  - 5 missions per building, with team parents from the train-like generator.
  - About 1,500 ID missions, 750 per family.
- **Methods** (mission plans: partition / assign / abstain / probe, certified per segment + footprint):
  - ours (the C-MAIN certifier composed over segments, footprint as a hard constraint);
  - `raw`;
  - `nominal` plug-in;
  - `biggest_body` (merge everything that fits the narrowest passage, nominal capability);
  - `greedy_feasible` (smallest footprint-feasible body whose nominal capability clears each segment);
  - `oracle`.
- **Metrics.**
  - E_mis = share of committed missions with ≥ 1 segment failure in truth (the mission-level unsafe rate);
  - mission success (served without failure), useful commits, mission time including probes and splits.
- **Bars.**
  1. **Validity.** ID: E_mis(ours) ≤ 0.10 (building-clustered upper ≤ 0.13). Each E-H family: E_mis ≤ 0.15 (empirical,
     stated now).
  2. **P7.** ID mission success: ours − max(nominal, biggest_body, greedy_feasible) ≥ 5 pts among missions where the
     oracle succeeds, paired CI lower bound > 0 (Holm).
  3. **Generalisation.** Success drop ID → each E-H family ≤ 10 pts for ours, and smaller than raw's drop (paired;
     descriptive if the CI covers 0).
  4. **Splits and merges are necessary (descriptive, pre-stated).** The share of oracle-successful missions whose
     oracle plan uses a merge or split differs from the parent-as-is plan in ≥ 20 % of ID missions. Otherwise the
     env adds no reconfiguration content, and the paper says so.

### 4b. Amendment A-ENV (21:55Z, before any confirmatory label exists)
1. **Exclusion.** The models role constructed confirmatory mission `make_mission("conf-ID", 1499)` (geometry only)
   before scoring. **conf-ID mission 1499 is excluded** from every C-ENV cell. ID therefore uses conf-ID missions
   0–1,498. It is the last index, so no other mission shifts.
2. **Bound construction, as read.**
   - `raw` = the posterior median, i.e. a plug-in point. The paper calls it "plug-in median", not a bound.
   - `amortized_comp` = q50 − q_comp·max(q50 − q05, s_floor), with q_comp the **T1-cal simultaneous compositional
     constant** and no mission-level refit. This stays the **primary** C-ENV certifier, because it is what
     "the same certifier composed over segments" means.
   - Amendment: s_floor = **0.01** (the label resolution, the H9 fix), not 1e-3.
3. **Validity status.**
   - E_mis bars for `amortized_comp` are **empirical transfer bars**, not guarantees. The T1-cal constant was
     calibrated over (candidate × leg) per generator episode at nominal floor conditions.
   - Missions evaluate (candidate × segment) under segment μ factors and payloads outside that calibration
     distribution.
   - So env.md's statement "keeps P(a committed segment fails) ≤ α" must not appear in the paper for this certifier.
4. **Secondary certifier `amortized_comp_envcal`.** The same quantiles, with q_env = the conformal (1 − α) quantile of
   the per-mission max normalised shortfall over feasible (candidate, segment) entries.
   - It is fitted on the **exploratory** ID env missions (400) only. ID confirmatory scoring uses it unchanged, and
     the E-H families use the same q_env (no per-family refit).
   - Its E_mis bars are the same as the primary's, and it carries the guarantee on ID missions by exchangeability.
   - Price of the T1 constant (descriptive): useful-commit difference between the two certifiers on ID.

## 5. Claim C-SCALE: module count to 16
- **Stream.** `conf_H4` (bins 7–8, 9–10) and `conf_H4b` (bins 11–13, 14–16), test only, **600 episodes per size
  bin**. The models were trained on
  ≤ 6 modules (b1 n6 weights as committed at 22:15Z).
- **Methods.** `amortized_comp`, the direct head `blackbox_gnn` / `st` direct, `capxfer`, `raw`, and
  `dominance_pi`. The same T1 cal margin is used: no refit per size.
- **Metrics.** E_ep, per-decision 90 % lower-bound coverage on the stratum, and CF@U* per bin. Theory adds the
  closed-form extrapolated share per N (T0, CPU), which is not a stochastic claim.
- **Bars.**
  1. **Validity of composition with size.** E_ep(amortized_comp) ≤ 0.15 in every bin (upper ≤ 0.18), and stratum
     coverage ≥ 0.85.
  2. **Composition degrades less than direct (H5 confirmatory).** The coverage slope over the bin index:
     comp − direct > 0, paired episode bootstrap CI lower bound > 0. Not in the Holm family (secondary).
  3. **Extrapolated share** (theory, deterministic) is reported against N. Its pre-stated direction is
     non-decreasing in N for yaw legs.

## 6. Claim C-HET: heterogeneous module types
- **Stream.** **1,000 episodes each** of:
  - `conf_het_test` (ID: each assembly is single-type, mecanum or conventional);
  - `conf_H1` (unseen omni type on the parent);
  - `conf_H2` (parents that mix mecanum and conventional modules; training never mixes types in one rigid body).

  This uses the generator's `assign_types` as committed. The models are trained on `het_train` only.
- **Sanity gate (test, before generation).** A conventional module has sway capability 0 (|ρ_sway| < 0.02).
- **Methods.**
  - `amortized_comp` with the per-wheel channel model **told the module type** (type is observable metadata; the
    honest default);
  - `amortized_comp_blind` (type withheld);
  - the direct head;
  - `metadata_nf`;
  - `raw`.
- **Bars.**
  1. **Validity.** E_ep(amortized_comp) ≤ 0.15 on H1 and H2 (upper ≤ 0.18).
  2. **Degradation.** Coverage and CF drop ID → H1/H2 ≤ 5 pts for `amortized_comp`, and smaller than the direct
     head's drop (paired; secondary).
  0. **What H1 tests (amendment A-HET, 21:40Z, before any label).** The generator sets OMNI_MU = 0.2 (1591e17).
     Healthy omni is nominally ≈ mecanum (|Δρ| < 0.01), but faults have different consequences: a single module
     with a braked wheel has diagonals 0.25 vs 0.41.
     - So H1 tests generalisation to **new fault behaviour** of an unseen wheel type, not to new nominal
       capability. The paper must state it that way.
     - For H1 the bars are judged on the **faulted-candidate** block. The all-decisions block is descriptive, because
       healthy omni candidates are in effect in-distribution.
  3. **Pre-stated interpretation.** If `amortized_comp` passes only when told the type, the claim is "composition
     transfers to new module types whose wheel channel model is known", not "learned type inference".

## 6b. Claim C-EXPLAIN (added 22:00Z, before any confirmatory label exists; secondary, not in the Holm family)
Faithful explanations of capability limits (owner's original ask). Exploratory T0 (H17): 89.5 % faithful, and 100 %
belief-consistent explainer. So the explainer can be **deterministic code** on the confirmatory rows, with no LLM in
the confirmatory path.
- **Rows.** C-MAIN (conf_test 40–3,039), T1, method `amortized_comp`. Every refused (candidate, leg, τ) decision in
  the stratum.
- **Needs.** Per-wheel posteriors (models' `wheels_<split>.npz`) and true latents (regenerated from the episode
  index).
- **Explainer.** Blamed wheels = the candidate's wheels with posterior P(mode ≠ ok) ≥ 0.5. Each refused leg gets one
  of two labels:
  - a **deficit** claim if the posterior median < τ;
  - a **margin** claim ("cannot confirm τ") if the median ≥ τ but the lower bound < τ. This is the H17 fix:
    conservative refusals no longer name a false short leg.
- **Truth.** A blamed wheel is correct iff it carries any injected fault. A deficit claim is correct iff
  ρ_dec^T1 < τ. Margin claims make no physical assertion; their share is reported.
- **Bars** (episode-clustered bootstrap, point estimate judged, CI printed):
  - wheel precision ≥ 0.95;
  - deficit-claim precision ≥ 0.90;
  - informative share (≥ 1 blamed wheel or a deficit claim) ≥ 0.80 among refusals of truly incapable decisions;
  - false-reason rate on truly *capable* refusals ≤ 0.05. This is the share of those refusals carrying a wrong
    deficit claim; it was 20/20 before the margin fix.
- **Exploratory add-on (not scored here).** An LLM paraphrase spot-check of 200 cases, run by the annotation
  role after 01:45Z, checked for belief consistency.

## 7. What is *not* confirmatory
- Probing policies (EVOI etc.); merges in env missions beyond bar 4.4; any number from SMC; plateless rows; α = 0.05
  rows; own-reference labels.
- All of these are reported as exploratory.

## 8. Scoring
- A single committed script `scripts/analysis/confirmatory.py` (owner: paper role, with research-lead review)
  reads only the streams above and prints, per bar: the power check, the estimate, the CI, PASS / FAIL /
  UNDERPOWERED, and the Holm table for P1–P7.
- It must be committed and unit-tested on mock data **before** the first confirmatory shard is pulled.
- Results go to `results/confirmatory/verdicts.csv` with the git sha of this file.

## 9. Scorer readings ratified, with three amendments (21:15Z, before any confirmatory label exists)
The research lead reviewed `scripts/analysis/confirmatory.py` (7fe5883) against `docs/board/paper.md` §5c.

**Ratified as read:**
- Point-only "≤ b" bars are judged on the one-sided 95 % upper bound ≤ b.
- The power check uses z = 1.96 (deliberately conservative against the one-sided bound; it reproduces deff ≲ 4.5
  at 20k), with the Kish fallback at an observed rate of 0.
- P5 is not power-gated on its excess part: an underpowered cell cannot falsely show E_dec > 1 %.
- The bound assignment:
  - matched point for C-MAIN E_dec / P1 / P2 and C-MERGE P4;
  - own bounds for E_ep everywhere, P3 and all of C-GATE.
- Holm over m = 7 fixed; a P-bar needs both Holm rejection and its point threshold.

**Amendments:**
1. **The block for the E_ep validity bar.** E_ep is the guaranteed event, so C-MAIN bar 1 and C-MERGE bar 2 judge E_ep
   on the **`all` block**: an episode or pair counts if *any* of its decisions is an unsafe commit under the
   method's own bound. Block-restricted E_ep (stratum, lone-faulted, …) is descriptive only.
2. **"≥" validity bars** (C-SCALE coverage ≥ 0.85, C-MERGE / P4 CF ≥ 85 %, C-HET drop ≤ 5 pts read as
   ID − split ≤ 0.05) are judged on the **point estimate**, with the one-sided 95 % bound printed. These are utility
   or descriptive thresholds, not safety bounds. The safety bars stay upper-bound-judged.
3. **P5's dominated contrast when underpowered.** If the ≥ 3-module *dominated* cell fails the power rule, its
   condition becomes: one-sided upper ≤ 2 %, i.e. clearly below the undominated excess threshold. It is printed as
   "dominated: UNDERPOWERED, contrast-judged". If it is powered, the original ≤ 1 % (upper ≤ 1.3 %) applies.
