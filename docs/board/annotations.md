# annotations (codex-lb batch annotation; owns `scripts/codex_annotate.py`, `scripts/validate_annotations.py`, `docs/annotations/`, `results/annotations/`)

**Status (2026-11-25 21:40Z).**
- Done:
  - pipeline and validators built and tested (`tests/test_annotate.py`, 18 pass);
  - **C complete** (528/528);
  - **B complete** (600 generated, 600 re-derived, 558 in the library).
- Running: **A** (50-case pilot, then 1181 cases) in a detached quota-aware driver; see "Pending".
- Models:
  - B generation and the first 503 C cases ran on the default `gpt-6.1-sol`, before the owner asked for a lower tier.
  - The last 25 C cases (on the old driver as the quota returned), and the A runs, use `gpt-6-luna`.
  - B's second pass uses `gpt-6.1-terra` (owner / coordinator, 21:17Z).

**Principle.** An LLM annotation is never ground truth. Every annotation is either:
- checked by code against simulator truth (latents, labels, held-out probe resolutions, the A5 ablation); or
- input whose physical requirement is recomputed by code (task library).

Codex runs `-s read-only` in an empty scratch directory and sees only the case JSON. Held-out truth lives in
`results/annotations/<x>_truth.jsonl` and is never sent.

## Pipeline
`scripts/codex_annotate.py run --name N --cases X.jsonl|csv --prompt T.md --schema S.json [--var K=@file]`
- Chunks of 25 cases, `codex exec --output-schema -o`, every item validated against the schema here, and the set of
  ids checked against the chunk.
- Missing or invalid items are re-queued, up to 3 attempts.
- AIMD concurrency: +1 slot after 4 successes, halved on a failure.
- Quota-aware: a "usage limit" / 502 "account unavailable" reply does not use up a retry. Dispatch pauses until the
  advertised reset time.
- Resumable by case id; nice 15; CPU only.
- Output: `results/annotations/N.jsonl` plus `N_calls.csv` (per call: seconds, status, tokens). Raw prompts, answers
  and events go to `$CFCAP_OUT/annotations/N/`.

## Ground-truth sources (workstream A cases; `scripts/validate_annotations.py`)
- **T0 decision pilot.** `pilot_t0_eps600.pkl` lacks the true latents and the particle hypotheses, so `t0-truth`
  replays all 600 episodes under a profiler hook.
  - The replay reproduces `rho_true` exactly in 600/600 episodes.
  - The particles map onto the pickled weights in 600/600 (checked against the uncommitted no-truncation edit of
    `t0data.py`).
  - Pair-level cases (candidate × task) under v2/sim0.1, v2/raw and all/sim0.1: 43 unsafe certificates, 138
    ambiguous pairs (= decision.md's 138), 2146 refused-capable, 378 refused-incapable
    (`results/annotations/t0_cases.jsonl`).
- **A2.** Replay of the 42 parents with an ambiguous v2 decision (oracle set, ρ known): 160 decisions (94 floor + 66
  sensor). The counts match the shard CSVs exactly. Every other probe set's resolution is held out.
- **A5.** Every T1→T2 `all` flip on ρ_dec from the npz (P0 2 shards + P1): 595 cases (single 415, pair 151, L3 21,
  square 6, 2x3 2). The per-condition ablation is held out.
- Workstream A set `a_all` has 1181 cases: T0 426 (unsafe + ambiguous + all 72 v2-refused-capable + 250 of 2074
  refused only after all probes), A2 160, A5 595. Pilot `a_pilot` has 50 (stratified).

## Findings
**C. Explanation faithfulness (exploratory, T0 pilot only).**
- `results/annotations/c_explain_summary.csv`, per case `c_explain_faithfulness.csv`; 528/528 explained.
- The robot explains a refused (candidate, task) from its *belief only*. Code then checks:
  - each blamed wheel is truly faulted;
  - each blamed leg is truly below r;
  - each blamed item is also supported by the belief codex saw (belief consistency).

| predictor | n | informative | faithful (95 % CI) | wheel precision | leg precision | wheel recall | belief-consistent |
|---|---|---|---|---|---|---|---|
| T0 posterior (v2) | 200 | 0.97 | **0.895 (0.845–0.935)** | 1.00 (n 193) | 0.854 (n 141) | 0.96 | 1.00 |
| — truly incapable pairs (correct refusals) | 128 | 1.00 | 0.992 | 1.00 | 0.995 | 1.00 | 1.00 |
| — truly capable (conservative refusals) | 72 | 0.90 | **0.722** | 1.00 | **0.00 (n 20)** | 0.90 | 1.00 |
| oracle (true latents, q = 0; refuses only incapable) | 128 | 1.00 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 |
| nominal plug-in (all healthy, q ≈ 30) | 200 | **0.00** | (1.000, vacuous) | — | — | **0.00** | 1.00 |

The partial run (503 cases) gave 0.904 (CI 0.86–0.94). The final number is **0.895 (0.845–0.935)**.

Readings:
1. Codex reports the belief faithfully in every case (belief-consistent 1.00). Every unfaithful explanation comes
   from a wrong posterior, not from the LLM.
2. Blamed wheels are always truly faulted (193/193).
3. Faithfulness fails exactly where the certificate is conservative. On capable-but-refused pairs, the robot names a
   "short" leg that is not truly short in 20/20 cases: its posterior mean is below r while the truth is above.
   - That is a trust hazard: the operator is told a false reason.
   - It is fixable by phrasing such refusals as margin or uncertainty refusals.
4. The nominal predictor's explanations are never wrong, but they are empty ("unsure"). Its fault recall is 0, so a
   calibrated-but-uninformed certificate cannot explain itself.
5. Candidate trustworthiness result: *explanations from the certified posterior are faithful 90 % of the time, and
   99 % when the refusal is correct; every error traces to the posterior, none to the explainer.*
   - Not yet tested: learned / amortized posteriors (no per-particle beliefs exported yet); T1.

**B. Task library (exploratory).**
- Output: `results/annotations/task_library.csv` with 558 tasks. Full audit trail in `task_library_all.csv`; report
  `task_library_report.json`.
- Generation: `tasks_gen.jsonl`, 600 scenarios over 4 domains × 10 settings × 10 emphases (warehouse, rescue,
  inspection, logistics).
- Codex supplies only physical needs: peak speeds in m/s and deg/s, directions, payload, team size.
- Code computes r = v_required / (nominal healthy achieved speed of the leg):
  - V_NOM·ρ_ref for surge and sway, |V_DIAG|·ρ_ref for diagonals, W_NOM·yaw_scale(team)·ρ_ref for yaw;
  - ρ_ref ≈ 0.91 (`$CFCAP_OUT/rho_ref_plated.json`);
  - min_modules = ⌈payload / 5 kg⌉ (the 5 kg per module is an ASSUMPTION).
- Code checks:
  - 600/600 in range (0 < r ≤ 1);
  - 0 duplicates (token Jaccard ≥ 0.6);
  - 581 consistent. 18 are flagged "body wider than the narrowest passage" and 1 "text needs reverse, no rearward
    leg", using negation-aware keyword rules for sway, yaw, diagonal and reverse.
- Independent second pass (`tasks_rederive.jsonl`, terra): re-derives the motions from the title, description,
  environment and payload only.
  - Leg set identical in 99.2 % of tasks (mean Jaccard 0.997).
  - Re-derived r is exactly equal in 65.5 %; the 90th percentile of max |Δr| is 0.073.
  - Agreement (Jaccard ≥ 0.75 and max |Δr| ≤ 0.15) in **96.2 %**.
  - Caveat: the descriptions state distances and times explicitly, so agreement partly measures recomputing the
    stated numbers. It shows internal consistency, not independent realism.
- Included = in range ∧ consistent ∧ not a duplicate ∧ agree: **558**.
  - Kind: payload 347 (≥ 2 modules), route 211. Domains: 138–141 each.
  - Legs required: surge+ 97 %, surge− 51 %, sway± 41–45 %, yaw± 24–25 %, diagonals 4–12 %.
  - r quantiles 0.07 / 0.37 / 0.88 (10/50/90 %). Realistic jobs ask for much less than the synthetic LEVELS 0.6–0.9
    on most legs, and lean on surge.
- **For decision:** use the named `r_<leg>` columns (NaN = not required) and `min_modules`. Map them by name: the T0
  and T1 leg orders differ.

**A. Failure-mode taxonomy (exploratory; T0 pilot, A2, A5).**
- Files: `results/annotations/a_taxonomy_by_category.csv`, `a_taxonomy_validated.csv`, `a_taxonomy_claims.csv`,
  `a_taxonomy_by_claim_kind.csv`, `a_taxonomy_table.csv`. Taxonomy: `docs/annotations/TAXONOMY.md` v1.
- Run: 1181/1181 cases annotated by luna (60 calls, 0 failures).
- Claims: 4522, of which 4491 are checkable against ground truth; **95.7 % true**.
- The pilot (50 cases, free labels) gave 222/225 true claims. Its labels map onto v1 with ≤ 3/50 unmapped ("fault
  outside the candidate"). `other` is 1.3 % in the full run. Both are below the 15 % rule, so **v1 stands**.

| category | n | claim acc | signature holds | validated | sources |
|---|---|---|---|---|---|
| near_threshold | 327 | 0.964 | 0.859 | **yes** | T0, A2, A5 |
| probe_overconfidence | 150 | 0.983 | 0.887 | **yes** | T0 |
| conformal_margin_slack | 126 | 0.969 | 0.976 | **yes** | T0 |
| passive_wheel_hidden | 89 | 0.991 | 0.921 | **yes** | T0, A2 |
| mode_confusion | 71 | 0.993 | 0.901 | **yes** | T0, A2 |
| spread_masked_weak | 13 | 0.945 | 0.923 | **yes** | T0, A2, A5 |
| screen_miss | 24 | 0.990 | 0.792 | no (just below 0.8; 19/21 on unsafe certificates) | T0 |
| friction_scatter | 141 | 0.947 | 0.780 | no | A5 |
| genuine_shortfall | 57 | 0.971 | 0.474 | no | all |
| kernel_numerics | 64 | 0.821 | 0.250 | no | A5 |
| dock_effects | 85 | 0.900 | **0.047** | no | A5 |
| unexcited_envelope / split_needed | 11 / 8 | 0.98 / 0.96 | 0.64 / 0.00 | no (n small) | A2, T0 |
| other | 15 | — | — | no | — |

Readings:
1. **Why T0 certificates fail or abstain (validated categories only, `a_taxonomy_table.csv`).**
   - Refused-capable pairs: conformal_margin_slack 110 and probe_overconfidence 132 of 265. The refusals are
     calibration effects, not posterior errors.
   - Ambiguous pairs: near_threshold and passive_wheel_hidden dominate.
   - A2 oracle ambiguity: mode_confusion 58, passive_wheel_hidden 54 and near_threshold 26 of 160. These are coast /
     brake wheels on single-module children that v2 cannot tell apart.
2. **Unsafe certificates.** Ground truth: 23/43 certified-but-incapable pairs are screen misses (truth_in_screen = 0;
   decision.md saw this for the 6 *committed* ones).
   - Codex labels them screen_miss with 19/21 correct.
   - The other unsafe certificates are hidden passive wheels (9, signature 1.0) and near-threshold labels (6).
   - So a second unsafe mechanism exists besides the screen: a coast or brake wheel in the child that v2 does not
     excite.
3. **The validator caught plausible but wrong LLM physics on A5.** Codex's misspecification attributions fail the
   ablation check:
   - its condition_flip claims are 156/286 true (55 %);
   - "dock effects" holds for 4.7 % of the cases it labelled;
   - "kernel numerics" holds for 25 %.

   The deterministic ablation itself gives the attribution (`a5_truth.jsonl`, 595 flips):
   - 566 (95 %) also flip with the roller-friction scatter alone (`aniso_mu`);
   - removing the scatter removes 391 (66 %);
   - the bare kernel flips 204 (34 %).

   This matches stress.md C-stress-1. So for A5 the paper table should cite the ablation, not the LLM category.
4. Claim kinds (`a_taxonomy_by_claim_kind.csv`):
   - factual claims (wheel_mode, leg_truth, leg_t1, post_mean, truth_mass) are ≥ 99 % true;
   - inferential claims are weaker: leg_margin 90 %, probe_resolves 10/16 checkable, condition_flip 55 %.

## Throughput and cost (`*_calls.csv`, `codex_annotate.py summary`)
| run | calls | valid items | wall | items/min | median call | max slots | input tok (cached) | output tok |
|---|---|---|---|---|---|---|---|---|
| tasks_gen | 24 | 600 | 1013 s | 35.6 | 129 s | 9 | 438k (260k) | 212k |
| c_explain | 22 | 528 | 178 s for the first 503 (the last chunk ran after the outage) | 170 | 45 s | 9 | 573k (213k) | 63k |
| tasks_rederive (terra) | 24 | 600 | 119 s | 303 | 20 s | 6 | 497k (401k) | 49k |
| a_pilot (luna; 1 call lost to the quota outage) | 6 | 50 | — | — | 37 s | 5 | 130k (0) | 13k |
| a_taxonomy (luna) | 60 | 1181 | 623 s | 114 | 56 s | 6 | 2.12M (505k) | 267k |

- No call failed before the quota hit.
- There is a long tail: one generation call took 757 s. AIMD reached 9 slots with no errors.
- Cost: no marginal $ (codex-lb subscription capacity). It is reported as tokens.
- The quota ran out at 20:38Z after about 1.0 M input tokens of my runs plus other sessions' tasks.

Note: `results/annotations/t0_cases.jsonl` (7 MB) is not committed. Regenerate it in about 4 s with `validate_annotations.py cases-t0` (needs `$CFCAP_OUT/annotations/t0_truth.pkl`).

## Pending
Nothing from codex. The driver finished (`$CFCAP_OUT/annotations/run_pending.log`). T1 and T2 sources slot in as their
rows land (see "Needs").

## Proposals
1. **Paper table "why certificates abstain or fail":**
   - Use the six validated categories of `a_taxonomy_by_category.csv` for T0 and A2 (`a_taxonomy_table.csv`).
   - For A5, use the deterministic ablation shares (reading 3), not LLM categories.
   - screen_miss (0.79) can enter if the next T1 batch clears 0.8.
2. **Explanation faithfulness as a trustworthiness result** (C): scale to T1 once decide-main lands, with the SMC and
   amortized posteriors. Report the refused-capable stratum separately (finding 3). About 2k cases, roughly 15 min of
   codex at 6 slots.
3. **Task library** as the realistic task distribution for T4 regret: rerun `decide_pilot` with tasks sampled from
   `task_library.csv` (include = 1). The low r values suggest the current synthetic levels overstate how often
   certificates must be tight.

## Needs from other roles
- **decision:** store the true latents in the T1 `decide_eps/*.pkl` dicts (modes, g, stall, noload, mu; the wheel
  order of `asm`). `cases-t0` then slots T1 in as a new source via `cases-a --extra`. Without them, wheel claims are
  unverifiable and are excluded.
- **stress:** keep the A5 npz layout. New T2 npz (`t2-eps`, `t2-extra`) slot into `cases-a5` if they carry
  `<shape>/lat/*` and the per-condition `rho`.
- **nuisance:** the cfcap-nuis2 ambiguous decisions slot in the same way, if the unit pickles keep the truth latents.

## Next step
Collect the pending runs, publish the A accuracy table and the B agreement, and add the T1/T2 sources as their rows
land.
