# decision (specialist board; owns `cfcap/decide`, `cfcap/probe`, paper table T4)

OPS: (ops, 21:25Z) cfcap-decide2 is submitted (18 jobs, 77c774e, bottom priority). The old cfcap-decide queue is cancelled.

OPS: (ops, 20:45Z) the 15 queued cfcap-decide jobs are cancelled (pre-fix posterior normalisation, evaluate leakage, screen); decide-canary and decide-main.s001 finish. Resubmit after N1 with a new plan campaign name; make_jobs now adds lineage.


**Status (2026-11-25 ~21:30Z):**
- Fixed: the round-0 review items and the codex screen and scorer bugs. Every number from before 2f6c236 is void.
- N1 passes its pre-registered bar (exploratory, T0).
- The paper file is now `results/decision/paper/probe_t0_n1.csv` (cert `simM0.1`). The voided T0 files are deleted.
- T1 campaign `cfcap-decide2` is ready for ops to resubmit. The T1 data is unaffected by the fixes.
- Envelope-class decisions (model-free) are running.
- The env mission interface is proposed below.

## What is built
- `problem.py` (PLAN §3).
  - **Tasks.** Each task is a requirement vector r over the 10 legs, with NaN for "don't care".
    - *route*: 1–2 leg groups (a ± pair, or a single leg) at levels 0.6–0.9.
    - *payload*: surge± and sway± ≥ 0.7, and at least 2 modules.
    - Tasks come from their own seed stream, independent of the truth.
  - **Options.** Every injective assignment of tasks to pairwise-disjoint pieces: the whole parent, its connected
    children, and merges with the other assembly.
    - Time cost = T_SPLIT_S × (parts − 1 + merges). The parts are the used pieces plus the connected components of
      the leftover modules.
    - Abstaining costs 0.
  - **Commit rule.** Only certified pairs are committed: every required leg needs a finite lower bound ≥ r, and the
    piece must be large enough. The rule maximises V·assigned − time. Empty, NaN or −inf bounds mean abstain.
  - **Utility** (in seconds): +V_TASK = 120 per task served, −C_UNSAFE = 600 per unsafe assignment, − split/merge
    time, − probe time. Both constants are assumptions.
  - **Oracle** = the same optimisation run on the true capability. Regret = U_oracle − U.
- `posterior.py`: the **interface** (next section).
- `policies.py`: the probe-selection policies.
  - `v2` (fixed): no extra probe.
  - `all`: clamp + moddiff + tug.
  - `random`.
  - `eig`: information gain on the discrete latent.
  - `epig`: summed binary-entropy reduction of the decision-relevant events {ρ[c, legs(t)] ≥ r_t}.
  - `evoi`: decision-EVOI greedy, argmax EVOI/t_u. It stops when EVOI ≤ t_u for every probe.
  - `always_split`: one diagnostic split of the child with the most decision entropy.
  - Expectations use S = 16 predictive draws. The non-EVOI policies probe only while some task is unserved, with at
    most 2 probes.
- `evaluate.py`: the certifiers, each **cross-fitted over 2 folds** (episode parity).
  - `sim0.1`: end-to-end SimultaneousLower over the episode's decision-relevant entries (every candidate × every
    required leg), fitted on the policy's *final* posteriors. The in-loop lookahead uses the q from v2.
  - `crc0.1` / `crc0.05`: `risk_control_margin` on the episode's unsafe-commit indicator, using
    lower = mean − λ·scale. The indicator is monotonised (max over λ′ ≥ λ), so the CRC premise holds. The probe
    trajectory is held fixed.
  - `raw`: q = 0 (plug-in).
  - Metrics, the T4 contract rows (`write_paper`), and the T0 episodes (`t0data.py`).

## Interface (agreed with nuisance / baselines / models)
`cfcap.decide.posterior.DecisionPosterior(w (P,), rho (P, C, L), latent (P,) int, probes {name: Probe})`.
- `Probe(sig (P, D), sd (P, D), t_s, y_true (D,))`: each particle's predicted signal for a not-yet-run probe, its noise
  sd, and its time cost. `y_true` is the observation the episode would return, used for evaluation only.
- A diagnostic split is just a probe, named `split:<cand>`, with cost T_SPLIT_S + the child's v2 duration.
- Any producer only has to give posterior draws pushed to the candidates. The probes are optional; without them, the
  only possible action is to decide.
  - nuisance: `cfcap/nuisance_post.py` draws. `task.py` builds them with the coordinator's label
    ρ_dec = ρ_abs / ρ_ref (ρ_ref = the nominal healthy body at point θ, no spread).
  - baselines: `posterior.propagate` output, via `from_baseline(rho, w)`.
  - models: latent samples propagated the same way, or quantile heads used as pseudo-samples.
- Oracle and nominal predictors are `degenerate(rho)` (a single particle).

## N1 pre-registration (written 2026-11-25 before the run; exploratory)
**Design.**
- 600 fresh T0 episodes, seeds 939000–939599, with tasks drawn from 949000 + j.
- Screen K = 5 (legacy statistic), M = 8 nuisance draws, particle budget 2048.
- **No budget abstention.** Truncation beyond 1e-6 mass or 2048 particles is recorded as `mass_dropped`. An empty
  screen still marks the support unknown, so the episode abstains.
- Marginal probe likelihood, Gaussian normaliser, 3-fold leak-free cross-fit (`evaluate.fold_plan`).
- Certifiers:
  - `sim0.1`: marginal end-to-end, which absorbs the screen and the truncation by exchangeability;
  - `simM0.1`: Mondrian by observable inference difficulty. Bins are `mass_dropped` ∈ {0, (0, 1e-3], > 1e-3} ×
    `n_screen` tercile (terciles taken on the calibration folds), with ≥ 60 calibration episodes per bin. Bins below
    that are merged across terciles within their mass bin first, then pooled.
  - `raw` (plus `simF0.1`, `bayes05`).

**Bar.**
- `simM0.1`: per-bin ep_any_unsafe ≤ 0.10, with an episode-bootstrap upper 95 % CI ≤ 0.15.
- Healthy-episode abstention ≤ 5 %.
- Useful commits ≥ 95 %.

**Verdict rule.** If the bins fail but `sim0.1` holds marginally (ep_any_unsafe ≤ 0.10), report the marginal claim
only. H11 then becomes "end-to-end calibration absorbs approximate inference", and the screen threshold is a compute
knob, not a safety knob.

## N1 results (exploratory, T0)
Source: `results/decision/t0_n1_{bins,overall,matched,verdict}`, `scripts/decide_n1_report.py`. 600 fresh episodes, as
pre-registered.

**Verdict: bar passed** (`t0_n1_verdict.json`).
- **Difficulty bins.** Every `simM0.1` bin has ep_any_unsafe ≤ 0.010, with bootstrap upper CI ≤ 0.024. Bars:
  ≤ 0.10 and ≤ 0.15.
- **Healthy-episode abstention:** 0 % (bar ≤ 5 %).
- **Useful commits:** 95.4 % (bar ≥ 95 %). The shortfall is the 4.5 % of episodes whose screen is empty; they abstain by
  design.
- **Marginal `sim0.1`:** ep_any_unsafe 0.67 %.
- **Truncation share.** Truncation dropped mass in 77.5 % of episodes, every one of them in mass bin (0, 1e-3]. No
  episode lands in the > 1e-3 bin, so the third Mondrian bin is empty.
- **Screen miss** (truth not among the screened hypotheses, K = 5): 5.2 %. All of it is caught: in the tables below it
  appears only as unsafe-free abstention or as calibrated margin.

**Certifiers at fixed v2** (episode unsafe / useful / healthy abstain / regret ± se in seconds):

| certifier | ep unsafe | useful | healthy abstain | regret |
|---|---|---|---|---|
| raw (posterior mean, q = 0) | 1.7 % | 95.0 % | 0 | 19.7 ± 3.9 |
| bayes05 (posterior 5 % quantile) | 0.17 % | 95.6 % | 0 | 9.5 ± 1.9 |
| sim0.1 | 0.67 % | 95.3 % | 0 | 13.2 ± 2.9 |
| simF0.1 (floor score) | 0.67 % | 95.5 % | 0 | 13.1 ± 2.8 |
| **simM0.1** (Mondrian difficulty) | 0.33 % | 95.4 % | 0 | 11.1 ± 2.3 |

The critic is right that the raw posterior is close. bayes05 is the best uncalibrated rule here, but it carries no
guarantee. simM0.1 keeps the guarantee and lands within 1.6 s of it.

**Monotonicity, before → after (review item 1).** Certified capable pairs along nested probe sets, on fixed episodes:
- **Before** (draw likelihood, 2-fold leaky cross-fit; `before_drawlik_*.csv`, void as an estimate but shows the bug):
  - sim0.1: v2 11,488 → all 9,421 (−18 %). q went from 1.0 to 5.2.
  - Neither the floor score (simF) nor the unnormalised score (simA) fixed it: both still gave 87 % useful commits after
    all probes.
- **After** (marginal probe likelihood, N1 episodes), v2 / +clamp / +clamp+moddiff / all:
  - simM0.1: 10,636 / 10,636 / 10,632 / 10,631;
  - sim0.1: 10,657 / 10,657 / 10,656 / 10,657.
  - Flat within 0.05 %, and useful commits after all probes are 95.5 %.
- **Root cause.** The probe likelihood, not the score. Each particle used its own prior nuisance draw, so probing
  collapsed the posterior onto one biased draw. Now probes discriminate hypotheses with the nuisance marginalised,
  exactly as v2 does.
- **Test.** `test_certification_monotone_along_nested_probe_sets` (well-specified synthetic: 0 → 34 → 1048 certified)
  guards it.

**Matched probe time (review item 2).** Policy, simM0.1: regret ± se, mean probe s.

| budget | policies |
|---|---|
| 0 s | v2: 11.1 ± 2.3, 0 s |
| 8.8 s | fix:clamp 19.5 ± 2.2 (8.4 s) · evoi@8.8 13.5 ± 1.9 (3.6 s) · epig@8.8 11.1 ± 2.3 · eig@8.8 12.0 ± 2.3 · random@8.8 12.3 ± 2.3 |
| 16.8 s | fix:clamp+moddiff 25.9 ± 1.8 (16.0 s) · evoi@16.8 16.8 ± 2.3 (5.7 s) |

Useful commits are 95.4–95.5 % for every policy. Under the physics-trusting T0 posterior no probe pays for its time.
Unbudgeted EVOI never probes, so it equals v2. The per-minute metric is dropped. The probing question moves to the
envelope class (next section).

## Envelope class: probing where ambiguity is real (exploratory; review item 2)
Source: `scripts/decide_envelope.py`, `cfcap/decide/envelope.py` → `results/decision/envelope_{rows,summary,by_law}.csv`.

**Setup.**
- Episodes: theory E1 `gate_episode`, seeds 0–479 from theory's pickles. 463 episodes after dropping knee
  non-convergence.
- Truth: knee motor law with severity = seed % 4. Severity 0 is the two-constant law, 120 episodes; knee episodes number
  343.
- Children only. 1–3 random tasks per episode, 3-fold leak-free cross-fit.
- Lower bounds:
  - `env`: the identified-set low end `rho_pi`, model-free;
  - `par`: the two-constant fit `rho_par`;
  - `_cal`: minus the end-to-end simultaneous margin.
- Probe: clamp (8.8 s) or not.
- evoi = probe iff E[V · Δ certified tasks] > 8.8 s. The predictive of the bound's gain is cross-fitted from the
  calibration folds by (leg kind × child ≥ 3 modules × v2-undominated) cell.

| predictor | policy | ep unsafe | useful | abstain | regret ± se (s) | probe s | ≥3-mod yaw cert |
|---|---|---|---|---|---|---|---|
| env_cal | v2 | 0.0 % | 83.4 % | 10.6 % | 31.4 ± 2.4 | 0 | 67 % |
| env_cal | clamp | 0.22 % | 95.3 % | 4.8 % | 19.4 ± 2.1 | 8.8 | 90 % |
| env_cal | **evoi** | 0.22 % | 94.9 % | 4.8 % | **14.3 ± 2.2** | 2.05 (23 % probed) | 74 % |
| env (raw, uncalibrated) | v2 / clamp / evoi | 0.4 / 1.3 / 1.5 % | 83.4 / 96.5 / 95.8 % | | 34.0 / 23.6 / 20.6 | | |
| par (raw physics fit) | v2 | **5.2 %** (knee episodes 7.0 %) | 97.9 % | | 41.0 ± 8.4 | 0 | |
| par_cal | v2 / clamp | 0.4 % | 71.6 / 71.5 % | 17.7 % | 53.9 / 63.0 | | |

**Readings.**
1. **Under the model-free class, probing earns its place.** Clamp lifts useful commits from 83 % to 95 % and nearly
   halves regret. Decision-EVOI gets the same useful commits with a quarter of the probe time (it probes 23 % of the
   episodes). Its regret, 14.3 s, is the best of all, against 19.4 for fixed clamp and 31.4 for v2 (differences about
   2–3 se).
2. **Trusting the physics is unsafe under a wrong motor law.** Raw `par` is 7.0 % unsafe on knee episodes and 0 % on
   two-constant ones. Calibrating it pooled over laws makes it safe, but only 72 % useful, worse than the model-free
   class.
3. **Yaw legs of children with ≥ 3 modules.** Clamp raises their certified share from 67 % to 90 % (env_cal), but raw
   `env` with clamp has 14 % unsafe among their incapable pairs. Calibration is needed exactly there, which fits Prop. 2b
   (translation probes do not dominate yaw operating points).
4. **Matched time.** Here the library is one probe, so evoi is "clamp when worth it". At equal mean probe time, evoi
   (2.05 s) beats any fixed policy. Fixed clamp at 8.8 s is 5.1 s worse in regret.
5. **Caveats.**
   - E1's faults are known to the envelope construction: rho_pi uses the true fault modes. So this isolates motor-law
     uncertainty from fault-mode uncertainty.
   - Split time uses a line layout for the children's leftovers.
   - T0 only.

### H3 pre-registration: EVOI under motor-law AND fault uncertainty (written 2026-11-25 before the run)
**Design.**
- Episodes: `cfcap/decide/envelope_post.py`. E1 truth (knee severity seed % 4) on parents of 2–4 modules.
- Noisy probe signals; fault modes come from a v2-only posterior (≤ 2 faults, top mass 0.99 capped at 6 hypotheses).
- Bounds per hypothesis follow the E1 construction:
  - env = min over the kept hypotheses of rho_pi;
  - par = the posterior mean of rho_par;
  - the dominance trigger's upper end = max rho_par.
- Seeds **2100–2699** (600 parents; untouched; 2000–2009 were used for a smoke test and are excluded).
- Tasks from 959000 + seed; 3-fold leak-free cross-fit; S = 32 predictive draws.
- Policies: v2, always-clamp, evoi, and `dom`, the closed-form rival: clamp iff some required, v2-undominated leg's
  interval [lower_v2, max rho_par] straddles r.

**Bars** (primary predictor `env_cal`, all episodes pooled):
- **B1:** mean EVOI probe time ≤ 0.5 × always-clamp (≤ 4.4 s).
- **B2:** paired episode bootstrap (4,000) of mean(regret_EVOI − regret_clamp): the upper 95 % end is ≤ 0.
- **B3:** useful(EVOI) ≥ useful(clamp) − 1 pt.
- **B4** (vs the closed-form rival): paired upper 95 % end of mean(regret_EVOI − regret_dom) ≤ 0.
  - If B4 fails, the claim becomes "decision-EVOI matches a dominance-triggered clamp", with no superiority.
- **Validity:** env_cal ep_any_unsafe ≤ 0.10.

**Reported without a bar:** the ≥ 3-module-child yaw unsafe rate per policy (expected unresolved by probing, Prop. 2b),
and the knee vs two-constant split.

**Verdict.** H3 holds at confirmatory grade iff B1–B3 and validity hold. B4 decides "beats" vs "matches" the rival.

**Conformalized screen.** On 600 calibration episodes (seeds 929000+), the truth's screen statistic has a heavy tail:
median 2.0, 99 % quantile 23.9, max 29.5. Single faults reach 22.2 and double faults 23.7 at their 99 % quantiles.
- The calibrated threshold K(α_s = 0.01) = 23.9 keeps 55–7,000 hypotheses per episode, at up to 212 s per 4-module
  episode. So at α_s = 0.01 the conformalized screen is a compute problem, not a safety fix.
- Following N1, the safety comes from end-to-end calibration of the whole screened procedure. The screen threshold is
  a compute knob.
- **Measured on the first 240 of the N1 seeds, conformal K = 23.9 (M = 4, budget 2048) vs K = 5 (M = 8)**
  (`t0_kcal_*.csv`):

  | | conformal K = 23.9 | K = 5 |
  |---|---|---|
  | screen miss | 2.5 % (6/240; nominal 1 %, P(≥ 6 \| 1 %) ≈ 0.04) | 5.2 % |
  | empty screen | 0.8 % | 4.6 % |
  | sim0.1 episode unsafe | 0.83 % | 0.42 % |
  | sim0.1 useful | 98.6 % | 95.8 % |
  | sim0.1 regret | 7.8 s | 10.9 s |
  | bayes05 useful | 98.8 % | — |
  | generation | 9.9 s/episode | 3.3 s/episode |
  | median screened hypotheses | 1,828 | — |

  - The larger screen buys about 3 points of useful commits, because fewer screens come up empty.
  - simM0.1 is worse at this n (useful 92 %, abstain 6.7 %): 80 calibration episodes per fold cannot fill ≥ 60-episode
    bins.

**Method paragraph (for the paper).** *End-to-end validity under approximate inference.*
- The posterior is computed by an approximate procedure: a hard hypothesis screen, a finite set of nuisance draws, and
  mass truncation to a particle budget.
- Coverage is never assumed for the posterior itself. Instead the whole pipeline (screen → truncation → probe policy →
  final posterior → lower bound) is treated as one fixed map from observations to bounds. Its simultaneous score is
  calibrated on held-out generator episodes, so the guarantee is marginal over episodes whatever the approximation
  dropped.
- Failures that carry no information are not renormalised into confidence: an empty screen, or a likelihood underflow
  after a probe, marks the support unknown, and that forces abstention.
- Information the pipeline *does* see about its own difficulty is used as a Mondrian taxonomy: the truncated mass and
  the screened-set size are observable. A per-bin quantile keeps the guarantee within each difficulty class.
- For the uncalibrated Bayes route, a conformalized screen (threshold = (1 − α_s) split quantile of the truth's screen
  statistic on calibration episodes) plus recorded pruned mass ε bounds the risk by α + α_s + ε.

**Audit fixes** (codex `scorer_audit.md`, `screen_bug.md`; tests `tests/codex/test_audit_decide.py` and
`test_screen_bug.py` pass):
- **Leakage.** The test fold's in-loop q came from the other fold's labels. Now `evaluate.fold_plan` uses 3 folds: an
  in-loop fold h, calibration folds C, and test fold f, with both the C and the f trajectories steered by h.
- **Likelihood.** The Gaussian likelihood now includes its 1/σ normaliser.
- **Margins.** An infinite present error now gives an infinite margin.
- **Empty screen.** It now abstains; the "healthy" substitute is gone.
- **Diagnostics.** `truth_in_final` and `mass_dropped` are recorded.
- **Retired test.** `test_particle_cap_also_marks_unknown_support` is retired by the coordinator (N1), and replaced by
  `test_truncation_recorded_not_abstained` and `test_truncation_recorded_and_end_to_end_within_alpha`. In the latter,
  truncation makes raw 19.5 % unsafe, against sim 2.5 % and simM 0 %.

## Paper contract
- `results/decision/paper/probe_t0_n1.csv` (cert simM0.1) follows contract 1b, with `test_plant`=T0. Policy names add
  `fixed_v2_clamp`, `fixed_v2_clamp_moddiff`, and the matched-budget `evoi@8.8` / `epig@8.8` / `eig@8.8` /
  `random@8.8` / `evoi@16.8`.
- `make_all` currently stops on `results/stress/paper/dec_merge.csv`, which lacks `status` (stress role), so T4 was not
  re-rendered.
  - Pairs = (candidate, task). `n_unsafe` = certified-but-incapable pairs.
  - Ambiguous = posterior P(capable) ∈ (0.05, 0.95).
  - Extra columns: `regret`, `n_unsafe_assigned`, `n_split`.
- **Coordinator:** t4_probing.py groups by policy only. When the T1 file (`probe_t1.csv`) lands, either delete the T0
  file or add a `test_plant` filter.
- No `dec_*.csv` from this role: T0 ρ is not on the T1 legs or labels.

## Proposal: full-scale T1 (decides T4 / claim C4)
**Primary (T1, physics-trusting posterior).** `scripts/campaigns/decide_pilot.json` (draft, for the coordinator to
submit).
- `decide-canary`: 3 units (one per shape, 2 episodes each) ≈ 0.15 box-h.
- `decide-main`: 16 shards. That is 72 units of 4 episodes = 288 episodes (96 per shape), with the same episode seeds
  as nuis-main (`--nseed 1909`), 192 particles and 64 draws.
  - Per episode: SMC, plus T1 labels for children and 10 merges, plus clamp/tug/moddiff, plus one diagnostic split per
    child.
  - **Measured locally (4070, shared GPU):** `/tmp/decide_smoke`, 2 pair episodes, 64 particles, 16 draws, 40 stages.
    It took 148 s/episode (SMC 132 s, probes + labels 16 s), with peak VRAM 0.13 GB. The rows and pickles were
    checked.
  - Scaled estimate: ~320 s/episode on average (pair 150 / square 300 / 2x3 500) → ~1300 s/unit →
    **make_jobs emit (validated): 18 jobs, 30.5 box-h, $5.5 at $0.18–0.2/h** (16 main jobs × ~110 min, plus a 4-min shrunk canary and a 1-unit stage-1 canary), plus boot (~$0.5). It fits the 8 h window.
  - If the canary is slower, cut `--episodes` to 64 per shape (~$3.5).
- Offline evaluation on CPU (≤ 10 min): `scripts/decide_t1_eval.py <pulled dirs> --tag t1` →
  `results/decision/paper/probe_t1.csv`.
- It serves table T4 and claim C4: EVOI vs fixed/EIG/EPIG/split per probe-second, and the end-to-end certificate's
  unsafe/useful/abstain/regret on T1.

**Second predictor class (model-free speed-ceiling envelope; theory finding: v2 ambiguity 7 %, clamp → 0.06 %).** It
costs no rented compute; it is CPU only.
- An envelope predictor is a `DecisionPosterior` whose draws are the corners of the identified set. Its probes shrink
  the set.
- On T0 it needs a theory-supplied `envelope_lower`. Then rerun `decide_pilot.py` with a second predictor
  (~10 min on 4 CPU). On T1 it would reuse the same pulled episodes.

**If T2 truths exist** for the same eids (stress), re-score with the probe trajectory fixed. This is offline and costs
nothing.

## Env missions interface (proposal to the env role; plan §2a; please mirror it in docs/board/env.md)
The decision layer consumes one **mission** as one `problem.Task`. This is implemented and tested in 111de5b
(`test_per_candidate_requirement_and_feasibility`):
- `Task(r=NaN*(L,), r_c=(C, L'), ok_c=(C,), value=V, kind="mission")`.
  - `r_c[c, l']` is the max over the mission's segments using extended leg l′ = (condition k, leg l) of
    `req[c, s] = v_req[s] / (rho_ref[c, l] · |cmd[c, l]|)`; NaN where no segment uses it.
  - `ok_c[c]` = footprint fits every door AND load rating ≥ payload. `False` makes the body ineligible.
- The capability axis L′ = n_cond × 10 is (condition-major) `rho_dec[c, k, l]` flattened. Truth labels come from
  `label_missions` on the same axis.
- A segment is certified iff `lower[c, k, l] ≥ req[c, s]`. The mission is certified iff every segment is certified
  (simultaneous over segments: that is the max-over-entries score already in use).
- I need from env, per mission episode:
  - candidates (same order as the gen episode);
  - `r_c`, `ok_c`, condition keys;
  - `rho_dec` truth (C, n_cond, 10);
  - the body episode id, so the posterior draws (nuisance_post / models) can be joined;
  - optional: a per-mission value and time budget.
- Assignment / partition logic is unchanged: several missions = several tasks.

## Needs from other roles
- **nuisance:** keep `nuisance_post` `setup / sample_truth / screen / Q / smc / fresh_split / concat / Probe / Sim /
  observe_feats` stable. `task.py` calls them read-only. Use the same `--nseed` (1909) as nuis-main, so that eids join.
- **theory:** done. `cfcap/decide/envelope.py` consumes `theory_task.gate_episode` rows (rho_pi / rho_par at v2 and
  v2+clamp). It reuses your `$CFCAP_OUT/theory/gate/gate_*.pkl` when they exist.
- **stress:** T2 truths for the same eids. They let T4 be re-scored on T2 with the probe trajectory unchanged.
- **ops/compute:** resubmit the cancelled T1 jobs as campaign **`cfcap-decide2`**
  (`scripts/campaigns/decide_pilot.json`; `make_jobs emit --validate` ok: 18 jobs, 30.5 box-h, $5.5). The T1 episode
  data is unaffected by the fixes; evaluation is offline (`scripts/decide_t1_eval.py`, now 3 folds and simM/simF).
- **stress:** `results/stress/paper/dec_merge.csv` is missing `status`, which breaks `make_all`.
- **env:** the mission fields above.

## Next step
- Add the model-free envelope predictor to T0 and to the T1 evaluation (the theory finding: clamp resolves 7 % → 0.06 %
  under that class).
- Run the T1 canary, then the main grid. Score policies on T1, and on T2 truth once it exists.
- Recalibrate V_TASK / C_UNSAFE as a sensitivity sweep.
