# decision (specialist board; owns `cfcap/decide`, `cfcap/probe`, paper table T4)

**Status:** decision layer built and tested (`cfcap/decide/`, `tests/test_decide.py` 43 pass). T0 CPU pilot done
(numbers below, exploratory). T1 runner task `cfcap/decide/task.py:TASK` written on the nuisance role's posterior;
campaign draft `scripts/campaigns/decide_pilot.json`. The T4 contract file is in place, and `make_all` fills T4 (checked; the paper outputs were not committed).

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

## Findings, T0 pilot (exploratory; `scripts/decide_pilot.py`, `results/decision/pilot_t0_*.csv`)
**Setup.**
- 600 episodes: parents of 2–4 modules, 79 % faulted, heterogeneity on, plus merges with a healthy single.
- 1–3 tasks per episode.
- The posterior is ≤ 2-fault enumeration × 8 prior nuisance draws, with the nuisance marginalised by sd inflation.
  - The truth survives the screen in 95.5 % of episodes.
  - Mean particle count 63, median ESS 8.5.
- Cross-fitted over 2 folds. Utility constants are V 120 s and C_unsafe 600 s. Generation 421 s, evaluation 60 s.

**Unsafe/useful rates are per assignment; "ep" rates are per episode.** Source: `pilot_t0_summary.csv`.

| predictor / policy / certifier | unsafe commit | ep any unsafe | useful commit | abstain | regret (s, ±se) | probe s |
|---|---|---|---|---|---|---|
| oracle (raw / sim) | 0 | 0 | 100 % | 0 | 0 | 0 |
| nominal, raw (plug-in healthy) | **8.7 %** | 15.7 % | 92.7 % | 0 | 116 ± 11 | 0 |
| nominal, sim0.1 (q = 30) | 0 | 0 | 0 | 100 % | 193 ± 3 | 0 |
| posterior, v2, raw | 1.1 % | 2.0 % | 98.9 % | 0 | 14.3 ± 4.1 | 0 |
| posterior, v2, **sim0.1** (q ≈ 1.0) | 0.54 % | 1.0 % | 99.4 % | 0 | **7.7 ± 2.9** | 0 |
| posterior, v2, crc0.01 (λ ≈ 5.5) | 0.57 % | 0.83 % | 77.7 % | 11.5 % | 45.6 ± 3.6 | 0 |
| posterior, evoi, sim0.1 | 0.72 % | 1.3 % | 99.3 % | 0 | 9.8 ± 3.3 | 0.18 |
| posterior, epig, sim0.1 | 0.63 % | 1.2 % | 99.3 % | 0 | 9.5 ± 3.1 | 0.67 |
| posterior, eig, sim0.1 | 0.63 % | 1.2 % | 99.3 % | 0 | 11.5 ± 3.2 | 2.7 |
| posterior, random, sim0.1 | 0.81 % | 1.5 % | 99.2 % | 0 | 18.3 ± 3.6 | 7.2 |
| posterior, all (clamp+moddiff+tug), sim0.1 (q = 5.2) | 0.71 % | 1.2 % | 87.4 % | 4.8 % | 55.2 ± 3.6 | 24.8 |
| posterior, always_split, sim0.1 | 0.54 % | 1.0 % | 99.5 % | 0 | 49.5 ± 2.9 | 42 |

**Probe value (T4 contract, `paper/probe_pilot_t0.csv`).** Pairs are (candidate, task).
- After v2, 138 of 12,862 pairs are ambiguous (1.1 %).
- Share of those resolved, with the resolved share ÷ extra probe-minutes (T4's "per extra min", ×100 there):
  - EVOI: 23 % at 0.18 s mean probe time (78).
  - EPIG: 20 % at 0.67 s (18).
  - EIG: 9 % at 2.7 s (2).
  - random: 19 % at 7.2 s (1.6).
  - fixed all: 99 % at 24.8 s (2.4).
  - always split: 92 % at 42 s (1.3).
- EVOI wins per probe-second.
- No probing policy improves regret over fixed v2: every difference is within 1 se. The only exceptions are the costly
  policies (all, split), which do worse.

**Decision-relevant stratum.** The faulted piece or |ρ − r| < 0.1 covers 8,135 pairs. Certified share is 99.0–99.1 %
for every policy except fixed all (72 %). Unsafe certified pairs are 3.3–3.7 % of the stratum's incapable pairs (fixed all 2.8 %).

**Readings** (exploratory; T0 only, and the T0 posterior is crude):
1. The oracle has zero unsafe commits and zero regret. The plug-in nominal predictor commits unsafely 8.7 % of the time.
   Conformalising the nominal predictor makes it safe but useless (100 % abstain).
2. Every residual unsafe commit (6/600 episodes, v2 sim) is a **screen miss**: a 2-fault truth outside the screened
   hypothesis set, with truth_in = 0 in 6/6. The posterior's misspecification sets the floor, not the decision rule.
3. **The end-to-end conformal certificate catches a probe-induced misspecification.** After all three probes the
   posterior is overconfident: its likelihood uses prior nuisance draws with only an sd inflation. The cross-fitted
   end-to-end q rises from 1.0 to 5.2, and useful commits fall to 87 %. A fixed q calibrated on v2 would have been
   unsafe there.
4. The CRC margin is λ = 0 at α = 0.1 / 0.05, because the plug-in risk of 2 % is already below α. At α = 0.01 the
   finite-sample term (B/(n+1) with n = 300 per fold) forces λ ≈ 5.5 and 11 % abstention. The simultaneous certificate
   (nominal α = 0.1) is far more conservative than its level but not in utility. Its realised episode risk (1 %) beats
   CRC@0.01 in regret.
5. Probing is secondary on T0, as Stage A found (C4 fallback holds on T0). The T1/model-free comparison is still open,
   see the proposal.

## Paper contract
- `results/decision/paper/probe_pilot_t0.csv` follows contract 1b, with `test_plant`=T0.
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

## Needs from other roles
- **nuisance:** keep `nuisance_post` `setup / sample_truth / screen / Q / smc / fresh_split / concat / Probe / Sim /
  observe_feats` stable. `task.py` calls them read-only. Use the same `--nseed` (1909) as nuis-main, so that eids join.
- **theory:** the speed-ceiling envelope class from `scripts/theory_coverage.py`, as a function
  `envelope_lower(asm, obs_by_probe) -> (C, L)`, or as corner draws. It plugs in as a degenerate posterior whose probes
  shrink the interval. This gives T4's second predictor class.
- **stress:** T2 truths for the same eids. They let T4 be re-scored on T2 with the probe trajectory unchanged.
- **compute:** submit `decide_pilot.json` (canary first).

## Next step
- Add the model-free envelope predictor to T0 and to the T1 evaluation (the theory finding: clamp resolves 7 % → 0.06 %
  under that class).
- Run the T1 canary, then the main grid. Score policies on T1, and on T2 truth once it exists.
- Recalibrate V_TASK / C_UNSAFE as a sensitivity sweep.
