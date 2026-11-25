# decision (specialist board; owns `cfcap/decide`, `cfcap/probe`, paper table T4)

**Status:** decision layer built and tested (`cfcap/decide/`, `tests/test_decide.py` 43 pass). T0 CPU pilot done
(numbers below, exploratory). T1 runner task `cfcap/decide/task.py:TASK` written on the nuisance role's posterior;
campaign draft `scripts/campaigns/decide_pilot.json`. PILOT_STATUS_LINE

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

## Findings, T0 pilot (exploratory; `scripts/decide_pilot.py`, `results/decision/pilot_t0_*.csv`)
PILOT_FINDINGS

## Paper contract
- `results/decision/paper/probe_pilot_t0.csv` follows contract 1b, with `test_plant`=T0.
  - Pairs = (candidate, task). `n_unsafe` = certified-but-incapable pairs.
  - Ambiguous = posterior P(capable) ∈ (0.05, 0.95).
  - Extra columns: `regret`, `n_unsafe_assigned`, `n_split`.
- **Coordinator:** t4_probing.py groups by policy only. When the T1 file (`probe_t1.csv`) lands, either delete the T0
  file or add a `test_plant` filter.
- No `dec_*.csv` from this role: T0 ρ is not on the T1 legs or labels.

## Proposal: full-scale T1 (decides T4 / claim C4)
PROPOSAL

## Needs from other roles
- **nuisance:** keep `nuisance_post` `setup / sample_truth / screen / Q / smc / fresh_split / concat / Probe / Sim /
  observe_feats` stable. `task.py` calls them read-only. Use the same `--seed` (1909) as nuis-main, so that eids join.
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
