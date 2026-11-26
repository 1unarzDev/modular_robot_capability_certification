# Weak points C: statistics and data-leakage audit (2026-11-25)

Scope: docs/CONFIRMATORY.md and its amendments, scripts/analysis/confirmatory.py, generator and stress/decision seed
streams. I did not open results/confirmatory or any conf_* output. The audit was time-boxed to 15 min, so items
marked "unverified" need a follow-up check.

Ranked by severity.

## 1. HIGH: P3's Holm row can PASS when its other conditions fail
- Evidence: confirmatory.py:451-470. CONFIRMATORY.md §1 bar 4 (lines 114-121) requires all three of: raw E_dec > 2 %
  (lower > 1 %), ours E_dec <= 2 %, and the paired E_ep difference > 0 (Holm).
- The row with status HOLM only carries `lo_gt=0.0`. `holm()` (1115-1130) checks that row's own threshold string and
  nothing else. The raw condition only goes into `note`, and the ours condition is never checked.
- Result: when raw <= 2 %, a DROPPED row is written, but the HOLM row can still resolve to PASS.
- Fix (10 min): put the two condition verdicts into the HOLM row (e.g. a `requires` field). In `holm()`, PASS only if
  both are PASS. If raw <= 2 %, set P3 to DROPPED and keep its p in the family (m = 7). Add a mock test. Before
  scoring, check P6' (around line 830) for the same pattern.

## 2. HIGH: UNDERPOWERED overrides FAIL, and A-POWER makes UNDERPOWERED more likely
- Evidence: `Scorer.bar` (confirmatory.py:305-309) checks power before PASS/FAIL. So a safety bar with a point estimate
  of 5 % in an underpowered cell prints UNDERPOWERED, not FAIL.
- A-POWER (CONFIRMATORY.md:404, 413) cut C-MAIN from 3,000 to 2,400 episodes, about 21.6k incapable decisions, just
  above the 20k floor. Passing the power rule now needs deff <= ~3.6 instead of ~4.5. For C-GATE the document already
  concedes E_dec is underpowered at any feasible n (line 406).
- The trim is a prefix taken without seeing labels, so it is not selection. But it shifts E_dec verdicts from
  "can FAIL" towards "cannot be judged", on an owner request made 25 min before scoring.
- Fix (5 min): if a cell fails the power rule and its point estimate or one-sided upper bound is above the bar, print
  "UNDERPOWERED (point/upper exceeds bar)" and count it as not shown valid.
- Disclosure: "A-POWER reduced C-MAIN to 2,400 episodes for compute reasons, which lowered the design-effect tolerance
  of the E_dec power rule from ~4.5 to ~3.6. E_dec cells that failed the power rule are reported with their point
  estimates and upper bounds and are not counted as validated."

## 3. MEDIUM: amendment timestamps do not match the git commits; A-SEED was committed 2.5 h after its stated time
- Evidence: git log (commit times are -06:00).
  - A-SEED says "22:05Z" (CONFIRMATORY.md:374), but commit 03387d0 is 18:31-06:00 = 00:31Z on 11-26.
  - A-GATE says 21:40Z but was committed at 21:17Z; A-ENV says 21:55Z, committed 21:25Z.
  - §0.1 says the amendment is "21:05Z" but it was committed at 20:57Z. The header says "~21:10Z", but the file was
    first committed at 20:53Z.
- Confirmatory model evaluation (eval2b) was queued by 00:31Z. The "before any confirmatory output from either seed
  exists" claim (line 378) is therefore unverified. Picking the seed for availability is harmless only if no st-n6-2
  conf rows were readable at 00:31Z.
- Fix (15 min): from the queue/ledger logs (compute/LEDGER.csv, agent logs), record when the first conf model shard
  landed for each seed. Correct the in-text times to the commit times, or add "(decided at X, committed at Y)".

## 4. MEDIUM: A-GATE is outcome-informed by exploratory E1, and `robust` is calibrated on s = 3
- Evidence: CONFIRMATORY.md:186-197. The primary certifier was replaced after exploratory E1 falsified the gate. This
  is legitimate, because no confirmatory label existed. But the new primary (`robust`) has its margin fitted on knee
  episodes pooled over s in {0, 1, 2, 3}.
- So its s = 3 validity (P6' and validity at every severity) is in-distribution for the margin and is not an
  extrapolation test. Only `lawcal` holds out s = 3.
- The calibration knee episodes come from the exploratory `cal` latents (gate_task.py:8, `--split cal --root
  92599,97`). That is disjoint from conf_test, so there is no leakage.
- Disclosure: "The primary C-GATE certifier was chosen after an exploratory falsification of the trust gate, and its
  margin was calibrated on episodes of all four knee severities, including s = 3. Its validity at s = 3 is therefore
  within-family; only `lawcal` is tested on a held-out severity."

## 5. MEDIUM: A-POWER also trims the calibration set (cal 0-1,999 to 0-999)
- Evidence: CONFIRMATORY.md:405; POWER_RANGES at confirmatory.py:89.
- The exploratory cal labels have been visible all along. Choosing how much of cal to use after the first registration
  is a degree of freedom over margins whose effect anyone could have previewed. A prefix limits the harm.
- Fix (10 min): report the matched-point margins fitted on cal 0-999 and on cal 0-1,999 side by side, as descriptive
  rows. Or disclose: "Cal was cut to its first 1,000 episodes for compute; margins from the full 2,000 are reported in
  Appendix X."

## 6. MEDIUM-LOW: the headline stratum is defined using the label
- Evidence: confirmatory.py:165-166; CONFIRMATORY.md:60. The stratum keeps decisions with |truth - tau| < 0.10.
- Restricting on the outcome means stratum E_dec and CF are not the quantity the certificate guarantees. The robot
  cannot form this stratum at deployment, and capable/incapable decisions near tau are over-represented by design.
- The matched point is fitted on the same label-defined stratum in cal (`stratum="rel"`, line 340). That is consistent,
  but it inherits the problem.
- Disclosure: "The decision-relevant stratum is defined using the true capability (|rho - tau| < 0.10). It is an
  evaluation lens, not a deployable filter. E_ep on all decisions is the guaranteed quantity."

## 7. LOW: in-sample margins in exploratory stress, used for A-POWER sizing
- N7 (scripts/stress_N7.py:30-60) replays A5m's own draws (SEED0 906000, same as stress_A5m.py:34). It fits q from
  T1 perturbations and evaluates T2 unsafe/CF on the same episodes. q uses no T2 label, so this is optimism rather
  than leakage. N7's margin feeds the C-GATE singles+pairs secondary cell.
- N4 `q_pert_by_nmod` (results/stress/N4_margins.json) is fitted on E2 exploratory pairs (seeds 907000), the same pairs
  behind "size-aware miscov 0.002-0.02". A-POWER sized C-MERGE (500 pairs) from that number (CONFIRMATORY.md:396,
  407), so the power estimate is optimistic. The confirmatory C-MERGE stream ([92599, 21, ...], merge_task.py:20) is
  disjoint.
- Disclosure: "Exploratory margins (N4, N7) were evaluated in-sample; confirmatory C-MERGE sizing used those in-sample
  rates."

## 8. LOW: RNG stream aliasing in the decision envelope (latent hazard; does not affect NEXT-1b)
- Evidence: cfcap/decide/envelope_post.py:128 uses `default_rng((60_000 + seed,))` and line 158 uses
  `default_rng((61_000 + seed,))`. numpy treats `(x,)` the same as `x` (verified). So the v2 nuisance-sd stream of
  calibration seed s (5000-5599) is identical to the clamp nuisance-sd stream of seed s-1000, which falls in the eval
  range 4000-4499 for s = 5000-5499.
- Line 158 only runs when `lik == "robust"`. NEXT-1b used the v2-only likelihood (docs/board/decision.md:503), so the
  knee-tolerant screen's calibration (5000-5599) and evaluation (3900-4499) are disjoint in practice. The truth
  streams (30_000+seed, (40_000+seed, sev), 50_000+seed) do not overlap between the two ranges.
- Fix (5 min): use tagged roots, e.g. `default_rng([60_000, seed])` and `default_rng([61_000, seed])`. Note the
  aliasing in decision.md if any robust-likelihood run spans both ranges.
- In the same vein: `merge_task.py:66` passes one scalar T2 seed to every pair. T2 noise is then common across pairs,
  which correlates clusters; this is not leakage. A per-pair seed would be cleaner.

## Checked and found sound
- Generator split separation: `episode_rng` (episode.py:165-167) roots conf_* at SEED_CONF 92599 with codes +64,
  versus SEED0 92511. The streams are fully disjoint from train/cal/test. The conf twins reuse the same RULES (line
  100), which is intended.
- The conf_test 0-39 exclusion, the canary shard 0 exclusion, and the conf-ID 1499 exclusion are justified by
  pre-label construction and implemented (confirmatory.py:148-160, ENV_RANGES). The spare/canary units of gate and
  merge use separate roots (92599,98 / 92599,99).
- Bootstrap: one multinomial weight matrix per cluster set, reused for paired contrasts (correct pairing). The C-GATE
  bootstrap is paired across severities. The clustering units match §0.3.
- Holm: step-down over m = 7 with missing P's keeping their slot. P5 uses an intersection-union max-p. The one-sided
  p = (1 + #{d* <= 0}) / (B + 1) has a minimum of 5e-4, which is below 0.05/7.
- One-sided bounds use the 95th/5th percentiles. The power rule matches §0.4 and the §9 ratification (z = 1.96, Kish
  fallback).
- Allowlists (TAG_ALLOW, lines 117-123): st-family is restricted to st-n6-2 and blackbox_gnn to bb-n6-1. Mixing tagged
  and untagged rows raises an error.
- `SCORING_OPENS` gate (line 1313). A-POWER prefixes are applied and the dropped rows are logged.
- Model selection vs evaluation: P2's "best trivial" is picked on cal (in-sample on cal, line 437); no conf data is
  involved. I found no conf episode used for selection. results/models/calib.csv is detector ECE; the split it was
  computed on was not verified.
