Exploratory audit completed; source code unchanged, CPU only.  
Final suite: **19 failed, 47 passed**, recorded in [audit run record](/home/lunarz/cfcap/tests/codex/test_audit_conformal.py:3).  
Highest-severity findings: empty-set certification and adaptive calibration/test label leakage.

Intended semantics reviewed:

- **Conformal:** corrected split quantile; episode-max normalized overprediction score; finite lower bounds meeting the requirement permit commit; CRC uses bounded episode losses.
- **Baselines:** nominal healthy prediction, privileged oracle, RLS intervals, posterior summaries, and capxfer’s minimum over consistent hypotheses all feed simultaneous calibration.
- **Decisions:** every required leg and size constraint must pass. Unsafe means assigned but incapable; useful means served. Regret subtracts realized utility from oracle utility.
- **Analysis:** matched margins fit calibration rows and remain frozen across evaluation plants; relevant means faulted **or** near threshold. Held-out coverage is empirical; probing metrics aggregate episode counts.
- **Nuisance ambiguity:** `q05 < τ <= q95`, using decision labels and excluding the truth row from posterior quantiles.

Threshold ties, ordinary cross-fit isolation, frozen matched-margin isolation, ragged nonempty candidates, utility/regret examples, held-out coverage filtering, and nuisance ambiguity boundaries passed.

Each row below represents a failing test and a **suspected bug**. Reproductions are in the linked audit files.

| Severity | Source | Input; expected → actual |
|---|---|---|
| **High** | [capxfer_cert.py:139](/home/lunarz/cfcap/cfcap/baselines/capxfer_cert.py:139) | Sets `[{0}, ∅]`, child wheel `[0]`: globally empty support must refuse → healthy child vector returned. [Test](/home/lunarz/cfcap/tests/codex/test_audit_baselines.py:73) |
| **High** | [evaluate.py:124](/home/lunarz/cfcap/cfcap/decide/evaluate.py:124) | Change only held-out fold truth from `.9` to `.1`: its margin should remain fixed → approximately `5` becomes `−.76923`, through calibration-fold adaptive probing. [Test](/home/lunarz/cfcap/tests/codex/test_audit_decide.py:99) |
| **High** | [posterior.py:83](/home/lunarz/cfcap/cfcap/decide/posterior.py:83) | Equal means, equal prior weights, particle SDs `[1,10]`, observation at the mean: weights `[10/11,1/11]` → `[.5,.5]`. Gaussian normalization is missing. [Test](/home/lunarz/cfcap/tests/codex/test_audit_decide.py:78) |
| **High** | [base.py:107](/home/lunarz/cfcap/cfcap/baselines/base.py:107) | Faulted-far and healthy-near candidates: relevant count `2` → `0`; baseline uses AND where paper uses OR. [Test](/home/lunarz/cfcap/tests/codex/test_audit_baselines.py:67) |
| **Medium** | [base.py:108](/home/lunarz/cfcap/cfcap/baselines/base.py:108) | Lower `+inf`, truth `.2`, τ `.7`: refuse, unsafe count `0` → `1`. Scoring disagrees with the actual decision rule. [Test](/home/lunarz/cfcap/tests/codex/test_audit_baselines.py:54) |
| **Medium** | [base.py:105](/home/lunarz/cfcap/cfcap/baselines/base.py:105) | Truth `[+inf,−inf,NaN]`: exclude invalid labels, decision count `0` → `2`. [Test](/home/lunarz/cfcap/tests/codex/test_audit_baselines.py:59) |
| **Medium** | [core.py:60](/home/lunarz/cfcap/cfcap/conformal/core.py:60) | Empty candidate arrays `(3,0)`: infinite calibration margin/refusal → reduction exception. [Test](/home/lunarz/cfcap/tests/codex/test_audit_conformal.py:41) |
| **Medium** | [evaluate.py:60](/home/lunarz/cfcap/cfcap/decide/evaluate.py:60) | Calibration episodes with zero errors plus an infinite present error: margin `+inf` → `0`; the infinite-error episode is dropped. [Test](/home/lunarz/cfcap/tests/codex/test_audit_decide.py:121) |
| **Medium** | [common.py:231](/home/lunarz/cfcap/scripts/analysis/common.py:231) | All calibration entries capable: matched rule should permit finite capable predictions → `k=−inf` produces lower bounds `−inf`, refusing everything. [Test](/home/lunarz/cfcap/tests/codex/test_audit_analysis.py:77) |
| **Medium** | [common.py:290](/home/lunarz/cfcap/scripts/analysis/common.py:290) | “Always go” uses lower `+inf`: expected certified/unsafe counts `(1,1)` → `(0,0)`, because the scorer rejects infinity. [Test](/home/lunarz/cfcap/tests/codex/test_audit_analysis.py:97) |
| **Medium** | [t3_grid.py:85](/home/lunarz/cfcap/scripts/analysis/t3_grid.py:85) | Amortized is more useful but less safe and slower: expected tradeoff verdict → “SMC wins on both.” Failure to dominate is treated as reverse dominance. [Test](/home/lunarz/cfcap/tests/codex/test_audit_analysis.py:123) |
| **Medium** | [t4_probing.py:33](/home/lunarz/cfcap/scripts/analysis/t4_probing.py:33) | No capable relevant pairs, numerator/denominator `0/0`: undefined → reported rate `0`. [Test](/home/lunarz/cfcap/tests/codex/test_audit_analysis.py:134) |
| **Low** | [core.py:59](/home/lunarz/cfcap/cfcap/conformal/core.py:59) | Present predictions `[1,1]`, truth `[1,NaN]`: reject invalid required label or worst-case score → score `0`. Requires distinguishing missing labels from padding. [Test](/home/lunarz/cfcap/tests/codex/test_audit_conformal.py:48) |
| **Low** | [core.py:38](/home/lunarz/cfcap/cfcap/conformal/core.py:38) | Calibration weights `[1,NaN]`: reject/refuse → quantile `0`. [Test](/home/lunarz/cfcap/tests/codex/test_audit_conformal.py:60) |
| **Low** | [core.py:38](/home/lunarz/cfcap/cfcap/conformal/core.py:38) | Calibration weights `[1,1]`, test weight `−2`: reject/refuse → quantile `0`. [Test](/home/lunarz/cfcap/tests/codex/test_audit_conformal.py:60) |
| **Low** | [core.py:41](/home/lunarz/cfcap/cfcap/conformal/core.py:41) | Selected weighted score is NaN: worst-case `+inf` → NaN. [Test](/home/lunarz/cfcap/tests/codex/test_audit_conformal.py:69) |
| **Low** | [core.py:91](/home/lunarz/cfcap/cfcap/conformal/core.py:91) | CRC losses `−inf`: reject invalid bounded losses/refuse → accepts margin `0`. [Test](/home/lunarz/cfcap/tests/codex/test_audit_conformal.py:81) |

The malformed-input cases are validation gaps, not observed corpus defects.

The requested directory run also exposed these existing failures; I did not edit their tests:

| Severity | Source | Input; expected → actual |
|---|---|---|
| **High** | [t0data.py:112](/home/lunarz/cfcap/cfcap/decide/t0data.py:112) | Empty screened support: expected refusal → healthy fallback yields finite bounds. Reproduced by [screen-miss test](/home/lunarz/cfcap/tests/codex/test_screen_bug.py:73). |
| **Medium** | [t0data.py:132](/home/lunarz/cfcap/cfcap/decide/t0data.py:132) | Particle cap leaves a single particle: existing test expects unknown-support refusal → finite bounds. This requires an explicit support-loss contract; raw posterior bounds carry no guarantee. [Test](/home/lunarz/cfcap/tests/codex/test_screen_bug.py:91) |

Files written:

- [test_audit_conformal.py](/home/lunarz/cfcap/tests/codex/test_audit_conformal.py)
- [test_audit_baselines.py](/home/lunarz/cfcap/tests/codex/test_audit_baselines.py)
- [test_audit_decide.py](/home/lunarz/cfcap/tests/codex/test_audit_decide.py)
- [test_audit_analysis.py](/home/lunarz/cfcap/tests/codex/test_audit_analysis.py)