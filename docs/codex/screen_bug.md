Exploratory: the screen drops truth in **27/600 episodes (4.5%)** ([meta CSV](/home/lunarz/cfcap/results/decision/pilot_t0_meta.csv)).
All **six unsafe episodes** exhaust the screen, then substitute healthy ([episode CSV](/home/lunarz/cfcap/results/decision/pilot_t0_episodes.csv), [replay test](/home/lunarz/cfcap/tests/codex/test_screen_bug.py)).
Wrote an unapplied abstention patch and verified the regression against it in memory.

The failure path:

- **Hard screen:** [t0data.py](/home/lunarz/cfcap/cfcap/decide/t0data.py:98) enumerates the complete discrete fault space, then rejects a hypothesis whenever any signal’s standardized residual exceeds **5**, using nominal nuisances and sigma inflated by **1.5**. Screening proceeds sequentially over probe legs; enumeration order does not prioritize single faults.
- **Nuisance mismatch:** [observe.py](/home/lunarz/cfcap/cfcap/t0/observe.py:50) estimates nuisance spread from healthy draws and adds heuristic motor terms. These are standard deviations, not bounds for fault-dependent nonlinear responses. Its wheel-speed inflation applies only when observed duty indicates saturation.
- **Observed rejection:** every unsafe episode rejects truth on a wheel-speed feature, with maximum residuals **5.14–27.42** ([replay fixtures](/home/lunarz/cfcap/tests/codex/test_screen_bug.py)). Their screens eventually become empty. The healthy replacement then receives normalized posterior weight, producing confident capability predictions.
- **Additional truncation:** the posterior keeps cumulative conditional mass **0.9999**, capped at **192 particles**, then renormalizes ([t0data.py](/home/lunarz/cfcap/cfcap/decide/t0data.py:129)). This can remove further true states. Sigma inflation softens likelihood concentration but cannot restore screened-out support.

Exploratory screen-miss rates by fault count are **24/250** for double faults, **3/222** for single faults, and **0/128** for healthy episodes ([meta CSV](/home/lunarz/cfcap/results/decision/pilot_t0_meta.csv)).

Metadata issue: `n_screen` conflates an empty-screen healthy replacement with a genuine singleton, while `truth_in` is recorded before particle truncation ([t0data.py](/home/lunarz/cfcap/cfcap/decide/t0data.py:111)).

Other screening paths:

- [baselines/posterior.py](/home/lunarz/cfcap/cfcap/baselines/posterior.py:265): legacy nuisance SMC ranks all hypotheses at mean nuisance with sigma multiplied by **3**, retaining the top **96**. Subsequent moves change nuisances only, so excluded fault states cannot recover.
- [bayes.py](/home/lunarz/cfcap/cfcap/baselines/bayes.py:25): exact enumeration is subsequently reduced to conditional mass **0.999**, capped at **32**; SMC baselines also subsample propagation particles. [episode.py](/home/lunarz/cfcap/cfcap/baselines/episode.py) constructs episodes and propagates capability; it contains no hypothesis screen.
- [nuisance_post.py](/home/lunarz/cfcap/cfcap/nuisance_post.py:286): the single-fault screen builds a **soft proposal**, mixing **30% prior** with **70% screened proposal**. Double faults retain positive proposal probability, and discrete moves can recover them. Finite sampling can still miss them; tempering provides no retention guarantee.
- [stagea.py](/home/lunarz/cfcap/cfcap/t0/stagea.py:146): the idealized screen uses the true continuous plant. That oracle consistency property does not transfer to nominal-nuisance screening.

The proposed patch preserves omitted support as an explicit, unbounded unknown flag. Screening, truncation, or weight underflow triggers abstention through both summary and Bayesian bounds; copies and probe updates preserve the flag. It also adds separate empty-screen and final-particle diagnostics.

**Tradeoff:** every pilot episode pruned hypotheses, so this fallback blocks every recorded episode ([meta CSV](/home/lunarz/cfcap/results/decision/pilot_t0_meta.csv)). Useful recovery requires independently calibrating the truth’s maximum residual with `split_quantile`, retaining the resulting set without a hard cap, and retaining the unknown guard whenever the compute budget prevents propagation. Screening coverage alone does not validate the approximate posterior.

The safety regressions fail on current source and pass with the diff loaded in memory; existing decision tests also pass. Reproduce using the harness documented in [the test](/home/lunarz/cfcap/tests/codex/test_screen_bug.py), through `scripts/heavy.sh`. No source patch was applied.

Files written:

- [docs/codex/screen_bug.patch](/home/lunarz/cfcap/docs/codex/screen_bug.patch)
- [tests/codex/test_screen_bug.py](/home/lunarz/cfcap/tests/codex/test_screen_bug.py)