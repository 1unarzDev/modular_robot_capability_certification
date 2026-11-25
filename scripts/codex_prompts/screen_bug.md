TASK screen_bug. In the T0 decision pilot (docs/board/decision.md), all 6 unsafe commits happened because the
posterior's hypothesis screen dropped the true 2-fault state. Find the screen(s): cfcap/decide/, cfcap/baselines/
(posterior.py, bayes.py, episode.py), cfcap/nuisance_post.py, cfcap/t0/. Explain exactly why the true state is
pruned (thresholds, top-k, single-fault-first search, tempering), quantify how often it happens if you can do so
cheaply on CPU (T0 only, through scripts/heavy.sh, ≤10 min), and propose a minimal fix that keeps the true state with
high probability, e.g. a coverage-guaranteeing screen, conformalized pruning, or a fallback to an "unknown" mass
that forces abstention. Write the proposed patch as docs/codex/screen_bug.patch (unified diff; do not apply it) plus
a test under tests/codex/test_screen_*.py that fails before the patch and passes after.
