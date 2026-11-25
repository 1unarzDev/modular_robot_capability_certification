Exploratory audit: the saved decision-stratum rates reproduce from the episode data; calibration and test episodes are disjoint.  
The recorded kill and both success bars fail; the first success bar’s probe scope is ambiguous in the preregistration.  
The audit tests pass; the sufficiency check does not enforce all of its stated conditions.

The table gives certified share among capable decisions and unsafe rate among incapable decisions, both in percent, with episode-bootstrap 95% intervals. Intervals condition on the fitted cross-fold bounds. Rates reproduce [gate_cert.csv](/home/lunarz/cfcap/results/theory/gate_cert.csv); intervals are computed from the episode pickles in `$CFCAP_OUT/theory/gate/` by [test_gate_audit.py](/home/lunarz/cfcap/tests/codex/test_gate_audit.py).

| Probe | Severity | Certifier | Certified % [95% CI] | Unsafe % [95% CI] |
|---|---:|---|---|---|
| v2 | 0 | par | 96.6 [96.1, 97.0] | 2.3 [0.9, 4.0] |
| v2 | 0 | pi | 75.1 [73.5, 76.5] | 0.2 [0.0, 0.6] |
| v2 | 0 | gate | 81.8 [79.7, 83.7] | 0.5 [0.0, 1.2] |
| v2 | 0 | raw | 99.2 [99.0, 99.4] | 6.3 [3.9, 9.2] |
| v2 | 1 | par | 96.6 [96.2, 97.0] | 3.0 [1.4, 5.3] |
| v2 | 1 | pi | 74.1 [72.8, 75.4] | 0.1 [0.0, 0.3] |
| v2 | 1 | gate | 79.5 [76.9, 81.8] | 0.7 [0.1, 1.6] |
| v2 | 1 | raw | 99.2 [99.1, 99.4] | 6.6 [4.2, 9.8] |
| v2 | 2 | par | 96.6 [96.1, 97.0] | 3.7 [2.5, 5.1] |
| v2 | 2 | pi | 73.0 [71.6, 74.5] | 0.1 [0.0, 0.3] |
| v2 | 2 | gate | 79.4 [77.2, 81.6] | 0.2 [0.0, 0.5] |
| v2 | 2 | raw | 99.3 [99.2, 99.4] | 8.0 [6.2, 10.0] |
| v2 | 3 | par | 97.0 [96.6, 97.4] | 29.2 [26.3, 32.5] |
| v2 | 3 | pi | 70.9 [69.3, 72.5] | 2.3 [1.7, 2.9] |
| v2 | 3 | gate | 72.1 [69.4, 74.9] | 1.7 [1.2, 2.4] |
| v2 | 3 | raw | 99.5 [99.3, 99.6] | 48.3 [44.7, 52.2] |
| v2+clamp | 0 | par | 96.1 [95.5, 96.6] | 1.4 [0.4, 2.7] |
| v2+clamp | 0 | pi | 92.7 [91.7, 93.7] | 0.3 [0.0, 0.7] |
| v2+clamp | 0 | gate | 94.1 [93.1, 95.1] | 1.0 [0.4, 1.9] |
| v2+clamp | 0 | raw | 99.3 [99.1, 99.4] | 6.5 [4.0, 9.4] |
| v2+clamp | 1 | par | 96.0 [95.5, 96.5] | 1.6 [0.3, 3.4] |
| v2+clamp | 1 | pi | 92.2 [91.2, 93.1] | 0.2 [0.0, 0.5] |
| v2+clamp | 1 | gate | 92.9 [91.8, 93.8] | 1.7 [0.4, 3.7] |
| v2+clamp | 1 | raw | 99.2 [99.1, 99.4] | 6.7 [4.3, 9.7] |
| v2+clamp | 2 | par | 95.9 [95.4, 96.4] | 2.1 [1.2, 3.2] |
| v2+clamp | 2 | pi | 90.8 [89.6, 91.8] | 2.6 [1.7, 3.7] |
| v2+clamp | 2 | gate | 87.9 [86.6, 89.1] | 0.6 [0.2, 1.1] |
| v2+clamp | 2 | raw | 99.3 [99.2, 99.4] | 8.5 [6.6, 10.7] |
| v2+clamp | 3 | par | 96.4 [95.8, 96.9] | 22.1 [19.6, 24.8] |
| v2+clamp | 3 | pi | 89.6 [88.3, 90.7] | 13.3 [11.4, 15.4] |
| v2+clamp | 3 | gate | 77.4 [75.6, 79.0] | 0.4 [0.2, 0.7] |
| v2+clamp | 3 | raw | 99.5 [99.3, 99.6] | 48.6 [45.1, 52.4] |

Against the bars in [theory.md](/home/lunarz/cfcap/docs/board/theory.md):

- **Kill: not met.** The saved undominated-stratum rates exceed the threshold; this agrees with [gate_verdict.txt](/home/lunarz/cfcap/results/theory/gate_verdict.txt).
- **Success 1: fails for v2.** The preregistered wording does not explicitly restrict this first bar to v2, although the recorded verdict does; v2+clamp passes using unrounded values.
- **Success 2: fails.** At the selected severity, gate misses both the unsafe-rate and certified-share requirements.

The cross-fit trace confirms that quantile fitting uses calibration episodes, with the gate threshold selected on calibration half A and its final margin fitted on half B. Two implementation discrepancies remain in [theory_gate.py](/home/lunarz/cfcap/scripts/theory_gate.py): the sufficiency check does not detect a completely missing expected cell, and it checks only the saved par and gate margins rather than every margin. The available saved cells and checked margins meet those checks, so these discrepancies do not change this run’s recorded verdict.

Validation: the exploratory audit tests passed using CPU via `scripts/heavy.sh`.

Files written: [test_gate_audit.py](/home/lunarz/cfcap/tests/codex/test_gate_audit.py).