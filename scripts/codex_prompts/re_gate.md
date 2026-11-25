TASK re_gate: audit theory's E1 trust-gate analysis (exploratory). Inputs: docs/board/theory.md (pre-registered E1
bars), results/theory/gate_cert.csv, scripts/theory_gate.py, cfcap/theory_task.py, cfcap/t0/envelope.py.
- Recompute the certified share and unsafe rate per certifier (physics-trusting, model-free, trust-gated, raw) ×
  knee severity × probe set, with episode-bootstrap CIs.
- Check that margins are fitted on the calibration fold only (cross-fit), and that the verdict follows the
  pre-written bars exactly.
Write the audit tests under tests/codex/test_gate_*.py (do not change source). Report pass/fail per bar and any
discrepancy.
