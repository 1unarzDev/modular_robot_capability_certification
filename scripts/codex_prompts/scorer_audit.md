TASK scorer_audit. An earlier paper in this program died on a scorer artifact; capxfer once had a bug where an empty
consistent set returned "go". Independently audit every scorer and metric in cfcap for correctness:
cfcap/conformal/core.py, cfcap/baselines/ (scoring + SimultaneousLower use), cfcap/decide/ (unsafe/useful/regret
metrics), scripts/analysis/ (T1–T4, the matched-1 % unsafe margin rule, decision-relevant stratum), and
cfcap/nuisance_post.py's ambiguity rule. For each: read the code, state its intended semantics, and write adversarial
unit tests (edge cases: empty sets, NaN/inf, ties at τ, all-capable/all-incapable, ragged candidate sets, n too small
for the quantile, strata with zero rows, label leakage between cal and test). Write tests ONLY under
tests/codex/test_audit_*.py and run them with `$PY -m pytest -q tests/codex` (CPU). Do not fix source code. Report
each failing test as a suspected bug: file:line, the input, expected vs actual, and severity.
