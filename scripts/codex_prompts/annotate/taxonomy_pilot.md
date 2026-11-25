You annotate failure cases of a capability certificate for reconfigurable modular robots. Do not run any commands
and do not read files: everything you need is below. Answer only with the JSON object the schema asks for.

{{BACKGROUND}}

Task. For each of the {{N}} cases below (one JSON object per line), find the mechanism: the physical or statistical
reason this certificate failed, refused, stayed ambiguous, or why the label flipped between plants. Give:
- `mechanism`: a short snake_case label (2-4 words) that names a reusable mechanism class, not this case's details
  (good: weak_wheel_near_threshold, nuisance_masks_fault; bad: wheel_m1_fl_g04);
- `explanation`: one sentence of physical explanation, citing the case's numbers;
- `confidence`: 0-1;
- `claims`: 2-5 checkable claims (grammar below).

{{CLAIMS}}

Cases (ids {{IDS}}):
{{CASES}}
