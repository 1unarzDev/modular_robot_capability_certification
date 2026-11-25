You annotate failure cases of a capability certificate for reconfigurable modular robots. Do not run any commands
and do not read files: everything you need is below. Answer only with the JSON object the schema asks for.

{{BACKGROUND}}

Fixed mechanism taxonomy (choose exactly one `category`; use `other` only if none fits, and then say why):
{{TAXONOMY}}

Task. For each of the {{N}} cases below (one JSON object per line), give:
- `category`: one name from the taxonomy;
- `explanation`: one sentence of physical explanation, citing the case's numbers;
- `confidence`: 0-1;
- `claims`: 2-5 checkable claims (grammar below). Include at least one claim that tests the category's mechanism
  (for example truth_in_screen, spread_masked, probe_resolves, condition_flip, leg_margin), not only facts you can
  read off the case.

{{CLAIMS}}

Cases (ids {{IDS}}):
{{CASES}}
