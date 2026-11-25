TASK e4_pairs (read-only feasibility check, ≤30 min). The field corpus at ~/field_corpus_real_evaluation/modular/ is
real hardware data. Corpus rules: never copy telemetry anywhere; read descriptive files and summary CSVs only; do not
investigate oddities; disregard the derived fields WHEEL contact_force_*, motor_torque_*, floor_mu. Read
reference/protocol.md, configs.csv, codebook.csv and the summary CSVs.

Question: how many matched "assembly → lone module" pairs exist? A pair is the same physical module (M1–M4) and plate
state, observed (a) inside a multi-module assembly (layouts 202 pair, 203 L3, 204 square) in probe-style trials and
(b) alone (layouts 201 / configs 2010x, 2020x) in probe-style trials on which a lone-capability ratio over
surge/sway/yaw could be computed from TRUTH. Count by module × plate state × assembly layout × session, and say which
sessions are calibration vs held-out test (protocol). Include the weak-wheel coupling trials (impairment 201). We need
≥30 pairs to re-score cfcap's counterfactual certificate on real data. Report the counts table and the exact trial
filters, and write the matching recipe (no data) as docs/codex/e4_pairs_recipe.md. Write nothing else.
