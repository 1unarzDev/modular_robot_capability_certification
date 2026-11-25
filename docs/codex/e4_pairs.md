Exploratory: **329 metadata candidate matches; 55 with unloaded parents**, but full-axis TRUTH labels remain unverified [S, C].
Exploratory: **43 weak-wheel coupling trials; no matching faulted lone probe trials established** [W, S].
The requested **≥30 scorable pairs is not established**; recipe-file creation is blocked by this session’s read-only filesystem.

All corpus paths below are relative to `/home/lunarz/field_corpus_real_evaluation/modular/`.

Sources: **[S]** `data/*/modular_probes.csv`, `modular_single.csv`, `modular_pair.csv`; **[C]** `configs.csv`; **[W]** `data/*/modular_coupling.csv`. Session-specific table rows use those files under `data/modular_<session>/`.

Each table cell gives **pair / L3 / square** candidate counts, followed by **(unloaded-parent counts)**. A candidate is one assembly-trial/module observation with an eligible lone observation in the same session; lone observations may be reused. These are availability counts, not independent scored pairs [S, C].

| Session | Plate | Role | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|---|
| 20261024 | on | calibration | 12/0/6 (5/0/1) | 0/6/6 (0/1/1) | 12/6/6 (5/1/1) | 0/6/6 (0/1/1) |
| 20261027 | on | calibration | 0/0/0 | 0/5/5 (0/1/1) | 0/0/0 | 0/0/0 |
| 20261102 | on | calibration | 5/0/4 (1/0/0) | 0/4/4 (0/0/0) | 5/4/4 (1/0/0) | 0/4/4 (0/0/0) |
| 20261105 | on | wet OOD | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| 20261109 | on | held-out test | 9/0/6 (1/0/1) | 0/5/6 (0/2/1) | 9/5/6 (1/2/1) | 0/5/6 (0/2/1) |
| 20261112 | on | wet OOD | 0/0/0 | 0/0/0 | 0/0/0 | 0/3/2 (0/0/0) |
| 20261116 | off | calibration | 6/0/6 (1/0/1) | 0/6/6 (0/1/1) | 6/6/6 (1/1/1) | 0/6/6 (0/1/1) |
| 20261119 | off | calibration | 5/0/5 (0/0/1) | 0/4/5 (0/0/1) | 5/4/5 (0/0/1) | 0/4/5 (0/0/1) |
| 20261120 | off | held-out test | 11/0/5 (1/0/1) | 0/5/5 (0/1/1) | 11/5/5 (1/1/1) | 0/5/5 (0/1/1) |

Totals: calibration **210 (34 unloaded)**; held-out test **114 (21 unloaded)**; wet OOD **5 (0 unloaded)** [S, C].

The split comes from [capxfer’s analysis protocol](/home/lunarz/worktrees/capxfer-cluster/capxfer/docs/ANALYSIS_PLAN_CORPUS.md). The corpus’s `reference/protocol.md` does **not** assign calibration/test sessions.

The label limitation is decisive: **no unloaded module/session has all surge, sway, and yaw represented by `modular_probes.csv` axis metadata** [S, C]. `modular_single.csv` has no axis/amplitude columns, and the descriptive protocol does not establish that these trials contain the full capability routine. Moreover, probe `achieved_gain` is explicitly an **ODOM peak**, not a TRUTH capability ratio (`reference/protocol.md`). Thus actual full-axis availability remains unknown under summary-only access.

Weak-wheel coupling availability is below. All are **test-only**, including those recorded in calibration sessions, under the capxfer analysis protocol.

| Session | Impairment-201 trials | Completed |
|---|---:|---:|
| 20261024 | 6 | 3 |
| 20261027 | 3 | 3 |
| 20261102 | 5 | 5 |
| 20261105 | 2 | 2 |
| 20261109 | 5 | 5 |
| 20261112 | 3 | 1 |
| 20261116 | 6 | 2 |
| 20261119 | 7 | 5 |
| 20261120 | 6 | 6 |
| **Total [W]** | **43** | **32** |

These are preassembled pair trials; the weak wheel belongs to M1. A healthy lone M1 observation cannot supply its faulted post-split capability label (`reference/protocol.md`, [W]). Coupling trials are listed separately because they are docking observations, not full-axis probes.

Matching recipe and exact filters, intended for `docs/codex/e4_pairs_recipe.md`:

- Read current-session summary CSVs only; exclude `prior/` to avoid duplicate observations. Never open or copy telemetry.
- Join each summary’s **actual `config_id`** to `configs.csv`. Resolve physical modules through `module_k_code`; derive per-module plate mass as `plate_kg / modules_n`.
- Define candidate probe-style portions as `probes`, `single`, and `pair`. Require `outcome_code == 1` and `abort_code == 0`. Exclude push, coast, wheelstep, navigation, and payload portions.
- Lone side: require layout `201`, unloaded configuration, and exact ID in `{20101,20105,20106,20107,20201,20205,20206,20207}` [C; `reference/protocol.md`].
- Assembly side: require layout in `{202,203,204}` and membership of the target physical module [C]. Retain parent payload as a covariate; report unloaded parents separately.
- Match within **session × physical module × per-module plate state**. Count each eligible assembly-trial/module once when any eligible lone observation exists; do not multiply by all possible lone trials. Preserve shared-label and shared-parent clustering.
- Before scoring, require documented surge/sway/yaw excitation and usable TRUTH coverage for the lone target. Do not substitute `achieved_gain`, endpoint displacement, or assembly TRUTH for lone capability. Freeze the command windows, signs, normalization, and estimator before scoring.
- Keep impairment state attached to the physical module/wheel. Include coupling `impair_code == 201` as a separate test-only stratum, retaining aborted trials in its availability inventory; require a matching faulted lone target before capability scoring.
- Preserve the existing session split. Never place either side of a test match into identification or calibration.

Files written: **none**. The requested recipe could not be saved because filesystem writes are prohibited by this session’s permissions.