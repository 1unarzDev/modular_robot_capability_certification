# Failure-mode taxonomy (workstream A) — v1, provisional

**Status (2026-11-25 14:50 CST).** v1 is frozen for the first full annotation run. It was drafted from the mechanisms
already documented on the board (decision.md readings 2–3, nuisance.md weak(g) ↔ spread ridge, stress.md C-stress-1,
theory's unexcited-envelope result, A3) and from the held-out quantities the validator can compute.
- The planned 50-case pilot (`results/annotations/a_pilot_cases.jsonl`, free-form labels) could not run first:
  codex-lb hit its usage limit at 14:38 CST (reset advertised for 18:52).
- The pilot runs first when quota returns (`docs/board/annotations.md`, "Pending"). **v1 is replaced if** more than
  15 % of the pilot's free labels fit no category below, or more than 15 % of the full run chooses `other`.

**Rule.** A category enters the paper table ("why certificates abstain or fail") only if it is *validated*:
- it has a signature (a deterministic predicate on simulator ground truth in `scripts/validate_annotations.py`,
  `_sig_<name>`);
- the signature holds on ≥ 80 % of the cases codex put in it;
- ≥ 90 % of its checkable claims are true;
- it has n ≥ 10 cases.

Categories without a signature are never validated. Signatures use held-out truth that codex never sees: whether the
true hypothesis survived the screen, the posterior mass on it, the single-probe resolution of each ambiguous decision,
and the A5 per-condition ablation.

The table is parsed by `load_taxonomy` (column 1 = category, column 2 = signature).

| category | signature | mechanism | typical source |
|---|---|---|---|
| `screen_miss` | `screen_miss` | The true fault hypothesis was removed by the hypothesis screen (the screen's nuisance model is too tight), so the posterior puts no mass on the truth and certifies a body that is truly short. | T0 unsafe_cert |
| `mode_confusion` | `mode_confusion` | The truth survived the screen, but another fault hypothesis (wrong wheel, wrong mode, or a different g) explains v2 about as well, and the posterior favours it. | T0 unsafe / ambiguous |
| `spread_masked_weak` | `spread_masked` | A weak wheel with g inside the motor-spread band (g ≥ ≈0.85) is the same plant as a healthy wheel with slower motors. No probe can identify it, and the error is irreducible. | T0, A2 |
| `passive_wheel_hidden` | `hidden_passive` | A coast or brake wheel whose effect on the required leg is not visible in v2 (below the back-drive threshold, or on a roller channel v2 never loads). | T0, A2 ambiguous |
| `unexcited_envelope` | `unexcited` | v2 never drives the wheels to the operating point the decision needs (duty clamp, friction limit, the child's own load split). One targeted probe (clamp / moddiff / tug) settles it. | A2, T0 ambiguous |
| `split_needed` | `split_only` | Only running the child on its own (the diagnostic split) settles the decision; no parent-side probe does. | A2, T0 ambiguous |
| `near_threshold` | `near_threshold` | The label lies within 0.05 of the requirement or τ, so a small posterior, plant or numerical error decides the side. | all |
| `conformal_margin_slack` | `margin_slack` | The posterior is right: its mean clears r on every required leg. The conformal margin q·(sd + 0.02) refuses a capable body. | T0 refused_capable |
| `probe_overconfidence` | `probe_overconfidence` | The pair is certified after v2 but refused after the extra probes. The probe likelihood is overconfident, so the cross-fitted end-to-end q inflates (1.0 → ≈5) and the bounds collapse. | T0 refused_capable (all) |
| `kernel_numerics` | `kernel_numerics` | A5: the flip already happens with the independent crane kernel alone (`base`: slip-ratio friction, Coulomb roller drag, massless wheels). Its cause is the plant formulation, not an added effect. | A5 |
| `friction_scatter` | `friction_scatter` | A5: removing the per-wheel roller-friction scatter from T2 `all` removes the flip. One wheel's friction error does not average out on small bodies. | A5 singles / pairs |
| `dock_effects` | `dock_effects` | A5: dock compliance / hook play, load transfer or roller-drag anisotropy alone flips the decision, and the bare kernel does not. | A5 multi-module |
| `genuine_shortfall` | `genuine_incapable` | The body truly falls more than 0.05 short on a required leg. The certificate's caution is correct, and the question is only why it was not more confident. | T0 ambiguous / A2 |
| `other` | none | None of the above; the explanation must say why. | — |

Notes.
- Signatures overlap by design (for example, a screen miss can also be near-threshold). Codex picks the *primary*
  mechanism. The validator checks that the chosen category's own signature holds; it does not check that no other
  signature holds.
- `near_threshold` is weak evidence: most A5 flips are near τ by construction. For A5 the informative categories are
  `kernel_numerics`, `friction_scatter` and `dock_effects`, which the ablation checks. For T0 they are `screen_miss`,
  `mode_confusion` and `probe_overconfidence`.
