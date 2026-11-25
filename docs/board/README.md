# cfcap specialist board (agentic loop)

The coordinator is the `/plan (Branch)` session. It runs specialists in rounds and proposes the results to the main
session (`certified-counterfactual-self-models`).

**Each specialist owns one file, `docs/board/<role>.md`**, and keeps it current:
- Status: one line.
- Findings: numbers, each citing its CSV/script; label each exploratory or confirmatory.
- Proposal for full-scale data collection / training: sizes, shards, box-hours, $ (3090 ≈ $0.18/h), what decision it
  serves.
- Needs from other roles: name the role.
- Next step if given another round.

**Shared rules**
- Heavy local work goes through `scripts/heavy.sh` (`--gpu` for GPU).
  - At most 2 heavy jobs at once across all specialists, and 1 GPU job at a time.
  - Keep each local pilot ≤ 15 min wall.
- Never edit capxfer (`~/worktrees/capxfer-cluster/capxfer`); import it read-only.
- Don't edit another role's files. Shared modules (`cfcap/t0`, `t1`, `conformal`) may only be extended additively:
  new functions or new files, no behaviour changes, and the existing tests must still pass.
- Commit only your own paths, with plain messages and no AI attribution.
- Results go to `results/<role>/` as CSV + the script that made it. Bulk output goes to `$CFCAP_OUT/<role>/`.
- No rented compute and no cluster submission from specialists: put the job spec in your proposal.

**Stage A state (2026-11-25):**
- A2: 0.02 % of decisions ambiguous after v2 for the idealized T0 oracle (0.37 % when nuisances use the prior).
  Oracle certifies 99.98 % vs capxfer 58–75 %. So probing is secondary, and the main target is certificate slack plus
  nuisance and misspecification.
- A3: not testable (24 ambiguous decisions).
- A1 (T1): with heterogeneity on, a weak wheel with g 0.75–0.9 is detected only 73–76 % of the time by the self-model.
- A6: exact enumeration costs 3.9 s/episode for 2x3 with nuisances known.
- A0: T1 runs 80k 3-s legs/s for 6 modules on the 4070.

## Coordinator decisions
- **2026-11-25 ~19:25Z, decision label.** The nuisance role found that, with ±8 % motor spread, weak g ≥ 0.85 is the
  same plant as a healthy wheel with slower motor constants. So ρ_rel (dividing by the episode's own fault-free
  rollout) depends on unobservable latents and is not a valid decision target.
  - **Primary label is now ρ_dec = ρ_abs / ρ_ref(shape, leg).** ρ_ref is the mean absolute ρ of the *nominal*
    healthy module or assembly (nominal θ, no spread) for that shape and leg, a fixed table computed once.
  - τ applies to ρ_dec.
  - ρ_rel stays as a secondary field, used only for comparison with capxfer.
  - Error from faults inside the motor spread is irreducible. Report it as such; it is not a method failure.

## Deferred (run once the data is in)
- **Paper refinement loop.** The owner asked for this on 2026-11-25. The playbook is
  `docs/PAPER_REFINEMENT_LOOP.md`, distilled from the owner's prompts in the `rewrite-conference-paper` session.
  - Start condition: the Stage B–D result CSVs in `docs/board/paper.md` exist, and `scripts/analysis/make_all.py`
    has run on real data.
  - Then run the multi-round, multi-perspective review and refinement loop exactly as the playbook says.

## Loop structure (owner, 2026-11-25 ~20:00Z: science over compute management)
- **ops** owns all cluster work: submissions, failures, output collection and the STATUS.md table. Roles put campaign
  plans in `scripts/campaigns/` and do not chase jobs.
- **research lead** keeps `docs/HYPOTHESES.md` current (which hypotheses hold, failed or are open) and keeps the
  ranked idea and experiment queue in `docs/board/research.md`.
- **independent critics** review the draft without seeing our conclusions (`docs/reviews/`).
- The coordinator turns the research lead's ranking and the critics' top fixes into role assignments. Every role
  reads HYPOTHESES.md before starting new work.

## Codex workers (owner, 2026-11-25: use the codex-lb capacity)
- Dispatch with `scripts/codex_task.sh NAME SANDBOX PROMPT_FILE`. The rules in `scripts/codex_prompts/_RULES.md` are
  prepended to every prompt.
- Reports go to `docs/codex/NAME.md`.
- The coordinator reviews and commits the output. Codex does not commit and does not change source except as patch
  files.
- Suited to independent test audits, reference verification, second-implementation re-analysis of results, failure
  annotation, and code review. Not suited to cluster operations.

## Owner direction (2026-11-25 ~20:35Z): research first, paper later
- Until all data collection is complete, the loop does **research**: critique each result as it lands, keep
  HYPOTHESES.md current, ideate improvements, and design and run the next experiments.
- **Paper writing** (prose, figures, the refinement loop in docs/PAPER_REFINEMENT_LOOP.md, including the round-0
  figure owners) is a later loop that starts only after all data are in. Analysis scripts that turn results into
  numbers stay in scope, because they are research.
- Round-0 artefacts so far (OUTLINE.md, DIAGRAM_STRATEGY.md, BIB_FIXES.md) are kept for that later loop.
