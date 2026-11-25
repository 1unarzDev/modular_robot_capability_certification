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
