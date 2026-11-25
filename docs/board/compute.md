# compute (cluster jobs, rented 3090s)

**Status:** the runner, job emitter and validator are built and tested. A local box simulation passes, including a
kill and resume. No job has been submitted or rented. The canary jobs are ready for the main session to submit.

## Findings (all infrastructure; nothing here is a study result)
- **`cfcap/runner.py`**: `python -m cfcap.runner <task> --shard K --n-shards N [task args]`.
  - **Outputs.** Columnar `units/<unit>.npz` plus a `.json` sidecar for each unit, and a `manifest.json`. They go to
    `/workspace/out/$CRANE_JOB_ID/` on a box. On a free worker they go to `$CRANE_RUN_DIR/artifacts/out/`, which is
    uploaded. Locally they go to `$CFCAP_OUT/runs/<task>/shardNNNN`.
  - **Progress.** `progress.json` is written at most 30 s apart, along with `RC-PROGRESS` lines.
  - **Metrics.** `$CRANE_RUN_DIR/metrics.json` holds at most 1000 finite numbers.
  - **Atomic writes and resume.** Every file is written to a temp name and renamed. A finished unit is skipped when its
    sha256 still matches. A unit may also checkpoint its own state (`ctx.save_state` / `ctx.load_state`).
  - **Changed arguments.** A run with different arguments into the same output directory is refused (exit 2).
  - **Signals.** SIGTERM, SIGUSR1 and SIGINT checkpoint the job and exit 75. A unit that ignores the signal is cut off
    after a grace period (20 s / 60 s), and never during a rename.
  - **Exit codes:** 0 done, 75 resumable, 2 usage, 3 missing input or module, 1 task error.
- **Task registry** (`TASKS`, lazy `module:attr`). The contract is in the `Task` docstring.
  - `selftest`: built in.
  - `gen_shard`: works now, through `cfcap/compute_tasks.py`, an adapter over the generator's `write_shard`.
  - `train` → `cfcap.models.train:TASK`, `smc` → `cfcap.baselines.smc:TASK`, `t2_eval` → `cfcap.t2.eval:TASK`. These
    roles still have to provide them. Any other task can be run with `--task-spec mod:attr`.
- **`scripts/make_jobs.py`**:
  - **Box contract (compute session, 2026-11-25 evening):** code goes as **pinned repos**, data as `inputs_rsync`.
    - Every job carries `repos: [cfcap@HEAD (~/cfcap), capxfer@<stage sha>]`. `research submit` uploads the bundles,
      and the worker checks them out at `<work>/src/<name>`.
    - `spec.run` uses `S=$PWD/src`, not `{repos[...]}`. The box backend formats only `bash -lc {spec[run]}`, once, so a
      placeholder inside `spec.run` would reach bash literally; I checked this against `command.py` and `boot.sh`.
    - `inputs_rsync` holds only `stage/theta` (→ `/workspace/in/theta`) plus any stage-2 `extra_inputs`.
  - `stage` builds `stage/capxfer`, a small git repo of capxfer `capxfer/`, `params/`, `configs/` and `theta/`, about
    3 MB with no corpus or results. The real capxfer worktree is ~100k files. `stage` also builds `stage/theta`.
  - `emit --first-shard-only` emits shard 0 of each group as a canary. Its job ids and bodies are identical to the full
    emit, so the later full submit returns `exists` for it.
  - `emit PLAN.json` writes rc-job/1 jobs:
    - `est_seconds` = overhead_s + units × unit_s, with no padding;
    - a group without `pilot` evidence is emitted only as `draft_jobs`, which cannot be submitted.
  - `validate` runs the coordinator's own TS checks, bundled read-only with its esbuild: purpose, estimate, submitter,
    campaign, `parseNeeds`, and the rcjob fields. It caught a short purpose, a local path in a purpose, and an unknown
    needs key.
  - The lanes:

    | Lane | Kind | Tags and needs | Inputs | Limit |
    |---|---|---|---|---|
    | `rented` | `gpu_command` | `[cloud-ok, torch\|jax, rented]`, `needs.gpu`, `vram_gb` | repos + θ by `inputs_rsync` | |
    | `cpu` | `py_command` | `[cloud-ok, cpu, rented]` | repos + θ by `inputs_rsync` | |
    | `free` | `py_command` | no `cloud-ok` tag, so it never rents | repos only (θ from the capxfer repo) | 60 min (innovation's cap) |

- **Dry run** (`scripts/box_dryrun.sh`; exploratory). It ran job `cfcap-canary.st-torch.s000` exactly as a box would:
  - the inputs were rsynced into `in/` the way crane-vast does it;
  - `spec.run` ran through `bash -lc` with a clean env, the system python3 (numpy only) and the backend's env vars;
  - the box root was moved to /tmp/cfcap-box-torch.

  What happened:
  - SIGTERM to the process group exited 75 with 2/6 units kept.
  - The resume, then kill -9, kept 3/6.
  - The next resume exited 0 with 6/6 units, 600k rows, hashes ok, 14 finite metrics, `units_resumed`=3 and one unit
    resumed mid-unit.
  - The 30 s units took 30 s.
  - The free-lane job (`st-free.s000`) ran from cloned bundles at the pinned commits, through `bash -lc` under
    nice: exit 0, 6/6 units, 300 s for 6 × 50 s units.
  - The `gen_shard` adapter is tested against a stub `write_shard` (shard mapping, resume). It has not run on the
    GPU: the GPU lock was held by other roles' pilots, so I cancelled my run.
  - `jobs.json` passed `validate`.
  - **Re-run after the contract change:** repos were checked out at their pinned refs into `work/src/`, and θ was
    rsynced to `in/theta` (box root /tmp/cfcap-box2). It gave the same outcome: SIGTERM → 75 with 2/6 units kept;
    kill -9 → 3/6 kept; resume → exit 0, 6/6 units, hashes ok, the commit recorded as cf70611 and clean.
  - Tests: `tests/test_runner.py` 23 pass. They cover the repo and θ layout, the input wait (late arrival, and the
    timeout → exit 3) and the first-shard emit.
- **Input wait.** `--require-inputs a,b`, which make_jobs adds on the rented lanes, waits up to 6 min, polling every
  15 s, for each `$CRANE_INPUTS_DIR/<x>`. Then it exits 3. `ctx.input()` waits the same way. A signal during the wait
  exits 75.
- **Path bug for other roles.** Inputs land at **`/workspace/in/<basename>`** (boot.sh `CRANE_INPUTS_DIR`, crane-vast
  `rsync SRC box:/workspace/in/`), not at `/workspace/inputs/`. The docstrings of `scripts/gen_shard.py` and
  `scripts/stageA0_throughput.py` give `/workspace/inputs/`.
  - Jobs emitted by make_jobs set `CAPXFER_ROOT=$S/capxfer` and `CAPXFER_THETA_DIR=/workspace/in/theta`.

## Capacity (read 2026-11-25 18:30–18:50 UTC)

**Rented.** `vast.json` now carries the owner override, valid until 2026-11-26T06:00Z:
- `max_instances` 8;
- `daily_budget_usd` 35;
- `interruptible_only` true, which also blocks the on-demand fallback after outbids.

The base is $15 a day and 3 boxes.

The `torch` profile:
- `python:3.12-slim` + pip torch;
- needs driver CUDA ≥ 13.0;
- costs ≤ $0.20/h at a 1.4× bid;
- falls back to `torch_full`, the pytorch 2.14 image with an 18 min pull.

The `jax` profile: slim + `jax[cuda12]`, with **no torch**, so capxfer and T1 cannot run on jax boxes.

**Money.**
- Credit is $19.46 before the owner's $15 top-up.
- The ledger shows $0.39 since 11-25 UTC, of which $0.21 was on-demand, earlier today.
- The guard's day is UTC.
- The 6-hour look-ahead check is `spent + burn × min(6, hours left in the UTC day) ≤ cap`.

**Free.**
- The coordinator has 9 approved workers, 3 of them live.
- Innovation runs `py_command` with 2 slots × 1 CPU and refuses jobs over 60 min. Its python3 has numpy only; the free
  lane uses the capxfer venv when it exists.
- No free `gpu_command` lane: the 4070 is for pilots only.

**Autoscale.**
- Per profile, it rents about backlog / 2 h boxes after a 10 min wait.
- The profiles share the cap of 8, and the biggest backlog is served first. So ≥ 16 box-h must be queued to fill 8 boxes,
  and a large torch backlog starves jax.
- Inputs: the crane-vast tick now re-pushes changed inputs to every live box of the profile, so no drain is needed.

## Proposal (draft plan `scripts/campaigns/cfcap_b1.draft.json`)

Every `unit_s` is an **estimate** from A0 (4070 Ti SUPER, 3090 assumed 1.25× slower) until the owning role pastes its
pilot. Generation is cheap: at 80k legs/s, 200k episodes with labels take minutes of GPU. So **training generates its
data on the fly from seeds**, and only reference splits are persisted. That avoids pushing GB datasets.

| Stage | Group (role) | Lane | Jobs × min | Box-h | $ |
|---|---|---|---|---|---|
| 1 | t2-episodes (stress) | jax | 4 × 21.7 | 1.4 | 0.26 |
| 1 | train: arch × diversity × 3 seeds + ensemble (models) | torch | 29 × 48 | 23.2 | 4.18 |
| 1 | smc-t1: 500 episodes × 1e6 sims (baselines) | torch | 8 × 18.8 | 2.5 | 0.45 |
| 1 | gen-ref: cal/test/H* splits (generator) | torch | 4 × 8 | 0.5 | 0.10 |
| 1 | t0-sweep (generator/theory) | free | 12 × 40.5 | 8.1 | 0 |
| 2 | t2-eval: weights + T2 episodes (stress) | torch | 8 × 15 | 2.0 | 0.36 |
| 2 | smc-t2 (baselines) | torch | 8 × 18.8 | 2.5 | 0.45 |

The rented total is 32 box-h ≈ **$5.8**, plus about 10 min of boot per box (~$0.5). That is far inside $35. The binding
limit today is wall time: 5.5 h are left in the UTC day, and the override lapses at 06:00Z. The plan serves the
decisions in §6 B/C: the diversity → generalisation curve, and the 2×2 grid {SMC-T1, amortized} × {T1, T2}.

**Schedule (keeps 8 boxes busy):**
1. **Canary.** Submit `cfcap-canary.json`: 2 torch shards and 1 free shard, ~$0.05. After that, submit each
   campaign's `emit --first-shard-only` file before its full file (two broken outcomes in a row trip a batch). Check that its outputs arrive under
   `~/.cache/vast-out/<box>/crane/cfcap-canary.*`.
2. **Stage 1, jax first.** Submit the jax group alone, so autoscale gives jax 1 box.
3. **Stage 1, torch 10 min later.** Submit the torch groups: about 26 box-h, which asks for 13 boxes and is capped at
   the 7 left. The lower priority number leases first: train (+10), then smc-t1 (+20), then gen-ref (+30), with the
   long jobs first and the short shards filling the tail. When the jax box leaves (~1.5 h), the 8th slot goes to torch.
   Expected wall time is ~3.8 h. The free t0-sweep runs alongside on innovation.
4. **Stage-2 handoff.** Stage 2 reads stage-1 outputs pulled back from the boxes (weights, T2 episodes), merged into
   `$CFCAP_OUT/compute/stage2/{b1-weights,b1-t2eps}`.
   - Because the tick re-pushes inputs to live boxes, stage 2 can be submitted while stage-1 boxes are still up.
   - The runner waits up to 6 min for the pushed inputs.
   - The remaining gap is the pull-back: outputs come back when a box leaves, so the last stage-1 jobs must be pulled
     and merged first.
5. **Stage 2.** ~4.5 box-h, about 3 boxes. Fill the other 5 with Stage D fillers (adversarial CEM, EVOI oracles) once
   their pilots exist.

**Draft `compute/REQUESTS.md` row** (not appended; the `{…}` fields come from the roles' pilots):

| 2026-11-25 | 09 cfcap | cfcap Stage B/C campaign `cfcap-b1`: stage 1 training grid (arch × diversity × 3 seeds), GPU-SMC on the T1 test split, T2 test episodes, reference splits; stage 2 T2 eval and SMC-T2 | torch (+1 jax), interruptible only, ≤ 8 boxes until 2026-11-26 06:00Z (owner) | {from `make_jobs.py emit` summary; draft 32.2 + ~1 boot} | {models: peak VRAM of one training run; baselines: SMC peak VRAM; stress: T2 JAX peak; generator: gen shard RAM/VRAM} | runner canary `scripts/box_dryrun.sh` (exit 75 on SIGTERM, resume, 30 s units = 30 s); {models: pilot cmd → path, s/epoch-chunk}; {baselines: cmd → path, s/episode}; {stress: cmd → path, s/episode}; {generator: `scripts/gen_shard.py` → path, s/shard} | {models: …; baselines: …; stress: …} (each must exist before submission) | repos: cfcap@{sha} (~/cfcap) and capxfer@{stage sha} (`$CFCAP_OUT/compute/stage/capxfer`: capxfer source + params + configs); inputs_rsync: `$CFCAP_OUT/compute/stage/theta` (θ JSONs); stage 2 adds `stage2/b1-weights`, `stage2/b1-t2eps`; no corpus | requested |

**Cost monitor:** `watch -n 300 ~/cfcap/scripts/cost_watch.sh 35 2026-11-25 cfcap-b1 crane`. It shows the boxes and
burn, the ledger spend since 11-25 against $35 (flagging any on-demand spend), and the campaign queue.

## Needs from other roles
- **generator:**
  - commit `cfcap/gen/*`: the stage exports HEAD, and untracked files never reach a box;
  - fix `/workspace/inputs` → `/workspace/in` in docstrings;
  - give the measured s per shard on the 4070, and the episodes per shard.
- **models:**
  - `cfcap/models/train.py:TASK`: one unit = an epoch chunk of ≤ 5 min, with `ctx.save_state` and `ctx.should_stop()`;
  - on-the-fly data, plus evaluation on the regenerated cal/test sets inside the job;
  - pilot s/chunk, peak VRAM, and the weights size (it decides bundle vs gate).
- **baselines:** `cfcap/baselines/smc.py:TASK` (unit = k episodes), with pilot s/episode and VRAM.
- **stress:**
  - `cfcap/t2/eval.py:TASK` in two modes, `simulate` (jax, no torch) and `evaluate` (torch);
  - say whether T2 needs torch;
  - pilot numbers.
- **All roles:** derive episode specs from numpy seeds on the CPU, so a regenerated test split is the same on every
  host.

## Next step
Once the main session submits the canary, check the pulled outputs and update the box facts. Then put the roles' pilot
lines into the draft plan, run `emit --validate`, and fill in the REQUESTS row.
