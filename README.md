# cfcap: certified counterfactual capability for reconfigurable modular robots (CRANE study 09)

Companion to capxfer (study 05). The plan is in docs/PLAN.md. Imports capxfer read-only from
`~/worktrees/capxfer-cluster/capxfer`; `source env.sh` first.

Layout: `cfcap/t0` quasi-static, `t1` dynamic (capxfer sim wrapper), `t2` independent stress plant (crane kernel),
`gen` episode generator, `models`, `conformal`, `decide`, `probe`, `baselines`, `field`. Results are CSVs in
`results/` plus the scripts that make them; bulk outputs go to `$CFCAP_OUT`.
