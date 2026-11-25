# source me: sets the interpreter and import paths for cfcap
export CAPXFER_ROOT=$HOME/worktrees/capxfer-cluster/capxfer
export PYTHONPATH=$HOME/cfcap:$CAPXFER_ROOT${PYTHONPATH:+:$PYTHONPATH}
export PY=$HOME/crane_explain/capxfer/.venv/bin/python
export CFCAP_OUT=$HOME/.cache/modular-runs/2026-11/cfcap
export CAPXFER_THETA_DIR=$HOME/.cache/modular-runs/2026-11/capxfer/sysid
mkdir -p $CFCAP_OUT
