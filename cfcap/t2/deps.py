"""Import paths for T2: crane's modular kernel (read-only, ~/crane_ml) and numba / JAX.

The cfcap interpreter (capxfer's venv) has torch but neither numba nor JAX. A crane venv with the same Python 3.12 and
numpy 2.5.5 has both; its site-packages is APPENDED to sys.path (never prepended), so numpy, torch and the NVIDIA
wheels still resolve from the cfcap interpreter first and only the missing packages come from the crane venv.

    CFCAP_CRANE_ML    crane checkout            (default $CRANE_INPUTS_DIR/crane_ml on a box, else ~/crane_ml;
                      scripts/stage_crane_t2.sh exports the 9 files T2 needs)
    CFCAP_T2_SITE     site-packages with numba/jax (default ~/.cache/crane-w4-sub-venv/lib/python3.12/site-packages)
"""
from __future__ import annotations

import os
import pathlib
import sys

def _crane_default():
    staged = pathlib.Path(os.environ.get("CRANE_INPUTS_DIR") or "/workspace/in") / "crane_ml"   # a rented box
    return staged if staged.is_dir() else pathlib.Path.home() / "crane_ml"


CRANE_ML = pathlib.Path(os.environ.get("CFCAP_CRANE_ML") or _crane_default())
T2_SITE = pathlib.Path(os.environ.get("CFCAP_T2_SITE", pathlib.Path.home()
                                      / ".cache/crane-w4-sub-venv/lib/python3.12/site-packages"))


def setup():
    """Append (never prepend) the crane checkout and the numba/jax site-packages. The staged box input
    ($CRANE_INPUTS_DIR/crane_ml) is appended even when it does not exist yet: a rented box may receive it during the
    runner's input wait, after this module was imported, and a missing sys.path entry is harmless until then."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")     # JAX shares the GPU with torch
    staged = pathlib.Path(os.environ.get("CRANE_INPUTS_DIR") or "/workspace/in") / "crane_ml"
    cands = [str(CRANE_ML), str(staged), str(T2_SITE)]
    for p in cands:
        if p not in sys.path and (pathlib.Path(p).exists() or p == str(staged)):
            sys.path.append(p)


setup()
