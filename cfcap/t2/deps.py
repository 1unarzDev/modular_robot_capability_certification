"""Import paths for T2: crane's modular kernel (read-only, ~/crane_ml) and numba / JAX.

The cfcap interpreter (capxfer's venv) has torch but neither numba nor JAX. A crane venv with the same Python 3.12 and
numpy 2.5.5 has both; its site-packages is APPENDED to sys.path (never prepended), so numpy, torch and the NVIDIA
wheels still resolve from the cfcap interpreter first and only the missing packages come from the crane venv.

    CFCAP_CRANE_ML    crane checkout            (default ~/crane_ml)
    CFCAP_T2_SITE     site-packages with numba/jax (default ~/.cache/crane-w4-sub-venv/lib/python3.12/site-packages)
"""
from __future__ import annotations

import os
import pathlib
import sys

CRANE_ML = pathlib.Path(os.environ.get("CFCAP_CRANE_ML", pathlib.Path.home() / "crane_ml"))
T2_SITE = pathlib.Path(os.environ.get("CFCAP_T2_SITE", pathlib.Path.home()
                                      / ".cache/crane-w4-sub-venv/lib/python3.12/site-packages"))


def setup():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")     # JAX shares the GPU with torch
    for p in (str(CRANE_ML), str(T2_SITE)):
        if p not in sys.path and pathlib.Path(p).exists():
            sys.path.append(p)


setup()
