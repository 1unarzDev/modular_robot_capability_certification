"""Sufficiency guard of scripts/theory_gate.py (codex audit re_gate.md, item 1)."""
import importlib.util
import pathlib

import numpy as np
import pandas as pd

_P = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "theory_gate.py"
_spec = importlib.util.spec_from_file_location("theory_gate", _P)
TG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TG)


def _R(n_ep=300, drop=None):
    rows = [dict(stratum="all", certifier="par", severity=s, probe=p, n_episodes=n_ep)
            for s in TG.SEVERITIES for p in TG.PROBES if (s, p) != drop]
    return pd.DataFrame(rows)


def _Q(bad=None):
    rows = []
    for p in TG.PROBES:
        for f in (0, 1):
            r = dict(probe=p, fold=f, q_gate=np.inf, **{m: 0.1 for m in TG.MARGINS})
            if bad and bad[:2] == (p, f):
                r[bad[2]] = np.inf
            rows.append(r)
    return pd.DataFrame(rows)


def test_sufficient():
    assert TG.sufficiency(_R(), _Q()) == []          # q_gate = inf is a threshold, allowed


def test_missing_cell():
    pr = TG.sufficiency(_R(drop=(3, "v2clamp")), _Q())
    assert any("missing" in x and "severity=3" in x for x in pr)


def test_small_cell():
    assert any("< 100 episodes" in x for x in TG.sufficiency(_R(n_ep=99), _Q()))


def test_every_margin_checked():
    for m in TG.MARGINS:
        pr = TG.sufficiency(_R(), _Q(bad=("v2", 1, m)))
        assert any(m in x for x in pr), m


def test_missing_fold_and_table():
    Q = _Q()
    assert any("fold=0 missing" in x for x in TG.sufficiency(_R(), Q[~((Q.probe == "v2") & (Q.fold == 0))]))
    assert TG.sufficiency(_R(), None) == ["no margin table"]
