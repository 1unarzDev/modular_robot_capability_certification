"""Exploratory T0 screen audit and regression for screen_bug.patch.

CPU only; run with PYTHONDONTWRITEBYTECODE=1 and pytest -p no:cacheprovider.
The audit executes the production episode prefix, stopping before propagation.
Seeds are unsafe posterior/v2/sim0.1 rows in
results/decision/pilot_t0_episodes.csv. No corpus or GPU access is needed.

To verify the unapplied diff without writing source files, run this file with
--proposed-patch; additional arguments are passed to pytest. For example:
  scripts/heavy.sh "$PY" -B tests/codex/test_screen_bug.py --proposed-patch tests/test_decide.py
Set PYTHONDONTWRITEBYTECODE=1 and CUDA_VISIBLE_DEVICES='' in the environment.
"""
import inspect
import textwrap

import numpy as np
import pytest

from cfcap.decide import policies as Pol
from cfcap.decide import posterior as DP
from cfcap.decide import problem as Pb
from cfcap.decide import t0data as T


# seed, first rejecting leg (zero based), feature (gz, wheel speeds, duties), z
# Exploratory replay of the production prefix; values rounded for portability.
REJECTIONS = [
    (909057, 2, 4, 27.420763),
    (909073, 2, 6, 18.721159),
    (909269, 4, 5, 9.328393),
    (909327, 4, 4, 5.562475),
    (909371, 4, 7, 5.142283),
    (909452, 0, 4, 22.955633),
]


@pytest.mark.parametrize("seed,leg,feature,z_expected", REJECTIONS)
def test_screen_rejection_provenance(seed, leg, feature, z_expected):
    # Execute the actual screen with unchanged RNG order and equations. Capture
    # the truth's residual only while it is still alive; no truth-based fix.
    src = textwrap.dedent(inspect.getsource(T.episode))
    src = src.split("\n    if len(alive) == 0:")[0]
    src = src.replace(
        "    for li in SA.V2_ORDER:",
        "    ti = SA.find_row(HM, HG, tm, tg)\n"
        "    trace = []\n    for li in SA.V2_ORDER:",
    )
    src = src.replace(
        "        alive = alive[z <= K_SCREEN]",
        '        hit = np.flatnonzero(alive == ti)\n'
        '        if len(hit):\n'
        '            zs = np.abs(O.signals(r)[hit[0], 0] - ytrue["v2"][li]) / sig_scr[li]\n'
        '            j = int(zs.argmax())\n'
        '            trace.append((li, j, float(zs[j])))\n'
        '        alive = alive[z <= K_SCREEN]',
    )
    src += "\n    return truth_in, trace, len(alive)\n"
    ns = dict(T.__dict__)
    exec(compile(src, __file__, "exec"), ns)
    truth_in, trace, n_alive = ns["episode"](seed)
    assert not truth_in
    assert trace[-1][:2] == (leg, feature)
    assert trace[-1][2] == pytest.approx(z_expected, abs=2e-5)
    assert trace[-1][2] > T.K_SCREEN
    assert n_alive == 0  # production currently substitutes a healthy particle


@pytest.fixture(scope="module")
def screen_miss():
    return T.episode(REJECTIONS[0][0])


def test_screen_miss_must_abstain_before_and_after_probe(screen_miss):
    e = screen_miss
    assert e["meta"]["truth_in"] == 0
    assert e["meta"]["n_screen"] == 1
    post = e["post"].copy()
    # Even q=0 must refuse when omitted support has no coverage guarantee.
    for name in (None, "clamp", "moddiff", "tug"):
        if name is not None:
            post.observe(name)
        lower = Pol.Certifier(0.0).lower(post)
        assert np.isneginf(lower).all(), "screen failure was renormalised into confidence"
        assert np.isneginf(post.bayes_lower()).all()
        req = np.full(10, np.nan)
        req[0] = 0.6
        chosen, _ = Pb.decide(e["inst"], [Pb.Task(req)], lower)
        assert (chosen < 0).all()


def test_particle_cap_also_marks_unknown_support(monkeypatch):
    # Keep every discrete hypothesis through screening, then truncate particles.
    monkeypatch.setattr(T, "K_SCREEN", np.inf)
    e = T.episode(909000, sizes=(2,), M=1, p_max=1, n_merge=0)
    assert e["meta"]["truth_in"] == 1
    assert e["meta"]["n_screen"] == e["meta"]["n_hyp"]
    assert np.isneginf(Pol.Certifier(0.0).lower(e["post"])).all()


def test_complete_support_keeps_existing_summary():
    post = DP.degenerate(np.ones((1, 10)))
    assert np.all(Pol.Certifier(0.0).lower(post) == 1.0)
    assert np.all(post.bayes_lower() == 1.0)


if __name__ == "__main__":
    # Verification harness: consume the delivered unified diff in memory only.
    import linecache
    from pathlib import Path
    import re
    import sys

    if "--proposed-patch" not in sys.argv:
        raise SystemExit("Use pytest for the original source, or pass --proposed-patch.")
    root = Path(__file__).resolve().parents[2]
    lines = (root / "docs/codex/screen_bug.patch").read_text().splitlines(True)
    i = 0
    while i < len(lines):
        assert lines[i].startswith("--- a/")
        path = lines[i][6:].strip()
        assert path in ("cfcap/decide/posterior.py", "cfcap/decide/t0data.py")
        i += 2
        old = (root / path).read_text().splitlines(True)
        new, pos = [], 0
        while i < len(lines) and not lines[i].startswith("--- a/"):
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", lines[i])
            assert match, lines[i]
            start = int(match.group(1)) - 1
            new.extend(old[pos:start])
            pos, i = start, i + 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- a/")):
                tag, body = lines[i][0], lines[i][1:]
                if tag in " -":
                    assert old[pos] == body, (path, pos)
                    pos += 1
                if tag in " +":
                    new.append(body)
                i += 1
        new.extend(old[pos:])
        src, full = "".join(new), str(root / path)
        module = DP if path.endswith("/posterior.py") else T
        linecache.cache[full] = (len(src), None, new, full)
        exec(compile(src, full, "exec"), module.__dict__)
    args = [a for a in sys.argv[1:] if a != "--proposed-patch"]
    raise SystemExit(pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", __file__, *args]))
