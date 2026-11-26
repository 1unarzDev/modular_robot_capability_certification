"""Post-hoc correction (reported as such) of the run-1 P1/P2 pairing defect.

Run 1 (77fc39b, scorer fb51ccb) computed each method's ratio in `_paired_diff` over the union bootstrap index,
so ours covered 1,050 episodes and capxfer / ghat_physics 210. This script re-runs the same scorer with ONE change:
the paired difference uses only the episodes both methods have (eid intersection), with one weight matrix over that
intersection. The run-1 verdicts stay in the record; this writes results/confirmatory/posthoc_p1p2/ only.
"""
import pathlib, shutil, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import confirmatory as CF  # noqa: E402

ROWS = []


def _paired_intersection(S, bt, claim, pid, metric, a, b, num, den, min_pts, ma, mb, plant, note=""):
    if pid not in ("P1", "P2") or a is None or b is None:
        return _orig(S, bt, claim, pid, metric, a, b, num, den, min_pts, ma, mb, plant, note)
    common = np.sort(np.intersect1d(np.asarray(a.index), np.asarray(b.index)))
    bi = CF.Boot(common)
    ea, ra = bi.ratio(a, num, den)
    eb, rb = bi.ratio(b, num, den)
    d = ra - rb
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    p = CF.p_greater(d)
    ROWS.append(dict(claim=claim, bar=pid, metric=f"{metric}: {ma} - {mb}", ours=ea, other=eb, diff=ea - eb,
                     lo95=lo, hi95=hi, p_one_sided=p, n_common_episodes=len(common),
                     n_ours=len(a.index), n_other=len(b.index), min_pts=min_pts,
                     passes=bool(ea - eb >= min_pts and lo > 0), note=note))
    return _orig(S, bt, claim, pid, metric, a, b, num, den, min_pts, ma, mb, plant, note)


_orig = CF._paired_diff
CF._paired_diff = _paired_intersection

if __name__ == "__main__":
    out = CF.OUT / "posthoc_p1p2"
    tmp = pathlib.Path(tempfile.mkdtemp(dir=pathlib.Path.home() / ".cache"))
    CF.run(pathlib.Path(CF.C.RES), tmp)
    shutil.rmtree(tmp)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ROWS)
    df.to_csv(out / "p1p2_paired.csv", index=False)
    print(df.to_string())
