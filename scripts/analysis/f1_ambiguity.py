"""F1, decision ambiguity vs per-wheel heterogeneity: Prop. 3's prediction (linear band, docs/theory/THEORY_cfcap.md
§3) against measured ambiguity.

Inputs:
  results/theory/nuisance_curve.csv   (theory role, existing) -> 'theory_linear' (amb_linear) and 'T0 exact'
                                       (amb_exact_corner), subset 'all', per identification variant
  contract 'amb' (results/*/paper/amb_het*.csv*) -> measured series (e.g. nuisance role's T1 SMC posterior)
Outputs: paper/figs/f1_ambiguity.pdf, macros cfFOne*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
import figstyle as S

THEORY = "theory/nuisance_curve.csv"
VARIANT = "ident_sensor"   # primary: parent's evidence identifies beta to the sensor noise (THEORY §3)


def run(res, paper):
    N = C.Numbers("f1_ambiguity.py")
    out = paper / "figs/f1_ambiguity.pdf"
    rows, srcs = [], []
    tf = res / THEORY
    if tf.exists():
        t = pd.read_csv(tf)
        t = t[t.subset == "all"]
        for v, d in t.groupby("variant"):
            for _, r in d.iterrows():
                rows.append(dict(source=f"Prop. 3 linear ({v})", delta=r.delta, amb_share=r.amb_linear,
                                 lo=np.nan, hi=np.nan, kind="theory", variant=v, status="exploratory"))
                rows.append(dict(source=f"T0 exact ({v})", delta=r.delta, amb_share=r.amb_exact_corner,
                                 lo=np.nan, hi=np.nan, kind="T0", variant=v, status="exploratory"))
        srcs.append(C.rel(tf))
    A, s2 = C.load("amb", res)
    if not A.empty:
        A = A[A.subset == "all"]
        for _, r in A.iterrows():
            rows.append(dict(source=str(r.source), delta=r.delta, amb_share=r.amb_share, lo=r.lo, hi=r.hi,
                             kind="measured", variant="", status=r.status))
        srcs += s2
    if not rows:
        S.placeholder(out, "F1: awaiting theory/nuisance_curve.csv and amb_het.csv")
        for k in ("f1.slope.theory", "f1.ratio.measured", "f1.amb.maxdelta"):
            N.ph(k)
        return N
    R = pd.DataFrame(rows)
    st = C.status_of(R)
    fig, ax = S.plt.subplots(figsize=(S.W1, 2.2))
    for i, (src, d) in enumerate(R.groupby("source", sort=False)):
        d = d.sort_values("delta")
        k = d.kind.iloc[0]
        if k == "theory" and VARIANT not in src:
            continue
        if k == "T0" and VARIANT not in src:
            continue
        col = S.PAL[0] if k == "theory" else S.PAL[1] if k == "T0" else S.PAL[2 + i % 4]
        ls = "--" if k == "theory" else "-"
        ax.plot(100 * d.delta, 100 * d.amb_share, ls, color=col, marker="o" if k != "theory" else None, ms=3,
                label=src)
        if d.lo.notna().any():
            ax.fill_between(100 * d.delta, 100 * d.lo, 100 * d.hi, color=col, alpha=0.2, lw=0)
    ax.set_xlabel(r"per-wheel motor spread $\delta$ (%)")
    ax.set_ylabel("ambiguous decisions (%)")
    ax.legend(frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    S.plt.close(fig)
    th = R[(R.kind == "theory") & R.source.str.contains(VARIANT)].sort_values("delta")
    if len(th) >= 2:
        sl = np.polyfit(th.delta, th.amb_share, 1)[0]
        N.num("f1.slope.theory", sl, srcs, st, fmt="{:.3f}")
        N.pct("f1.amb.maxdelta", th.amb_share.iloc[-1], srcs, st, d=2)
        N.pct("f1.maxdelta", th.delta.iloc[-1], srcs, st, d=0)
    else:
        N.ph("f1.slope.theory"), N.ph("f1.amb.maxdelta"), N.ph("f1.maxdelta")
    me = R[R.kind == "measured"]
    if len(me) and len(th):
        m = me.merge(th[["delta", "amb_share"]], on="delta", suffixes=("", "_th"))
        m = m[m.amb_share_th > 0]
        N.num("f1.ratio.measured", float(np.median(m.amb_share / m.amb_share_th)) if len(m) else np.nan,
              srcs, st, fmt="{:.2f}")
    else:
        N.ph("f1.ratio.measured")
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
