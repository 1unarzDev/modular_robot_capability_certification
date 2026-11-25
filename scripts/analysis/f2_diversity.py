"""F2, training diversity -> generalization to larger and unseen morphologies (models role's arch x diversity x seed
grid, PLAN §6 B). x = test morphology size N, y = metric (certified fraction at the nominal bound; pinball in panel
b); one line per training-diversity level (mean over seeds, min-max band); held-out test morphologies hollow.

Input: contract 'div' (long format; metric in {cert_frac, unsafe_commit, pinball, cov90, det_1pct}).
Outputs: paper/figs/f2_diversity.pdf, macros cfFTwo*.
"""
from __future__ import annotations

import numpy as np

import common as C
import figstyle as S

ARCH = "st"
PANELS = (("cert_frac", "certified (%)", 100), ("pinball", r"pinball loss of $\rho$", 1))


def run(res, paper):
    N = C.Numbers("f2_diversity.py")
    out = paper / "figs/f2_diversity.pdf"
    D, srcs = C.load("div", res)
    if not D.empty:
        D = D[D.arch == ARCH]
    if D.empty:
        S.placeholder(out, "F2: awaiting models/paper/diversity.csv")
        for k in ("f2.cert.lowdiv.maxN", "f2.cert.highdiv.maxN", "f2.maxN", "f2.gain.pts"):
            N.ph(k)
        return N
    st = C.status_of(D)
    divs = D.drop_duplicates("diversity").sort_values("n_train_families").diversity.tolist()
    fig, axes = S.plt.subplots(1, 2, figsize=(S.W2, 2.2))
    for ax, (met, lab, sc) in zip(axes, PANELS):
        d = D[D.metric == met]
        for i, dv in enumerate(divs):
            g = d[d.diversity == dv].groupby(["test_n", "heldout"]).value.agg(["mean", "min", "max"]).reset_index()
            g = g.sort_values("test_n")
            col = S.PAL[i % len(S.PAL)]
            mx = int(D[D.diversity == dv].max_train_n.iloc[0])
            nf = int(D[D.diversity == dv].n_train_families.iloc[0])
            ax.plot(g.test_n, sc * g["mean"], "-", color=col, label=f"{nf} families, N$\\leq${mx}")
            ax.fill_between(g.test_n, sc * g["min"], sc * g["max"], color=col, alpha=0.15, lw=0)
            for ho, mk in ((0, "o"), (1, "o")):
                h = g[g.heldout == ho]
                ax.plot(h.test_n, sc * h["mean"], mk, color=col, ms=3.5,
                        mfc=col if ho == 0 else "white")
        ax.set_xlabel("test morphology size N (modules)")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    S.plt.close(fig)
    c = D[D.metric == "cert_frac"]
    maxN = c.test_n.max()
    at = c[c.test_n == maxN].groupby("diversity").value.mean()
    lo_, hi_ = divs[0], divs[-1]
    N.num("f2.maxN", maxN, srcs, st)
    N.pct("f2.cert.lowdiv.maxN", at.get(lo_, np.nan), srcs, st)
    N.pct("f2.cert.highdiv.maxN", at.get(hi_, np.nan), srcs, st)
    N.num("f2.gain.pts", 100 * (at.get(hi_, np.nan) - at.get(lo_, np.nan)), srcs, st, fmt="{:.1f}")
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
