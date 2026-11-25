"""F4, the trust-gated certificate as the motor law degrades (E1, regime 2 "law wrong"; docs/board/research.md §0, E1).

Input: contract 'gate' = results/theory/gate_cert.csv (theory role, T0). One row per (law, severity, probe, certifier,
stratum). severity 0 / law two_constant is the correct law. Certifiers: par (two-constant fit + simultaneous), pi
(step-envelope identified set, model-free), gate (par where the evidence vouches for it, else pi), raw (par point, q=0),
nominal, detector_gated.
Figure: one column per probe set; top row certified share of capable decisions, bottom row unsafe-commit rate, both on
the decision stratum; one line per certifier per non-baseline law, the two-constant point at severity 0.
Success rule (pre-written in research.md E1; decision stratum):
  two-constant truth: gate CF >= par CF - 2 pts;
  worst knee severity at which par unsafe > 2 % (v2): gate unsafe <= 1 % and gate CF >= pi CF + 5 pts (v2 only).
  Kill: par unsafe <= 1 % at every knee severity.
Outputs: paper/figs/f4_gate.pdf, paper/tables/f4_gate.csv, macros cfFFour*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
import figstyle as S

STRATUM = "decision"
BASE_LAW = "two_constant"
PROBE = "v2"
CERTS = (("par", "parametric (two-constant)"), ("pi", "identified set (model-free)"), ("gate", "trust-gated (ours)"),
         ("raw", "raw posterior"), ("nominal", "nominal"), ("detector_gated", "detector-gated"))
KEYS = ("f4.gate.minus.par.twoc", "f4.severity.worst", "f4.par.unsafe.worst", "f4.gate.unsafe.worst",
        "f4.gate.cf.worst", "f4.pi.cf.worst", "f4.gate.minus.pi.worst", "f4.verdict", "f4.undom.share")


def _get(G, law, sev, cert, probe=PROBE):
    r = G[(G.law == law) & np.isclose(G.severity, sev) & (G.certifier == cert) & (G.probe == probe)]
    return r.iloc[0] if len(r) else None


def run(res, paper):
    N = C.Numbers("f4_gate.py")
    out = paper / "figs/f4_gate.pdf"
    G, srcs = C.load("gate", res)
    if G.empty:
        S.placeholder(out, "F4: awaiting results/theory/gate_cert.csv (E1)")
        for k in KEYS:
            N.ph(k)
        return N
    G["severity"] = G["severity"].astype(float)
    G.loc[G.law == BASE_LAW, "severity"] = 0.0
    st = C.status_of(G) if G["status"].notna().any() else "exploratory"
    Gd = G[G.stratum == STRATUM]
    probes = sorted(Gd.probe.unique(), key=lambda p: (p != PROBE, p))
    laws = [l for l in Gd.law.unique() if l != BASE_LAW]
    fig, axes = S.plt.subplots(2, max(1, len(probes)), figsize=(S.W2, 3.6), squeeze=False, sharex=True)
    lss = ("-", "--", ":", "-.")
    for j, pr in enumerate(probes):
        for i, (cert, lab) in enumerate(CERTS):
            for li, law in enumerate(laws or [BASE_LAW]):
                d = Gd[(Gd.probe == pr) & (Gd.certifier == cert) & Gd.law.isin([BASE_LAW, law])].sort_values("severity")
                if d.empty:
                    continue
                kw = dict(color=S.PAL[i % len(S.PAL)], ls=lss[li % len(lss)], marker="o", ms=2.5,
                          label=f"{lab}" + (f" ({law})" if len(laws) > 1 else "") if j == 0 else None)
                axes[0, j].plot(d.severity, 100 * d.cert_frac_capable, **kw)
                kw.pop("label")
                axes[1, j].plot(d.severity, 100 * d.unsafe_rate, **kw)
        axes[0, j].set_title(f"probe {pr}")
        axes[1, j].axhline(100 * C.U_STAR, color="0.5", lw=0.6, ls=":")
        axes[1, j].set_xlabel("motor-law misspecification severity (0 = two-constant)")
    axes[0, 0].set_ylabel("certified (%)")
    axes[1, 0].set_ylabel("unsafe commit (%)")
    axes[0, 0].legend(frameon=False, fontsize=6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    S.plt.close(fig)
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    G.drop(columns="_src").to_csv(paper / "tables/f4_gate.csv", index=False)

    # ---- success rule
    gd = Gd
    g0, p0 = _get(gd, BASE_LAW, 0.0, "gate"), _get(gd, BASE_LAW, 0.0, "par")
    if g0 is not None and p0 is not None:
        N.num("f4.gate.minus.par.twoc", 100 * (g0.cert_frac_capable - p0.cert_frac_capable), srcs, st, fmt="{:.1f}")
    else:
        N.ph("f4.gate.minus.par.twoc")
    par_bad = gd[(gd.certifier == "par") & (gd.probe == PROBE) & (gd.law != BASE_LAW)]
    worst = par_bad[par_bad.unsafe_rate > 0.02]
    if len(worst):
        w = worst.loc[worst.severity.idxmax()]
        g, pi = _get(gd, w.law, w.severity, "gate"), _get(gd, w.law, w.severity, "pi")
        N.num("f4.severity.worst", w.severity, srcs, st, fmt="{:g}")
        N.pct("f4.par.unsafe.worst", w.unsafe_rate, srcs, st, d=1)
        N.pct("f4.gate.unsafe.worst", g.unsafe_rate if g is not None else np.nan, srcs, st, d=2)
        N.pct("f4.gate.cf.worst", g.cert_frac_capable if g is not None else np.nan, srcs, st)
        N.pct("f4.pi.cf.worst", pi.cert_frac_capable if pi is not None else np.nan, srcs, st)
        if g is not None and pi is not None:
            gap = 100 * (g.cert_frac_capable - pi.cert_frac_capable)
            N.num("f4.gate.minus.pi.worst", gap, srcs, st, fmt="{:.1f}")
            twoc_ok = g0 is not None and p0 is not None and g0.cert_frac_capable >= p0.cert_frac_capable - 0.02
            ok = twoc_ok and g.unsafe_rate <= 0.01 and gap >= 5
            N.set("f4.verdict", "E1 passes" if ok else "E1 fails its pre-written success rule", srcs, st)
        else:
            N.ph("f4.gate.minus.pi.worst"), N.ph("f4.verdict")
    else:
        for k in ("f4.severity.worst", "f4.par.unsafe.worst", "f4.gate.unsafe.worst", "f4.gate.cf.worst",
                  "f4.pi.cf.worst", "f4.gate.minus.pi.worst"):
            N.ph(k, "par never exceeds 2 % unsafe")
        if len(par_bad) and (par_bad.unsafe_rate <= 0.01).all():
            N.set("f4.verdict", "E1 killed: the parametric certificate stays safe at every knee severity", srcs, st)
        else:
            N.ph("f4.verdict")
    u = G[(G.law == BASE_LAW) & (G.certifier == "par") & (G.probe == PROBE)]
    if "cert_share_undominated" in u and u.cert_share_undominated.notna().any():
        N.pct("f4.undom.share", float(u.cert_share_undominated.dropna().iloc[0]), srcs, st, d=0)
    else:
        N.ph("f4.undom.share", "needs optional column cert_share_undominated")
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
