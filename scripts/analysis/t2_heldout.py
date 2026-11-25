"""T2, held-out family coverage: does each method's own lower bound (nominal alpha) still cover on morphology
families never seen in training or calibration (generator splits H3 topology, H4 scale, H6, H7)?

Input: contract 'dec' (truth = rho_dec), test_plant T1, splits test (in-family reference) and H*. Reported per
method x split x stratum: entry coverage P(lower <= truth), episode-simultaneous coverage (target 1 - alpha for
simultaneous routes), certified fraction and unsafe-commit rate at the nominal bound. HEADLINE = decision-relevant
stratum (common.py: faulted candidate or |rho_dec - tau| < REL_BAND; for coverage, near any tau): an episode is
covered if every relevant entry is. Empirical only: no held-out guarantee is claimed (PLAN §2 "Not claimed").
Outputs: paper/tables/t2_heldout.tex (rel), t2_heldout_all.tex, t2_heldout.csv, macros cfTTwo* (unsuffixed = rel).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
from t1_certificate import ORDER, name, sfx

SKIP = {"always_go", "never_go", "metadata", "metadata_nf"}


def _table(R, methods, splits, stratum, srcs, st, path):
    Rs = R[R.stratum == stratum]
    cols = " & ".join(f"\\multicolumn{{2}}{{c}}{{{s}}}" for s in splits)
    lines = [r"\begin{tabular}{l" + "rr" * len(splits) + "}", r"\toprule", f"Method & {cols}\\\\",
             " & " + " & ".join(["sim.\\ cov", "cert"] * len(splits)) + r"\\", r"\midrule"]
    for m in methods:
        cells = [name(m)]
        for s in splits:
            r = Rs[(Rs.method == m) & (Rs.split == s)]
            cells += [C.fmt_pct(r.ep_cov.iloc[0]), C.fmt_pct(r.cert_frac.iloc[0])] if len(r) else [r"\textemdash"] * 2
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(path, "\n".join(lines), srcs,
                  f"{stratum} stratum: episode-simultaneous coverage / certified %, nominal alpha [{st}]")


def run(res, paper):
    N = C.Numbers("t2_heldout.py")
    dec, srcs = C.load("dec", res)
    dec, _ = C.split_merge_rows(dec)   # E2 merge-only rows are scored in merge_table.py
    out, out_a = paper / "tables/t2_heldout.tex", paper / "tables/t2_heldout_all.tex"
    N.pct("t2.alpha", 1 - C.ALPHA, "scripts/analysis/common.py ALPHA", d=0)
    dec = dec[(dec["test_plant"] == "T1") & (dec["split"] != "cal")] if not dec.empty else dec
    splits = sorted(dec["split"].unique(), key=lambda s: (s != "test", s)) if not dec.empty else []
    if dec.empty or not any(s.startswith("H") for s in splits):
        C.placeholder_table(out, "tTwoTable", 6, "awaiting dec rows with split H3/H4/H6/H7")
        C.placeholder_table(out_a, "tTwoTableAll", 6, "awaiting dec rows with split H3/H4/H6/H7")
        for s in C.STRATA:
            for k in ("t2.epcov.worst", "t2.epcov.worst.split", "t2.epcov.comp.worst"):
                N.ph(k + sfx(s))
        return N
    st = C.status_of(dec)
    methods = sorted((m for m in dec["method"].unique() if m not in SKIP),
                     key=lambda m: (ORDER.index(m) if m in ORDER else 99, m))
    rows = []
    for m in methods:
        for s in splits:
            d = dec[(dec.method == m) & (dec.split == s)]
            if d.empty:
                continue
            for stratum in C.STRATA:
                cv = C.coverage(d, stratum=stratum)
                dc = C.decision_counts(d, stratum=stratum)
                rows.append(dict(method=m, split=s, stratum=stratum, route=d.route.iloc[0], **cv,
                                 cert_frac=dc["cert_frac"], unsafe=dc["unsafe_commit"]))
    R = pd.DataFrame(rows)
    R.to_csv(paper / "tables/t2_heldout.csv", index=False)
    _table(R, methods, splits, "rel", srcs, st, out)
    _table(R, methods, splits, "all", srcs, st, out_a)
    for _, r in R.iterrows():
        x = sfx(r.stratum)
        N.pct(f"t2.epcov.{r.method}.{r.split}{x}", r.ep_cov, srcs, st)
        N.pct(f"t2.entrycov.{r.method}.{r.split}{x}", r.entry_cov, srcs, st)
        N.pct(f"t2.cert.{r.method}.{r.split}{x}", r.cert_frac, srcs, st)
    for stratum in C.STRATA:
        x = sfx(stratum)
        H = R[(R.stratum == stratum) & R.split.str.startswith("H")
              & R.route.isin(["simultaneous", "compositional", "marginal"])].dropna(subset=["ep_cov"])
        if len(H):
            w = H.loc[H.ep_cov.idxmin()]
            N.pct("t2.epcov.worst" + x, w.ep_cov, srcs, st)
            N.set("t2.epcov.worst.split" + x, f"{w.split} ({name(w.method)})", srcs, st)
        else:
            N.ph("t2.epcov.worst" + x), N.ph("t2.epcov.worst.split" + x)
        Hc = H[H.route == "compositional"] if len(H) else H
        N.pct("t2.epcov.comp.worst" + x, Hc.ep_cov.min() if len(Hc) else np.nan, srcs, st)
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
