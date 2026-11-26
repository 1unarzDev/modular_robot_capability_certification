"""Confirmatory run-1 tables (read-only formatting of results/confirmatory/{cells,verdicts}.csv).

tables/tc_main.tex      Table: own bound and matched point per method, C-MAIN T1 (decision-relevant stratum, lone faulted)
tables/tc_merge.tex     Table: never-co-assembled merges (C-MERGE), T1 and T2
tables/tc_transfer.tex  Table: larger parents (C-SCALE) and new module types (C-HET), validity and usefulness
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

DISPLAY = [("amortized_comp", "Ours"), ("capxfer", "CapXfer"), ("ghat_physics", r"$\hat g$-physics"),
           ("min_over_members", "Min over members"), ("copy_parent", "Copy parent"),
           ("detector_gated", "Detector-gated"), ("metadata_nf", "Shape metadata"), ("oracle", "Oracle (full state)")]


def f(x, d=1):
    if x is None or not np.isfinite(x):
        return r"\textemdash"
    return f"{100 * x:.{d}f}"


def sg(x):
    t = f"{100 * x:+.1f}"
    return t.replace("-", "$-$") if t not in ("+0.0", "-0.0") else "0.0"


def run(res, paper):
    N = C.Numbers("tc_tables.py")
    cp, vp = res / "confirmatory/cells.csv", res / "confirmatory/verdicts.csv"
    if not cp.exists():
        return N
    c = pd.read_csv(cp)
    v = pd.read_csv(vp)
    src = ["results/confirmatory/cells.csv", "results/confirmatory/verdicts.csv"]

    # ---- main
    m = c[(c.claim == "C-MAIN") & (c.plant == "T1")]
    rows = []
    for code, name in DISPLAY:
        s = m[(m.method == code) & (m.block == "stratum")]
        lo = m[(m.method == code) & (m.block == "lone-faulted")]
        if s.empty:
            continue
        s, lo = s.iloc[0], lo.iloc[0]
        ours = code == "amortized_comp"
        b = (lambda t: rf"\textbf{{{t}}}") if ours else (lambda t: t)
        rows.append(" & ".join([b(name), f"{int(s.n_ep):,}", b(f(s.CF_own)), f(s.E_dec_own, 2), f(s.E_ep, 2),
                                b(f(lo.CF_own)), f(lo.E_dec_own, 2), f(s.CF_matched), f(s.E_dec_matched, 2),
                                f(lo.CF_matched), f(lo.E_dec_matched, 2)]) + r" \\")
        if ours:
            rows.append(r"\midrule")
    body = (r"\begin{tabular}{@{}lrrrrrrrrrr@{}}\toprule" "\n"
            r"& & \multicolumn{5}{c}{Own bound ($\alpha=0.10$)} & \multicolumn{4}{c}{Matched point (target $U^\ast=1\,\%$)}\\"
            r"\cmidrule(lr){3-7}\cmidrule(l){8-11}" "\n"
            r"& & \multicolumn{3}{c}{Stratum} & \multicolumn{2}{c}{Lone faulted} & \multicolumn{2}{c}{Stratum} & \multicolumn{2}{c}{Lone faulted}\\"
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}" "\n"
            r"Method & Episodes & CF & $E_{\mathrm{dec}}$ & $E_{\mathrm{unsafe}}$ & CF & $E_{\mathrm{dec}}$ & CF & $E_{\mathrm{dec}}$ & CF & $E_{\mathrm{dec}}$\\\midrule" "\n"
            + "\n".join(rows) + "\n" r"\bottomrule\end{tabular}")
    C.write_table(paper / "tables/tc_main.tex", body, src, "confirmatory run 1 (77fc39b); percentages")
    plain = (body.replace(r"\multicolumn{3}{c}{Stratum}", r"\multicolumn{3}{c}{Hard cases}")
             .replace(r"\multicolumn{2}{c}{Stratum}", r"\multicolumn{2}{c}{Hard cases}")
             .replace(r"\multicolumn{2}{c}{Lone faulted}", r"\multicolumn{2}{c}{Faulty module alone}"))
    C.write_table(paper / "tables/tc_main_plain.tex", plain, src, "IEEE wording; same numbers as tc_main.tex")

    # ---- merge
    def vb(claim, bar, metric=None, plant=None):
        r = v[(v.claim == claim) & (v.bar == bar)]
        if metric is not None:
            r = r[r.metric == metric]
        if plant is not None:
            r = r[r.plant == plant]
        return r.iloc[0]

    def ci(r, d=1):
        return rf"{f(r.estimate, d)} {{\scriptsize[{f(r.ci_lo, d)}, {f(r.ci_hi, d)}]}}"

    mr = []
    for plant in ("T1", "T2"):
        fl = vb("C-MERGE", "1 delta_comp", plant=plant)
        mc = vb("C-MERGE", "2 validity", "miscoverage per pair", plant)
        ep = vb("C-MERGE", "2 validity", "E_ep per pair", plant)
        if plant == "T1":
            cf = vb("C-MERGE", "P4", "CF@U* comp_enum_size")
        else:
            cf = vb("C-MERGE", "P4 (T2, reported)", "CF@U* comp_enum_size")
        mr.append(f"{plant} & {f(fl.estimate, 2)} {{\\scriptsize($\\le${f(fl.upper95, 2)})}} & {ci(mc, 1)} & {f(ep.estimate, 2)} & {ci(cf, 1)}" + r" \\")
    body = (r"\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}lrrrr@{}}\toprule" "\n"
            r"Plant & Flips & Miscov. & $E_{\mathrm{unsafe}}$ & CF \\\midrule" "\n" + "\n".join(mr) + "\n"
            r"\bottomrule\end{tabular}")
    C.write_table(paper / "tables/tc_merge.tex", body, src, "confirmatory run 1; percentages [95% CI]")

    # ---- transfer
    tr = []
    for cell, lab in (("N 7-8", "7--8"), ("N 9-10", "9--10")):
        e = v[(v.claim == "C-SCALE") & (v.metric == "E_ep") & (v.cell == cell)].iloc[0]
        cv = v[(v.claim == "C-SCALE") & (v.metric == "stratum coverage") & (v.cell == cell)].iloc[0]
        tr.append(f"{lab} modules & {ci(e)} & {f(cv.estimate)} & \\textemdash & \\textemdash" + r" \\")
    for cell, lab, met in (("conf_H1", "Omniwheel", "E_ep (faulted)"), ("conf_H2", "Mixed types", "E_ep (all)")):
        e = v[(v.claim == "C-HET") & (v.metric == met) & (v.cell == cell) & (v.method == "amortized_comp")].iloc[0]
        tag = "conf_H1 (faulted)" if cell == "conf_H1" else "conf_H2 (all)"
        cd = v[(v.claim == "C-HET") & (v.metric == f"coverage drop ID -> {tag}")].iloc[0]
        fd = v[(v.claim == "C-HET") & (v.metric == f"CF drop ID -> {tag}")].iloc[0]
        tr.append(f"{lab} & {ci(e)} & \\textemdash & {sg(-cd.estimate)} & {sg(-fd.estimate)}" + r" \\")
    body = (r"\setlength{\tabcolsep}{3.5pt}\begin{tabular}{@{}lrrrr@{}}\toprule" "\n"
            r"Shift & $E_{\mathrm{unsafe}}$ & Cov. & $\Delta$Cov. & $\Delta$CF \\\midrule" "\n" + "\n".join(tr) + "\n"
            r"\bottomrule\end{tabular}")
    C.write_table(paper / "tables/tc_transfer.tex", body, src, "confirmatory run 1; percentages")
    return N
