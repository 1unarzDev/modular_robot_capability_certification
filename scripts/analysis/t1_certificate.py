"""T1, the certificate table: certified fraction at matched unsafe-commit rate per method x candidate shape x plant.

Input: contract 'dec' (results/*/paper/dec_*.csv*), plus the trivial baselines computed here (always_go, never_go,
metadata, metadata_nf).
Matched operating point: for each method one margin k is chosen on its T1 *cal* rows so that the pooled unsafe-commit
rate (over shapes and TAUS) is U_STAR, and applied unchanged to T1 test and T2 test. A method without cal rows is
matched on its own T1 test rows instead and marked with a dagger (an operating-point comparison, not a guarantee).
The second table reports each method's own certified lower bound at its nominal alpha (the guarantee that ships).
Outputs: paper/tables/t1_certificate.tex, paper/tables/t1_nominal.tex, macros cfTOne*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

NAMES = {  # display names; unknown methods print their raw name
    "always_go": "Always go (constant)", "never_go": "Never go (constant)",
    "metadata_nf": "Shape metadata only", "metadata": "Metadata + true fault flag",
    "nominal": "Nominal (healthy) model", "rls_interval": "RLS + intervals", "capxfer": "capxfer certificate",
    "capxfer_raw": "capxfer certificate (raw)", "enum_known": "Exact enumeration (nuisance known)",
    "nuisance_smc": "GPU-SMC (simulator likelihood)", "blackbox_gnn": "Black-box counterfactual GNN",
    "ensemble": "Deep ensemble", "amortized_marg": "Amortized, marginal", "amortized_sim": "Amortized, simultaneous",
    "amortized_comp": "Amortized, compositional", "oracle": "Full-state oracle",
}
ORDER = list(NAMES)
GROUPS = (("pair", lambda d: d["shape"] == "pair"), ("square", lambda d: d["shape"] == "square"),
          ("2x3", lambda d: d["shape"] == "2x3"), ("merge", lambda d: d["kind"] == "merge"),
          ("all", lambda d: np.ones(len(d), bool)))
NO_GUARANTEE = {"always_go", "never_go", "metadata", "metadata_nf", "nominal", "oracle"}


def tex(s):
    return str(s).replace("_", r"\_")


def name(m):
    return NAMES.get(m, tex(m))


def run(res, paper):
    N = C.Numbers("t1_certificate.py")
    dec, srcs = C.load("dec", res)
    out_m, out_n = paper / "tables/t1_certificate.tex", paper / "tables/t1_nominal.tex"
    if dec.empty:
        C.placeholder_table(out_m, "tOneTable", 11, "awaiting dec_*.csv from baselines/models/stress")
        C.placeholder_table(out_n, "tOneNominal", 5, "awaiting dec_*.csv")
        for k in ("t1.best.cert.T1", "t1.best.cert.T2", "t1.best.method", "t1.capxfer.cert.T1", "t1.gain.pts",
                  "t1.ustar"):
            N.ph(k)
        N.pct("t1.ustar", C.U_STAR, "scripts/analysis/common.py U_STAR")
        return N
    dec = pd.concat([dec, C.trivial_methods(dec)], ignore_index=True)
    st = C.status_of(dec[dec["split"] != "cal"])
    methods = sorted(dec["method"].unique(), key=lambda m: (ORDER.index(m) if m in ORDER else 99, m))
    res_rows, nom_rows = [], []
    for m in methods:
        dm = dec[dec["method"] == m]
        fit = dm[(dm["test_plant"] == "T1") & (dm["split"] == "cal")]
        dagger = fit.empty
        if dagger:
            fit = dm[(dm["test_plant"] == "T1") & (dm["split"] == "test")]
        if fit.empty:
            continue
        for plant in ("T1", "T2"):
            ev = dm[(dm["test_plant"] == plant) & (dm["split"] == "test")].reset_index(drop=True)
            if ev.empty:
                continue
            lo, k = C.matched_lower(fit, ev)
            for g, sel in GROUPS:
                s = np.asarray(sel(ev), bool)
                if not s.any():
                    continue
                sub = ev[s]
                cnt = C.ep_counts(sub, lo[s])
                cf = cnt["cert"].sum() / max(cnt["cap"].sum(), 1)
                us = cnt["uns"].sum() / max(cnt["inc"].sum(), 1)
                lohi = C.boot_ratio(cnt["cert"], cnt["cap"]) if g == "all" else (np.nan, np.nan)
                res_rows.append(dict(method=m, plant=plant, group=g, cf=cf, unsafe=us, k=k, dagger=dagger,
                                     lo=lohi[0], hi=lohi[1], n_ep=len(cnt)))
            if m not in {"metadata", "metadata_nf"}:
                c = C.decision_counts(ev, "lower")
                cnt = C.ep_counts(ev, ev["lower"].to_numpy(float))
                nom_rows.append(dict(method=m, plant=plant, **c,
                                     unsafe_ci=C.boot_ratio(cnt["uns"], cnt["inc"])))
    R = pd.DataFrame(res_rows)
    Rn = pd.DataFrame(nom_rows)
    paper.joinpath("tables").mkdir(parents=True, exist_ok=True)
    R.to_csv(paper / "tables/t1_certificate.csv", index=False)

    # ---- table: matched
    hdr = " & ".join(["Method"] + [f"{g}" for _ in ("T1", "T2") for g, _s in GROUPS])
    lines = [r"\begin{tabular}{l" + "r" * 5 + "|" + "r" * 5 + "}", r"\toprule",
             r" & \multicolumn{5}{c|}{test on T1 (training simulator)} & \multicolumn{5}{c}{test on T2 (independent plant)}\\",
             hdr + r"\\", r"\midrule"]
    for m in methods:
        cells = [name(m) + (r"$^\dagger$" if R[(R.method == m)].dagger.any() else "")]
        for plant in ("T1", "T2"):
            for g, _s in GROUPS:
                r = R[(R.method == m) & (R.plant == plant) & (R.group == g)]
                cells.append(C.fmt_pct(r.cf.iloc[0]) if len(r) else r"\textemdash")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(out_m, "\n".join(lines), srcs,
                  f"certified % of capable decisions at unsafe-commit {C.U_STAR:.0%} (margin from T1 cal) [{st}]")

    # ---- table: nominal guarantee
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Method & plant & certified (\%) & unsafe commit (\%) & episodes with any unsafe (\%)\\", r"\midrule"]
    for _, r in Rn.iterrows():
        lines.append(f"{name(r.method)} & {r.plant} & {C.fmt_pct(r.cert_frac)} & {C.fmt_pct(r.unsafe_commit, 2)} "
                     f"[{C.fmt_pct(r.unsafe_ci[0], 2)}, {C.fmt_pct(r.unsafe_ci[1], 2)}] & "
                     f"{C.fmt_pct(r.ep_any_unsafe)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(out_n, "\n".join(lines), srcs, f"own lower bound at nominal alpha [{st}]")

    # ---- macros
    N.pct("t1.ustar", C.U_STAR, "scripts/analysis/common.py U_STAR")
    for (m, plant), r in R[R.group == "all"].set_index(["method", "plant"]).iterrows():
        N.pct(f"t1.cert.{m}.{plant}", r.cf, srcs, st)
        N.pct(f"t1.certlo.{m}.{plant}", r.lo, srcs, st)
        N.pct(f"t1.certhi.{m}.{plant}", r.hi, srcs, st)
    for _, r in Rn.iterrows():
        N.pct(f"t1.nom.unsafe.{r.method}.{r.plant}", r.unsafe_commit, srcs, st, d=2)
        N.pct(f"t1.nom.cert.{r.method}.{r.plant}", r.cert_frac, srcs, st)
        N.pct(f"t1.nom.epunsafe.{r.method}.{r.plant}", r.ep_any_unsafe, srcs, st)
    cand = R[(R.group == "all") & (R.plant == "T1") & ~R.method.isin(NO_GUARANTEE)]
    if len(cand):
        b = cand.loc[cand.cf.idxmax()]
        N.set("t1.best.method", name(b.method), srcs, st)
        N.pct("t1.best.cert.T1", b.cf, srcs, st)
        t2 = R[(R.group == "all") & (R.plant == "T2") & (R.method == b.method)]
        N.pct("t1.best.cert.T2", t2.cf.iloc[0] if len(t2) else np.nan, srcs, st)
        cx = R[(R.group == "all") & (R.plant == "T1") & (R.method == "capxfer")]
        if len(cx):
            N.pct("t1.capxfer.cert.T1", cx.cf.iloc[0], srcs, st)
            N.num("t1.gain.pts", 100 * (b.cf - cx.cf.iloc[0]), srcs, st, fmt="{:.1f}")
        else:
            N.ph("t1.capxfer.cert.T1"), N.ph("t1.gain.pts")
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
