"""T1, the certificate table: certified fraction at matched unsafe-commit rate per method x candidate shape x plant,
for the decision-relevant stratum (HEADLINE) and for all decisions (appendix).

Input: contract 'dec' (results/*/paper/dec_*.csv*; truth = rho_dec), plus the trivial baselines computed here
(always_go, never_go, metadata, metadata_nf) unless a role exports them. The E3 trivial and partial-identification
baselines arrive as results/baselines/paper/dec_trivial.csv (detector_gated, ghat_physics, copy_parent,
credible_lower, min_over_members, dominance_pi) through the same contract. E2's dec_merge.csv rows are excluded here
(scored in merge_table.py).
Strata (common.py): 'rel' = candidate holds a faulted module OR |rho_dec - tau| < REL_BAND; 'all' = every decision.
Matched operating point, per stratum: for each method one margin k is chosen on its T1 *cal* rows so that the unsafe-
commit rate among that stratum's incapable decisions (pooled over shapes and TAUS) is U_STAR, and applied unchanged to
T1 test and T2 test. A method without cal rows is matched on its own T1 test rows and marked with a dagger (an
operating-point comparison, not a guarantee). The nominal table reports each method's own bound at its nominal alpha
in both strata (the guarantee that ships).
Outputs: paper/tables/t1_certificate.tex (rel), t1_certificate_all.tex (all), t1_nominal.tex, t1_by_size.tex (rel:
lone faulted module and candidate size 1 / 2 / >=3), t1_certificate.csv;
macros cfTOne* (unsuffixed = rel stratum; 'all' variants carry 'all' in the name).
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
    "amortized_comp": "Amortized, compositional",
    # E3 (baselines role): strong trivial and partial-identification baselines
    "detector_gated": "Detector-gated nominal", "ghat_physics": "Detected gain + physics",
    "copy_parent": "Copy parent capability", "min_over_members": "Min over member modules",
    "credible_lower": "Posterior credible lower (Bayes q05)", "raw": "Raw posterior (q = 0)",
    "cfBH": "Conformal selection (cfBH, FDR 0.1)", "dominance_pi": "Dominance identified set",
    "compositional_enum": "Compositional (per-assembly enumeration)",
    "oracle": "Full-state oracle",
}
ORDER = list(NAMES)
GROUPS = (("pair", lambda d: d["shape"] == "pair"), ("square", lambda d: d["shape"] == "square"),
          ("2x3", lambda d: d["shape"] == "2x3"), ("merge", lambda d: d["kind"] == "merge"),
          ("all", lambda d: np.ones(len(d), bool)))
SIZE_GROUPS = (("lone faulted", lambda d: (d["cand_size"] == 1) & (d["has_fault"] > 0.5)),
               ("size 1", lambda d: d["cand_size"] == 1), ("size 2", lambda d: d["cand_size"] == 2),
               ("size $\\geq$3", lambda d: d["cand_size"] >= 3))
# E3 success rule (docs/board/research.md E3, pre-written): decision stratum, T1 test, CF@U*
E3_TRIVIAL = ("detector_gated", "ghat_physics", "copy_parent", "metadata_nf")
E3_PI = "dominance_pi"
POSTERIOR_BASED = ("smc", "amortized", "enum")
NO_GUARANTEE = {"always_go", "never_go", "metadata", "metadata_nf", "nominal", "oracle"}
STRATUM_LABEL = {"rel": "decision-relevant decisions (faulted candidate or $|\\rho_{\\mathrm{dec}}-\\tau|<0.1$)",
                 "all": "all decisions"}


def tex(s):
    return str(s).replace("_", r"\_")


def name(m):
    return NAMES.get(m, tex(m))


def sfx(stratum):
    """Macro key suffix: the headline (rel) stratum is unsuffixed."""
    return "" if stratum == "rel" else ".all"


def _matched_table(R, methods, stratum, srcs, st, path):
    hdr = " & ".join(["Method"] + [g for _ in ("T1", "T2") for g, _s in GROUPS])
    lines = [r"\begin{tabular}{l" + "r" * 5 + "|" + "r" * 5 + "}", r"\toprule",
             r" & \multicolumn{5}{c|}{test on T1 (training simulator)} & \multicolumn{5}{c}{test on T2 (independent plant)}\\",
             hdr + r"\\", r"\midrule"]
    Rs = R[R.stratum == stratum]
    for m in methods:
        cells = [name(m) + (r"$^\dagger$" if Rs[Rs.method == m].dagger.any() else "")]
        for plant in ("T1", "T2"):
            for g, _s in GROUPS:
                r = Rs[(Rs.method == m) & (Rs.plant == plant) & (Rs.group == g)]
                cells.append(C.fmt_pct(r.cf.iloc[0]) if len(r) else r"\textemdash")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(path, "\n".join(lines), srcs,
                  f"certified % of capable {stratum} decisions at {stratum} unsafe-commit {C.U_STAR:.0%} "
                  f"(margin from T1 cal) [{st}]")


def run(res, paper):
    N = C.Numbers("t1_certificate.py")
    dec, srcs = C.load("dec", res)
    dec, _merge = C.split_merge_rows(dec)
    if "_src" in dec and not dec.empty:
        srcs = sorted(dec["_src"].unique())
    out_m, out_a, out_n = (paper / "tables/t1_certificate.tex", paper / "tables/t1_certificate_all.tex",
                           paper / "tables/t1_nominal.tex")
    out_s = paper / "tables/t1_by_size.tex"
    N.pct("t1.ustar", C.U_STAR, "scripts/analysis/common.py U_STAR")
    N.num("t1.relband", C.REL_BAND, "scripts/analysis/common.py REL_BAND", fmt="{:.2f}")
    if dec.empty:
        C.placeholder_table(out_m, "tOneTable", 11, "awaiting dec_*.csv from baselines/models/stress")
        C.placeholder_table(out_a, "tOneTableAll", 11, "awaiting dec_*.csv")
        C.placeholder_table(out_n, "tOneNominal", 7, "awaiting dec_*.csv")
        C.placeholder_table(out_s, "tOneBySize", 9, "awaiting dec_*.csv")
        for k in ("t1.e3.gain.trivial", "t1.e3.gain.pi", "t1.e3.verdict", "t1.e3.best.trivial"):
            N.ph(k)
        for s in C.STRATA:
            for k in ("t1.best.cert.T1", "t1.best.cert.T2", "t1.capxfer.cert.T1", "t1.gain.pts"):
                N.ph(k + sfx(s))
        for k in ("t1.best.method", "t1.relshare", "t1.relshare.capable", "t1.labels.below"):
            N.ph(k)
        return N
    dec = pd.concat([dec, C.trivial_methods(dec)], ignore_index=True)
    st = C.status_of(dec[dec["split"] != "cal"])
    methods = sorted(dec["method"].unique(), key=lambda m: (ORDER.index(m) if m in ORDER else 99, m))

    # stratum size on the shared labels (first method's T1 test rows)
    ref = dec[(dec.method == methods[0]) & (dec.test_plant == "T1") & (dec.split == "test")]
    if len(ref):
        a = C.decision_counts(ref, "lower", stratum="all")
        r_ = C.decision_counts(ref, "lower", stratum="rel")
        N.pct("t1.relshare", C._ratio(r_["n_dec"], a["n_dec"]), srcs, st)
        N.pct("t1.relshare.capable", C._ratio(r_["n_capable"], r_["n_dec"]), srcs, st)
        N.pct("t1.labels.below", float((ref.truth < max(C.TAUS)).mean()), srcs, st)

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
            for stratum in C.STRATA:
                if m in C.SELECTION_METHODS:
                    break                 # selection rule: no matched point; reported in the nominal table only
                lo, k = C.matched_lower(fit, ev, stratum=stratum)
                for g, sel in GROUPS + SIZE_GROUPS:
                    s = np.asarray(sel(ev), bool)
                    if not s.any():
                        continue
                    cnt = C.ep_counts(ev[s], lo[s], stratum=stratum)
                    cf = C._ratio(cnt["cert"].sum(), cnt["cap"].sum())
                    us = C._ratio(cnt["uns"].sum(), cnt["inc"].sum())
                    lohi = C.boot_ratio(cnt["cert"], cnt["cap"]) if g == "all" else (np.nan, np.nan)
                    res_rows.append(dict(method=m, plant=plant, stratum=stratum, group=g, cf=cf, unsafe=us, k=k,
                                         dagger=dagger, lo=lohi[0], hi=lohi[1], n_ep=len(cnt),
                                         n_capable=int(cnt["cap"].sum()), n_incapable=int(cnt["inc"].sum())))
            if m not in {"metadata", "metadata_nf"}:
                row = dict(method=m, plant=plant)
                for stratum in C.STRATA:
                    c = C.decision_counts(ev, "lower", stratum=stratum)
                    cnt = C.ep_counts(ev, ev["lower"].to_numpy(float), stratum=stratum)
                    row.update({f"{k}_{stratum}": v for k, v in c.items()})
                    row[f"unsafe_ci_{stratum}"] = C.boot_ratio(cnt["uns"], cnt["inc"])
                nom_rows.append(row)
    R = pd.DataFrame(res_rows)
    Rn = pd.DataFrame(nom_rows)
    paper.joinpath("tables").mkdir(parents=True, exist_ok=True)
    R.to_csv(paper / "tables/t1_certificate.csv", index=False)
    _matched_table(R, methods, "rel", srcs, st, out_m)
    _matched_table(R, methods, "all", srcs, st, out_a)
    _size_table(R, methods, srcs, st, out_s)

    # ---- nominal guarantee, both strata (rel first)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{decision-relevant} & \multicolumn{2}{c}{all decisions}\\",
             r"Method & plant & cert.\ (\%) & unsafe (\%) & ep.\ any unsafe (\%) & cert.\ (\%) & unsafe (\%)\\",
             r"\midrule"]
    for _, r in Rn.iterrows():
        ci = r.unsafe_ci_rel
        lines.append(f"{name(r.method)} & {r.plant} & {C.fmt_pct(r.cert_frac_rel)} & "
                     f"{C.fmt_pct(r.unsafe_commit_rel, 2)} [{C.fmt_pct(ci[0], 2)}, {C.fmt_pct(ci[1], 2)}] & "
                     f"{C.fmt_pct(r.ep_any_unsafe_all)} & {C.fmt_pct(r.cert_frac_all)} & "
                     f"{C.fmt_pct(r.unsafe_commit_all, 2)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(out_n, "\n".join(lines), srcs, f"own lower bound at nominal alpha, both strata [{st}]")

    # ---- macros (unsuffixed = rel stratum)
    for (m, plant, stratum), r in R[R.group == "all"].set_index(["method", "plant", "stratum"]).iterrows():
        N.pct(f"t1.cert.{m}.{plant}{sfx(stratum)}", r.cf, srcs, st)
        N.pct(f"t1.certlo.{m}.{plant}{sfx(stratum)}", r.lo, srcs, st)
        N.pct(f"t1.certhi.{m}.{plant}{sfx(stratum)}", r.hi, srcs, st)
    for _, r in Rn.iterrows():
        for stratum in C.STRATA:
            N.pct(f"t1.nom.unsafe.{r.method}.{r.plant}{sfx(stratum)}", r[f"unsafe_commit_{stratum}"], srcs, st, d=2)
            N.pct(f"t1.nom.cert.{r.method}.{r.plant}{sfx(stratum)}", r[f"cert_frac_{stratum}"], srcs, st)
        N.pct(f"t1.nom.epunsafe.{r.method}.{r.plant}", r.ep_any_unsafe_all, srcs, st)
    best = None
    for stratum in C.STRATA:
        Rs = R[(R.group == "all") & (R.stratum == stratum)]
        cand = Rs[(Rs.plant == "T1") & ~Rs.method.isin(NO_GUARANTEE)]
        if not len(cand):
            for k in ("t1.best.cert.T1", "t1.best.cert.T2", "t1.capxfer.cert.T1", "t1.gain.pts"):
                N.ph(k + sfx(stratum))
            continue
        if stratum != "rel" and (cand.method == best).any():
            b = cand[cand.method == best].iloc[0]       # same method as the headline, for comparability
        else:
            b = cand.loc[cand.cf.idxmax()]
        if stratum == "rel":
            best = b.method
            N.set("t1.best.method", name(b.method), srcs, st)
        N.pct("t1.best.cert.T1" + sfx(stratum), b.cf, srcs, st)
        N.pct("t1.best.certlo.T1" + sfx(stratum), b.lo, srcs, st)
        N.pct("t1.best.certhi.T1" + sfx(stratum), b.hi, srcs, st)
        t2 = Rs[(Rs.plant == "T2") & (Rs.method == b.method)]
        N.pct("t1.best.cert.T2" + sfx(stratum), t2.cf.iloc[0] if len(t2) else np.nan, srcs, st)
        cx = Rs[(Rs.plant == "T1") & (Rs.method == "capxfer")]
        if len(cx):
            N.pct("t1.capxfer.cert.T1" + sfx(stratum), cx.cf.iloc[0], srcs, st)
            N.num("t1.gain.pts" + sfx(stratum), 100 * (b.cf - cx.cf.iloc[0]), srcs, st, fmt="{:.1f}")
        else:
            N.ph("t1.capxfer.cert.T1" + sfx(stratum)), N.ph("t1.gain.pts" + sfx(stratum))
    if best is None:
        N.ph("t1.best.method")
    _e3(N, R, dec, srcs, st)
    return N


def _size_table(R, methods, srcs, st, path):
    Rs = R[R.stratum == "rel"]
    gs = [g for g, _ in SIZE_GROUPS]
    lines = [r"\begin{tabular}{l" + "r" * len(gs) + "|" + "r" * len(gs) + "}", r"\toprule",
             f" & \\multicolumn{{{len(gs)}}}{{c|}}{{test on T1}} & \\multicolumn{{{len(gs)}}}{{c}}{{test on T2}}\\\\",
             "Method & " + " & ".join(gs * 2) + r"\\", r"\midrule"]
    for m in methods:
        cells = [name(m)]
        for plant in ("T1", "T2"):
            for g in gs:
                r = Rs[(Rs.method == m) & (Rs.plant == plant) & (Rs.group == g)]
                cells.append(C.fmt_pct(r.cf.iloc[0]) if len(r) else r"\textemdash")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(path, "\n".join(lines), srcs,
                  f"decision-relevant stratum by body size, CF@U*={C.U_STAR:.0%} (margin from T1 cal) [{st}]")


def _e3(N, R, dec, srcs, st):
    """E3 success rule: best posterior-based method beats max(trivial) by >= 10 pts and dominance_pi by >= 5 pts."""
    post = dec.drop_duplicates("method").set_index("method")["posterior"]
    Rs = R[(R.stratum == "rel") & (R.plant == "T1") & (R.group == "all")].set_index("method")["cf"].dropna()
    pb = Rs[[m for m in Rs.index if post.get(m) in POSTERIOR_BASED]]
    tv = Rs[[m for m in Rs.index if m in E3_TRIVIAL]]
    if not len(pb) or not len(tv):
        for k in ("t1.e3.gain.trivial", "t1.e3.gain.pi", "t1.e3.verdict", "t1.e3.best.trivial"):
            N.ph(k)
        return
    g_tv = 100 * (pb.max() - tv.max())
    N.num("t1.e3.gain.trivial", g_tv, srcs, st, fmt="{:.1f}")
    N.set("t1.e3.best.trivial", name(tv.idxmax()), srcs, st)
    if E3_PI in Rs.index:
        g_pi = 100 * (pb.max() - Rs[E3_PI])
        N.num("t1.e3.gain.pi", g_pi, srcs, st, fmt="{:.1f}")
    else:
        g_pi = np.nan
        N.ph("t1.e3.gain.pi")
    ok = g_tv >= 10 and (np.isnan(g_pi) or g_pi >= 5)
    N.set("t1.e3.verdict", ("E3 passes" if ok else "E3 fails: reword C3 to latency / v2-only / gate (F3)")
          + ("" if np.isfinite(g_pi) else " (dominance\\_pi missing)"), srcs, st)


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
