"""T3, the 2x2 grid {GPU-SMC with the T1 simulator as likelihood, amortized posterior trained with wide DR} x
{test on T1, test on T2}. This is the test of the ML claim (PLAN §2 contribution 4): amortization must win on
misspecification robustness (T2 column) and/or decision latency; if SMC wins on both, the word "amortized" goes.

Input: contract 'dec' rows (truth = rho_dec) with posterior in {smc, amortized}, split test, the route given by ROUTE
(simultaneous, the guarantee both share). Per cell and stratum: unsafe-commit (CI over episodes), episodes with any
unsafe go, certified fraction at the nominal bound, median seconds per decision. HEADLINE and verdict = the
decision-relevant stratum (common.py); the all-decisions numbers are reported beside it.
Outputs: paper/tables/t3_grid.tex, macros cfTThree* (unsuffixed = rel).
"""
from __future__ import annotations

import numpy as np

import common as C
from t1_certificate import sfx

ROUTE = "simultaneous"
# models 8c96915: exactly one method has posterior "amortized": amortized_comp (st-n6-2), route compositional; the
# ensemble / black-box carry posterior "none", the direct heads "amortized_direct". Each posterior uses its own route.
ROUTES = {"smc": "simultaneous", "amortized": "compositional"}
POST = (("smc", "GPU-SMC (T1 likelihood)"), ("amortized", "Amortized (wide DR)"))


def _dominates(x, y):
    """x is at least as safe AND at least as useful as y on T2 (NaN never dominates)."""
    return bool(x["unsafe_commit"] <= y["unsafe_commit"] and x["cert_frac"] >= y["cert_frac"])


def verdict(a2, s2) -> str:
    """Robustness = Pareto dominance on (T2 unsafe-commit, T2 certified share); latency = median s/decision.
    Failing to dominate is NOT reverse dominance (codex audit, t3_grid.py:85): when neither posterior dominates the
    other, the robustness result is a tradeoff, and 'SMC wins on both' needs SMC to dominate AND be faster."""
    a_rob, s_rob = _dominates(a2, s2), _dominates(s2, a2)
    if a_rob and s_rob:
        rob = "tie on T2 robustness"
    elif a_rob:
        rob = "amortized more robust on T2"
    elif s_rob:
        rob = "SMC more robust on T2"
    else:
        more_useful = "amortized" if a2["cert_frac"] > s2["cert_frac"] else "SMC"
        safer = "amortized" if a2["unsafe_commit"] < s2["unsafe_commit"] else "SMC"
        rob = f"T2 tradeoff ({more_useful} certifies more, {safer} is safer)"
    fin = np.isfinite(a2["sec"]) and np.isfinite(s2["sec"])
    a_fast = bool(fin and a2["sec"] < s2["sec"])
    s_fast = bool(fin and s2["sec"] < a2["sec"])
    if a_rob and not s_rob and a_fast:
        return "amortized wins on T2 robustness and latency"
    if s_rob and not a_rob and s_fast:
        return "SMC wins on both: drop 'amortized'"
    lat = "amortized faster" if a_fast else "SMC faster" if s_fast else "latency tie or unknown"
    return f"{rob}; {lat}"


LAT_NAMES = {"envpost_ktol": "Exact posterior (knee-tolerant)", "envpost_k5": "Exact posterior (K = 5)",
             "nuisance_smc": "GPU-SMC (T1 likelihood)", "enum_known": "Exact enumeration (nuisance known)"}


def latency(res, paper, N):
    """Latency half of the ML claim: wall time per decision by method and parent size (common.latency_table)."""
    L, srcs = C.latency_table(res)
    out = paper / "tables/t3_latency.tex"
    if L.empty:
        C.placeholder_table(out, "tThreeLatency", 4, "awaiting latency*.csv")
        return
    L = L.copy()
    L["size"] = L["shape"].astype(str)
    lines = [r"\begin{tabular}{llrl}", r"\toprule", r"Method & parent & s / decision & hardware\\", r"\midrule"]
    for _, r in L.sort_values(["method", "n_wheels"]).iterrows():
        nm = LAT_NAMES.get(r.method, str(r.method).replace("_", r"\_"))
        v = r.sec_per_decision
        cell = f"{1000 * v:.1f} ms" if v < 0.1 else f"{v:.2f} s"
        lines.append(f"{nm} & {r['size'].replace('_', ' ')} & {cell} & {str(r.hardware).replace('_', ' ')}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(out, "\n".join(lines), srcs, "wall time per decision (pilot timings; hardware differs by row)")
    st = C.status_of(L)
    for m, d in L[L.n_wheels.notna()].groupby("method"):
        N.num(f"t3.lat.{m}.min", d.sec_per_decision.min(), srcs, st, fmt="{:.3g}")
        N.num(f"t3.lat.{m}.max", d.sec_per_decision.max(), srcs, st, fmt="{:.3g}")


def run(res, paper):
    N = C.Numbers("t3_grid.py")
    latency(res, paper, N)
    dec, srcs = C.load("dec", res)
    dec, _ = C.split_merge_rows(dec)   # E2 merge-only rows are scored in merge_table.py
    out = paper / "tables/t3_grid.tex"
    if not dec.empty:
        sel = np.zeros(len(dec), bool)
        for p_, rt in ROUTES.items():
            sel |= ((dec.posterior == p_) & (dec.route == rt)).to_numpy()
        dec = dec[(dec.split == "test").to_numpy() & sel]
    if dec.empty:
        C.placeholder_table(out, "tThreeTable", 7, "awaiting dec rows: posterior smc (simultaneous) / amortized "
                            "(compositional)")
        for p, _ in POST:
            for pl in ("T1", "T2"):
                for s in C.STRATA:
                    for k in ("unsafe", "cert", "epunsafe"):
                        N.ph(f"t3.{k}.{p}.{pl}{sfx(s)}")
            N.ph(f"t3.sec.{p}")
        N.ph("t3.verdict")
        return N
    st = C.status_of(dec)
    cell = {}
    for p, _ in POST:
        for pl in ("T1", "T2"):
            d = dec[(dec.posterior == p) & (dec.test_plant == pl)]
            if d.empty:
                continue
            if d.method.nunique() > 1:
                raise C.SchemaError(f"T3: several methods for posterior={p}, route={ROUTES[p]}: "
                                    f"{sorted(d.method.unique())}")
            sec = float(np.nanmedian(d.drop_duplicates("eid")["seconds"])) if d["seconds"].notna().any() else np.nan
            for s in C.STRATA:
                c = C.decision_counts(d, stratum=s)
                cnt = C.ep_counts(d, d["lower"].to_numpy(float), stratum=s)
                c["ci"] = C.boot_ratio(cnt["uns"], cnt["inc"])
                c["sec"] = sec
                cell[p, pl, s] = c
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{test T1} & \multicolumn{3}{c}{test T2}\\",
             r" & \multicolumn{2}{c}{decision-relevant} & all & \multicolumn{2}{c}{decision-relevant} & all\\",
             r"Posterior & unsafe (\%) & cert.\ (\%) & cert.\ (\%) & unsafe (\%) & cert.\ (\%) & cert.\ (\%)\\",
             r"\midrule"]
    for p, label in POST:
        cells = [label]
        for pl in ("T1", "T2"):
            c, ca = cell.get((p, pl, "rel")), cell.get((p, pl, "all"))
            if c is None:
                cells += [r"\textemdash"] * 3
            else:
                cells += [f"{C.fmt_pct(c['unsafe_commit'], 2)} [{C.fmt_pct(c['ci'][0], 2)}, {C.fmt_pct(c['ci'][1], 2)}]",
                          C.fmt_pct(c["cert_frac"]), C.fmt_pct(ca["cert_frac"])]
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(out, "\n".join(lines), srcs, f"routes {ROUTES}, nominal alpha [{st}]")
    for (p, pl, s), c in cell.items():
        N.pct(f"t3.unsafe.{p}.{pl}{sfx(s)}", c["unsafe_commit"], srcs, st, d=2)
        N.pct(f"t3.cert.{p}.{pl}{sfx(s)}", c["cert_frac"], srcs, st)
        N.pct(f"t3.epunsafe.{p}.{pl}{sfx(s)}", c["ep_any_unsafe"], srcs, st)
    for p, _ in POST:
        secs = [c["sec"] for (pp, _, _), c in cell.items() if pp == p]
        N.num(f"t3.sec.{p}", np.nanmin(secs) if secs else np.nan, srcs, st, fmt="{:.3g}")
    # verdict per PLAN §2.4, mechanical, on the decision-relevant stratum
    a2, s2 = cell.get(("amortized", "T2", "rel")), cell.get(("smc", "T2", "rel"))
    if a2 and s2:
        N.set("t3.verdict", verdict(a2, s2), srcs, st)
    else:
        N.ph("t3.verdict")
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
