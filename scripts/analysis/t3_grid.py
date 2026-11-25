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
POST = (("smc", "GPU-SMC (T1 likelihood)"), ("amortized", "Amortized (wide DR)"))


def run(res, paper):
    N = C.Numbers("t3_grid.py")
    dec, srcs = C.load("dec", res)
    out = paper / "tables/t3_grid.tex"
    if not dec.empty:
        dec = dec[(dec.split == "test") & dec.posterior.isin([p for p, _ in POST]) & (dec.route == ROUTE)]
    if dec.empty:
        C.placeholder_table(out, "tThreeTable", 7, f"awaiting dec rows posterior smc/amortized, route {ROUTE}")
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
                raise C.SchemaError(f"T3: several methods for posterior={p}, route={ROUTE}: {sorted(d.method.unique())}")
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
    C.write_table(out, "\n".join(lines), srcs, f"route {ROUTE}, nominal alpha [{st}]")
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
        rob = a2["unsafe_commit"] <= s2["unsafe_commit"] and a2["cert_frac"] >= s2["cert_frac"]
        fast = bool(np.isfinite(a2["sec"]) and np.isfinite(s2["sec"]) and a2["sec"] < s2["sec"])
        v = ("amortized wins on T2 robustness and latency" if rob and fast else
             "amortized wins on latency only" if fast else
             "amortized wins on T2 robustness only" if rob else "SMC wins on both: drop 'amortized'")
        N.set("t3.verdict", v, srcs, st)
    else:
        N.ph("t3.verdict")
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
