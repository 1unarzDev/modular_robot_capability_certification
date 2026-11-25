"""T3, the 2x2 grid {GPU-SMC with the T1 simulator as likelihood, amortized posterior trained with wide DR} x
{test on T1, test on T2}. This is the test of the ML claim (PLAN §2 contribution 4): amortization must win on
misspecification robustness (T2 column) and/or decision latency; if SMC wins on both, the word "amortized" goes.

Input: contract 'dec' rows with posterior in {smc, amortized}, split test, the route given by ROUTE (simultaneous,
the guarantee both share). Per cell: unsafe-commit (CI over episodes), episodes with any unsafe go, certified fraction
at the nominal bound, median seconds per decision. Outputs: paper/tables/t3_grid.tex, macros cfTThree*.
"""
from __future__ import annotations

import numpy as np

import common as C

ROUTE = "simultaneous"
POST = (("smc", "GPU-SMC (T1 likelihood)"), ("amortized", "Amortized (wide DR)"))


def run(res, paper):
    N = C.Numbers("t3_grid.py")
    dec, srcs = C.load("dec", res)
    out = paper / "tables/t3_grid.tex"
    if not dec.empty:
        dec = dec[(dec.split == "test") & dec.posterior.isin([p for p, _ in POST]) & (dec.route == ROUTE)]
    if dec.empty:
        C.placeholder_table(out, "tThreeTable", 5, f"awaiting dec rows posterior smc/amortized, route {ROUTE}")
        for p, _ in POST:
            for pl in ("T1", "T2"):
                for k in ("unsafe", "cert", "epunsafe"):
                    N.ph(f"t3.{k}.{p}.{pl}")
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
            c = C.decision_counts(d)
            cnt = C.ep_counts(d, d["lower"].to_numpy(float))
            c["ci"] = C.boot_ratio(cnt["uns"], cnt["inc"])
            c["sec"] = float(np.nanmedian(d.drop_duplicates("eid")["seconds"])) if d["seconds"].notna().any() else np.nan
            cell[p, pl] = c
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{test T1} & \multicolumn{2}{c}{test T2}\\",
             r"Posterior & unsafe (\%) & cert.\ (\%) & unsafe (\%) & cert.\ (\%)\\", r"\midrule"]
    for p, label in POST:
        cells = [label]
        for pl in ("T1", "T2"):
            c = cell.get((p, pl))
            if c is None:
                cells += [r"\textemdash"] * 2
            else:
                cells += [f"{C.fmt_pct(c['unsafe_commit'], 2)} [{C.fmt_pct(c['ci'][0], 2)}, {C.fmt_pct(c['ci'][1], 2)}]",
                          C.fmt_pct(c["cert_frac"])]
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(out, "\n".join(lines), srcs, f"route {ROUTE}, nominal alpha [{st}]")
    for (p, pl), c in cell.items():
        N.pct(f"t3.unsafe.{p}.{pl}", c["unsafe_commit"], srcs, st, d=2)
        N.pct(f"t3.cert.{p}.{pl}", c["cert_frac"], srcs, st)
        N.pct(f"t3.epunsafe.{p}.{pl}", c["ep_any_unsafe"], srcs, st)
    for p, _ in POST:
        secs = [c["sec"] for (pp, _), c in cell.items() if pp == p]
        N.num(f"t3.sec.{p}", np.nanmin(secs) if secs else np.nan, srcs, st, fmt="{:.3g}")
    # verdict per PLAN §2.4, mechanical: amortized must win on T2 unsafe or on latency
    a2, s2 = cell.get(("amortized", "T2")), cell.get(("smc", "T2"))
    if a2 and s2:
        rob = a2["unsafe_commit"] <= s2["unsafe_commit"] and a2["cert_frac"] >= s2["cert_frac"]
        lat = [c["sec"] for (pp, _), c in cell.items() if pp == "amortized"]
        lsm = [c["sec"] for (pp, _), c in cell.items() if pp == "smc"]
        fast = bool(lat and lsm and np.nanmin(lat) < np.nanmin(lsm))
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
