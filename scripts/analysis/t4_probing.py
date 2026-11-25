"""T4, decision metrics per probe policy (fixed v2, random, EIG, EPIG, decision-EVOI, always split, ...).

Input: contract 'probe' (one row per episode x policy, same episodes for every policy; pairs = (candidate, task),
labels rho_dec). Per (test_plant, policy): certified fraction and unsafe-commit rate (CI over episodes) in the
decision-relevant stratum (HEADLINE; columns n_*_rel: pairs whose candidate holds a faulted module or whose label is
within REL_BAND of the task requirement) and over all pairs; share of v2-ambiguous pairs resolved; mean probe time;
ambiguity resolved per extra probe-minute vs fixed_v2; abstain rate. The table shows the T1 plant if present, else
the first plant found (e.g. the decision role's T0 pilot), and says which.
Stage A3 found too few ambiguous decisions to test probing (results/stageA/A3_verdict.json), so this table may land
in the appendix (docs/board/paper.md, claim C4).
The headline table is test_plant == HEADLINE_PLANT ("T0-knee-post-ktol", decision's probe_t0n1bK.csv) when present;
every other plant (the *-appendix ablations "T0-knee-post-K5-appendix", "T0-knee-truemodes-appendix", and older
plants) goes to t4_probing_appendix.tex.
Research-lead ruling (23:00Z): the HEADLINE columns are useful commits (sum n_served / sum oracle_served), regret
(mean, s) with the paired difference vs always-clamp (fixed_v2_clamp) and vs the dominance trigger (dominance_clamp)
[95 % episode-paired bootstrap CI], the episode-unsafe rate E_ep (share of episodes with an unsafe assignment,
n_unsafe_assigned > 0, else n_unsafe > 0) [CI], and probe seconds; plus, when decision provides them, the
task-relevant certified share (n_certified_task / n_capable_task). The candidate-level certified share moves to the
secondary table t4_probing_candidates.tex, labelled "certifies what the task needs, not every candidate". If decision
provides `clamp_bought` (per episode: clamp's extra certifications bought utility), the count is a macro.
Outputs: paper/tables/t4_probing.tex (headline), t4_probing_candidates.tex (secondary), t4_probing_appendix.tex,
t4_probing.csv, macros cfTFour* (unsuffixed = rel).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
from t1_certificate import sfx

HEADLINE_PLANT = "T0-knee-post-ktol"     # decision role's headline (probe_t0n1bK.csv); *-appendix plants -> appendix
NAMES = {"fixed_v2": "Fixed v2", "fixed_v2_clamp": "Fixed v2 + clamp", "dominance_clamp": "Dominance + clamp",
         "fixed_v2_clamp_moddiff": "Fixed v2 + clamp + module-differential", "fixed_all": "Fixed full library", "random": "Random probe",
         "eig": "EIG (parameters)", "epig": "EPIG (predictions)", "evoi": "Decision-EVOI (ours)",
         "always_split": "Always split first", "dad": "Amortized policy (DAD/TNDP)",
         "oracle_probe": "Oracle probe choice"}
ORDER = list(NAMES)
REFS = (("fixed_v2_clamp", "clamp"), ("dominance_clamp", "dominance"))   # paired regret references
B_PAIR = 2000
KEY = ("t4.useful.evoi", "t4.regret.evoi", "t4.epunsafe.evoi", "t4.dregret.evoi.clamp", "t4.dregret.evoi.dominance",
       "t4.resolved.evoi", "t4.resolved.epig", "t4.unsafe.evoi", "t4.cert.evoi", "t4.permin.evoi",
       "t4.namb.fixedv2", "t4.plant")


def _ep_unsafe(d):
    col = "n_unsafe_assigned" if "n_unsafe_assigned" in d and d["n_unsafe_assigned"].notna().any() else "n_unsafe"
    return (d[col].fillna(0) > 0).astype(float).to_numpy()


def _mean_ci(x, B=B_PAIR, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    reps = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), (float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975)))


def paired_regret(Pp, pol, ref):
    """Mean paired regret difference (pol - ref) over episodes both policies ran, with a 95 % bootstrap CI."""
    if "regret" not in Pp or Pp["regret"].isna().all():
        return np.nan, (np.nan, np.nan), 0
    a = Pp[Pp.policy == pol].groupby("eid").regret.mean()
    b = Pp[Pp.policy == ref].groupby("eid").regret.mean()
    j = a.index.intersection(b.index)
    if not len(j) or pol == ref:
        return np.nan, (np.nan, np.nan), 0
    est, ci = _mean_ci((a[j] - b[j]).to_numpy())
    return est, ci, len(j)


def _ratio_ci(d, num, den):
    if num not in d or d[num].isna().all():
        return np.nan, (np.nan, np.nan)
    return C._ratio(d[num].sum(), d[den].sum()), C.boot_ratio(d[num].to_numpy(float), d[den].to_numpy(float))


def _appendix(Ra, P, srcs, path):
    """Every non-headline plant (the *-appendix ablations, and older plants such as T0), labelled by plant."""
    if Ra.empty:
        C.placeholder_table(path, "tFourAppendix", 7, "no appendix plants")
        return
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Plant & policy & probe time (s) & ambiguity resolved (\%) & cert.\ (\%) & unsafe (\%) & "
             r"cert.\ all (\%)\\", r"\midrule"]
    for pl, d in Ra.groupby("plant", sort=True):
        for _, r in d.iterrows():
            nm = NAMES.get(r.policy, r.policy.replace("_", r"\_"))
            lines.append(f"{pl} & {nm} & {r.t:.1f} & {C.fmt_pct(r.resolved)} & {C.fmt_pct(r.cert)} & "
                         f"{C.fmt_pct(r.unsafe, 2)} & {C.fmt_pct(r.cert_all)}\\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    C.write_table(path, "\n".join(lines), srcs, "appendix plants (ablations); decision-relevant stratum + all")


def run(res, paper):
    N = C.Numbers("t4_probing.py")
    P, srcs = C.load("probe", res)
    out = paper / "tables/t4_probing.tex"
    if not P.empty:
        P = P[P.split == "test"].copy()
        P["test_plant"] = P["test_plant"].fillna("T1").astype(str)
    if P.empty:
        C.placeholder_table(out, "tFourTable", 8, "awaiting probe_*.csv (decision role)")
        for k in KEY:
            N.ph(k)
        N.ph("t4.cert.evoi.all")
        return N
    plants = sorted(P.test_plant.unique(), key=lambda p: (p != "T1", p))
    rows = []
    for (plant, pol), d in P.groupby(["test_plant", "policy"]):
        base_t = P[(P.test_plant == plant) & (P.policy == "fixed_v2")].probe_time_s.mean()
        amb_b, amb_a = d.n_amb_before.sum(), d.n_amb_after.sum()
        res_ = 1 - amb_a / amb_b if amb_b else np.nan
        extra_min = (d.probe_time_s.mean() - base_t) / 60 if np.isfinite(base_t) else np.nan
        cert_r, _ = _ratio_ci(d, "n_certified_rel", "n_capable_rel")
        uns_r, uns_r_ci = _ratio_ci(d, "n_unsafe_rel", "n_incapable_rel")
        rows.append(dict(plant=plant, policy=pol, n_ep=len(d), status=C.status_of(d),
                         cert=cert_r, unsafe=uns_r, unsafe_ci=uns_r_ci,
                         cert_all=C._ratio(d.n_certified.sum(), d.n_capable.sum()),
                         unsafe_all=C._ratio(d.n_unsafe.sum(), d.n_incapable.sum()),
                         resolved=res_, n_amb=amb_b, t=d.probe_time_s.mean(),
                         per_min=(res_ / extra_min) if extra_min and extra_min > 0 and np.isfinite(res_) else np.nan,
                         utility=d.utility.mean(), abstain=(d.action == "abstain").mean(),
                         useful=(C._ratio(d.n_served.sum(), d.oracle_served.sum())
                                 if "n_served" in d and d.n_served.notna().any() else np.nan),
                         regret=d.regret.mean() if "regret" in d and d.regret.notna().any() else np.nan,
                         ep_unsafe=_mean_ci(_ep_unsafe(d))[0], ep_unsafe_ci=_mean_ci(_ep_unsafe(d))[1],
                         cert_task=(C._ratio(d.n_certified_task.sum(), d.n_capable_task.sum())
                                    if "n_certified_task" in d and d.n_certified_task.notna().any() else np.nan)))
    R = pd.DataFrame(rows)
    R["o"] = R.policy.map(lambda p: ORDER.index(p) if p in ORDER else 99)
    R = R.sort_values(["plant", "o", "policy"])
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    R.drop(columns="o").to_csv(paper / "tables/t4_probing.csv", index=False)
    plant = HEADLINE_PLANT if HEADLINE_PLANT in plants else plants[0]
    Rp = R[R.plant == plant]
    st = C.status_of(P[P.test_plant == plant])
    if "_src" in P:
        srcs = sorted(P[P.test_plant == plant]["_src"].unique())           # headline numbers cite only their file
    _appendix(R[R.plant != plant], P, sorted(P[P.test_plant != plant]["_src"].unique()) if "_src" in P else srcs,
              paper / "tables/t4_probing_appendix.tex")
    Pp = P[P.test_plant == plant]
    dreg = {}
    for pol in Rp.policy:
        for ref, tag in REFS:
            dreg[pol, tag] = paired_regret(Pp, pol, ref)
    has_task = Rp.cert_task.notna().any()
    ci = lambda c: (f"[{c[0]:+.1f}, {c[1]:+.1f}]" if np.isfinite(c[0]) else "")              # noqa: E731
    hdr = (r"Policy & useful commits (\%) & regret (s) & $\Delta$ regret vs always-clamp & $\Delta$ regret vs "
           r"dominance trigger & $E_{\mathrm{ep}}$ (\%) & probe (s)" + (r" & task-relevant cert.\ (\%)" if has_task
                                                                        else "") + r"\\")
    lines = [r"\begin{tabular}{l" + "r" * (7 if has_task else 6) + "}", r"\toprule", hdr, r"\midrule"]
    for _, r in Rp.iterrows():
        nm = NAMES.get(r.policy, r.policy.replace("_", r"\_"))
        cells = [nm, C.fmt_pct(r.useful), f"{r.regret:.1f}" if np.isfinite(r.regret) else r"\textemdash"]
        for _, tag in REFS:
            e, c, _n = dreg[r.policy, tag]
            cells.append(f"{e:+.1f} {ci(c)}" if np.isfinite(e) else r"\textemdash")
        cells += [f"{C.fmt_pct(r.ep_unsafe, 1)} [{C.fmt_pct(r.ep_unsafe_ci[0], 1)}, {C.fmt_pct(r.ep_unsafe_ci[1], 1)}]",
                  f"{r.t:.1f}"]
        if has_task:
            cells.append(C.fmt_pct(r.cert_task))
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(out, "\n".join(lines), srcs, f"plant {plant}; headline (research-lead ruling); paired by episode [{st}]")
    # secondary: candidate-level certification
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{decision-relevant} & all & \\",
             r"Policy & ambiguity resolved (\%) & per extra min & cert.\ (\%) & unsafe (\%) & cert.\ (\%) & "
             r"abstain (\%)\\", r"\midrule"]
    for _, r in Rp.iterrows():
        nm = NAMES.get(r.policy, r.policy.replace("_", r"\_"))
        pm = f"{100 * r.per_min:.0f}" if np.isfinite(r.per_min) else r"\textemdash"
        lines.append(f"{nm} & {C.fmt_pct(r.resolved)} & {pm} & {C.fmt_pct(r.cert)} & {C.fmt_pct(r.unsafe, 2)} & "
                     f"{C.fmt_pct(r.cert_all)} & {C.fmt_pct(r.abstain)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    C.write_table(paper / "tables/t4_probing_candidates.tex", "\n".join(lines), srcs,
                  "secondary: candidate-level certified share -- certifies what the task needs, not every candidate")
    for _, r in Rp.iterrows():
        p = r.policy
        N.pct(f"t4.useful.{p}", r.useful, srcs, st)
        N.num(f"t4.regret.{p}", r.regret, srcs, st, fmt="{:.1f}")
        N.pct(f"t4.epunsafe.{p}", r.ep_unsafe, srcs, st, d=1)
        N.pct(f"t4.epunsafehi.{p}", r.ep_unsafe_ci[1], srcs, st, d=1)
        if np.isfinite(r.cert_task):
            N.pct(f"t4.certtask.{p}", r.cert_task, srcs, st)
        for _, tag in REFS:
            e, c, _n = dreg[p, tag]
            if np.isfinite(e):
                N.num(f"t4.dregret.{p}.{tag}", e, srcs, st, fmt="{:+.1f}")
                N.num(f"t4.dregretlo.{p}.{tag}", c[0], srcs, st, fmt="{:+.1f}")
                N.num(f"t4.dregrethi.{p}.{tag}", c[1], srcs, st, fmt="{:+.1f}")
    if "clamp_bought" in Pp and Pp.clamp_bought.notna().any():
        cb = Pp[Pp.policy == "fixed_v2_clamp"] if (Pp.policy == "fixed_v2_clamp").any() else Pp
        cb = cb.dropna(subset=["clamp_bought"]).drop_duplicates("eid")
        N.set("t4.clamp.bought", f"{int(cb.clamp_bought.sum())}/{len(cb)}", srcs, st)
    else:
        N.ph("t4.clamp.bought", "needs decision's clamp_bought column")
    N.set("t4.plant", plant, srcs, st)
    for _, r in Rp.iterrows():
        p = r.policy
        N.pct(f"t4.resolved.{p}", r.resolved, srcs, st)
        N.pct(f"t4.unsafe.{p}", r.unsafe, srcs, st, d=2)
        N.pct(f"t4.cert.{p}", r.cert, srcs, st)
        N.pct(f"t4.unsafe.{p}{sfx('all')}", r.unsafe_all, srcs, st, d=2)
        N.pct(f"t4.cert.{p}{sfx('all')}", r.cert_all, srcs, st)
        N.num(f"t4.time.{p}", r.t, srcs, st, fmt="{:.1f}")
        if p != "fixed_v2":
            N.pct(f"t4.permin.{p}", r.per_min, srcs, st, d=0)
    v2 = Rp[Rp.policy == "fixed_v2"]
    N.num("t4.namb.fixedv2", float(v2.n_amb.iloc[0]) if len(v2) else np.nan, srcs, st)
    for k in KEY:
        if C.macro_name(k) not in N.items:
            N.ph(k)
    if C.macro_name("t4.cert.evoi.all") not in N.items:
        N.ph("t4.cert.evoi.all")
    return N


if __name__ == "__main__":
    res, paper = C.env_paths()
    for k, v in run(res, paper).items.items():
        print(k, v[0])
