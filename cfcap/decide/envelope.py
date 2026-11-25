"""Decision layer under the model-free envelope class (theory E1; cfcap/theory_task.py:gate_episode).

Input: gate_episode rows (one per parent seed x child x leg): rho_true (knee-law truth, severity seed % 4),
rho_par_<p> (two-constant fit at probe set p) and rho_pi_<p> (identified-set low end over the speed-ceiling class,
model-free) for p in {v2, v2clamp}, has_fault, cand_size, undom_speed_<p>.

Episode = one parent seed: candidates = its connected children (no whole parent, no merges in E1), tasks from
problem.make_tasks (seed stream 959000 + seed, T0 leg order). Predictors (lower bounds per (cand, leg)):
  env      rho_pi_<p>                 model-free; valid whenever the class contains the truth (no calibration);
  env_cal  rho_pi_<p> - q             q = end-to-end simultaneous margin (simA score: unnormalised) on cal folds;
  par      rho_par_<p>                plug-in physics fit (trusts the two-constant law);
  par_cal  rho_par_<p> - q            same calibration.
Probe policies (probe library: v2 only, or + clamp at 8.8 s):
  v2       never probe;
  clamp    always probe (fixed v2+clamp);
  evoi     probe clamp iff E[V x certified tasks after clamp] - V x certified tasks now > t_clamp, the expectation over
           a cross-fitted empirical predictive of the bound's gain Delta = lower_clamp - lower_v2, drawn from the
           calibration folds' entries with the same (leg kind, child size >= 3, v2-undominated) cell (S draws, the
           draws of one episode share one donor episode per cell, preserving within-cell correlation roughly).
Each (predictor, policy) is scored with problem.decide / outcome against rho_true; time = probe time + split time.
Strata: all, and pairs whose task needs a yaw leg on a child of >= 3 modules (Prop. 2b: undominated by translation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfcap.conformal import core

from . import evaluate as Ev
from . import problem as Pb

T_CLAMP = 8.8
TASK_SEED0 = 959000
LEG_KIND = np.array(["surge", "surge", "sway", "sway", "diag", "diag", "diag", "diag", "yaw", "yaw"])


class _C:
    def __init__(self, modules, size, name):
        self.kind, self.modules, self.size, self.name = "child", modules, size, name


def _modules(name):
    """'parentK/c0-2' -> (0, 2)."""
    return tuple(int(x) for x in str(name).rsplit("/c", 1)[1].split("-"))


def episodes_from_gate(df, legs):
    eps = []
    for seed, g in df.groupby("seed"):
        names = list(dict.fromkeys(g.cand))
        sizes = g.groupby("cand", sort=False).cand_size.first().reindex(names).to_numpy()
        mods = [_modules(n) for n in names]
        m = max(max(t) for t in mods) + 1
        cells = [(k, 0) for k in range(m)]               # cells only feed split-time leftovers; E1 children are
        inst = Pb.Instance.from_candidates(cells, [_C(t, int(s), n) for t, s, n in zip(mods, sizes, names)],
                                           include_parent=False)
        arr = lambda col: np.stack([g[g.cand == n].sort_values("leg")[col].to_numpy(float) for n in names])
        e = dict(inst=inst, rho_true=arr("rho_true"), has_fault=np.array([bool(g[g.cand == n].has_fault.iloc[0])
                                                                           for n in names]),
                 sizes=sizes, law=int(g.law.iloc[0]), seed=int(seed), fold=int(seed) % 3,
                 tasks=Pb.make_tasks(np.random.default_rng(TASK_SEED0 + int(seed)), legs),
                 meta=dict(seed=int(seed), parent_size=m, n_faults=int(g.has_fault.max())))
        for p in ("v2", "v2clamp"):
            for c in ("rho_par", "rho_pi", "undom_speed", "rho_par_hi"):
                if f"{c}_{p}" in g:
                    e[f"{c}_{p}"] = arr(f"{c}_{p}")
            e.setdefault(f"rho_par_hi_{p}", e[f"rho_par_{p}"])
        eps.append(e)
    return eps


def _margin(eps, ca, col):
    pts = [eps[i][col] for i in ca]
    return Ev.fit_q(pts, [np.ones_like(p) for p in pts], [eps[i] for i in ca], 0.1)


def _cell(e):
    big = (e["sizes"] >= 3)[:, None]
    und = e["undom_speed_v2"] > 0.5
    kind = np.broadcast_to(LEG_KIND[None], e["rho_true"].shape)
    return np.char.add(np.char.add(kind, np.where(big, "B", "s")), np.where(und, "U", "d"))


def _certified_tasks(e, lower):
    ch, tc = Pb.decide(e["inst"], e["tasks"], lower)
    return int((ch >= 0).sum()), ch, tc


def run(eps, preds=("env", "env_cal", "par", "par_cal"), S=32, seed=0, cal_eps=None):
    """cal_eps: optional external calibration episodes (held-out law check): q and the gain predictive come from them
    and every episode of eps is a test episode (no cross-fit)."""
    if cal_eps is not None:
        allx = list(eps) + list(cal_eps)
        idx = {0: list(range(len(eps))), 1: list(range(len(eps), len(allx)))}
        plan = {0: (None, [1])}
        eps = allx
    else:
        plan = Ev.fold_plan(np.array([e["fold"] for e in eps]))
        idx = {f: [i for i, e in enumerate(eps) if e["fold"] == f] for f in plan}
    rows = []
    rng = np.random.default_rng(seed)
    for f, (h, C) in plan.items():
        ca = [i for c in C for i in idx[c]]
        base = {}
        for pred in preds:
            src = "rho_pi" if pred.startswith("env") else "rho_par"
            q = {p: (_margin(eps, ca, f"{src}_{p}") if pred.endswith("_cal") else 0.0) for p in ("v2", "v2clamp")}
            # empirical predictive of the gain, by cell, from the calibration folds
            pool = {}
            for i in ca:
                e = eps[i]
                with np.errstate(invalid="ignore"):
                    d = (e[f"{src}_v2clamp"] - q["v2clamp"]) - (e[f"{src}_v2"] - q["v2"])
                d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                cl = _cell(e)
                for k in np.unique(cl):
                    pool.setdefault(k, []).append(d[cl == k])
            for i in idx[f]:
                e = eps[i]
                lo = {p: np.where(np.isfinite(e[f"{src}_{p}"] - q[p]), e[f"{src}_{p}"] - q[p], -np.inf)
                      for p in ("v2", "v2clamp")}
                n0, _, _ = _certified_tasks(e, lo["v2"])
                cl = _cell(e)
                gain = 0.0
                if n0 < len(e["tasks"]):
                    for _ in range(S):
                        dd = np.zeros_like(lo["v2"])
                        for k in np.unique(cl):
                            pk = pool.get(k)
                            if pk:
                                donor = pk[int(rng.integers(len(pk)))]
                                dd[cl == k] = rng.choice(donor, size=int((cl == k).sum()), replace=True)
                        gain += _certified_tasks(e, lo["v2"] + dd)[0] - n0
                    gain /= S
                evoi = Pb.V_TASK * gain - T_CLAMP
                # dominance trigger (closed-form rival): clamp iff some required, v2-undominated leg of a size-feasible
                # candidate has its interval [lower_v2, rho_par_hi_v2] straddling the requirement
                R, mm, _ = Pb.task_arrays(e["tasks"], 10)
                Rc = Pb.req_ctl(R, len(e["sizes"]))                               # (C, T, L)
                okc = (e["sizes"][:, None] >= mm[None, :])[..., None]
                und = (e["undom_speed_v2"] > 0.5)[:, None, :]
                hi = np.nan_to_num(e["rho_par_hi_v2"], nan=np.inf)[:, None, :]
                strad = okc & und & ~np.isnan(Rc) & (lo["v2"][:, None, :] < Rc) & (hi >= Rc)
                dom = bool(strad.any())
                for pol, use in (("v2", "v2"), ("clamp", "v2clamp"), ("evoi", "v2clamp" if evoi > 0 else "v2"),
                                 ("dom", "v2clamp" if dom else "v2")):
                    _, ch, tc = _certified_tasks(e, lo[use])
                    tp = T_CLAMP if use == "v2clamp" else 0.0
                    o = Pb.outcome(e["inst"], e["tasks"], ch, tc, e["rho_true"], tp)
                    orc = Pb.oracle(e["inst"], e["tasks"], e["rho_true"])[2]
                    pc = Ev.pair_counts(e, lo[use], np.ones(1), np.ones(1), e["rho_true"][None])
                    R, mm, _ = Pb.task_arrays(e["tasks"], 10)
                    yawbig = (np.isfinite(R[:, 8:]).any(1)[None, :] & (e["sizes"] >= 3)[:, None])
                    cap = Pb.capable(e["rho_true"], R, mm, e["sizes"])
                    cer = Pb.certified(lo[use], R, mm, e["sizes"])
                    och = Pb.oracle(e["inst"], e["tasks"], e["rho_true"])[0]       # oracle plan's (cand, task) pairs
                    opairs = [(int(c), t) for t, c in enumerate(och) if c >= 0]
                    rows.append(dict(seed=e["seed"], law=e["law"], fold=f, predictor=pred, policy=pol, probed=int(tp > 0),
                                     regret=orc["U"] - o["U"], oracle_served=orc["n_served"], **o,
                                     n_capable=pc["n_capable"], n_incapable=pc["n_incapable"],
                                     n_certified=pc["n_certified"], n_unsafe_cert=pc["n_unsafe_cert"],
                                     n_dec=pc["n_dec"], n_capable_rel=pc["n_capable_rel"],
                                     n_incapable_rel=pc["n_incapable_rel"], n_certified_rel=pc["n_certified_rel"],
                                     n_unsafe_rel=pc["n_unsafe_rel"], parent_size=e["meta"]["parent_size"],
                                     yb_capable=int((cap & yawbig).sum()), yb_certified=int((cer & cap & yawbig).sum()),
                                     yb_unsafe=int((cer & ~cap & yawbig).sum()), yb_incapable=int((~cap & yawbig).sum()),
                                     q=q[use], oracle_pairs=len(opairs),
                                     oracle_pairs_cert=int(sum(cer[c, t] for c, t in opairs))))
    return pd.DataFrame(rows)


def summarise(df):
    g = df.groupby(["predictor", "policy"])
    return g.apply(lambda d: pd.Series(dict(
        n_ep=len(d), ep_any_unsafe=(d.n_unsafe > 0).mean(),
        unsafe_commit_rate=d.n_unsafe.sum() / max(d.n_assigned.sum(), 1),
        useful=d.n_served.sum() / max(d.oracle_served.sum(), 1), abstain=d.abstain.mean(),
        regret=d.regret.mean(), regret_se=d.regret.std(ddof=1) / np.sqrt(len(d)), probe_s=d.probe_s.mean(),
        probed_share=d.probed.mean(),
        cert_pairs=d.n_certified.sum() / max(d.n_capable.sum(), 1),
        unsafe_pairs=d.n_unsafe_cert.sum() / max(d.n_incapable.sum(), 1),
        task_rel_cert=d.oracle_pairs_cert.sum() / max(d.oracle_pairs.sum(), 1),
        yawbig_cert=d.yb_certified.sum() / max(d.yb_capable.sum(), 1),
        yawbig_unsafe=d.yb_unsafe.sum() / max(d.yb_incapable.sum(), 1))), include_groups=False).reset_index()


PAPER_POLICY = {"v2": "fixed_v2", "clamp": "fixed_v2_clamp", "evoi": "evoi", "dom": "dominance_clamp"}


def write_paper(rows, path, pred="env_cal", v2_s=12.0, test_plant="T0-knee", status="exploratory"):
    """T4 contract rows (docs/board/paper.md 1b) for one envelope predictor; predictor_class = envelope. Ambiguity
    is undefined for a bound-only predictor (n_amb_* = NaN)."""
    d = rows[rows.predictor == pred]
    out = pd.DataFrame(dict(policy=d.policy.map(PAPER_POLICY), predictor_class="envelope", predictor=pred,
                            split="test", status=status, eid=d.seed, shape=d.parent_size.map(lambda m: f"{m}mod"),
                            n_probes=d.probed, probe_time_s=d.probe_s + v2_s,
                            action=np.where(d.abstain > 0, "abstain", "commit"), n_dec=d.n_dec,
                            n_capable=d.n_capable, n_incapable=d.n_incapable, n_certified=d.n_certified,
                            n_unsafe=d.n_unsafe_cert, n_amb_before=np.nan, n_amb_after=np.nan, utility=d.U,
                            test_plant=test_plant, regret=d.regret, n_unsafe_assigned=d.n_unsafe, law=d.law,
                            n_capable_rel=d.n_capable_rel, n_incapable_rel=d.n_incapable_rel,
                            n_certified_rel=d.n_certified_rel, n_unsafe_rel=d.n_unsafe_rel,
                            oracle_pairs=d.oracle_pairs, oracle_pairs_cert=d.oracle_pairs_cert,
                            task_rel_cert_share=d.oracle_pairs_cert / d.oracle_pairs.where(d.oracle_pairs > 0)))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def paired(rows, pred, a, b, col="regret", B=4000, seed=0):
    """Paired episode bootstrap of mean(col | policy a) - mean(col | policy b) for one predictor: (mean, lo, hi)."""
    d = rows[rows.predictor == pred].pivot_table(index="seed", columns="policy", values=col)
    x = (d[a] - d[b]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), len(x)
