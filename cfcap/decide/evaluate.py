"""End-to-end evaluation of (predictor, probe policy, certifier) on decision episodes; shared by the T0 pilot script and
the T1 runner task.

An episode dict carries: inst (problem.Instance), post (posterior.DecisionPosterior incl. probes with y_true),
rho_true (C, L), rho_nom (C, L), tasks (list of problem.Task), fold (int), meta.

Certifiers (all commit only certified pairs; cross-fitted by `fold_plan`: an episode's q / lambda never depends on the
labels of its own fold, also not through the in-loop q that steers adaptive probing):
  sim     SimultaneousLower (alpha) on the episode's decision-relevant entries (every candidate x every leg some task
          requires), fitted on the policy's *final* posteriors (end-to-end over the probe policy); in-loop lookahead
          uses q fitted on the v2 posteriors.
  crc     conformal risk control (`risk_control_margin`) on the decision loss: lower = point - lambda x scale, loss =
          the episode's unsafe-commit indicator of the certified decision, monotonised in lambda (max over
          lambda' >= lambda, so the CRC premise holds and the bound covers the raw loss), probe trajectory fixed.
  simF    (fix for the round-0 review item 3) hybrid score with a calibration-set scale floor: scale' = max(scale,
          sigma_min), sigma_min = the unnormalised simultaneous margin = (1 - alpha) split quantile of the per-episode
          max |point - truth| over decision-relevant entries, fitted on one half of the calibration folds; q of the
          normalised score with scale' fitted on the other half (so sigma_min is a fixed function when q is fitted).
          A collapsed posterior (sd at the 0.02 floor, nuisance-biased point) can then never inflate q beyond
          margin / sigma_min ~ 1. In-loop lookahead uses the same certifier fitted on the v2 posteriors.
  simA    unnormalised score (scale = 1): lower = point - q.
  raw     q = 0 (plug-in posterior mean; no guarantee).
  bayes05 the raw posterior 5 % quantile (uncalibrated Bayes lower bound; no guarantee).
"""
from __future__ import annotations

import numpy as np

from cfcap.conformal import core

from . import policies as Pol
from . import problem as Pb

LAMBDAS = np.unique(np.r_[np.linspace(0, 5, 101), np.linspace(5, 60, 111)])


def relevant(ep):
    """(C, L) bool: entries whose leg some task of the episode requires."""
    R, _, _ = Pb.task_arrays(ep["tasks"], ep["rho_true"].shape[-1])
    legs = np.isfinite(R).any(0)
    return np.broadcast_to(legs[None], ep["rho_true"].shape)


def fit_q(points, scales, eps, alpha):
    """SimultaneousLower q over episodes (lists of (C_i, L) arrays, ragged C) on decision-relevant entries."""
    if not eps:
        return np.inf
    W = max(p.size for p in points)
    P = np.full((len(eps), W), np.nan); S = np.ones((len(eps), W)); Tr = np.full((len(eps), W), np.nan)
    for i, (p, s, e) in enumerate(zip(points, scales, eps)):
        m = relevant(e)
        pp = np.where(m, p, np.nan).ravel()
        P[i, :pp.size], S[i, :pp.size], Tr[i, :pp.size] = pp, s.ravel(), e["rho_true"].ravel()
    return core.SimultaneousLower(alpha).fit(P, S, Tr).q


def margin_abs(points, eps, alpha):
    """Unnormalised simultaneous margin: split quantile of per-episode max (point - truth)_+ over relevant entries."""
    sc = []
    for p, e in zip(points, eps):
        r = np.where(relevant(e), np.asarray(p, float) - e["rho_true"], np.nan)
        present = ~np.isnan(r)
        sc.append(float(np.max(r[present])) if present.any() else -np.inf)
    sc = np.asarray(sc, float)
    return core.split_quantile(sc[np.isfinite(sc) | (sc == np.inf)], alpha)


def fit_floor(points, scales, eps, ca, alpha):
    """simF: (q, sigma_min) from calibration indices ca: sigma_min on ca[0::2], q on ca[1::2]."""
    a, b = ca[0::2], ca[1::2]
    smin = margin_abs([points[i] for i in a], [eps[i] for i in a], alpha)
    if not np.isfinite(smin):
        return (np.inf, 1.0)
    smin = max(smin, 1e-3)
    q = fit_q([points[i] for i in b], [np.maximum(scales[i], smin) for i in b], [eps[i] for i in b], alpha)
    return (q, smin)


def crossfit(values_by_ep, folds, fit):
    """fit(list of indices) -> parameter; returns per-episode parameter from the other folds."""
    out = np.empty(len(folds), dtype=object)
    for f in np.unique(folds):
        te = np.flatnonzero(folds == f)
        ca = np.flatnonzero(folds != f)
        v = fit(ca)
        for i in te:
            out[i] = v
    return out


def episode_loss_curve(ep, point, scale, lambdas=LAMBDAS):
    """Monotonised unsafe-commit indicator of the certified decision for every lambda."""
    raw = np.empty(len(lambdas))
    for k, lam in enumerate(lambdas):
        ch, tc = Pb.decide(ep["inst"], ep["tasks"], np.where(np.isfinite(point), point - lam * scale, -np.inf))
        raw[k] = Pb.outcome(ep["inst"], ep["tasks"], ch, tc, ep["rho_true"])["n_unsafe"] > 0
    return np.maximum.accumulate(raw[::-1])[::-1]


def _run_one(job):
    pol, e, q, seed, max_probes, S = job
    qq, smin = q if isinstance(q, tuple) else (q, 0.0)
    ch, tc, tp, used, post = Pol.run(pol, e["inst"], e["tasks"], e["post"], Pol.Certifier(qq, floor=smin),
                                     np.random.default_rng(list(seed)), max_probes=max_probes, S=S)
    pt, sc = post.summary()
    return dict(point=pt, scale=sc, probe_s=tp, used=used, w=post.w, unknown=post.unknown)


def fold_plan(folds):
    """Cross-fit plan without label leakage. For each test fold f: an in-loop fold h (its v2 posteriors fit the q the
    policy uses for lookahead and stopping) and calibration folds C (their *final* posteriors, run with h's q, fit the
    end-to-end certificate applied to f). With K >= 3 folds h = the fold after next and C = the rest, so the test and
    calibration trajectories are the same function of fold-h data and exchangeable; with K = 2 the single other fold
    is both h and C (no test-label leakage; calibration-fold trajectories then depend on calibration labels, so the
    level is approximate). Returns {f: (h, C)}."""
    u = sorted(set(int(x) for x in folds))
    K = len(u)
    plan = {}
    for i, f in enumerate(u):
        if K == 1:
            plan[f] = (None, [])
        elif K == 2:
            g = u[(i + 1) % 2]
            plan[f] = (g, [g])
        else:
            h = u[(i + 2) % K]
            plan[f] = (h, [x for x in u if x not in (f, h)])
    return plan


def run_policies(eps, policies, alpha=0.1, alpha_crc=(0.1, 0.05), S=16, max_probes=2, seed=0, log=None, pmap=None,
                 in_loop="simF"):
    """Per-episode rows (list of dicts) for posterior-based policies and the degenerate predictors (see fold_plan)."""
    folds = np.array([int(e["fold"]) for e in eps])
    plan = fold_plan(folds)
    idx = {f: np.flatnonzero(folds == f) for f in plan}
    m0 = [e["post"].summary() for e in eps]
    qin = {}
    for h in {v[0] for v in plan.values() if v[0] is not None}:
        ca = idx[h]
        if in_loop == "simF":
            qin[h] = fit_floor([m[0] for m in m0], [m[1] for m in m0], eps, ca, alpha)
        else:
            qin[h] = fit_q([m0[i][0] for i in ca], [m0[i][1] for i in ca], [eps[i] for i in ca], alpha)
    rows = []
    orc = [Pb.oracle(e["inst"], e["tasks"], e["rho_true"])[2] for e in eps]
    for pol in policies:
        # runs[(h, i)]: episode i run with the in-loop certificate fitted on fold h
        need = sorted({(h, int(i)) for f, (h, C) in plan.items() if h is not None
                       for i in np.r_[idx[f], np.concatenate([idx[c] for c in C]) if C else []].astype(int)})
        jobs = [(pol, eps[i], qin[h], (seed, i, Pol.policy_id(pol)), max_probes, S) for h, i in need]
        runs = dict(zip(need, (pmap or map)(_run_one, jobs)))
        if log:
            log(f"policy {pol} done ({len(jobs)} runs)")
        for f, (h, C) in plan.items():
            if h is None:
                continue
            ca = np.concatenate([idx[c] for c in C]).astype(int)
            fin = {i: runs[(h, int(i))] for i in np.r_[idx[f], ca].astype(int)}
            pts = {i: fin[i]["point"] for i in fin}
            scs = {i: fin[i]["scale"] for i in fin}
            L = lambda d, ii: [d[i] for i in ii]
            q_end = fit_q(L(pts, ca), L(scs, ca), [eps[i] for i in ca], alpha)
            ptl, scl_ = [pts.get(i) for i in range(len(eps))], [scs.get(i) for i in range(len(eps))]
            qF = fit_floor(ptl, scl_, eps, ca, alpha)
            qA = fit_q(L(pts, ca), [np.ones_like(scs[i]) for i in ca], [eps[i] for i in ca], alpha)
            lam = {}
            if alpha_crc:
                curves = {int(i): episode_loss_curve(eps[i], pts[i], scs[i]) for i in ca}
                for a_ in alpha_crc:
                    lam[a_] = core.risk_control_margin(
                        lambda l: np.array([curves[int(i)][int(np.searchsorted(LAMBDAS, l))] for i in ca]),
                        LAMBDAS, a_)
            for i in idx[f]:
                e, fr = eps[i], fin[int(i)]
                one = np.ones_like(fr["scale"])
                certs = {f"sim{alpha}": (q_end, fr["scale"]),
                         f"simF{alpha}": (qF[0], np.maximum(fr["scale"], qF[1])),
                         f"simA{alpha}": (qA, one), "raw": (0.0, fr["scale"]),
                         **{f"crc{a_}": (lam[a_], fr["scale"]) for a_ in alpha_crc}}
                for cname, (q, scl) in certs.items():
                    lo = fr["point"] - q * scl if np.isfinite(q) else np.full_like(fr["point"], -np.inf)
                    lo = np.where(np.isfinite(lo), lo, -np.inf)
                    ch, tc = Pb.decide(e["inst"], e["tasks"], lo)
                    rows.append(_row(e, "posterior", pol, cname, q, ch, tc, fr["probe_s"], fr["used"], orc[i],
                                     lo, e["post"].w, fr["w"], e["post"].rho))
                # uncalibrated Bayes lower bound: weighted 5 % quantile of the final posterior
                pe = e["post"].copy()
                pe.unknown = fr.get("unknown", e["post"].unknown)
                lo = pe.bayes_lower(0.05, w=fr["w"])
                lo = np.where(np.isfinite(lo), lo, -np.inf)
                ch, tc = Pb.decide(e["inst"], e["tasks"], lo)
                rows.append(_row(e, "posterior", pol, "bayes05", 0.0, ch, tc, fr["probe_s"], fr["used"], orc[i],
                                 lo, e["post"].w, fr["w"], e["post"].rho))
    # degenerate predictors (no probing): oracle, nominal (raw and conformal)
    for pred in ("oracle", "nominal"):
        pts = [e["rho_true"] if pred == "oracle" else e["rho_nom"] for e in eps]
        sc = [np.full(p.shape, 0.02) for p in pts]
        for f, (h, C) in plan.items():
            ca = np.concatenate([idx[c] for c in C]).astype(int) if C else np.zeros(0, int)
            q = fit_q([pts[i] for i in ca], [sc[i] for i in ca], [eps[i] for i in ca], alpha)
            for i in idx[f]:
                e = eps[i]
                for cname, qq in ((f"sim{alpha}", q), ("raw", 0.0)):
                    lo = pts[i] - qq * sc[i] if np.isfinite(qq) else np.full_like(pts[i], -np.inf)
                    lo = np.where(np.isfinite(lo), lo, -np.inf)
                    ch, tc = Pb.decide(e["inst"], e["tasks"], lo)
                    rows.append(_row(e, pred, "v2", cname, qq, ch, tc, 0.0, [], orc[i], lo, np.ones(1), np.ones(1),
                                     pts[i][None]))
    return rows


AMB = (0.05, 0.95)
REL_BAND = 0.10        # scripts/analysis/common.py REL_BAND


def pair_counts(e, lower, w0, w1, rho):
    """Pair-level (candidate x task) counts for the paper's probe contract: capable / incapable (truth), certified
    and capable, certified and incapable (unsafe certificate), ambiguous before / after probing
    (posterior P(capable) in (0.05, 0.95))."""
    R, mm, _ = Pb.task_arrays(e["tasks"], e["rho_true"].shape[-1])
    cap = Pb.capable(e["rho_true"], R, mm, e["inst"].sizes)
    cer = Pb.certified(lower, R, mm, e["inst"].sizes)
    cs = Pb.capable(rho, R, mm, e["inst"].sizes).astype(float)
    p0, p1 = np.tensordot(w0, cs, 1), np.tensordot(w1, cs, 1)
    amb = lambda p: int(((p > AMB[0]) & (p < AMB[1])).sum())
    # decision-relevant stratum (docs/board/paper.md 1.0): the piece holds a faulted module, or some required leg's
    # true label lies within REL_BAND of its requirement
    hf = np.asarray(e.get("has_fault", np.zeros(cap.shape[0], bool)), bool)
    near = (np.abs(e["rho_true"][:, None, :] - R[None]) < REL_BAND).any(-1)          # (C, T); NaN r -> False
    rel = hf[:, None] | near
    return dict(n_capable_rel=int((cap & rel).sum()), n_incapable_rel=int((~cap & rel).sum()),
                n_certified_rel=int((cer & cap & rel).sum()), n_unsafe_rel=int((cer & ~cap & rel).sum()),
                n_dec=int(cap.size), n_capable=int(cap.sum()), n_incapable=int((~cap).sum()),
                n_certified=int((cer & cap).sum()), n_cert_any=int(cer.sum()), n_unsafe_cert=int((cer & ~cap).sum()),
                n_amb_before=amb(p0), n_amb_after=amb(p1))


def _row(e, pred, pol, cname, q, ch, tc, tp, used, orc, lower, w0, w1, rho):
    o = Pb.outcome(e["inst"], e["tasks"], ch, tc, e["rho_true"], tp)
    o.update(pair_counts(e, lower, w0, w1, rho))
    return dict(eid=e["meta"]["seed"], fold=e["fold"], parent_size=e["meta"]["parent_size"],
                n_faults=e["meta"]["n_faults"], truth_in=e["meta"].get("truth_in", 1), predictor=pred, policy=pol,
                cert=cname, q=float(q), n_probes=len(used), n_split=sum(u.startswith("split:") for u in used),
                probes="+".join(used), oracle_U=orc["U"], oracle_served=orc["n_served"], regret=orc["U"] - o["U"],
                **o)


def summarise(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby(["predictor", "policy", "cert"])
    out = g.apply(lambda d: pd.Series(dict(
        n_ep=len(d),
        unsafe_commit_rate=d.n_unsafe.sum() / max(d.n_assigned.sum(), 1),
        ep_any_unsafe=(d.n_unsafe > 0).mean(),
        useful_commit_rate=d.n_served.sum() / max(d.oracle_served.sum(), 1),
        served_frac=d.n_served.sum() / d.n_tasks.sum(),
        abstain_rate=d.abstain.mean(),
        regret_mean=d.regret.mean(), regret_se=d.regret.std(ddof=1) / np.sqrt(len(d)),
        U_mean=d.U.mean(), oracle_U_mean=d.oracle_U.mean(),
        probe_s_mean=d.probe_s.mean(), probes_mean=d.n_probes.mean(), split_share=(d.n_split > 0).mean(),
        q_median=d.q.median())), include_groups=False).reset_index()
    return df, out


PAPER_POLICY = {"v2": "fixed_v2", "all": "fixed_all", "fix:clamp": "fixed_v2_clamp", "fix:clamp+moddiff": "fixed_v2_clamp_moddiff"}


def write_paper(df, eps, path, v2_s, test_plant, cert="sim0.1"):
    """docs/board/paper.md contract 1b (probe_*.csv): one row per (policy, episode), posterior predictor with the
    end-to-end simultaneous certificate (sim0.1). Pairs = (candidate, task); n_unsafe = certified-but-incapable pairs.
    probe_time_s includes the 12 s v2 routine. Cross-fitted folds: every episode is a test episode."""
    d = df[(df.predictor == "posterior") & (df.cert == cert)].copy()
    shape = {e["meta"]["seed"]: e["meta"].get("shape", f"{e['meta']['parent_size']}mod") for e in eps}
    out = dict(policy=d.policy.map(lambda p: PAPER_POLICY.get(p, p)), split="test", status="exploratory", eid=d.eid,
               shape=d.eid.map(shape), n_probes=d.n_probes, probe_time_s=d.probe_s + v2_s,
               action=np.where(d.abstain > 0, "abstain", "commit"), n_dec=d.n_dec, n_capable=d.n_capable,
               n_incapable=d.n_incapable, n_certified=d.n_certified, n_unsafe=d.n_unsafe_cert,
               n_amb_before=d.n_amb_before, n_amb_after=d.n_amb_after, utility=d.U, test_plant=test_plant,
               regret=d.regret, n_unsafe_assigned=d.n_unsafe, n_split=d.n_split,
               n_capable_rel=d.n_capable_rel, n_incapable_rel=d.n_incapable_rel,
               n_certified_rel=d.n_certified_rel, n_unsafe_rel=d.n_unsafe_rel)
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
