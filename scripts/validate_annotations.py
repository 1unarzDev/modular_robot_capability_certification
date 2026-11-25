"""Ground truth for codex annotations: case builders (what codex sees), held-out truth (what only this script sees),
and the deterministic validators. LLM output is never ground truth here; every claim is checked against the
simulator's latents, labels or plant effects, or recomputed by code.

Subcommands (`source env.sh`; heavy ones through scripts/heavy.sh):
  t0-truth      replay the T0 decision pilot episodes (seeds of $CFCAP_OUT/decision/pilot_t0_eps600.pkl) and keep
                the latents the pickle does not store (true modes/g/nuisances, particle hypotheses); checks the
                replay reproduces the pickled rho_true exactly            -> $CFCAP_OUT/annotations/t0_truth.pkl
  cases-t0      pair-level (episode, candidate, task) cases of the T0 pilot: unsafe certificates, ambiguous pairs,
                refused capable / incapable pairs under v2 sim0.1, v2 raw and all sim0.1
  cases-a2      replay the Stage A2 parents with an ambiguous v2 decision (oracle consistent set, rho known) and
                record each ambiguous (candidate, leg, tau) with its consistent set
  cases-a5      every A5 decision whose truth flips T1 -> T2 `all` (rho_dec), arms P0 and P1, with the held-out
                per-condition ablation
  cases-a       merge the sources into the workstream-A file (stratified pilot subset with --pilot)
  validate-a    check every claim of an annotation file and the category signatures -> per-category accuracy
  tasks-slots / tasks-build      workstream B: generation slots; library build + validation + agreement
  cases-c / validate-c           workstream C: refused decisions explained from the posterior only; faithfulness

Files: visible cases `results/annotations/<x>_cases.jsonl` (sent to codex), held-out truth
`results/annotations/<x>_truth.jsonl` (never sent), annotations `results/annotations/<name>.jsonl`.
Leg names follow each source's own order; cases always carry legs by name.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RES = REPO / "results" / "annotations"
CFOUT = Path(os.environ.get("CFCAP_OUT", Path.home() / ".cache/modular-runs/2026-11/cfcap"))
OUT = CFOUT / "annotations"
MODES = {0: "ok", 1: "weak", 2: "coast", 3: "brake"}          # capxfer.sim OK, WEAK, COAST, BRAKE
DELTA = 0.08                                                   # per-wheel motor spread half-width (HET_NUIS / t0)
AMB = (0.05, 0.95)

# ---- physical leg scales (capxfer.probes: V_NOM 0.30 m/s, V_DIAG 0.21/0.21 m/s, W_NOM 1.0 rad/s; yaw scaled so the
# farthest wheel stays below WHEEL_SPEED_CAP 0.45 m/s; rho_ref of the nominal healthy body at 3 s ~ 0.91 on every
# leg and shape, $CFCAP_OUT/rho_ref_plated.json)
V_NOM, V_DIAG, W_NOM, RHO_REF = 0.30, 0.21 * math.sqrt(2), 1.0, 0.91
YAW_SCALE = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.911, 6: 0.911, 7: 0.708, 8: 0.708}   # named shapes, t0.qs.yaw_scale
PAYLOAD_PER_MODULE_KG = 5.0     # ASSUMPTION (not measured): rated payload per plated module for the min-team rule
LEGS10 = ("surge+", "surge-", "sway+", "sway-", "yaw+", "yaw-", "diag++", "diag+-", "diag-+", "diag--")  # t1 order


def yaw_scale_n(n):
    return YAW_SCALE.get(int(n), 0.708)


def wheel_names(asm, prefix="m"):
    """Body-frame corner names m<k>_<fl|fr|rl|rr> per wheel (module k = index in asm.modules)."""
    wp, wm, mp = np.asarray(asm.wheel_pos), np.asarray(asm.wheel_module), np.asarray(asm.module_pos)
    out = []
    for i in range(len(wm)):
        d = wp[i] - mp[wm[i]]
        out.append(f"{prefix}{wm[i]}_{'f' if d[0] > 0 else 'r'}{'l' if d[1] > 0 else 'r'}")
    return out


def _w(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=_np) + "\n")


def _np(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _module_mu(mu_per_wheel, names):
    """Per-module friction factor from the per-wheel array (t0.observe.sample_het repeats it on the module's wheels)."""
    out = {}
    for v, n in zip(mu_per_wheel, names):
        out.setdefault(n.split("_")[0], _f(v))
    return out


def _r(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _f(x, nd=3):
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


# ============================================================================= T0 decision pilot
def _replay_t0(seed):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from cfcap.decide import t0data as T
    cap = {}
    code = T.episode.__code__

    def prof(frame, event, arg):
        if event == "return" and frame.f_code is code:
            cap.update(frame.f_locals)
    sys.setprofile(prof)
    try:
        T.episode(seed)
    finally:
        sys.setprofile(None)
    parent, other = cap["parent"], cap["other"]
    pn, on = wheel_names(parent, "m"), wheel_names(other, "o")
    cands = []
    for cd in cap["cands"]:
        src = np.asarray(cd.src)
        cands.append(dict(name=cd.name, kind=cd.kind, modules=list(map(int, cd.modules)),
                          wheels=[pn[s] if s >= 0 else on[-(s + 1)] for s in src], size=int(cd.size)))
    return dict(seed=seed, modules=[list(m) for m in parent.modules], wheels=pn, tm=cap["tm"], tg=cap["tg"],
                stall=cap["st"], noload=cap["nl"], mu=cap["mu"], other_modules=[list(m) for m in other.modules],
                truth_in=bool(cap["truth_in"]), pm=cap["pm"].astype(np.int8), pg=cap["pg"].astype(np.float32),
                ph=cap["ph"], w=cap["w"], cands=cands, rho_true=cap["rho_true"])


def cmd_t0_truth(a):
    from multiprocessing import get_context
    eps = pickle.loads(Path(a.eps).read_bytes())
    seeds = [e["meta"]["seed"] for e in eps]
    with get_context("spawn").Pool(a.workers) as pool:
        tr = pool.map(_replay_t0, seeds, chunksize=4)
    bad = [t["seed"] for t, e in zip(tr, eps) if not np.allclose(t["rho_true"], e["rho_true"], equal_nan=True,
                                                                 atol=1e-9)]
    print(f"replayed {len(tr)} episodes; rho_true mismatch in {len(bad)}: {bad[:10]}")
    if bad:
        raise SystemExit("replay is not deterministic; refusing to write truth")
    # the pickled posterior keeps the top-mass particles of the same weight vector (t0data before the uncommitted
    # no-truncation change); map them and check the weights match, else drop the particle hypotheses of that episode
    n_map = 0
    for t, e in zip(tr, eps):
        P0 = len(e["post"].w)
        o = np.argsort(-t["w"])[:P0]
        ww = t["w"][o] / t["w"][o].sum()
        t["particles_ok"] = bool(len(t["w"]) >= P0 and np.allclose(ww, e["post"].w, rtol=1e-6, atol=1e-12))
        if t["particles_ok"]:
            t["pm"], t["pg"], t["ph"], t["w"] = t["pm"][o], t["pg"][o], t["ph"][o], ww
            n_map += 1
    print(f"particle hypotheses mapped onto the pickled posterior in {n_map}/{len(tr)} episodes")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t0_truth.pkl").write_bytes(pickle.dumps({t["seed"]: t for t in tr}))


def _hyp_str(modes, gs, names):
    f = [f"{names[i]} {MODES[int(modes[i])]}" + (f" g={gs[i]:.1f}" if int(modes[i]) == 1 else "")
         for i in np.flatnonzero(modes)]
    return "; ".join(f) if f else "healthy"


def posterior_view(w, pm, pg, names, truth_key=None, top=6):
    """Robot-visible posterior summary: top fault hypotheses (mass summed over nuisance draws), per-wheel fault
    probability (>= 2 %), ESS."""
    keys = [_hyp_str(m, g, names) for m, g in zip(pm, pg)]
    mass = defaultdict(float)
    for k, ww in zip(keys, w):
        mass[k] += float(ww)
    tops = sorted(mass.items(), key=lambda kv: -kv[1])[:top]
    pf = (np.asarray(w)[:, None] * (np.asarray(pm) != 0)).sum(0)
    wf = {names[i]: round(float(pf[i]), 3) for i in np.flatnonzero(pf >= 0.02)}
    out = dict(n_particles=int(len(w)), ess=_f(1 / (np.asarray(w) ** 2).sum(), 1),
               top_hypotheses=[dict(faults=k, mass=round(v, 3)) for k, v in tops], wheel_fault_prob=wf)
    tm = None if truth_key is None else round(mass.get(truth_key, 0.0), 3)
    return out, tm


def _t0_configs(e, q_by):
    """(config, point, scale, q, weights) for v2 sim0.1, v2 raw, all sim0.1."""
    post = e["post"]
    sd = e["meta"]["seed"]
    pt, sc = post.summary()
    out = [("v2/sim0.1", pt, sc, q_by[(sd, "v2")], post.w), ("v2/raw", pt, sc, 0.0, post.w)]
    p2 = post.copy()
    for k in ("clamp", "moddiff", "tug"):
        if k in p2.probes:
            p2.observe(k)
    pt2, sc2 = p2.summary(p2.w)
    out.append(("all/sim0.1", pt2, sc2, q_by[(sd, "all")], p2.w))
    return out


def _probe_after(post, name, w0):
    p = post.copy()
    p.w = w0.copy()
    if name not in p.probes:
        return None
    p.observe(name)
    return p.w


def cmd_cases_t0(a):
    import pandas as pd
    from cfcap.decide import problem as Pb
    eps = pickle.loads(Path(a.eps).read_bytes())
    truth = pickle.loads((OUT / "t0_truth.pkl").read_bytes())
    df = pd.read_csv(a.episodes_csv)
    d = df[(df.predictor == "posterior") & (df.cert == "sim0.1")]
    q_by = {(int(r.eid), r.policy): float(r.q) for r in d.itertuples()}
    rows = {}
    for e in eps:
        sd = e["meta"]["seed"]
        T = truth[sd]
        if not T.get("particles_ok", False):
            n_skip = locals().get("n_skip", 0) + 1
            print(f"episode {sd}: particle hypotheses do not map onto the pickle; skipped")
            continue
        names = T["wheels"]
        legs = list(e["legs"])
        R, mm, _ = Pb.task_arrays(e["tasks"], len(legs))
        sizes = e["inst"].sizes
        cap = Pb.capable(e["rho_true"], R, mm, sizes)
        truth_key = _hyp_str(T["tm"], T["tg"], names)
        rho = e["post"].rho
        for cfg, pt, sc, q, w in _t0_configs(e, q_by):
            lo = np.where(np.isfinite(pt - q * sc), pt - q * sc, -np.inf)
            cer = Pb.certified(lo, R, mm, sizes)
            ch, _ = Pb.decide(e["inst"], e["tasks"], lo)
            pcap = np.tensordot(w, Pb.capable(rho, R, mm, sizes).astype(float), 1)
            for c in range(cap.shape[0]):
                for t in range(cap.shape[1]):
                    if sizes[c] < mm[t]:
                        continue                                       # too small: trivial refusal
                    kinds = []
                    if cer[c, t] and not cap[c, t]:
                        kinds.append("unsafe_cert")
                    if cfg == "v2/sim0.1" and AMB[0] < pcap[c, t] < AMB[1]:
                        kinds.append("ambiguous")
                    if not cer[c, t]:
                        kinds.append("refused_capable" if cap[c, t] else "refused_incapable")
                    key = (sd, c, t)
                    r = rows.setdefault(key, dict(configs={}, e=e, T=T, c=c, t=t, legs=legs, R=R,
                                                  truth_key=truth_key, cap=bool(cap[c, t])))
                    r["configs"][cfg] = dict(kinds=kinds, q=float(q), committed=bool(ch[t] == c),
                                             lower={legs[l]: _f(lo[c, l]) for l in np.flatnonzero(np.isfinite(R[t]))},
                                             mean={legs[l]: _f(pt[c, l]) for l in np.flatnonzero(np.isfinite(R[t]))},
                                             sd={legs[l]: _f(sc[c, l]) for l in np.flatnonzero(np.isfinite(R[t]))},
                                             p_capable=_f(pcap[c, t]), w=w)
    vis, hid = [], []
    for (sd, c, t), r in rows.items():
        kinds = sorted({k for v in r["configs"].values() for k in v["kinds"]})
        if not kinds:
            continue
        e, T, legs, R = r["e"], r["T"], r["legs"], r["R"]
        cd = T["cands"][c] if c < len(T["cands"]) else None
        req = [legs[l] for l in np.flatnonzero(np.isfinite(R[t]))]
        cid = f"t0-{sd}-c{c}-t{t}"
        cw = set(cd["wheels"])
        wheels_truth = [dict(wheel=n, mode=MODES[int(T["tm"][i])], g=_f(T["tg"][i], 2), stall=_f(T["stall"][i]),
                             noload=_f(T["noload"][i]), in_candidate=int(n in cw)) for i, n in enumerate(T["wheels"])]
        pv, tmass = posterior_view(r["configs"]["v2/sim0.1"]["w"], T["pm"], T["pg"], T["wheels"], r["truth_key"])
        task = e["tasks"][t]
        leg_rows = {}
        for l in np.flatnonzero(np.isfinite(R[t])):
            ln = legs[l]
            leg_rows[ln] = dict(r=_f(R[t, l], 2), truth=_f(e["rho_true"][c, l]),
                                **{f"{k}_{cfg.split('/')[0]}{'' if cfg.endswith('sim0.1') else '_raw'}":
                                   r["configs"][cfg][k][ln] for cfg in r["configs"] for k in (("lower",) if cfg == "v2/raw" else ("mean", "sd", "lower"))})
        # per-leg P(rho >= r) under the v2 posterior
        w0 = r["configs"]["v2/sim0.1"]["w"]
        for l in np.flatnonzero(np.isfinite(R[t])):
            leg_rows[legs[l]]["p_leg_v2"] = _f(np.dot(w0, np.nan_to_num(e["post"].rho[:, c, l], nan=-1) >= R[t, l]))
        cfg_view = {cfg: dict(kinds=v["kinds"], q=_f(v["q"]), committed=v["committed"], p_capable=v["p_capable"])
                    for cfg, v in r["configs"].items()}
        vis.append(dict(
            case_id=cid, source="t0_decision", plant="T0 quasi-static", kinds=kinds,
            parent=dict(n_modules=len(T["modules"]), modules=[dict(module=f"m{k}", cell=[m[1], m[2]], yaw_quarter=m[3])
                                                               for k, m in enumerate(T["modules"])]),
            candidate=dict(name=cd["name"], kind=cd["kind"] if c > 0 else "parent", size=cd["size"],
                           wheels=cd["wheels"], includes_other_assembly=any(x.startswith("o") for x in cd["wheels"])),
            task=dict(kind=task.kind, min_modules=task.min_modules, required_legs=req),
            truth=dict(wheels=[x for x in wheels_truth if x["mode"] != "ok" or x["in_candidate"]],
                       modules_mu=_module_mu(T["mu"], T["wheels"]),
                       motor_spread_halfwidth=DELTA, capable=r["cap"]),
            probes_observed=dict(v2="parent v2 routine (all configs)", all="v2 + clamp + moddiff + tug"),
            posterior_v2=pv, posterior_mass_on_truth=tmass, legs=leg_rows, configs=cfg_view,
            certificate="lower = posterior mean - q * (posterior sd + 0.02); certified iff lower >= r on every "
                        "required leg; q fitted cross-fold on other episodes (sim0.1), raw: q = 0"))
        # held-out: does one extra probe settle an ambiguous pair (unambiguous and on the right side)?
        pr = {}
        if "ambiguous" in kinds:
            capm = Pb.capable(e["post"].rho, R, np.array([x.min_modules for x in e["tasks"]]), e["inst"].sizes)
            for name in ["clamp", "moddiff", "tug"] + [k for k in e["post"].probes if k == f"split:{c}"]:
                w1 = _probe_after(e["post"], name, w0)
                if w1 is None:
                    continue
                p1 = float(np.dot(w1, capm[:, c, t]))
                pr["split" if name.startswith("split") else name] = int(
                    (p1 >= AMB[1] and r["cap"]) or (p1 <= AMB[0] and not r["cap"]))
        hid.append(dict(case_id=cid, source="t0_decision", truth_in_screen=int(T["truth_in"]),
                        truth_mass=tmass, probe_resolves=pr, cap=r["cap"]))
    _w(RES / "t0_cases.jsonl", vis)
    _w(RES / "t0_truth.jsonl", hid)
    print(f"t0 cases {len(vis)}:", Counter(k for v in vis for k in v["kinds"]))


# ============================================================================= Stage A2 ambiguous decisions
def _replay_a2(seed):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from cfcap.t0 import qs, stagea as SA
    P, th = qs.load_params("plated"), qs.load_theta("plated")
    orig = SA.decision_counts
    rec = []

    def hook(rho_h, mask, rho_t, tau):
        out = orig(rho_h, mask, rho_t, tau)
        f = sys._getframe(1)
        L = f.f_locals
        sn, p, rm = L.get("sn"), L.get("p"), L.get("rm")
        if sn == "v2" and rm == "known" and out[1] > 0:
            cand, parent = L["cand"], L["parent"]
            pn = wheel_names(parent, "m")
            on = wheel_names(L["other"], "o")
            src = np.asarray(cand.src)
            sel = np.asarray(mask, bool)
            lo, hi = rho_h[sel].min(0), rho_h[sel].max(0)
            amb_legs = np.flatnonzero((lo < tau) & (hi >= tau))
            other_sets = {}
            for (sn2, p2), m2 in L["cmasks"].items():
                if p2 != p or sn2 == "v2":
                    continue
                s2 = np.asarray(m2, bool)
                l2, h2 = rho_h[s2].min(0), rho_h[s2].max(0)
                other_sets[sn2] = [int(not ((l2[k] < tau) and (h2[k] >= tau))) for k in amb_legs]
            HMU, HGU = L["HMU"], L["HGU"]
            hs = [dict(faults=_hyp_str(HMU[i], HGU[i], pn), rho={qs.LEG_NAMES[k]: _f(rho_h[i, k]) for k in amb_legs})
                  for i in np.flatnonzero(sel)][:12]
            rec.append(dict(seed=seed, noise=p, tau=float(tau), cand=cand.name, kind=cand.kind, size=int(cand.size),
                            wheels=[pn[s] if s >= 0 else on[-(s + 1)] for s in src],
                            legs=[qs.LEG_NAMES[k] for k in amb_legs], set_lo=[_f(lo[k]) for k in amb_legs],
                            set_hi=[_f(hi[k]) for k in amb_legs], truth=[_f(rho_t[k]) for k in amb_legs],
                            set_size=int(sel.sum()), hyps=hs, other_sets=other_sets,
                            parent_modules=[list(m) for m in parent.modules],
                            tm=L["tm"].tolist(), tg=np.round(L["tg"], 2).tolist(), het=bool(L["het"]),
                            stall=[round(float(x), 3) for x in L["st"]],
                            noload=[round(float(x), 3) for x in L["nl"]], mu=[round(float(x), 3) for x in L["mu"]],
                            wheel_names=pn))
        return out
    SA.decision_counts = hook
    try:
        SA.run_episode(seed, P, th)
    finally:
        SA.decision_counts = orig
    return rec


def cmd_cases_a2(a):
    import pandas as pd
    from multiprocessing import get_context
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(CFOUT / "stageA" / "decisions_plated_*.csv.gz")))])
    s = D[(D.probe_set == "v2") & (D.rho_model == "known") & (D.n_amb > 0)]
    seeds = sorted(s.seed.unique().tolist())
    cache = OUT / "a2_replay.pkl"
    if cache.exists() and not a.fresh:
        recs = pickle.loads(cache.read_bytes())
    else:
        with get_context("spawn").Pool(a.workers) as pool:
            recs = [r for rr in pool.map(_replay_a2, seeds, chunksize=1) for r in rr]
        OUT.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(recs))
    vis, hid = [], []
    n_check = s.groupby(["seed", "noise"]).n_amb.sum().to_dict()
    got = defaultdict(int)
    for r in recs:
        got[(r["seed"], r["noise"])] += len(r["legs"])
    mism = {k: (v, got.get(k, 0)) for k, v in n_check.items() if got.get(k, 0) != v}
    print(f"A2: {len(seeds)} seeds, {len(recs)} (candidate, tau, noise) records; count mismatches vs CSV: {mism}")
    for j, r in enumerate(recs):
        for k, leg in enumerate(r["legs"]):
            cid = f"a2-{r['seed']}-{r['noise']}-{r['cand']}-{leg}-{r['tau']}"
            names = r["wheel_names"]
            cw = set(r["wheels"])
            wt = [dict(wheel=n, mode=MODES[r["tm"][i]], g=r["tg"][i], stall=r["stall"][i], noload=r["noload"][i],
                       in_candidate=int(n in cw)) for i, n in enumerate(names) if r["tm"][i] or n in cw]
            vis.append(dict(
                case_id=cid, source="a2_ambiguous", plant="T0 quasi-static, idealised oracle consistent set",
                kinds=["ambiguous"],
                parent=dict(n_modules=len(r["parent_modules"]),
                            modules=[dict(module=f"m{i}", cell=[m[1], m[2]], yaw_quarter=m[3])
                                     for i, m in enumerate(r["parent_modules"])]),
                candidate=dict(name=r["cand"], kind=r["kind"], size=r["size"], wheels=r["wheels"]),
                truth=dict(wheels=wt, heterogeneity_on=r["het"], motor_spread_halfwidth=DELTA if r["het"] else 0.0,
                           modules_mu=_module_mu(r["mu"], r["wheel_names"])),
                probes_observed=dict(v2=f"parent v2 routine, noise preset '{r['noise']}', 3-sigma consistency"),
                posterior_v2=dict(kind="consistent set of <=2-fault hypotheses (rho of each on the candidate's true "
                                       "continuous plant)", set_size=r["set_size"], hypotheses=r["hyps"]),
                legs={leg: dict(tau=r["tau"], truth=r["truth"][k], set_min=r["set_lo"][k], set_max=r["set_hi"][k])},
                certificate="go iff min over the consistent set >= tau; ambiguous iff min < tau <= max"))
            hid.append(dict(case_id=cid, source="a2_ambiguous", truth_in_screen=1,
                            probe_resolves={sn: v[k] for sn, v in r["other_sets"].items()},
                            cap=bool(r["truth"][k] >= r["tau"])))
    _w(RES / "a2_cases.jsonl", vis)
    _w(RES / "a2_truth.jsonl", hid)
    print(f"a2 cases {len(vis)}")


# ============================================================================= A5 T1 -> T2 flips
A5_CONDS = ("base", "compliance", "load_transfer", "aniso", "aniso_mu", "all-aniso_mu", "all")


def cmd_cases_a5(a):
    from cfcap.t1 import params as P
    par = P.base_params("plated")
    S = CFOUT / "stress"
    R = np.load(S / "A5_refs.npz")
    vis, hid = [], []
    for f1 in sorted(glob.glob(str(S / "A5_*_t1.npz"))):
        m = re.search(r"A5_(P\d)_n(\d+)_s(\d+)_t1", f1)
        arm, shard = m.group(1), int(m.group(3))
        f2 = f1.replace("_t1.npz", "_t2_all.npz")
        if not os.path.exists(f2):
            continue
        z1, z2 = np.load(f1), np.load(f2)
        for shape in ("single", "pair", "L3", "square", "2x3"):
            asm = P.named_assembly(shape, par)
            names = wheel_names(asm, "m")
            r1 = z1[f"{shape}/T1/rho"] / R[f"{shape}/T1"][None]
            rc = {c: z2[f"{shape}/{c}/rho"] / R[f"{shape}/{c}"][None] for c in A5_CONDS if f"{shape}/{c}/rho" in z2}
            lat = {k.split("/lat/")[1]: z1[k] for k in z1.files if k.startswith(f"{shape}/lat/")}
            for ep in range(r1.shape[0]):
                for li, leg in enumerate(LEGS10):
                    for tau in (0.7, 0.8):
                        a1, a2 = r1[ep, li] >= tau, rc["all"][ep, li] >= tau
                        if a1 == a2:
                            continue
                        cid = f"a5-{arm}-s{shard}-{shape}-e{ep}-{leg}-{tau}"
                        wt = [dict(wheel=n, mode=MODES[int(lat["mode"][ep, i])], g=_f(lat["g"][ep, i], 3),
                                   stall=_f(lat["stall_w"][ep, i]), noload=_f(lat["noload_w"][ep, i]),
                                   yaw_err_rad=_f(lat["yaw_err"][ep, i], 4), load_z=_f(lat["load_z"][ep, i]))
                              for i, n in enumerate(names)]
                        dom = {k[4:]: _f(v[ep]) for k, v in lat.items() if k.startswith("dom.") and v.ndim == 1}
                        vis.append(dict(
                            case_id=cid, source="a5_flip", kinds=["flip_" + ("down" if a1 else "up")],
                            plant="T1 (capxfer-based training plant) vs T2 (independent crane kernel + compliance, "
                                  "load transfer, roller-drag anisotropy, per-wheel roller-friction scatter)",
                            arm="no payload" if arm == "P0" else "0-10 kg payload at a random deck point",
                            parent=dict(shape=shape, n_modules=len(asm.modules),
                                        modules=[dict(module=f"m{i}", cell=[mm[1], mm[2]], yaw_quarter=mm[3])
                                                 for i, mm in enumerate(asm.modules)]),
                            candidate=dict(name=shape, kind="child", size=len(asm.modules), wheels=names),
                            truth=dict(wheels=wt, global_latents=dom,
                                       modules=dict(vbatt_offset=[round(float(x), 3) for x in lat["vbatt_off"][ep]],
                                                    mu_factor=[round(float(x), 3) for x in lat["mu_mod"][ep]])),
                            legs={leg: dict(tau=tau, t1=_f(r1[ep, li]), t2=_f(rc["all"][ep, li]))},
                            all_legs=dict(t1={l: _f(r1[ep, k]) for k, l in enumerate(LEGS10)},
                                          t2={l: _f(rc["all"][ep, k]) for k, l in enumerate(LEGS10)}),
                            label="rho_dec = rho_abs / rho_ref(shape, leg), each plant with its own nominal reference",
                        ))
                        hid.append(dict(case_id=cid, source="a5_flip",
                                        condition_flip={c: int((rc[c][ep, li] >= tau) != a1) for c in rc if c != "all"},
                                        cap=bool(a2)))
    _w(RES / "a5_cases.jsonl", vis)
    _w(RES / "a5_truth.jsonl", hid)
    print(f"a5 cases {len(vis)}:", Counter(v["parent"]["shape"] for v in vis), Counter(v["kinds"][0] for v in vis))


# ============================================================================= workstream A: merge, pilot
def cmd_cases_a(a):
    srcs = ["t0", "a2", "a5"] + [s for s in a.extra.split(",") if s]
    vis, hid = [], []
    for s in srcs:
        v = _r(RES / f"{s}_cases.jsonl")
        h = {x["case_id"]: x for x in _r(RES / f"{s}_truth.jsonl")}
        if s == "t0":                       # workstream A: failures, abstentions, ambiguity (not correct refusals)
            v = [x for x in v if set(x["kinds"]) - {"refused_incapable"}]
            # refusals of capable pairs that occur only after all probes (the overconfident `all` posterior) are
            # ~2000 near-identical cases: keep a random subset of a.t0_all_only
            only_all = [x for x in v if set(x["kinds"]) == {"refused_capable"} and not any(
                "refused_capable" in x["configs"][c]["kinds"] for c in ("v2/sim0.1", "v2/raw"))]
            drop = set(x["case_id"] for x in only_all) - set(
                x["case_id"] for x in [only_all[i] for i in np.random.default_rng(5).permutation(len(only_all))[
                    :a.t0_all_only]])
            v = [x for x in v if x["case_id"] not in drop]
        vis += v
        hid += [h[x["case_id"]] for x in v]
    rng = np.random.default_rng(0)
    if a.pilot:
        strata = defaultdict(list)
        for i, x in enumerate(vis):
            strata[(x["source"], tuple(sorted(x["kinds"]))[0])].append(i)
        keys = sorted(strata)
        pick = []
        while len(pick) < a.pilot and any(strata[k] for k in keys):
            for k in keys:
                if strata[k] and len(pick) < a.pilot:
                    pick.append(strata[k].pop(int(rng.integers(len(strata[k])))))
        vis, hid = [vis[i] for i in pick], [hid[i] for i in pick]
    else:
        o = rng.permutation(len(vis))                      # mixed chunks
        vis, hid = [vis[i] for i in o], [hid[i] for i in o]
    _w(RES / f"{a.out}_cases.jsonl", vis)
    _w(RES / f"{a.out}_truth.jsonl", hid)
    print(f"{a.out}: {len(vis)} cases", Counter(x["source"] for x in vis))


# ============================================================================= claim checking
def truth_index(case, hidden):
    """Flatten a visible case + its held-out truth into the lookup the claim checker uses."""
    t = case.get("truth", {})
    wheels = {w["wheel"]: w for w in t.get("wheels", [])}
    cand_w = set(case.get("candidate", {}).get("wheels", []))
    legs = {}
    for ln, v in case.get("legs", {}).items():
        thr = v.get("r", v.get("tau"))
        tru = v.get("truth", v.get("t2"))
        legs[ln] = dict(v, thr=thr, truth=tru)
    return dict(wheels=wheels, cand_wheels=cand_w, legs=legs, has_fault=int(any(
        w["mode"] != "ok" for n, w in wheels.items() if n in cand_w)), mu=t.get("modules_mu", {}),
        latents=t.get("global_latents", {}), hidden=hidden or {}, case=case)


def _cmp(x, op, v, tol):
    if x is None or v is None:
        return None
    if op == "lt":
        return x < v + tol
    if op == "gt":
        return x > v - tol
    if op == "eq":
        return abs(x - v) <= max(tol, 1e-9)
    return None


def check_claim(ix, cl, tol=0.005):
    """-> (True | False | None, note). None = not checkable against this case's ground truth."""
    k, tgt, op, v, mode = cl.get("kind"), cl.get("target", ""), cl.get("op", "eq"), cl.get("value"), cl.get("mode", "")
    W, Lg, H = ix["wheels"], ix["legs"], ix["hidden"]
    if k in ("wheel_mode", "wheel_gain", "wheel_stall", "wheel_noload", "spread_masked"):
        w = W.get(tgt)
        if w is None:
            # a wheel absent from the truth list of a T0/A2 case is healthy (only faulted + candidate wheels listed)
            if k == "wheel_mode" and re.fullmatch(r"[mo]\d+_[fr][lr]", tgt or "") and ix["case"]["source"] != "a5_flip":
                return (mode == "ok"), "unlisted wheel = healthy"
            return None, "unknown wheel"
        if k == "wheel_mode":
            return w["mode"] == mode, w["mode"]
        if k == "wheel_gain":
            return _cmp(w["g"] if w["mode"] == "weak" else 1.0, op, v, tol), w["g"]
        if k == "spread_masked":
            d = ix["case"].get("truth", {}).get("motor_spread_halfwidth", DELTA)
            masked = w["mode"] == "weak" and w["g"] >= (1 - d) / (1 + d) - 1e-9
            return masked == bool(v), w["g"]
        key = "stall" if k == "wheel_stall" else "noload"
        return _cmp(w.get(key), op, v, tol), w.get(key)
    if k == "wheel_in_candidate":
        return (tgt in ix["cand_wheels"]) == bool(v), None
    if k == "candidate_has_fault":
        return ix["has_fault"] == int(bool(v)), ix["has_fault"]
    if k == "module_mu":
        x = ix["mu"].get(tgt)
        if x is None and ix["case"]["source"] == "a5_flip":
            mf = ix["case"]["truth"]["modules"]["mu_factor"]
            i = int(tgt[1:]) if re.fullmatch(r"m\d+", tgt or "") else -1
            x = mf[i] if 0 <= i < len(mf) else None
        return _cmp(x, op, v, tol), x
    if k == "latent":
        x = ix["latents"].get(tgt)
        return _cmp(x, op, v, tol), x
    if k in ("leg_truth", "leg_t1", "leg_margin", "leg_shift", "post_mean", "post_sd", "p_leg", "set_min", "set_max"):
        g = Lg.get(tgt)
        if g is None:
            return None, "leg not in case"
        if k == "leg_truth":
            x = g["truth"]
        elif k == "leg_t1":
            x = g.get("t1")
        elif k == "leg_margin":
            base = g.get("t1", g["truth"])
            x = None if base is None else abs(base - g["thr"])
        elif k == "leg_shift":
            x = None if g.get("t1") is None else g["t2"] - g["t1"]
        elif k == "post_mean":
            x = g.get("mean_v2")
        elif k == "post_sd":
            x = g.get("sd_v2")
        elif k == "p_leg":
            x = g.get("p_leg_v2")
        else:
            x = g.get(k)
        return _cmp(x, op, v, max(tol, 0.01 if k in ("post_mean", "post_sd") else tol)), x
    if k == "truth_mass":
        return _cmp(H.get("truth_mass"), op, v, 0.01), H.get("truth_mass")
    if k == "truth_in_screen":
        x = H.get("truth_in_screen")
        return (None if x is None else x == int(bool(v))), x
    if k == "probe_resolves":
        pr = H.get("probe_resolves", {})
        name = {"v2+clamp": "clamp", "v2+moddiff": "moddiff", "v2+tug": "tug", "diag_split": "split"}.get(tgt, tgt)
        x = pr.get(name, pr.get(tgt))
        if x is None and ix["case"]["source"] == "a2_ambiguous":
            x = pr.get(f"v2+{name}")
        return (None if x is None else x == int(bool(v))), x
    if k == "condition_flip":
        x = H.get("condition_flip", {}).get(tgt)
        return (None if x is None else x == int(bool(v))), x
    return None, "unknown kind"


# ---- category signatures (docs/annotations/TAXONOMY.md): deterministic predicates on ground truth. A category whose
# signature is None has no code check; it is never validated and never enters the paper table.
def _faulted_in_cand(ix):
    return [w for n, w in ix["wheels"].items() if n in ix["cand_wheels"] and w["mode"] != "ok"]


def _near(ix, band):
    return any(g.get("thr") is not None and g.get("t1", g["truth"]) is not None and
               abs(g.get("t1", g["truth"]) - g["thr"]) < band for g in ix["legs"].values())


def _sig_screen_miss(ix):
    return ix["hidden"].get("truth_in_screen") == 0


def _sig_spread_masked(ix):
    d = ix["case"].get("truth", {}).get("motor_spread_halfwidth", DELTA) or DELTA
    return any(w["mode"] == "weak" and w["g"] >= (1 - d) / (1 + d) - 0.051 for w in _faulted_in_cand(ix))


def _sig_mode_confusion(ix):
    """The posterior mass on the true hypothesis is small although the truth survived the screen."""
    h = ix["hidden"]
    return h.get("truth_in_screen") == 1 and (h.get("truth_mass") or 0) < 0.5


def _sig_near_threshold(ix, band=0.05):
    return _near(ix, band)


def _sig_margin_slack(ix):
    """Refused although capable, with the plug-in posterior mean above r on every required leg (the conformal margin,
    not the posterior, refuses)."""
    c = ix["case"]
    if "refused_capable" not in c["kinds"]:
        return False
    return all((g.get("mean_v2") is not None and g["mean_v2"] >= g["thr"]) for g in ix["legs"].values())


def _sig_probe_overconfidence(ix):
    """Refused only after the extra probes (all sim0.1), i.e. certified under v2 sim0.1."""
    cf = ix["case"].get("configs", {})
    return ("all/sim0.1" in cf and any(k.startswith("refused") for k in cf["all/sim0.1"]["kinds"])
            and not any(k.startswith("refused") for k in cf.get("v2/sim0.1", {}).get("kinds", ["refused"])))


def _sig_kernel_numerics(ix):
    return ix["hidden"].get("condition_flip", {}).get("base") == 1


def _sig_friction_scatter(ix):
    """Removing the per-wheel roller-friction scatter from T2 `all` removes the flip."""
    cf = ix["hidden"].get("condition_flip", {})
    return cf.get("all-aniso_mu") == 0


def _sig_dock_effects(ix):
    cf = ix["hidden"].get("condition_flip", {})
    return any(cf.get(k) == 1 for k in ("compliance", "load_transfer", "aniso")) and cf.get("base") == 0


def _sig_unexcited(ix):
    """A2 / T0 ambiguity that a targeted probe settles (the v2 routine never excited the decisive envelope)."""
    pr = ix["hidden"].get("probe_resolves", {})
    return any(v == 1 for k, v in pr.items() if "split" not in k)


def _sig_split_only(ix):
    pr = ix["hidden"].get("probe_resolves", {})
    return any(v == 1 for k, v in pr.items() if "split" in k) and not any(
        v == 1 for k, v in pr.items() if "split" not in k)


def _sig_hidden_passive(ix):
    """A coast / brake wheel in the candidate."""
    return any(w["mode"] in ("coast", "brake") for w in _faulted_in_cand(ix))


def _sig_genuine_incapable(ix):
    return any(g["truth"] is not None and g["thr"] is not None and g["truth"] < g["thr"] - 0.05
               for g in ix["legs"].values())


SIGNATURES = {}          # filled from TAXONOMY (category name -> predicate or None); see load_taxonomy


def load_taxonomy(path=REPO / "docs" / "annotations" / "TAXONOMY.md"):
    """Categories and their signature names from the TAXONOMY.md table rows `| name | signature | ... |`."""
    fns = {k[5:]: v for k, v in globals().items() if k.startswith("_sig_")}
    cats = {}
    if not Path(path).exists():
        return cats
    for l in Path(path).read_text().splitlines():
        m = re.match(r"\|\s*`([a-z_]+)`\s*\|\s*`?([a-z_]+|none)`?\s*\|", l)
        if m:
            cats[m.group(1)] = fns.get(m.group(2))
    return cats


def cmd_validate_a(a):
    import pandas as pd
    cats = load_taxonomy()
    ann = {x["case_id"]: x for x in _r(RES / f"{a.name}.jsonl")}
    cases = {x["case_id"]: x for x in _r(RES / f"{a.cases}_cases.jsonl")}
    hid = {x["case_id"]: x for x in _r(RES / f"{a.cases}_truth.jsonl")}
    rows, crow = [], []
    for cid, an in ann.items():
        if cid not in cases:
            continue
        ix = truth_index(cases[cid], hid.get(cid))
        res = [check_claim(ix, cl) for cl in an.get("claims", [])]
        for cl, (ok, x) in zip(an.get("claims", []), res):
            crow.append(dict(case_id=cid, category=an.get("category"), kind=cl.get("kind"), target=cl.get("target"),
                             op=cl.get("op"), value=cl.get("value"), mode=cl.get("mode"),
                             result="unverifiable" if ok is None else ("true" if ok else "false"), truth_value=x,
                             text=cl.get("text", "")[:200]))
        chk = [r for r, _ in res if r is not None]
        sig = cats.get(an.get("category"))
        rows.append(dict(case_id=cid, source=cases[cid]["source"], kinds="+".join(cases[cid]["kinds"]),
                         category=an.get("category"), n_claims=len(res), n_checkable=len(chk), n_true=sum(chk),
                         all_true=int(len(chk) > 0 and all(chk)),
                         signature=None if sig is None else int(bool(sig(ix))),
                         explanation=an.get("explanation", "")[:300]))
    D = pd.DataFrame(rows)
    C = pd.DataFrame(crow)
    D.to_csv(RES / f"{a.name}_validated.csv", index=False)
    C.to_csv(RES / f"{a.name}_claims.csv", index=False)
    g = D.groupby("category")
    summ = g.apply(lambda d: pd.Series(dict(
        n=len(d), sources="+".join(sorted(d.source.unique())),
        claims=int(d.n_claims.sum()), checkable=int(d.n_checkable.sum()),
        claim_acc=d.n_true.sum() / max(d.n_checkable.sum(), 1),
        case_all_true=d.all_true.mean(),
        has_signature=int(d.signature.notna().any()),
        signature_rate=d.signature.mean() if d.signature.notna().any() else np.nan)),
        include_groups=False).reset_index()
    summ["validated"] = ((summ.claim_acc >= a.min_claim_acc) & (summ.signature_rate >= a.min_sig)
                         & (summ.n >= a.min_n) & (summ.has_signature == 1)).astype(int)
    summ.to_csv(RES / f"{a.name}_by_category.csv", index=False)
    by_kind = C.groupby("kind").result.value_counts().unstack(fill_value=0)
    by_kind.to_csv(RES / f"{a.name}_by_claim_kind.csv")
    tot = C[C.result != "unverifiable"]
    print(f"{len(D)} annotations, {len(C)} claims, {len(tot)} checkable, accuracy {(tot.result == 'true').mean():.3f}")
    print(summ.round(3).to_string())
    print(by_kind.to_string())


# ============================================================================= workstream B: task library
DOMAINS = {
    "warehouse": ["narrow-aisle picking", "pallet shuttle across a cross-dock", "shelf-to-station tote transfer",
                  "inventory scan along racks", "loading-bay trailer unload", "cold-storage aisle",
                  "mezzanine conveyor feed", "returns sorting area", "high-density mobile racking",
                  "pharmacy fulfilment cell"],
    "rescue": ["collapsed-corridor search", "smoke-filled office floor", "debris field at a parking garage",
               "flooded basement (shallow water, flat floor)", "tunnel inspection after a quake",
               "hospital evacuation corridor", "stairwell landing (flat section only)", "warehouse fire aftermath",
               "train carriage interior", "mine refuge chamber"],
    "inspection": ["data-centre cold aisle", "solar-farm inverter room", "aircraft hangar floor under a wing",
                   "ship engine room deck", "nuclear plant turbine hall", "pipe gallery under a plant",
                   "bridge deck underside walkway", "wind-turbine nacelle floor", "semiconductor cleanroom bay",
                   "parking-structure column scan"],
    "logistics": ["hospital supply run", "hotel room-service corridor", "airport baggage hall",
                  "factory line-side kitting", "library book return", "lab sample transport",
                  "retail backroom restock", "construction-site indoor slab", "film-studio camera dolly",
                  "museum gallery after hours"],
}
EMPHASIS = ["tight lateral clearance", "frequent in-place reorientation", "heavy payload", "long straight runs",
            "reversing out of dead ends", "diagonal approach to a docking point", "precise slow alignment",
            "mixed: several motion types", "light payload but fast transit", "carrying a fragile load"]


def cmd_tasks_slots(a):
    rng = np.random.default_rng(1)
    rows, i = [], 0
    while len(rows) < a.n:
        for dom, subs in DOMAINS.items():
            for s in subs:
                if len(rows) >= a.n:
                    break
                rows.append(dict(case_id=f"task{i:04d}", domain=dom, setting=s,
                                 emphasis=EMPHASIS[int(rng.integers(len(EMPHASIS)))],
                                 team_hint=int(rng.choice([1, 2, 2, 3, 4, 4, 6]))))
                i += 1
    _w(RES / "tasks_slots.jsonl", rows)
    print(f"{len(rows)} task slots")


DIR_LEGS = dict(surge=("surge+", "surge-"), sway=("sway+", "sway-"), yaw=("yaw+", "yaw-"),
                diag=("diag++", "diag+-", "diag-+", "diag--"))


def task_r(m):
    """Requirement vector r (rho_dec units) over LEGS10 from physical motion needs m: surge/sway/diag m/s, yaw deg/s,
    the directions used (`<axis>_dirs`; surge/sway default to both signs), and the team size (yaw reference).
    r = required speed / nominal healthy achieved speed of that leg: V_NOM*RHO_REF (surge, sway), |V_DIAG|*RHO_REF
    (diagonals), W_NOM*yaw_scale(team)*RHO_REF (yaw). NaN = leg not needed."""
    r = dict.fromkeys(LEGS10, np.nan)
    nom = dict(surge=V_NOM * RHO_REF, sway=V_NOM * RHO_REF, diag=V_DIAG * RHO_REF,
               yaw=W_NOM * yaw_scale_n(m.get("team_modules", 1)) * RHO_REF)
    for ax in ("surge", "sway", "diag", "yaw"):
        v = float(m.get("yaw_dps" if ax == "yaw" else f"{ax}_mps", 0) or 0)
        if v <= 0:
            continue
        if ax == "yaw":
            v = math.radians(v)
        dirs = m.get(f"{ax}_dirs")
        if not dirs:
            dirs = DIR_LEGS[ax][:2] if ax in ("surge", "sway", "yaw") else DIR_LEGS[ax]
        for d in dirs:
            if d in r:
                r[d] = v / nom[ax]
    return r


KEYWORDS = dict(
    sway=r"\b(sideways|side-?step|lateral(ly)?|strafe|strafing|crab(bing)?|slide (left|right)|shift (left|right))\b",
    yaw=r"\b(turn(s|ing)? in place|rotate|rotating|rotation|pivot|spin|reorient|re-orient|u-turn|about-face)\b",
    diag=r"\b(diagonal(ly)?|at an angle|45.?deg|45°)\b",
    reverse=r"\b(revers(e|ing)|back(s|ing)? (out|up|away)|backward)\b")


BODY_DIMS = {1: (0.2, 0.2), 2: (0.4, 0.2), 3: (0.4, 0.4), 4: (0.4, 0.4), 5: (0.6, 0.4), 6: (0.6, 0.4), 7: (0.8, 0.4),
             8: (0.8, 0.4)}                 # most compact body per size (m); module 0.2 m square
NEG = r"\b(no|not|without|never|avoid(ing)?|nor|zero)\b(\W+\w+){0,4}\W+$"


def _mentions(pat, txt):
    """A positive (not negated within the 5 preceding words) mention of the motion pattern."""
    for mt in re.finditer(pat, txt):
        if not re.search(NEG, txt[max(0, mt.start() - 60):mt.start()]):
            return True
    return False


def consistency(row, m):
    """Flags (list of strings) of internal inconsistencies between the description and the motion fields, and of
    physical impossibilities (body wider than the narrowest passage, team too small for the payload)."""
    txt = (row.get("title", "") + ". " + row.get("description", "")).lower()
    fl = []
    if _mentions(KEYWORDS["sway"], txt) and not m.get("sway_mps"):
        fl.append("text needs sway, no sway")
    if _mentions(KEYWORDS["yaw"], txt) and not m.get("yaw_dps"):
        fl.append("text needs yaw, no yaw")
    if _mentions(KEYWORDS["diag"], txt) and not m.get("diag_mps"):
        fl.append("text needs diagonal, no diagonal")
    rear = {"surge-", "diag-+", "diag--"} & set((m.get("surge_dirs") or []) + (m.get("diag_dirs") or []))
    if _mentions(KEYWORDS["reverse"], txt) and not rear:
        fl.append("text needs reverse, no rearward leg")
    team = int(row.get("team_modules", 1))
    aw = float(row.get("environment", {}).get("aisle_width_m") or 0)
    if 0 < aw < min(BODY_DIMS.get(team, (0.8, 0.4))) - 1e-9:
        fl.append("body wider than the narrowest passage")
    pk = float(row.get("payload_kg", 0) or 0)
    if pk > 0 and team < math.ceil(pk / PAYLOAD_PER_MODULE_KG):
        fl.append("team too small for payload")
    if not any(m.get(k) for k in ("surge_mps", "sway_mps", "diag_mps", "yaw_dps")):
        fl.append("no motion")
    return fl


def _tokens(s):
    return set(re.findall(r"[a-z]{3,}", s.lower()))


def cmd_tasks_build(a):
    import pandas as pd
    gen = {x["case_id"]: x for x in _r(RES / f"{a.gen}.jsonl")}
    red = {x["case_id"]: x for x in _r(RES / f"{a.rederive}.jsonl")} if a.rederive else {}
    rows, seen = [], []
    for cid in sorted(gen):
        g = gen[cid]
        m = g["motion"]
        team = max(int(g.get("team_modules", 1)), 1)
        min_mod = max(1, math.ceil(float(g.get("payload_kg", 0) or 0) / PAYLOAD_PER_MODULE_KG))
        r = task_r(dict(m, team_modules=team))
        fin = [v for v in r.values() if np.isfinite(v)]
        in_range = int(bool(fin) and all(0 < v <= 1.0 for v in fin) and 0 <= float(g.get("payload_kg", 0)) <= 60)
        flags = consistency(g, m)
        tok = _tokens(g["title"] + " " + g["description"])
        dup = ""
        for sid, st in seen:
            if len(tok & st) / max(len(tok | st), 1) >= a.dup_jaccard:
                dup = sid
                break
        seen.append((cid, tok))
        row = dict(task_id=cid, domain=g["domain"], title=g["title"], description=g["description"],
                   payload_kg=float(g.get("payload_kg", 0) or 0), team_modules=team, min_modules=min_mod,
                   kind="payload" if min_mod >= 2 else "route",
                   **{f"phys_{k}": (v if not isinstance(v, list) else "+".join(v)) for k, v in m.items()},
                   **{f"r_{l}": (round(v, 3) if np.isfinite(v) else np.nan) for l, v in r.items()},
                   in_range=in_range, consistent=int(not flags), flag_list="; ".join(flags), duplicate_of=dup)
        if cid in red:
            r2 = task_r(dict(red[cid]["motion"], team_modules=team))
            s1 = {l for l, v in r.items() if np.isfinite(v)}
            s2 = {l for l, v in r2.items() if np.isfinite(v)}
            both = s1 & s2
            row.update(rederived_legs="+".join(sorted(s2)), leg_set_jaccard=round(len(both) / max(len(s1 | s2), 1), 3),
                       leg_set_equal=int(s1 == s2),
                       level_mad=round(float(np.mean([abs(r[l] - r2[l]) for l in both])), 3) if both else np.nan,
                       r2_max_abs_diff=round(float(max([abs(r[l] - r2[l]) for l in both], default=np.nan)), 3)
                       if both else np.nan)
            row["rederive_agree"] = int(row["leg_set_jaccard"] >= 0.75 and (
                not both or row["r2_max_abs_diff"] <= a.level_tol))
        row["include"] = int(in_range and not flags and not dup and row.get("rederive_agree", 1))
        rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(RES / "task_library_all.csv", index=False)
    keep = D[D.include == 1]
    cols = ["task_id", "domain", "kind", "title", "description", "payload_kg", "min_modules", "team_modules"] + \
           [f"r_{l}" for l in LEGS10]
    keep[cols].to_csv(RES / "task_library.csv", index=False)
    rep = dict(generated=len(D), in_range=int(D.in_range.sum()), consistent=int(D.consistent.sum()),
               duplicates=int((D.duplicate_of != "").sum()),
               rederived=int(D.get("rederive_agree", pd.Series(dtype=float)).notna().sum()),
               rederive_agree=float(D["rederive_agree"].mean()) if "rederive_agree" in D else None,
               leg_set_equal=float(D["leg_set_equal"].mean()) if "leg_set_equal" in D else None,
               leg_set_jaccard_mean=float(D["leg_set_jaccard"].mean()) if "leg_set_jaccard" in D else None,
               level_mad_median=float(D["level_mad"].median()) if "level_mad" in D else None,
               included=len(keep), by_domain=keep.domain.value_counts().to_dict(),
               by_kind=keep.kind.value_counts().to_dict(),
               legs_required_share={l: round(float(keep[f"r_{l}"].notna().mean()), 3) for l in LEGS10},
               r_quantiles={q: round(float(np.nanquantile(keep[[f"r_{l}" for l in LEGS10]].to_numpy(), q)), 3)
                            for q in (0.1, 0.5, 0.9)} if len(keep) else {},
               flag_counts=Counter(f for x in D["flag_list"] for f in x.split("; ") if f))
    (RES / "task_library_report.json").write_text(json.dumps(rep, indent=1, default=_np))
    print(json.dumps(rep, indent=1, default=_np))


# ============================================================================= workstream C: explanation faithfulness
def cmd_cases_c(a):
    """Refused (candidate, task) pairs explained from the robot's own posterior only. Predictors: the T0 enumeration
    posterior (v2, sim0.1), the oracle (true latents, a single particle) and the nominal plug-in (all healthy)."""
    vis_t0 = {x["case_id"]: x for x in _r(RES / "t0_cases.jsonl")}
    truth = pickle.loads((OUT / "t0_truth.pkl").read_bytes())
    rng = np.random.default_rng(3)
    refused = [x for x in vis_t0.values() if any(k.startswith("refused") for k in x["configs"]["v2/sim0.1"]["kinds"])]
    cap_r = [x for x in refused if x["truth"]["capable"]]
    inc_r = [x for x in refused if not x["truth"]["capable"]]
    pick = lambda xs, n: [xs[i] for i in rng.permutation(len(xs))[:n]]          # noqa: E731
    cap_pick = pick(cap_r, a.n_post // 2)
    base = pick(inc_r, a.n_post - len(cap_pick)) + cap_pick
    import pandas as pd
    eps = {e["meta"]["seed"]: e for e in pickle.loads(Path(a.eps).read_bytes())}
    dfe = pd.read_csv(a.episodes_csv)
    qn = {int(r.eid): float(r.q) for r in dfe[(dfe.predictor == "nominal") & (dfe.cert == "sim0.1")].itertuples()}
    vis, hid = [], []
    for x in base:
        sd, c, t = (int(re.search(r"t0-(\d+)-c(\d+)-t(\d+)", x["case_id"]).group(i)) for i in (1, 2, 3))
        T = truth[sd]
        e = eps[sd]
        lidx = {n: i for i, n in enumerate(e["legs"])}
        for pred in ("posterior", "oracle", "nominal")[:a.n_preds]:
            if pred == "oracle" and x["truth"]["capable"]:
                continue                       # the oracle (q = 0) never refuses a capable pair
            legs = {}
            for ln, g in x["legs"].items():
                if pred == "posterior":
                    legs[ln] = dict(r=g["r"], believed_mean=g["mean_v2"], believed_sd=g["sd_v2"],
                                    certified_lower=g["lower_v2"], p_meets_r=g["p_leg_v2"])
                elif pred == "oracle":
                    legs[ln] = dict(r=g["r"], believed_mean=g["truth"], believed_sd=0.0,
                                    certified_lower=None, p_meets_r=float(g["truth"] >= g["r"]))
                else:
                    rn = float(e["rho_nom"][c, lidx[ln]])
                    legs[ln] = dict(r=g["r"], believed_mean=_f(rn), believed_sd=0.0,
                                    certified_lower=_f(rn - qn[sd] * 0.02), p_meets_r=float(rn >= g["r"]))
            if pred == "posterior":
                belief = x["posterior_v2"]
            elif pred == "oracle":
                fk = _hyp_str(T["tm"], T["tg"], T["wheels"])
                belief = dict(n_particles=1, top_hypotheses=[dict(faults=fk, mass=1.0)],
                              wheel_fault_prob={T["wheels"][i]: 1.0 for i in np.flatnonzero(T["tm"])})
            else:
                belief = dict(n_particles=1, top_hypotheses=[dict(faults="healthy", mass=1.0)], wheel_fault_prob={},
                              note=f"nominal plug-in: every wheel assumed healthy, nominal nuisances; certificate "
                                   f"margin q = {qn[sd]:.1f} x 0.02")
            if pred == "oracle":                   # oracle certificate: q = 0, lower = the true capability
                for ln in legs:
                    legs[ln]["certified_lower"] = legs[ln]["believed_mean"]
            cid = f"c-{pred}-{x['case_id']}"
            vis.append(dict(case_id=cid, predictor=pred, candidate=x["candidate"], parent=x["parent"],
                            task=x["task"], belief=belief, legs=legs,
                            decision="refused: the certificate did not approve this body for this task"))
            hid.append(dict(case_id=cid, base=x["case_id"], predictor=pred))
    _w(RES / "c_cases.jsonl", vis)
    _w(RES / "c_truth.jsonl", hid)
    print(f"c cases {len(vis)} ({len(base)} refused pairs x {a.n_preds} predictors); "
          f"{len(inc_r)} incapable + {len(cap_r)} capable refused pairs available")


def faithful(ix, ex):
    """Deterministic faithfulness of an explanation {blamed_wheels, blamed_legs, blames_uncertainty} against truth:
    a blamed wheel must be faulted in the true latents; a blamed leg must truly be below its requirement.
    An explanation that blames only uncertainty (no wheel, no leg) is scored on its own: it is faithful iff the pair
    is truly capable or the true fault is absent from the robot's belief (the robot could not know)."""
    W = ix["wheels"]
    bw = [w for w in ex.get("blamed_wheels", []) if w]
    bl = [l for l in ex.get("blamed_legs", []) if l]
    wf = [W.get(w, {"mode": "ok"})["mode"] != "ok" for w in bw]
    lf = []
    for l in bl:
        g = ix["legs"].get(l)
        lf.append(None if g is None else bool(g["truth"] < g["thr"]))
    lf_ok = [x for x in lf if x is not None]
    wp = float(np.mean(wf)) if wf else np.nan
    lp = float(np.mean(lf_ok)) if lf_ok else np.nan
    true_f = [n for n, w in W.items() if w["mode"] != "ok" and n in ix["cand_wheels"]]
    wr = float(np.mean([n in bw for n in true_f])) if true_f else np.nan
    ok = (all(wf) if wf else True) and (all(lf_ok) if lf_ok else True) and (bool(wf) or bool(lf_ok) or
                                                                             ex.get("blames_uncertainty", False))
    return dict(wheel_precision=wp, leg_precision=lp, wheel_recall=wr, faithful=bool(ok),
                n_blamed_wheels=len(bw), n_blamed_legs=len(bl), unknown_legs=sum(x is None for x in lf))


def belief_consistency(vis, ex):
    """Is the explanation true to what the robot believes (the case codex saw)? A blamed leg must have believed mean
    < r (or P(meets r) < 0.5); a blamed wheel must carry >= 0.5 believed fault probability. Separates codex
    misreporting from a wrong posterior."""
    wp = vis.get("belief", {}).get("wheel_fault_prob", {})
    bw = [w for w in ex.get("blamed_wheels", []) if w]
    bl = [l for l in ex.get("blamed_legs", []) if l]
    wok = [wp.get(w, 0.0) >= 0.5 for w in bw]
    lok = []
    for l in bl:
        g = vis.get("legs", {}).get(l)
        if g is None or g.get("believed_mean") is None:
            lok.append(False)
        else:
            lok.append(g["believed_mean"] < g["r"] or (g.get("p_meets_r") is not None and g["p_meets_r"] < 0.5))
    return dict(belief_ok=bool(all(wok) and all(lok)), informative=int(bool(bw) or bool(bl)))


def cmd_validate_c(a):
    import pandas as pd
    ann = {x["case_id"]: x for x in _r(RES / f"{a.name}.jsonl")}
    hid = {x["case_id"]: x for x in _r(RES / "c_truth.jsonl")}
    vis_c = {x["case_id"]: x for x in _r(RES / "c_cases.jsonl")}
    t0 = {x["case_id"]: x for x in _r(RES / "t0_cases.jsonl")}
    t0h = {x["case_id"]: x for x in _r(RES / "t0_truth.jsonl")}
    rows = []
    for cid, ex in ann.items():
        h = hid.get(cid)
        if h is None:
            continue
        base = t0[h["base"]]
        ix = truth_index(base, t0h[h["base"]])
        f = faithful(ix, ex)
        f.update(belief_consistency(vis_c[cid], ex))
        rows.append(dict(case_id=cid, predictor=h["predictor"], capable=int(base["truth"]["capable"]),
                         truth_in_screen=t0h[h["base"]]["truth_in_screen"], cand_has_fault=ix["has_fault"],
                         blames_uncertainty=int(ex.get("blames_uncertainty", False)), **f,
                         explanation=ex.get("explanation", "")[:300]))
    D = pd.DataFrame(rows)
    D.to_csv(RES / f"{a.name}_faithfulness.csv", index=False)

    def agg(d):
        inf = d[d.informative == 1]
        return pd.Series(dict(n=len(d), informative=d.informative.mean(), faithful=d.faithful.mean(),
                              faithful_given_informative=inf.faithful.mean() if len(inf) else np.nan,
                              belief_consistent=d.belief_ok.mean(), wheel_precision=d.wheel_precision.mean(),
                              wheel_precision_n=int(d.wheel_precision.notna().sum()),
                              leg_precision=d.leg_precision.mean(), leg_precision_n=int(d.leg_precision.notna().sum()),
                              wheel_recall=d.wheel_recall.mean(), blames_uncertainty=d.blames_uncertainty.mean()))
    S = D.groupby(["predictor", "capable"]).apply(agg, include_groups=False).reset_index()
    S2 = D.groupby(["predictor"]).apply(agg, include_groups=False).reset_index().assign(capable="all")
    out = pd.concat([S2, S])
    B = []
    rng = np.random.default_rng(0)
    for p, d in D.groupby("predictor"):                  # bootstrap CI of the faithful rate (cases)
        v = d.faithful.to_numpy(float)
        bs = [v[rng.integers(len(v), size=len(v))].mean() for _ in range(2000)]
        B.append(dict(predictor=p, faithful_all_lo=np.percentile(bs, 2.5), faithful_all_hi=np.percentile(bs, 97.5)))
    out = out.merge(pd.DataFrame(B), on="predictor", how="left")
    out.to_csv(RES / f"{a.name}_summary.csv", index=False)
    print(out.round(3).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    eps_default = str(CFOUT / "decision" / "pilot_t0_eps600.pkl")
    s = sub.add_parser("t0-truth"); s.add_argument("--eps", default=eps_default); s.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("cases-t0"); s.add_argument("--eps", default=eps_default)
    s.add_argument("--episodes-csv", default=str(REPO / "results/decision/pilot_t0_episodes.csv"))
    s = sub.add_parser("cases-a2"); s.add_argument("--workers", type=int, default=4)
    s.add_argument("--fresh", action="store_true", help="re-run the replay even if $CFCAP_OUT/annotations/a2_replay.pkl exists")
    sub.add_parser("cases-a5")
    s = sub.add_parser("cases-a"); s.add_argument("--out", default="a_all"); s.add_argument("--pilot", type=int, default=0)
    s.add_argument("--extra", default="", help="more <x>_cases/<x>_truth sources (T1 campaigns)")
    s.add_argument("--t0-all-only", type=int, default=250)
    s = sub.add_parser("validate-a"); s.add_argument("--name", required=True); s.add_argument("--cases", default="a_all")
    s.add_argument("--min-claim-acc", type=float, default=0.9); s.add_argument("--min-sig", type=float, default=0.8)
    s.add_argument("--min-n", type=int, default=10)
    s = sub.add_parser("tasks-slots"); s.add_argument("--n", type=int, default=600)
    s = sub.add_parser("tasks-build"); s.add_argument("--gen", default="tasks_gen"); s.add_argument("--rederive", default="tasks_rederive")
    s.add_argument("--dup-jaccard", type=float, default=0.6); s.add_argument("--level-tol", type=float, default=0.15)
    s = sub.add_parser("cases-c"); s.add_argument("--n-post", type=int, default=200); s.add_argument("--n-preds", type=int, default=3)
    s.add_argument("--eps", default=eps_default)
    s.add_argument("--episodes-csv", default=str(REPO / "results/decision/pilot_t0_episodes.csv"))
    s = sub.add_parser("validate-c"); s.add_argument("--name", default="c_explain")
    a = ap.parse_args(argv)
    fn = {"t0-truth": cmd_t0_truth, "cases-t0": cmd_cases_t0, "cases-a2": cmd_cases_a2, "cases-a5": cmd_cases_a5,
          "cases-a": cmd_cases_a, "validate-a": cmd_validate_a, "tasks-slots": cmd_tasks_slots,
          "tasks-build": cmd_tasks_build, "cases-c": cmd_cases_c, "validate-c": cmd_validate_c}[a.cmd]
    fn(a)


if __name__ == "__main__":
    main()
