"""Export pulled `train` runs to the paper contracts (docs/board/paper.md §1a, 1d, 1e):

  results/models/paper/dec_models.csv      amortized_{marg,sim,comp}, blackbox_gnn, ensemble (headline diversity)
  results/models/paper/diversity.csv       every grid run x test morphology x metric
  results/models/paper/latency_models.csv  s/decision (episode) per method and shape

  $PY scripts/models_export.py RUN_ROOT [RUN_ROOT ...] [--headline n6] [--status exploratory]

A run dir is any directory with manifest.json of task `train` (its args give grid + shard -> arch, div, seed).
Label: rho_dec (truth column of the eval units). Lower bounds at ALPHA = 0.10, calibrated on the run's own eval-cal
rows (T1 cal), applied unchanged to test / H3 / H4. Routes:
  marginal      per-entry split conformal on (point - truth) / scale
  simultaneous  cfcap.conformal.core.SimultaneousLower (max-normalised over every candidate x leg of an episode)
  compositional simultaneous layer on the compositional quantiles (cal/test episodes that carry them)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cfcap.conformal import core  # noqa: E402
from cfcap.models import train as TR  # noqa: E402

ALPHA = 0.10
TAUS = (0.6, 0.7, 0.8)
KIND = {0: "self", 1: "child", 2: "merge"}
TOPO = ("single", "chain", "L", "path", "cyclic", "grid", "T", "star", "branched")
NAMED = {("single", 1): "single", ("chain", 2): "pair", ("L", 3): "L3", ("grid", 4): "square", ("grid", 6): "2x3"}


def shape_name(topo, n):
    t = TOPO[int(topo)]
    return NAMED.get((t, int(n)), f"{t}{int(n)}")


def complete_manifests(roots, task):
    """One manifest per job: among every pulled copy of a job_id (a job is pulled once per box it ran on, and
    preempted attempts leave partial copies), only the copy whose manifest status is "complete" is used (the most
    recently updated if several). Jobs without a complete copy are skipped. Returns [(dir, manifest)]."""
    best = {}
    for r in roots:
        for m in pathlib.Path(r).rglob("manifest.json"):
            try:
                man = json.loads(m.read_text())
            except ValueError:
                continue
            if man.get("task") != task or man.get("status") != "complete":
                continue
            key = man.get("job_id") or str(m.parent)
            if key not in best or str(man.get("updated", "")) > str(best[key][1].get("updated", "")):
                best[key] = (m.parent, man)
    return list(best.values())


def find_runs(roots):
    """Complete train runs, one per grid entry (resubmissions b1 / b1x / b1w / ... of the same entry: the most
    recently updated complete copy)."""
    best = {}
    for mdir, man in complete_manifests(roots, "train"):
        argv = list(man.get("task_argv") or [])
        grid_name = argv[argv.index("--grid") + 1] if "--grid" in argv else "b1"
        shard = int(man.get("shard", 0))
        if shard >= len(TR.grid(grid_name)):
            continue
        key = (grid_name, shard)
        if key not in best or str(man.get("updated", "")) > str(best[key][1].get("updated", "")):
            best[key] = (mdir, man)
    return [dict(dir=d, cfg=TR.grid(k[0])[k[1]], grid=k[0], host=man.get("host", "")) for k, (d, man) in
            sorted(best.items())]


def load_unit(d, split):
    f = d / "units" / f"eval-{split}.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        df = pd.DataFrame({k: z[k] for k in z.files})
    meta = json.loads((d / "units" / f"eval-{split}.json").read_text()).get("meta", {})
    return df, meta


def to_matrix(df, point, scale):
    """Episode x (candidate, leg) matrices (NaN-padded) of point, scale, truth; rows keep df order via (eid, col)."""
    e = df.eid.to_numpy()
    ue, ei = np.unique(e, return_inverse=True)
    col = (df.cand.to_numpy() * 10 + df.leg.to_numpy())
    W = int(col.max()) + 1
    P_, S_, Y_ = (np.full((len(ue), W), np.nan) for _ in range(3))
    P_[ei, col], S_[ei, col], Y_[ei, col] = point, scale, df.truth.to_numpy()
    return P_, S_, Y_, ei, col


def lowers(cal, tests, point_fn, route):
    """Fit on cal (DataFrame) with point/scale from point_fn(df) -> (point, scale); return lower per test df."""
    pc, sc = point_fn(cal)
    ok = np.isfinite(pc) & np.isfinite(sc)
    out = []
    if route == "marginal":
        q = core.split_quantile((pc[ok] - cal.truth.to_numpy()[ok]) / np.maximum(sc[ok], 1e-6), ALPHA)
        for t in tests:
            p, s = point_fn(t)
            out.append(np.where(np.isfinite(p), p - q * np.maximum(s, 1e-6), np.nan))
        return out, q
    c = cal[ok]
    Pm, Sm, Ym, _, _ = to_matrix(c, pc[ok], sc[ok])
    sl = core.SimultaneousLower(ALPHA).fit(Pm, Sm, Ym)
    for t in tests:
        p, s = point_fn(t)
        out.append(np.where(np.isfinite(p), sl.lower(p, s), np.nan))
    return out, sl.q


direct = lambda df: (df.q50.to_numpy(float), np.maximum(df.q50 - df.q05, 1e-3).to_numpy(float))
comp = lambda df: (df.cq50.to_numpy(float), np.maximum(df.cq50 - df.cq05, 1e-3).to_numpy(float))


def dec_rows(method, route, posterior, splits, lower, point_fn, seconds_fn, status):
    out = []
    for (split, df), lo in zip(splits, lower):
        p, s = point_fn(df)
        keep = np.isfinite(p)
        d = df[keep]
        out.append(pd.DataFrame(dict(
            method=method, route=route, posterior=posterior, train="T1", test_plant="T1", split=split, status=status,
            eid=d.eid.to_numpy(), shape=[shape_name(t, n) for t, n in zip(d.parent_topo, d.parent_n)],
            family=d.cand_family.to_numpy(), cand=d.cand.to_numpy(), kind=d.kind.map(KIND).to_numpy(),
            cand_size=d.cand_size.to_numpy(), has_fault=d.has_fault.to_numpy(), leg=d.leg.to_numpy(),
            truth=d.truth.to_numpy(), point=p[keep], scale=s[keep], lower=np.where(np.isfinite(lo[keep]), lo[keep],
                                                                                    -np.inf),
            alpha=ALPHA, seconds=seconds_fn(split, d))))
    return out


def decision_metrics(lower, truth):
    lo = np.nan_to_num(lower, nan=-np.inf)
    cf, uc = [], []
    for tau in TAUS:
        go, capable = lo >= tau, truth >= tau
        cf.append(((go & capable).sum(), capable.sum()))
        uc.append(((go & ~capable).sum(), (~capable).sum()))
    r = lambda x: sum(a for a, _ in x) / max(1, sum(b for _, b in x))
    return r(cf), r(uc)


def pinball(q, y, taus=(0.05, 0.25, 0.5, 0.75, 0.95)):
    d = y[:, None] - q
    t = np.asarray(taus)
    return float(np.mean(np.maximum(t * d, (t - 1) * d)))


def load_eval2(roots):
    """Stage-2 runs (task models_eval2, complete copies only, deduplicated by (job, unit) and then by (model, split,
    episode) across resubmitted campaigns): {tag: {split: DataFrame}} with end-to-end sec_direct / sec_comp."""
    parts, seen_units = {}, set()
    for mdir, man in complete_manifests(roots, "models_eval2"):
        job = man.get("job_id") or str(mdir)
        for f in sorted((mdir / "units").glob("*.npz")):
            if f.stem.startswith("env-") or (job, f.stem) in seen_units:
                continue
            seen_units.add((job, f.stem))
            meta = json.loads(f.with_suffix(".json").read_text()).get("meta", {})
            tags = str(meta.get("models", "")).split(",")
            with np.load(f) as z:
                df = pd.DataFrame({k: z[k] for k in z.files})
            split = f.stem.rsplit("-", 2)[0]
            for mi, g in df.groupby("model"):
                parts.setdefault(tags[int(mi)], {}).setdefault(split, []).append(
                    g.drop(columns="model").assign(_hw=meta.get("hardware", "")))
    out = {}
    for t, d in parts.items():
        out[t] = {}
        for s_, v in d.items():
            df = pd.concat(v, ignore_index=True)
            out[t][s_] = df.drop_duplicates(["eid", "cand", "leg"]).reset_index(drop=True)
    return out


def headline_from_eval2(e2, status, calib_out=None):
    """dec + latency rows from stage-2 evaluations (end-to-end seconds), and the T1-cal conformal constants."""
    dec, lat, calib = [], [], {}
    order = ("cal", "test", "H3", "H4")
    for tag, by in sorted(e2.items()):
        if "cal" not in by:
            continue
        arch, div_, seed = tag.split("-")
        evs = [(s, by[s]) for s in order if s in by]
        cal = by["cal"]
        sec_d = lambda split, df: df.sec_direct.to_numpy()
        sec_c = lambda split, df: df.sec_comp.to_numpy()
        sim_lo, q_sim = lowers(cal, [df for _, df in evs], direct, "simultaneous")
        mar_lo, q_marg = lowers(cal, [df for _, df in evs], direct, "marginal")
        calib[tag] = dict(alpha=ALPHA, q_sim=float(q_sim), q_marg=float(q_marg))
        head = "1" if arch in ("bb", "bbh") else "2"                     # A-SEED: st headline = seed 2
        sfx = "" if seed == head else f"_s{seed}"
        if arch == "bb":
            dec += dec_rows(f"blackbox_gnn{sfx}", "simultaneous", "none", evs, sim_lo, direct, sec_d, status)
            continue
        name = "amortized" if arch == "st" else f"amortized_{arch}"
        # direct head of the posterior network: posterior "amortized_direct" (not "ours"; ours = *_comp)
        dec += dec_rows(f"{name}_marg{sfx}", "marginal", "amortized_direct", evs, mar_lo, direct, sec_d, status)
        dec += dec_rows(f"{name}_sim{sfx}", "simultaneous", "amortized_direct", evs, sim_lo, direct, sec_d, status)
        if np.isfinite(cal.cq50).any():
            c_lo, q_comp = lowers(cal, [df for _, df in evs], comp, "compositional")
            calib[tag]["q_comp"] = float(q_comp)
            ours = arch == "st" and sfx == ""                       # exactly one "ours" (st-n6-2, compositional)
            dec += dec_rows(f"{name}_comp{sfx}", "compositional", "amortized" if ours else "amortized_variant", evs,
                            c_lo, comp, sec_c, status)
        for s, df in evs:
            ep = df.groupby("eid").first()
            for (topo, n), g in ep.groupby(["parent_topo", "parent_n"]):
                hw = str(g._hw.iloc[0])
                for meth, col in ((f"{name}_sim{sfx}", "sec_direct"), (f"{name}_comp{sfx}", "sec_comp")):
                    v = g[col].dropna()
                    if len(v):
                        lat.append(dict(method=meth, shape=shape_name(topo, n), n_wheels=int(4 * n),
                                        sec_per_decision=float(v.median()), hardware=hw, status=status,
                                        n=int(len(v)), split=s, end_to_end=1))
    if calib_out:
        pathlib.Path(calib_out).mkdir(parents=True, exist_ok=True)
        for tag, c in calib.items():
            (pathlib.Path(calib_out) / f"calib_{tag}.json").write_text(json.dumps(c, indent=1))
    return dec, lat


# ------------------------------------------------------------------ confirmatory (docs/board/paper.md §5a)
CONF_OPEN = "2026-11-26T01:45:00Z"
CLAIM_OF = {"conf_test": "C-MAIN", "conf_H4": "C-SCALE", "conf_H4b": "C-SCALE", "conf_het_test": "C-HET",
            "conf_H1": "C-HET", "conf_H2": "C-HET"}
CAL_OF = {"C-MAIN": "cal", "C-SCALE": "cal", "C-HET": "het_cal"}
# (claim, method) -> (model tag, quantile column prefix: "" direct, "c" compositional, "b" type-blind compositional)
CONF_METHODS = {
    ("C-MAIN", "amortized_comp"): ("st-n6-2", "c"), ("C-MAIN", "credible_lower"): ("st-n6-2", "c"),
    ("C-SCALE", "amortized_comp"): ("st-n6-2", "c"), ("C-SCALE", "st_direct"): ("st-n6-2", ""),
    ("C-SCALE", "blackbox_gnn"): ("bb-n6-1", ""),
    ("C-HET", "amortized_comp"): ("sth-het-1", "c"), ("C-HET", "amortized_comp_blind"): ("sth-het-1", "b"),
    ("C-HET", "st_direct"): ("sth-het-1", ""), ("C-HET", "blackbox_gnn"): ("bbh-het-1", ""),
}


# CONFIRMATORY A-SEED (§10): the confirmatory model is st-n6-2; st-n6-1, if it lands, is a descriptive
# seed-variation column only (its own file, outside the dec_* glob, never a bar)
CONF_MODEL = "st-n6-2"
SEEDVAR_METHODS = {
    ("C-MAIN", "amortized_comp"): ("st-n6-1", "c"),
    ("C-SCALE", "amortized_comp"): ("st-n6-1", "c"), ("C-SCALE", "st_direct"): ("st-n6-1", ""),
}


def _qfn(prefix):
    return lambda df: (df[prefix + "q50"].to_numpy(float),
                       np.maximum(df[prefix + "q50"] - df[prefix + "q05"], 1e-3).to_numpy(float))


def confirmatory_wheels(roots, out_path, model=CONF_MODEL):
    """C-EXPLAIN input (CONFIRMATORY §6b): per-wheel mode posteriors of `model` on conf_test 40-3039, from the
    models_eval2 wheel sidecars, as one npz next to dec_conf_models: eid, asm_role (0 parent, 1.. merge partners),
    wheel (index within its assembly, 4 k + corner), ep_wheel (index over the episode's assemblies in role order, the
    generator's wheel order), p (W, 4) = P(ok, weak, coast, brake), g_mean (Beta mean of the weak gain). The per-leg
    posterior median rho_dec of each candidate is `point` of the amortized_comp rows in dec_conf_models. Truth is
    not stored (regenerate it from the episode index)."""
    parts = []
    for mdir, man in complete_manifests(roots, "models_eval2"):
        for f in sorted(mdir.glob("wheels/conf_test-*.npz")):
            with np.load(f) as z:
                if str(z["model"]) != model:
                    continue
                d = {k: z[k] for k in ("eid", "asm_role", "wheel", "ep_wheel", "p", "g_mean")}
            keep = (d["eid"] >= 40) & (d["eid"] < 3040)
            parts.append({k: v[keep] for k, v in d.items()})
    if not parts:
        return None
    out = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
    o = np.lexsort((out["ep_wheel"], out["eid"]))
    out = {k: v[o] for k, v in out.items()}
    _, first = np.unique(np.stack([out["eid"], out["ep_wheel"]], 1), axis=0, return_index=True)   # dedup copies
    out = {k: v[np.sort(first)] for k, v in out.items()}
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out["model_tag"] = np.full(len(out["eid"]), model)
    np.savez_compressed(out_path, classes=np.array(["ok", "weak", "coast", "brake"]), model=np.array(model), **out)
    return out_path


def train_cal_rows(runs):
    """Fallback cal rows per model tag from the train runs' own eval-cal / eval-het_cal units (exploratory cal split,
    same generator episodes 0..eval_n-1; compositional quantiles on the first compose_n), shaped like eval2 rows."""
    out = {}
    for run in runs:
        c = run["cfg"]
        tag = f"{c['arch']}-{c['div']}-{c['seed']}"
        for split in ("cal", "het_cal"):
            u = load_unit(run["dir"], split)
            if u is None:
                continue
            df, meta = u
            df = df.assign(sec_comp=df.get("comp_seconds", np.nan),
                           sec_direct=meta.get("infer_ms_per_episode", np.nan) / 1e3, shard=-1)
            out.setdefault(tag, {})[split] = df
    return out


def confirmatory_rows(e2, now=None, allow_early=False, methods=None, fallback=None):
    """dec_conf_models rows: per (claim, method) the T1-cal (exploratory) rows and the conf_* rows with lower
    (simultaneous) and lower_marg (marginal), fixed-T1-reference rho_dec truth, claim, shard, parent_n, variant."""
    import datetime as dt
    now = now or dt.datetime.now(dt.timezone.utc)
    if not allow_early and now < dt.datetime(2026, 11, 26, 1, 45, tzinfo=dt.timezone.utc):
        raise SystemExit(f"confirmatory labels stay closed until {CONF_OPEN} (now {now:%Y-%m-%dT%H:%MZ})")
    out = []
    for (claim, method), (tag, pre) in (CONF_METHODS if methods is None else methods).items():
        by = dict(e2.get(tag) or {})
        cal_split = CAL_OF[claim]
        cal_src = "eval2"
        if cal_split not in by and fallback and cal_split in fallback.get(tag, {}):
            by[cal_split] = fallback[tag][cal_split]            # stage-2 cal job missing: train-run eval-cal rows
            cal_src = "train-run-eval-cal"
            print(f"confirmatory: {claim}/{method}: {tag} {cal_split} rows from the train run's eval unit")
        if not by or cal_split not in by:
            print(f"confirmatory: {claim}/{method}: no {tag} {cal_split} rows, skipped")
            continue
        tests = [(s, df) for s, df in by.items() if CLAIM_OF.get(s) == claim]
        if claim == "C-MAIN":
            tests = [(s, df[(df.eid >= 40) & (df.eid < 3040)]) for s, df in tests]
        if not tests:
            continue
        cal = by[cal_split]
        fn = _qfn(pre)
        if not np.isfinite(fn(cal)[0]).any():
            continue
        evs = [("cal", cal)] + tests
        if method == "credible_lower":      # Bayes q05 of the compositional posterior samples (no conformal layer)
            sim_lo = mar_lo = [df[pre + "q05"].to_numpy(float) for _, df in evs]
            q = 0.0
        else:
            sim_lo, q = lowers(cal, [df for _, df in evs], fn, "simultaneous")
            mar_lo, _ = lowers(cal, [df for _, df in evs], fn, "marginal")
        secs = (lambda split, df: df.sec_comp.to_numpy()) if pre else (lambda split, df: df.sec_direct.to_numpy())
        for (split, df), lo, lm in zip(evs, sim_lo, mar_lo):
            p_, s_ = fn(df)
            keep = np.isfinite(p_)
            d = df[keep]
            st = "exploratory" if split == "cal" else "confirmatory"
            out.append(pd.DataFrame(dict(
                claim=claim, method=method, model_tag=tag, variant=(f"{tag}:comp_posterior_q05" if method == "credible_lower" else
                                                    f"{tag}:{'comp' if pre == 'c' else 'comp_blind' if pre == 'b' else 'direct'}"
                                                    f"+simultaneous(q={q:.3f})") + f";cal={cal_src}",
                route=("none" if method == "credible_lower" else "compositional" if pre else "simultaneous"),
                posterior="amortized", train="T1", test_plant="T1",
                split=split, status=st, eid=d.eid.to_numpy(), shard=d["shard"].to_numpy() if "shard" in d else -1,
                shape=[shape_name(t, n) for t, n in zip(d.parent_topo, d.parent_n)], parent_n=d.parent_n.to_numpy(),
                family=d.cand_family.to_numpy(), cand=d.cand.to_numpy(), kind=d.kind.map(KIND).to_numpy(),
                cand_size=d.cand_size.to_numpy(), has_fault=d.has_fault.to_numpy(), leg=d.leg.to_numpy(),
                truth=d.truth.to_numpy(), point=p_[keep], scale=s_[keep],
                lower=np.where(np.isfinite(lo[keep]), lo[keep], -np.inf),
                lower_marg=np.where(np.isfinite(lm[keep]), lm[keep], -np.inf), alpha=ALPHA,
                seconds=secs(split, d))))
    return out


def load_env(roots):
    """C-ENV rows of models_eval2 env units (complete copies only; deduplicated by (family, mission, cand, seg))."""
    parts = []
    for mdir, man in complete_manifests(roots, "models_eval2"):
        for f in sorted((mdir / "units").glob("env-*.npz")):
            with np.load(f) as z:
                parts.append(pd.DataFrame({k: z[k] for k in z.files}))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(["family", "mission", "cand", "seg"])


ENV_FLOOR = 0.01                     # scale floor = label resolution (CONFIRMATORY A-ENV 2, the H9 fix)


def env_truth(dirs):
    """Exploratory ID env truth rows (family test, missions 0-399) from pulled env_missions runs; conf families are
    never read here."""
    from cfcap.env import buildings as B
    parts = []
    for mdir, man in complete_manifests(dirs, "env_missions"):
        for f in sorted(mdir.glob("units/env-test-*.npz")):
            with np.load(f) as z:
                if "truth" not in z.files:
                    continue
                parts.append(pd.DataFrame({k: z[k] for k in ("family", "mission", "cand", "seg", "truth", "fp_ok",
                                                              "load_ok")}))
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts, ignore_index=True)
    return t[(t.family == B.FAM_CODE["test"]) & (t.mission < 400)]


def q_envcal(env, truth):
    """CONFIRMATORY A-ENV 4: q_env = conformal (1 - alpha) quantile of the per-mission max normalised shortfall
    (q50 - truth) / max(q50 - q05, floor) over feasible (candidate, segment) entries of the 400 exploratory ID
    missions."""
    from cfcap.env import buildings as B
    e = env[(env.status_conf == 0) & (env.family == B.FAM_CODE["test"]) & (env.mission < 400)]
    m = e.merge(truth, on=["family", "mission", "cand", "seg"])
    m = m[m.fp_ok.astype(bool) & m.load_ok.astype(bool)]
    if m.empty:
        return None, 0
    r = (m.q50 - m.truth) / np.maximum(m.q50 - m.q05, ENV_FLOOR)
    sc = r.groupby(m.mission).max().to_numpy()
    return float(core.split_quantile(sc, ALPHA)), len(sc)


def env_lowers(env, q_comp, out_dir, conf, q_env=None, model_tag="st-n6-2"):
    """Per-row bounds for scripts/env_score.py --lower NAME=PATH (columns family, mission, cand, seg, lower):
      raw                    plug-in posterior median (q = 0): a point, NOT a bound (bound_kind plugin_median)
      amortized_comp         q50 - q_comp * max(q50 - q05, 0.01), q_comp the T1-cal simultaneous compositional
                             constant (primary C-ENV certifier; empirical transfer, no mission-level guarantee)
      amortized_comp_envcal  the same quantiles with q_env fitted on the 400 exploratory ID missions (secondary)
    conf-ID mission 1499 is dropped (CONFIRMATORY A-ENV 1)."""
    from cfcap.env import buildings as B
    d = env[env.status_conf == int(conf)]
    d = d[~((d.family == B.FAM_CODE.get("conf-ID", -1)) & (d.mission == 1499))]
    if conf:                                # A-POWER (CONFIRMATORY §11): conf-ID 0-599, each conf-E-H 0-299
        lim = np.where(d.family == B.FAM_CODE.get("conf-ID", -1), 600, 300)
        d = d[d.mission < lim]
    if d.empty:
        return []
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["family", "mission", "building", "cand", "seg", "leg", "cond", "shard"]
    sc = np.maximum(d.q50 - d.q05, ENV_FLOOR)
    meths = [("raw", d.q50, "plugin_median"), ("amortized_comp", d.q50 - q_comp * sc, f"bound_q_comp={q_comp:.4f}"),
             ("credible_lower", d.q05, "posterior_q05")]
    if q_env is not None:
        meths.append(("amortized_comp_envcal", d.q50 - q_env * sc, f"bound_q_env={q_env:.4f}"))
    files = []
    for name, lo, kind in meths:
        f = out_dir / f"{'env_conf' if conf else 'env'}_lower_{name}.csv.gz"
        d[keep].assign(lower=np.where(np.isfinite(lo), lo, -np.inf), point=d.q50, scale=sc, method=name,
                       model_tag=model_tag,
                       bound_kind=kind, status="confirmatory" if conf else "exploratory").to_csv(f, index=False)
        files.append(f)
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--headline", default="n6")
    ap.add_argument("--status", default="exploratory")
    ap.add_argument("--hardware", default="RTX 3090 (vast)")
    ap.add_argument("--out", default=str(ROOT / "results/models/paper"))
    ap.add_argument("--stage-weights", default=None,
                    help="also copy each train run's manifest.json + weights.pt to DIR/<arch>-<div>-<seed>/ (stage-2 input)")
    ap.add_argument("--confirmatory", action="store_true",
                    help="write results/models/confirmatory/dec_conf_models.csv.gz (refused before 2026-11-26 01:45Z)")
    ap.add_argument("--env-model", default=CONF_MODEL)
    ap.add_argument("--env-truth", nargs="*", default=[],
                    help="pulled env_missions run dirs (exploratory test family) for the envcal constant q_env")
    ap.add_argument("--env-out", default=str(ROOT / "results/models"), help="env/ and confirmatory/ go under it")
    ap.add_argument("--allow-early", action="store_true", help="(mock data only) skip the 01:45Z guard")
    ap.add_argument("--conf-out", default=str(ROOT / "results/models/confirmatory/dec_conf_models.csv.gz"))
    ap.add_argument("--calib-out", default=str(ROOT / "results/models/calib"),
                    help="T1-cal conformal constants per stage-2 model (calib_<tag>.json, for cfcap.models.predict)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(a.roots)
    print(f"{len(runs)} train runs")
    if a.stage_weights:
        import shutil
        for run in runs:
            w = run["dir"] / "weights.pt"
            if w.exists():
                c = run["cfg"]
                dst = pathlib.Path(a.stage_weights) / f"{c['arch']}-{c['div']}-{c['seed']}"
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(w, dst / "weights.pt")
                shutil.copy2(run["dir"] / "manifest.json", dst / "manifest.json")
    dec, div, lat, div_rel = [], [], [], []
    ens = {}
    for run in runs:
        cfg, d = run["cfg"], run["dir"]
        units = {s: load_unit(d, s) for s in TR.EVAL_SPLITS}
        if units["cal"] is None or units["test"] is None:
            continue
        cal = units["cal"][0]
        evs = [(s, units[s][0]) for s in ("cal", "test", "H3", "H4") if units[s] is not None]
        bb = TR.ARCHS[cfg["arch"]].get("blackbox")
        sim_lo, _ = lowers(cal, [df for _, df in evs], direct, "simultaneous")
        inf_s = {s: units[s][1].get("infer_ms_per_episode", np.nan) / 1e3 for s, _ in evs}
        sec_direct = lambda split, df: np.full(len(df), inf_s[split])
        sec_comp = lambda split, df: df.comp_seconds.to_numpy()
        # diversity rows (every run)
        d_info = TR.DIVERSITY[cfg["div"]]
        for (split, df), lo in zip(evs, sim_lo):
            if split == "cal":
                continue
            df = df.assign(_rel=TR.relevant(df), _lo=lo)
            for (topo, n, rel_only), g in [(k + (False,), g) for k, g in df.groupby(["parent_topo", "parent_n"])] + \
                    [(k + (True,), g) for k, g in df[df._rel].groupby(["parent_topo", "parent_n"])]:
                lo_g = g._lo.to_numpy()
                cf, uc = decision_metrics(lo_g, g.truth.to_numpy())
                q = g[list(TR.QCOLS)].to_numpy(float)
                y = g.truth.to_numpy()
                held = int(split in ("H3", "H4") or n > d_info["max_n"] or (
                    d_info.get("topologies") is not None and TOPO[int(topo)] not in d_info["topologies"]))
                base = dict(arch=cfg["arch"], diversity=cfg["div"], n_train_families=TR.n_train_families(cfg["div"]),
                            max_train_n=d_info["max_n"], seed=cfg["seed"], test_morph=shape_name(topo, n),
                            test_n=int(n), heldout=held, status=a.status, split=split)
                for metric, v in (("cert_frac", cf), ("unsafe_commit", uc), ("pinball", pinball(q, y)),
                                  ("cov90", float(((y >= q[:, 0]) & (y <= q[:, -1])).mean()))):
                    (div_rel if rel_only else div).append(dict(base, metric=metric, value=v))
            wf = d / f"wheels_{split}.npz"
            if wf.exists() and not bb:
                with np.load(wf) as z:
                    side = {k: z[k] for k in z.files}
                for n in np.unique(side["parent_n"]):
                    sub = {k: v[side["parent_n"] == n] for k, v in side.items()}
                    det = TR.detection(sub)
                    if "det_all" in det:
                        div.append(dict(arch=cfg["arch"], diversity=cfg["div"],
                                        n_train_families=TR.n_train_families(cfg["div"]), max_train_n=d_info["max_n"],
                                        seed=cfg["seed"], test_morph=f"n{int(n)}", test_n=int(n),
                                        heldout=int(split in ("H3", "H4") or n > d_info["max_n"]), status=a.status,
                                        split=split, metric="det_1pct", value=det["det_all"]))
        # decision rows: headline diversity, seed 1
        if cfg["div"] == a.headline and ((cfg["arch"] == "st" and cfg["seed"] == 2) or
                                         (cfg["arch"] == "bb" and cfg["seed"] == 1)):     # A-SEED: st seed 2
            if bb:
                dec += dec_rows("blackbox_gnn", "simultaneous", "none", evs, sim_lo, direct, sec_direct, a.status)
            else:
                mar_lo, _ = lowers(cal, [df for _, df in evs], direct, "marginal")
                dec += dec_rows("amortized_marg", "marginal", "amortized_direct", evs, mar_lo, direct, sec_direct,
                                a.status)
                dec += dec_rows("amortized_sim", "simultaneous", "amortized_direct", evs, sim_lo, direct, sec_direct,
                                a.status)
                cevs = [(s, df) for s, df in evs if s in ("cal", "test")]
                if np.isfinite(cal.cq50).any():
                    c_lo, _ = lowers(cal, [df for _, df in cevs], comp, "compositional")
                    dec += dec_rows("amortized_comp", "compositional", "amortized", cevs, c_lo, comp, sec_comp,
                                    a.status)
                    for s, df in cevs:
                        ok = np.isfinite(df.cq50)
                        for (topo, n), g in df[ok].groupby(["parent_topo", "parent_n"]):
                            v = g.comp_seconds.dropna()
                            lat.append(dict(method="amortized_comp", shape=shape_name(topo, n), n_wheels=int(4 * n),
                                            sec_per_decision=float(v.median()), hardware=a.hardware,
                                            status=a.status, n=int(g.eid.nunique()), split=s))
                for s, df in evs:
                    lat.append(dict(method="amortized_sim", shape="all", n_wheels=-1, sec_per_decision=inf_s[s],
                                    hardware=a.hardware, status=a.status, n=int(df.eid.nunique()), split=s))
        if cfg["div"] == a.headline and cfg["arch"] == "st":
            ens[cfg["seed"]] = evs
    # ensemble over st seeds of the headline diversity (point = mean median, scale = mean half-width + seed spread)
    if len(ens) >= 2:
        seeds = sorted(ens)
        base = ens[seeds[0]]
        for k in range(len(base)):
            dfs = [ens[s][k][1] for s in seeds]
            q50 = np.stack([d_.q50.to_numpy(float) for d_ in dfs])
            hw = np.stack([(d_.q50 - d_.q05).to_numpy(float) for d_ in dfs])
            df = base[k][1].copy()
            df["q50"], df["q05"] = q50.mean(0), q50.mean(0) - (hw.mean(0) + q50.std(0))
            base[k] = (base[k][0], df)
        e_lo, _ = lowers(base[0][1], [df for _, df in base], direct, "simultaneous")
        # posterior "none": a deep ensemble of DIRECT heads is not the amortized posterior; with posterior
        # "amortized" the T3 grid (posterior in {smc, amortized}, route simultaneous) picked it up as "Amortized"
        dec += dec_rows("ensemble", "simultaneous", "none", base, e_lo, direct,
                        lambda s, df: np.full(len(df), np.nan), a.status)
    e2 = load_eval2(a.roots)
    env = load_env(a.roots)
    if a.confirmatory:
        fb = train_cal_rows(runs)
        rows = confirmatory_rows(e2, allow_early=a.allow_early, fallback=fb)
        f = pathlib.Path(a.conf_out)
        f.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            pd.concat(rows).to_csv(f, index=False)
        print(f"confirmatory rows {sum(len(r) for r in rows)} -> {f}")
        w = confirmatory_wheels(a.roots, f.parent / "wheels_conf_models.npz", model=CONF_MODEL)
        print(f"confirmatory wheels -> {w}")
        sv = confirmatory_rows(e2, allow_early=a.allow_early, methods=SEEDVAR_METHODS, fallback=fb)
        if sv:                              # descriptive seed variation (A-SEED): not a dec_* file, never scored
            sv = [r.assign(method=r.method + "_seedvar", seedvar=1) for r in sv]
            pd.concat(sv).to_csv(f.parent / "seedvar_conf_models.csv.gz", index=False)
            print(f"seed-variation rows {sum(len(r) for r in sv)} -> {f.parent / 'seedvar_conf_models.csv.gz'}")
        e2 = {k: {s_: v for s_, v in d.items() if not s_.startswith("conf_")} for k, d in e2.items()}
    else:                                   # never look at confirmatory rows in the exploratory export
        e2 = {k: {s_: v for s_, v in d.items() if not s_.startswith("conf_")} for k, d in e2.items()}
    if e2:                                  # stage-2 (same eids, end-to-end timing) supersedes the train-run rows
        print(f"stage-2 evaluations: {sorted(e2)}")
        dec2, lat2 = headline_from_eval2(e2, a.status, a.calib_out)
        if dec2:
            ens_rows = [d_ for d_ in dec if len(d_) and d_.method.iloc[0] == "ensemble"]
            dec = dec2 + ens_rows
            lat = [r for r in lat if not str(r["method"]).startswith("amortized")] + lat2
    if dec:
        pd.concat(dec).to_csv(out / "dec_models.csv.gz", index=False)
    pd.DataFrame(div).to_csv(out / "diversity.csv", index=False)
    # decision-relevant stratum (faulted candidates + |rho_dec - tau| < 0.1): outside the paper glob, so the F2 script
    # never double counts; same columns plus stratum
    strata = out.parent / "strata"
    strata.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(div_rel).assign(stratum="relevant").to_csv(strata / "diversity_relevant.csv", index=False)
    loc = ROOT / "results/models/latency.csv"
    if loc.exists():                                  # local pilot timings (batch 1, own pilot data)
        for r in pd.read_csv(loc).itertuples():
            n = {"pair": 2, "square": 4, "2x3": 6}.get(r.shape, 0)
            lat.append(dict(method=f"pilot_{r.method}", shape=r.shape, n_wheels=4 * n,
                            sec_per_decision=r.ms_per_episode / 1e3, hardware="RTX 4070 Ti SUPER (local pilot)",
                            status="exploratory", n=0))
    pd.DataFrame(lat).to_csv(out / "latency_models.csv", index=False)
    if not env.empty:                       # C-ENV lower bounds (exploratory always; conf only with --confirmatory)
        cf = pathlib.Path(a.calib_out) / f"calib_{a.env_model}.json"
        q_comp = json.loads(cf.read_text()).get("q_comp") if cf.exists() else None
        if q_comp is None:                  # no stage-2 cal rows: the train run's own exploratory eval-cal rows
            tc = train_cal_rows(runs).get(a.env_model, {}).get("cal")
            if tc is not None and np.isfinite(tc.cq50).any():
                _, q_comp = lowers(tc, [], comp, "compositional")
                q_comp = float(q_comp)
                cf.parent.mkdir(parents=True, exist_ok=True)
                cf.write_text(json.dumps(dict(alpha=ALPHA, q_comp=q_comp, source="train-run eval-cal (fallback)"),
                                         indent=1))
                print(f"env: q_comp {q_comp:.4f} from the train run's eval-cal rows (fallback)")
        if q_comp is None:
            print(f"env: no q_comp in {cf}; run with the env model's stage-2 cal rows first")
        else:
            q_env = None
            if a.env_truth:
                q_env, n_m = q_envcal(env, env_truth(a.env_truth))
                print(f"env: q_env {q_env} from {n_m} exploratory ID missions")
                if q_env is not None:
                    (pathlib.Path(a.calib_out) / f"calib_env_{a.env_model}.json").write_text(
                        json.dumps(dict(alpha=ALPHA, q_env=q_env, n_missions=n_m, floor=ENV_FLOOR), indent=1))
            fs = env_lowers(env, q_comp, pathlib.Path(a.env_out) / "env", conf=False, q_env=q_env,
                            model_tag=a.env_model)
            if a.confirmatory:
                confirmatory_rows({}, allow_early=a.allow_early, methods={})   # the 01:45Z guard only
                fs += env_lowers(env, q_comp, pathlib.Path(a.env_out) / "confirmatory", conf=True, q_env=q_env,
                                 model_tag=a.env_model)
            print("env lower files:", [str(f) for f in fs])
    print(f"dec rows {sum(len(x) for x in dec)}  diversity rows {len(div)}  latency rows {len(lat)}  -> {out}")


if __name__ == "__main__":
    main()
