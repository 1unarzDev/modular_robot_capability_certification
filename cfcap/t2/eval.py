"""cfcap.runner task `t2_eval` (role stress): T2 labels and T2 probe observations of Stage B episodes, and the
evaluation of a predictor against them. EVALUATION ONLY: nothing here may feed training or calibration.

    python -m cfcap.runner t2_eval --shard K --n-shards N --mode simulate --split test --episodes 4000
    python -m cfcap.runner t2_eval --shard K --n-shards N --mode evaluate --episodes-dir DIR --predictor mod:fn \
        --models DIR

simulate. Episodes are the generator's (cfcap.gen.episode.make_episode(plate, split, index): numpy seeds on the CPU,
so the same episode on every host as the T1 split). One unit = `--per-unit` consecutive episode indices. For every
candidate (self, children, merges) the T2 plant (`--conds`, default `all`) gives rho, rho_healthy, rho_rel and the
primary decision label rho_dec = rho / rho_ref (T2's own nominal healthy reference of that morphology,
cfcap.t2.capability.rho_ref) on the 10 decision legs; a candidate without a faulted wheel has rho_healthy = rho (no twin run, as the generator does).
Rows = candidates (columnar units/<unit>.npz): plate, split, ep_index, cand, kind, n_mod, n_wheel, n_fault, and
`<cond>.rho_rel.<leg index>` etc. The parent's v2 probe under T2 `all` (the generator's probe command and RNG
stream, robot-visible channels as cfcap.gen.simrun.observe_np) goes to OUT/obs/<unit>.npz (ragged, with offsets).
T2 effect draws are seeded by (plate, split, index).

evaluate. Reads the simulate units from --episodes-dir, calls the predictor `fn(models_dir, episode, v2_obs)` ->
either lower bounds (n_cand, L) on rho_rel or {method: dict(lower, point, scale, route, posterior, train, alpha)}
(arrays (n_cand, L)), and keeps one row per (method, episode, candidate, leg) with the paper's dec_*.csv columns
(docs/board/paper.md §1a; test_plant T2, truth = T2 rho_rel); unsafe-go and certified fraction at --taus go to the
metrics. `export_dec_csv(out_dirs, "results/stress/paper/dec_t2.csv")` concatenates the units. The predictor is the
models / baselines roles' (cfcap.t2 provides none); its bounds must be calibrated on T1 cal only.

Requires torch (CPU only: the episode generator and capxfer imports) and numba (default backend) or JAX
(`--backend jax`, GPU); plus the crane kernel files (scripts/stage_crane_t2.sh -> /workspace/in/crane_ml).
"""
from __future__ import annotations

import io
import json
import zlib

import numpy as np

from cfcap.runner import InputMissing, Stop, Task, atomic_bytes, shard_range
from cfcap.t2 import deps  # noqa: F401  (crane kernel + numba/jax paths before the runner checks `requires`)


def _args(p):
    p.add_argument("--mode", choices=("simulate", "evaluate"), required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--plate", default="plated")
    p.add_argument("--episodes", type=int, default=4000, help="episode indices 0..episodes-1 of the split")
    p.add_argument("--per-unit", type=int, default=20)
    p.add_argument("--conds", default="all", help="comma list of cfcap.t2.build.ABLATIONS names")
    p.add_argument("--backend", default="auto", choices=("auto", "numba", "jax", "numpy"),
                   help="auto: numba if importable, else jax, else numpy (no torch.compile anywhere in T2)")
    p.add_argument("--no-obs", action="store_true", help="skip the parent's v2 probe observation")
    p.add_argument("--misfit", action="store_true",
                   help="per-candidate v2 and clamp misfit T2 vs T1 with latents known (cfcap.t2.misfit; ~2x cost)")
    p.add_argument("--episodes-dir", default=None)
    p.add_argument("--predictor", default=None, help="module:function (evaluate)")
    p.add_argument("--models", default=None)
    p.add_argument("--taus", default="0.6,0.7,0.8")
    p.add_argument("--label", default="rho_dec", choices=("rho_dec", "rho_rel"),
                   help="truth of evaluate: rho_dec = rho / rho_ref (primary, board 2026-11-25) or rho_rel")


def _units(a, shard, n_shards):
    if a.mode == "simulate":
        nu = -(-a.episodes // a.per_unit)
        return [f"t2-{a.plate}-{a.split}-u{j:05d}" for j in shard_range(nu, shard, n_shards)]
    import glob
    import os
    if not a.episodes_dir or not os.path.isdir(a.episodes_dir):
        raise InputMissing(f"--episodes-dir {a.episodes_dir} (the simulate units) is not on this host")
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(a.episodes_dir, "**/units/t2-*.npz"),
                                                                   recursive=True))
    return [f"ev-{n}" for n in names[shard::n_shards]]


def episode_seed(plate, split, index):
    return int(zlib.crc32(f"t2|{plate}|{split}|{int(index)}".encode()))


FEAT_COLS = ("faults", "min_g", "payload_kg", "mu_fac", "mu_fac_min", "mu_fac_max", "drag_pos", "drag_neg",
             "D_imu_v2", "D_imu_clamp", "D_duty_v2", "D_duty_clamp")


def candidate_features(asm, lat, par, seed, effects):
    """Per-candidate features for the decision rows: faults [[wheel, mode, g]], min g, payload, the T2 per-wheel
    roller-friction factors (aniso_mu) and roller-drag factors (aniso) of `effects` with this candidate's seed."""
    from cfcap.t2 import build
    from engine.kernels import mecanum as K
    mode, g = build._np(lat["mode"])[0], build._np(lat["g"])[0]
    S = build.episode_specs(asm, lat, par, effects, seed)
    wp = S["p"][:, K.G:].reshape(1, -1, K.Q)
    d = lat["dom"]
    base = float(build._np(d["mu"])[0]) * build._np(lat["mu_mod"])[0][np.asarray(asm.wheel_module)]
    muf = wp[0, :, K.IW["mu_t"]] / base
    cr = float(build._np(d["c_roller"])[0]) * build.V_REF / S["N0"][0]
    pl = lat.get("payload")
    return dict(faults=json.dumps([[int(w), int(mode[w]), round(float(g[w]), 3)] for w in np.nonzero(mode)[0]]),
                min_g=float(g.min()), payload_kg=float(build._np(pl["mass"])[0]) if pl is not None else 0.0,
                mu_fac=json.dumps(np.round(muf, 3).tolist()), mu_fac_min=float(muf.min()), mu_fac_max=float(muf.max()),
                drag_pos=json.dumps(np.round(S["mufp"][0] / cr, 3).tolist()),
                drag_neg=json.dumps(np.round(S["mufn"][0] / cr, 3).tolist()))


def simulate_episode(ep, conds, backend="numba", obs=True, misfit=False):
    """T2 labels (per condition) for every candidate of a generator episode, and the parent's v2 probe."""
    from cfcap.gen import episode as E
    from cfcap.gen import simrun as SR
    from cfcap.t2 import build, capability as cap2
    par = E.params(ep.plate)
    L = len(cap2.LEGS)
    seed = episode_seed(ep.plate, ep.split, ep.index)
    lats = [E.plant_latents(ep, c.modules, c.load_z, c.payload, B=1) for c in ep.cands]
    faulted = np.array([bool((l["mode"][0] != 0).any()) for l in lats])
    out = {}
    for cond in conds:
        eff = build.ABLATIONS[cond]
        rho = np.zeros((len(ep.cands), L))
        rh = np.zeros_like(rho)
        for sel, rel in ((faulted, True), (~faulted, False)):
            idx = np.nonzero(sel)[0]
            if not len(idx):
                continue
            # one effect seed per candidate index: its draws do not depend on which others are faulted
            targets = [(ep.cands[i].asm, lats[i]) for i in idx]
            o = _capability_seeded(targets, par, eff, rel, backend, [seed + 7919 * int(i) for i in idx])
            for k, i in enumerate(idx):
                rho[i] = o[k]["rho"][0]
                rh[i] = o[k]["rho_healthy"][0] if rel else o[k]["rho"][0]
        ref = np.stack([cap2.rho_ref(c.asm, par, ep.plate, eff, n_draws=1, backend=backend) for c in ep.cands])
        out[cond] = dict(rho=rho, rho_healthy=rh, rho_rel=rho / rh, rho_ref=ref, rho_dec=rho / ref)
    eff0 = build.ABLATIONS[conds[0]]
    feats = []
    for i, c in enumerate(ep.cands):
        f = candidate_features(c.asm, lats[i], par, seed + 7919 * i, eff0)
        if misfit:
            from cfcap.t2 import misfit as MF
            mf = MF.probe_misfit(c.asm, lats[i], par, eff0, seed + 7919 * i, backend=backend,
                                 rng=E.episode_rng(ep.plate, ep.split, ep.index, stream=200 + i))
            f.update(D_imu_v2=float(mf["v2"][0][0]), D_duty_v2=float(mf["v2"][1][0]),
                     D_imu_clamp=float(mf["clamp"][0][0]), D_duty_clamp=float(mf["clamp"][1][0]))
        else:
            f.update(D_imu_v2=np.nan, D_duty_v2=np.nan, D_imu_clamp=np.nan, D_duty_clamp=np.nan)
        feats.append(f)
    out["_features"] = feats
    ob = None
    if obs:
        a = ep.parent
        rng = E.episode_rng(ep.plate, ep.split, ep.index, stream=1 + 16 * a.role + SR.PROBE_KINDS.index("v2"))
        cmd, _, info = SR.probe_tables("v2", a.asm, par, rng)
        lat = E.plant_latents(ep, [(a.role, k) for k in range(a.m)], a.load_z, E.parent_payload(ep),
                              gyro=(a.gyro_bias, a.acc_bias))
        T = SR.probe_T("v2")
        R = cap2.simulate(a.asm, lat, par, cmd, T, build.ABLATIONS[conds[0]], backend=backend, seed=seed)   # obs under the first condition
        ob = dict(_observe_np(R, lat, rng), cmd=cmd[:R["t"].shape[0]], **info)
    return out, faulted, ob


def _capability_seeded(targets, par, eff, rel, backend, seeds):
    """cfcap.t2.capability.capability with one effect seed per target (batched by wheel count)."""
    from cfcap.t2 import build, kernel, capability as cap2
    L = len(cap2.LEGS)
    K = cap2._n_ticks(cap2.T_LEG, par)
    jobs = []
    for (asm, lat), sd in zip(targets, seeds):
        tabs, tw = cap2.cap1.leg_tables(asm, par)
        for healthy in ((False, True) if rel else (False,)):
            S = build.episode_specs(asm, lat, par, eff, sd, clear_faults=healthy)
            jobs.append((build.repeat(S, L), tabs, tw, asm.char_length()))
    by_w = {}
    for q, j in enumerate(jobs):
        by_w.setdefault(j[0]["px0"].shape[1], []).append(q)
    tw_res = {}
    for W, qs in by_w.items():
        S = build.concat([jobs[q][0] for q in qs])
        tables = np.concatenate([jobs[q][1] for q in qs], 0).astype(np.float64)
        which = np.concatenate([np.arange(L) + L * k for k in range(len(qs))])
        tw = kernel.run(S, tables, which, K, backend)["twist"]
        for k, q in enumerate(qs):
            tw_res[q] = tw[k * L:(k + 1) * L]
    t = np.arange(K) * cap2.DT_OBS
    outs, q = [], 0
    for _ in targets:
        _, _, tw, Lc = jobs[q]
        r, d = cap2.rho_from_twist(tw_res[q][None], t, tw, cap2.T_LEG, Lc)
        o = dict(rho=r, drift=d)
        q += 1
        if rel:
            rh, _ = cap2.rho_from_twist(tw_res[q][None], t, tw, cap2.T_LEG, Lc)
            o.update(rho_healthy=rh, rho_rel=r / rh)
            q += 1
        outs.append(o)
    return outs


def _observe_np(R, lat, rng):
    """cfcap.gen.simrun.observe_np on T2 records (B = 1): pre_duty, encoder count increments dcnt, imu (gz, ax, ay)."""
    import math
    from cfcap.t1 import params as P
    from cfcap.t1.plant import IMU_NOISE
    cpr = P.CPR
    cnt = np.floor(R["angle"][0] / (2 * math.pi) * cpr)
    dcnt = np.zeros_like(cnt)
    dcnt[1:] = np.diff(cnt, axis=0)
    duty = R["duty"][0].astype(np.float32)
    T, n = duty.shape
    p = float(lat["drop_p"])
    if p > 0:
        drop = rng.uniform(size=(T, n)) < p
        drop[0] = False
        idx = np.maximum.accumulate(np.where(drop, 0, np.arange(T)[:, None]), axis=0)
        duty, dcnt = np.take_along_axis(duty, idx, 0), np.take_along_axis(dcnt, idx, 0)
    gb = float(lat["gyro_bias"][0])
    ab = np.asarray(lat["acc_bias"][0])
    imu = np.stack([R["gz"][0] + IMU_NOISE[0] * rng.normal(size=T) + gb,
                    R["ax"][0] + IMU_NOISE[1] * rng.normal(size=T) + ab[0],
                    R["ay"][0] + IMU_NOISE[2] * rng.normal(size=T) + ab[1]], -1).astype(np.float32)
    return dict(duty=duty, dcnt=dcnt.astype(np.int32), imu=imu)


def _run_simulate(unit, ctx):
    import time
    from cfcap.gen import episode as E
    a = ctx.args
    j = int(unit.rsplit("-u", 1)[1])
    idx = list(range(j * a.per_unit, min(a.episodes, (j + 1) * a.per_unit)))
    conds = tuple(a.conds.split(","))
    st = ctx.load_state() or {"done": [], "rows": [], "obs": []}
    t0 = time.time()
    for k, ei in enumerate(idx):
        if ei in st["done"]:
            continue
        if ctx.should_stop():
            ctx.save_state(st)
            raise Stop("t2_eval checkpoint")
        ep = E.make_episode(a.plate, a.split, ei)
        lab, faulted, ob = simulate_episode(ep, conds, a.backend, obs=not a.no_obs, misfit=a.misfit)
        for ci, c in enumerate(ep.cands):
            r = dict(plate=E.PLATES.index(a.plate), split=E.SPLITS[a.split], ep_index=ei, cand=ci, kind=c.kind,
                     n_mod=len(c.modules), n_wheel=c.asm.n, n_fault=int(faulted[ci]), **lab["_features"][ci])
            for cond in conds:
                for key in ("rho", "rho_healthy", "rho_rel", "rho_ref", "rho_dec"):
                    for li, v in enumerate(lab[cond][key][ci]):
                        r[f"{cond}.{key}.{li}"] = float(v)
            st["rows"].append(r)
        if ob is not None:
            st["obs"].append((ei, ob))
        st["done"].append(ei)
        ctx.save_state(st)
        ctx.progress((k + 1) / len(idx), phase=f"t2 simulate ep {ei}")
    ctx.meta.update(episodes=len(idx), candidates=len(st["rows"]), s_per_episode=(time.time() - t0) / max(1, len(idx)))
    if st["obs"]:
        buf = io.BytesIO()
        o = st["obs"]
        np.savez_compressed(buf, ep_index=np.array([e for e, _ in o]),
                            T=np.array([x["duty"].shape[0] for _, x in o]), n=np.array([x["duty"].shape[1] for _, x in o]),
                            amp=np.array([x["amp"] for _, x in o]), lead=np.array([x["lead"] for _, x in o]),
                            duty=np.concatenate([x["duty"].ravel() for _, x in o]),
                            dcnt=np.concatenate([x["dcnt"].ravel() for _, x in o]),
                            imu=np.concatenate([x["imu"] for _, x in o]), cmd=np.concatenate([x["cmd"] for _, x in o]))
        atomic_bytes(ctx.out_dir / "obs" / f"{unit}.npz", buf.getvalue())
    rows = st["rows"]
    if not rows:
        return None
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


DEC_COLS = ("method", "route", "posterior", "train", "test_plant", "split", "eid", "shape", "family", "cand", "kind",
            "cand_size", "has_fault", "leg", "truth", "point", "scale", "lower", "alpha", "seconds")
KIND_NAMES = {0: "self", 1: "child", 2: "merge"}


def _as_methods(out):
    """Predictor output -> {method: dict(lower, point, scale, route, posterior, train, alpha)}."""
    if isinstance(out, dict) and "lower" in out:
        out = {"predictor": out}
    elif not isinstance(out, dict):
        out = {"predictor": dict(lower=out)}
    return out


def _run_evaluate(unit, ctx):
    """Per-row decision table (docs/board/paper.md dec_*.csv), one row per (method, episode, candidate, leg); taus are
    applied by the paper scripts (metrics here use --taus)."""
    import glob
    import importlib
    import os
    import time
    from cfcap.gen import episode as E
    a = ctx.args
    if not a.predictor:
        raise InputMissing("--predictor module:function (models / baselines role) is required for --mode evaluate")
    mod, _, fn = a.predictor.partition(":")
    pred = getattr(importlib.import_module(mod), fn)
    name = unit[3:]
    f = glob.glob(os.path.join(a.episodes_dir, f"**/units/{name}.npz"), recursive=True)
    fo = glob.glob(os.path.join(a.episodes_dir, f"**/obs/{name}.npz"), recursive=True)
    if not f:
        raise InputMissing(f"{name}.npz not under {a.episodes_dir}")
    z = np.load(f[0])
    ob = np.load(fo[0]) if fo else None
    taus = [float(t) for t in a.taus.split(",")]
    cond = a.conds.split(",")[0]
    L = 10
    obs = {}
    if ob is not None:
        o1 = o2 = 0
        for i, e in enumerate(ob["ep_index"]):
            T, n = int(ob["T"][i]), int(ob["n"][i])
            obs[int(e)] = dict(duty=ob["duty"][o1:o1 + T * n].reshape(T, n), dcnt=ob["dcnt"][o1:o1 + T * n].reshape(T, n),
                               imu=ob["imu"][o2:o2 + T], cmd=ob["cmd"][o2:o2 + T], amp=float(ob["amp"][i]),
                               lead=float(ob["lead"][i]))
            o1, o2 = o1 + T * n, o2 + T
    plate, split = E.PLATES[int(z["plate"][0])], E.SPLIT_NAMES[int(z["split"][0])]
    rows = {k: [] for k in DEC_COLS}
    st = {"n": 0, "unsafe": 0, "go_cap": 0, "cap": 0}
    for ei in np.unique(z["ep_index"]):
        if ctx.should_stop():
            raise Stop("t2_eval evaluate: unit restarts from its first episode")
        m = z["ep_index"] == ei
        ep = E.make_episode(plate, split, int(ei))
        t0 = time.time()
        meths = _as_methods(pred(a.models, ep, obs.get(int(ei))))
        sec = (time.time() - t0) / max(1, len(ep.cands))
        rr = np.stack([z[f"{cond}.{a.label}.{l}"][m] for l in range(L)], 1)
        cand, nf = z["cand"][m], z["n_fault"][m]
        shape = ep.parent.topo
        for meth, d in meths.items():
            lower = np.asarray(d["lower"], float)
            nanL = np.full_like(lower, np.nan)
            point = np.asarray(d.get("point", nanL), float)
            scale = np.asarray(d.get("scale", nanL), float)
            for k, ci in enumerate(cand):
                c = ep.cands[int(ci)]
                for l in range(L):
                    vals = dict(method=meth, route=d.get("route", "none"), posterior=d.get("posterior", "none"),
                                train=d.get("train", "T1"), test_plant="T2", split=split, eid=int(ei), shape=shape,
                                family=str(c.topo), cand=int(ci), kind=KIND_NAMES.get(int(c.kind), str(c.kind)),
                                cand_size=len(c.modules), has_fault=int(nf[k]), leg=l, truth=float(rr[k, l]),
                                point=float(point[ci, l]), scale=float(scale[ci, l]), lower=float(lower[ci, l]),
                                alpha=float(d.get("alpha", np.nan)), seconds=sec)
                    for key in DEC_COLS:
                        rows[key].append(vals[key])
                    for key in FEAT_COLS:
                        if key in z.files:
                            rows.setdefault(key, []).append(z[key][m][k])
                    for t in taus:
                        rows.setdefault(f"margin_{t}", []).append(float(rr[k, l] - t))
                    for t in taus:
                        go, tr = lower[ci, l] >= t, rr[k, l] >= t
                        st["n"] += 1
                        st["unsafe"] += int(go and not tr)
                        st["go_cap"] += int(go and tr)
                        st["cap"] += int(tr)
    ctx.meta.update(n=st["n"], unsafe_go=st["unsafe"], go_capable=st["go_cap"], capable=st["cap"])
    if not rows["eid"]:
        return None
    return {k: np.array(v) for k, v in rows.items()}


def export_dec_csv(out_dirs, path):
    """Concatenate evaluate units (columnar npz) into results/stress/paper/dec_t2.csv (docs/board/paper.md §1a)."""
    import csv
    import glob
    import os
    import pathlib
    files = sorted(f for d in out_dirs for f in glob.glob(os.path.join(d, "**/units/ev-*.npz"), recursive=True))
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        cols = None
        for f in files:
            z = np.load(f)
            if cols is None:
                cols = list(DEC_COLS) + [k for k in z.files if k not in DEC_COLS]   # + features, margins
                w.writerow(cols)
            for r in zip(*[z[k] for k in cols]):
                w.writerow([x.item() if hasattr(x, "item") else x for x in r])
    return len(files)


def resolve_backend(b):
    if b != "auto":
        return b
    import importlib.util
    for name in ("numba", "jax"):
        if importlib.util.find_spec(name) is not None:
            return name
    return "numpy"


def _run(unit, ctx):
    ctx.args.backend = resolve_backend(ctx.args.backend)
    ctx.meta["backend"] = ctx.args.backend
    return _run_simulate(unit, ctx) if ctx.args.mode == "simulate" else _run_evaluate(unit, ctx)


def _summary(recs, a):
    m = [r.get("meta", {}) for r in recs]
    if a.mode == "simulate":
        eps = sum(x.get("episodes", 0) for x in m)
        sp = [x["s_per_episode"] for x in m if "s_per_episode" in x]
        return {"t2.episodes": eps, "t2.candidates": sum(x.get("candidates", 0) for x in m),
                "t2.s_per_episode_mean": float(np.mean(sp)) if sp else 0.0}
    n = sum(x.get("n", 0) for x in m)
    cap_ = sum(x.get("capable", 0) for x in m)
    return {"t2.decisions": n, "t2.unsafe_go_rate": sum(x.get("unsafe_go", 0) for x in m) / max(1, n),
            "t2.certified_frac": sum(x.get("go_capable", 0) for x in m) / max(1, cap_)}


TASK = Task("t2_eval", _units, _run, _args, _summary, ("numpy", "torch"),
            "T2 stress-plant labels + v2 observations of generator episodes (simulate), predictor evaluation (evaluate)")


def refuse_all(models_dir, episode, v2_obs):
    """Reference predictor (tests, smoke runs): refuses every decision (lower = -inf)."""
    n = len(episode.cands)
    return {"refuse_all": dict(lower=np.full((n, 10), -np.inf), route="none", posterior="none", train="none")}
