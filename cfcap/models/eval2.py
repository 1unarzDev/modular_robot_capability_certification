"""Stage-2 evaluation of trained `train` runs (role models): `python -m cfcap.runner models_eval2 --task-spec
cfcap.models.eval2:TASK --shard K --n-shards N --runs DIR --splits cal:2000,test:2000,H3:500,H4:500`.

Loads each trained run's `weights.pt` once, regenerates the generator's episodes of each split by index (the same ids
as the reference shards cfcap-gen-ref2 and the baselines' rows; `cfcap.gen.episode.make_episode` is deterministic),
and writes the same decision rows as the train task's eval units plus END-TO-END per-decision timing:
  sec_direct  per episode, batch 1: features -> encoder -> direct head for every candidate (synchronised GPU wall)
  sec_comp    per episode: the posterior (as above) + compositional rollouts of every candidate (S samples, one
              batched t1.capability call per episode), i.e. what one decision costs in the probing loop
  sec_gen     generator time per episode (probe + label simulation; not part of a decision)
Units: <split>-<lo>-<hi> episode chunks (--per-unit), sharded contiguously. --compose-every k composes every k-th
episode of a unit (1 = all). --runs points at a directory of pulled train runs (each with manifest.json and
weights.pt); --select keeps only the named grid entries (arch-div-seed, e.g. st-n6-1,bb-n6-1).
"""
from __future__ import annotations

import io
import json
import pathlib
import time

import numpy as np

from cfcap.runner import InputMissing, Stop, Task, atomic_bytes, chunk_units, parse_chunk, shard_range, unit_seed

QCOLS = ("q05", "q25", "q50", "q75", "q95")


def add_args(p):
    p.add_argument("--runs", required=False, default=None, help="directory of pulled train runs (or an input name)")
    p.add_argument("--select", default="st-n6-1,st-n6-2,st-n6-3,bb-n6-1,ds-n6-1,stl-n6-1")
    p.add_argument("--splits", default="cal:2000,test:2000,H3:500,H4:500")
    p.add_argument("--per-unit", type=int, default=100)
    p.add_argument("--compose-S", type=int, default=16)
    p.add_argument("--compose-every", type=int, default=1)
    p.add_argument("--compose-select", default="st-n6-1", help="models that also run the compositional route")
    p.add_argument("--plate", default="plated")
    p.add_argument("--reserve-shard0", action="store_true",
                   help="shard 0 gets no units (the confirmatory scorer excludes shard 0, the canary's)")
    p.add_argument("--wheels-model", default="st-n6-1",
                   help="model whose per-wheel mode posteriors go to OUT/wheels/<unit>.npz (C-EXPLAIN)")
    p.add_argument("--env-families", default="",
                   help="C-ENV missions, family:n or family:lo:hi (e.g. test:600,E-H1:300,conf-ID:0:1500)")
    p.add_argument("--env-model", default="st-n6-1")
    p.add_argument("--env-S", type=int, default=12)
    p.add_argument("--blind-splits", default="het_cal,het_test,H1,H2,conf_het_test,conf_H1,conf_H2",
                   help="splits where the compositional route also runs type-blind (module type withheld)")


def _items(a):
    """--splits name:n (indices 0..n-1) or name:lo:hi, comma separated."""
    out = []
    for s in filter(None, a.splits.split(",")):
        f = s.split(":")
        lo, hi = (0, int(f[1])) if len(f) == 2 else (int(f[1]), int(f[2]))
        out += chunk_units(f[0], range(lo, hi), a.per_unit)
    for s in filter(None, getattr(a, "env_families", "").split(",")):
        f = s.split(":")
        lo, hi = (0, int(f[1])) if len(f) == 2 else (int(f[1]), int(f[2]))
        out += chunk_units(f"env-{f[0]}", range(lo, hi), a.per_unit)
    return out


def units(a, shard, n_shards):
    it = _items(a)
    if a.reserve_shard0 and n_shards > 1:
        return [] if shard == 0 else [it[i] for i in shard_range(len(it), shard - 1, n_shards - 1)]
    return [it[i] for i in shard_range(len(it), shard, n_shards)]


def find_models(root, select):
    """{tag: (weights path, cfg)} for the selected grid entries under root (pulled train runs)."""
    from cfcap.models import train as TR
    want = set(select.split(",")) if select else None
    out = {}
    for m in pathlib.Path(root).rglob("manifest.json"):
        try:
            man = json.loads(m.read_text())
        except ValueError:
            continue
        if man.get("task") != "train" or not (m.parent / "weights.pt").exists():
            continue
        argv = list(man.get("task_argv") or [])
        g = argv[argv.index("--grid") + 1] if "--grid" in argv else "b1"
        cfg = TR.grid(g)[int(man.get("shard", 0))]
        tag = f"{cfg['arch']}-{cfg['div']}-{cfg['seed']}"
        if want is None or tag in want:
            out[tag] = (m.parent / "weights.pt", cfg)
    return out


_MODELS = {}


def load_models(ctx):
    import torch
    from cfcap.models import train as TR
    a = ctx.args
    root = a.runs
    if root and not pathlib.Path(root).exists():
        root = str(ctx.input(root))
    if not root:
        raise InputMissing("--runs DIR (pulled train runs with weights.pt)")
    if not _MODELS:
        for tag, (w, cfg) in sorted(find_models(root, a.select).items()):
            m = TR.build_model(cfg).cuda()
            m.load_state_dict(torch.load(w, map_location="cuda", weights_only=False)["state"])
            _MODELS[tag] = (m.eval(), cfg)
        if not _MODELS:
            raise InputMissing(f"no weights.pt for {a.select} under {root}")
    return _MODELS


def cand_index(D):
    """Per candidate row of a generator columnar dict: its index in make_episode's candidate list (ep.cands), the
    key the confirmatory scorers regenerate truth with. _assemble writes ep.cands in order, so it is the row offset
    within the episode."""
    ce = D["cand_ep"]
    return np.arange(len(ce)) - D["ep_cand_off"][ce]


def run_unit(unit, ctx):
    if unit.startswith("env-"):
        return run_env_unit(unit, ctx)
    import torch
    from cfcap.gen import episode as E
    from cfcap.models import compose as CO
    from cfcap.models import gendata as G
    from cfcap.models import posterior as M
    from cfcap.models import train as TR
    a = ctx.args
    split = unit.rsplit("-", 2)[0]
    idx = list(parse_chunk(unit))
    if split not in E.SPLITS:
        raise InputMissing(f"split {split!r} is not in this generator version ({sorted(E.SPLITS)})")
    models = load_models(ctx)
    t0 = time.time()
    D, _, st = G.episodes(a.plate, split, len(idx), indices=idx)
    gs = G.GenSet(D, a.plate)
    sec_gen = (time.time() - t0) / max(1, gs.E)
    par, dr = E.params(a.plate), E.domain_dr(a.plate)
    rng = np.random.default_rng(unit_seed("models-eval2", unit))
    L = 10
    ce = D["cand_ep"]
    base = dict(eid=np.repeat(D["ep_index"][ce], L),
                cand=np.repeat(cand_index(D), L), kind=np.repeat(D["cand_kind"], L),
                cand_size=np.repeat(np.diff(D["cand_cm_off"]), L),
                has_fault=np.repeat((D["cand_n_fault"] > 0).astype(np.int8), L),
                cand_family=np.repeat(D["cand_family"], L),
                parent_n=np.repeat(gs.asm_m[D["ep_asm_off"][:-1]][ce], L),
                parent_topo=np.repeat(D["asm_topo"][D["ep_asm_off"][:-1]][ce], L),
                leg=np.tile(np.arange(L), gs.C), truth=D["cand_rho_dec"].reshape(-1),
                truth_rel=D["cand_rho_rel"].reshape(-1))
    blind = split in a.blind_splits.split(",")
    cols = {k: [] for k in list(base) + ["model"] + list(QCOLS) + ["c" + q for q in QCOLS] +
            ["b" + q for q in QCOLS] + ["sec_direct", "sec_comp", "sec_gen", "shard"]}
    wsd = {k: [] for k in ("eid", "asm_role", "wheel", "ep_wheel", "p", "g_mean")}
    for mi, (tag, (model, cfg)) in enumerate(models.items()):
        bb = TR.ARCHS[cfg["arch"]].get("blackbox")
        cf = np.full((gs.C, L, 5), np.nan, np.float32)
        comp = np.full((gs.C, L, 5), np.nan, np.float32)
        compb = np.full((gs.C, L, 5), np.nan, np.float32)
        s_dir = np.full(gs.E, np.nan)
        s_comp = np.full(gs.E, np.nan)
        for e in range(gs.E):
            if ctx.should_stop():
                raise Stop("eval2: unit restarts")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                b = gs.batch([e])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = G.forward(model, b)
                o = {k: v.float() for k, v in o.items()}
                s = M.posterior_summary(o)
                q = o["cf"].cpu().numpy()
            torch.cuda.synchronize()
            s_dir[e] = time.perf_counter() - t1
            cs = gs.ep_cand[e]
            cf[cs] = q
            if tag == a.wheels_model:      # per-wheel mode posterior sidecar (C-EXPLAIN); no truth is stored
                mk = b["mask"]
                pw = s["p"][mk].cpu().numpy()
                ab_ = o["ab"][mk].cpu().numpy()
                for ai, asm_i in enumerate(gs.ep_asm[e]):
                    n_a = int(gs.asm_n[asm_i])
                    o0 = int(gs.asm_n[gs.ep_asm[e][:ai]].sum())
                    wsd["eid"].append(np.full(n_a, int(D["ep_index"][e])))
                    wsd["asm_role"].append(np.full(n_a, int(D["asm_role"][asm_i])))
                    wsd["wheel"].append(np.arange(n_a))
                    wsd["ep_wheel"].append(o0 + np.arange(n_a))
                    wsd["p"].append(pw[o0:o0 + n_a])
                    wsd["g_mean"].append(ab_[o0:o0 + n_a, 0] / ab_[o0:o0 + n_a].sum(1))
            if bb or (e % a.compose_every) or tag not in a.compose_select.split(","):
                continue
            m, mm = b["mask"], b["mod_mask"] > 0
            post = dict(p=s["p"][m], ab=o["ab"][m], nu_mu=s["nu_mu"][mm], nu_sd=s["nu_sd"][mm])
            w0 = int(gs.asm_w0[gs.ep_asm[e][0]])
            m0 = int(D["asm_mod_off"][gs.ep_asm[e][0]])
            jobs = []
            ep_obj = E.make_episode(a.plate, split, int(D["ep_index"][e]))
            for ci, c in enumerate(cs):
                k0, k1 = D["cand_cm_off"][c:c + 2]
                mods = D["cm_mod"][k0:k1]
                widx = (4 * mods[:, None] + np.arange(4)).reshape(-1) - w0
                jobs.append((ep_obj.cands[ci].asm, CO.sample_from_episode(ep_obj, ep_obj.cands[ci], post, widx,
                                                                          list(mods - m0), a.compose_S, rng, dr)))
            res = CO.compositional_quantiles(jobs, par, ref=gs.ref[cs])
            torch.cuda.synchronize()
            s_comp[e] = time.perf_counter() - t1
            for c, (qq, _) in zip(cs, res):
                comp[c] = qq
            if blind:                       # module type withheld: every wheel composed as mecanum
                jb = [(ca, {k: v for k, v in lat.items() if k != "wtype"}) for ca, lat in jobs]
                for c, (qq, _) in zip(cs, CO.compositional_quantiles(jb, par, ref=gs.ref[cs])):
                    compb[c] = qq
            ctx.progress((mi * gs.E + e + 1) / (len(models) * gs.E), f"{tag}")
        for k, v in base.items():
            cols[k].append(v)
        cols["model"].append(np.full(gs.C * L, mi))
        for k, qn in enumerate(QCOLS):
            cols[qn].append(cf[..., k].reshape(-1))
            cols["c" + qn].append(comp[..., k].reshape(-1))
            cols["b" + qn].append(compb[..., k].reshape(-1))
        cols["sec_direct"].append(np.repeat(s_dir[ce], L))
        cols["sec_comp"].append(np.repeat(s_comp[ce], L))
        cols["sec_gen"].append(np.full(gs.C * L, sec_gen))
        cols["shard"].append(np.full(gs.C * L, ctx.runner.args.shard))
        ctx.meta[f"{tag}/sec_direct_median"] = float(np.nanmedian(s_dir))
        if np.isfinite(s_comp).any():
            ctx.meta[f"{tag}/sec_comp_median"] = float(np.nanmedian(s_comp))
    if wsd["eid"]:
        buf = io.BytesIO()
        np.savez_compressed(buf, split=np.array(split), model=np.array(a.wheels_model),
                            **{k: np.concatenate(v).astype(np.float32 if k in ("p", "g_mean") else np.int64)
                               for k, v in wsd.items()})
        atomic_bytes(ctx.out_dir / "wheels" / f"{unit}.npz", buf.getvalue())
    ctx.meta["models"] = ",".join(models)
    ctx.meta["hardware"] = torch.cuda.get_device_name(0)
    out = {}
    for k, v in cols.items():
        x = np.concatenate(v)
        out[k] = x.astype(np.float32) if x.dtype.kind == "f" else x.astype(np.int64)
    return out


def run_env_unit(unit, ctx):
    """C-ENV: per (mission, candidate, segment) the compositional rho_dec quantiles of the segment's leg under the
    segment's condition (mu factor on every module, payload at the candidate centre when loaded), from the env-model's
    posterior on the mission's body episode (v2 observations of every assembly; no labels are simulated here).
    Rows carry point = q50 and q05 / q95; the lower bounds (raw: q = 0 -> q50; amortized_comp: the T1-cal
    simultaneous constant) are applied by scripts/models_export.py --env."""
    import torch
    from cfcap.env import buildings as B
    from cfcap.env import missions as MS
    from cfcap.gen import episode as E
    from cfcap.gen import shard as SH
    from cfcap.gen import simrun as SR
    from cfcap.models import compose as CO
    from cfcap.models import gendata as G
    from cfcap.models import posterior as M
    from cfcap.t1 import capability as cap
    a = ctx.args
    fam = unit[4:].rsplit("-", 2)[0]
    idx = list(parse_chunk(unit))
    models = load_models(ctx)
    if a.env_model not in models:
        raise InputMissing(f"env model {a.env_model} not loaded (--select)")
    model, cfg = models[a.env_model]
    par, dr = E.params(a.plate), E.domain_dr(a.plate)
    rng = np.random.default_rng(unit_seed("models-eval2-env", unit))
    cols = {k: [] for k in ("family", "mission", "building", "cand", "seg", "leg", "cond", "q05", "q50", "q95",
                            "seconds", "shard")}
    L = len(cap.LEGS)
    for j, mi in enumerate(idx):
        if ctx.should_stop():
            raise Stop("env unit restarts")
        m = B.make_mission(fam, mi)
        segs = MS.segments(m)
        if not segs:
            continue
        ep = MS.body_episode(m, a.plate)
        keys, ci = MS.conditions(segs, m.payload_kg)
        t0 = time.perf_counter()
        obs = SR.probe_episodes([ep], ("v2",))
        C = len(ep.cands)
        one = np.ones((C, L), np.float32)
        lab = [dict(rho=one, rho_healthy=one, rho_rel=one, drift=0 * one, rho_ref=one, rho_dec=one,
                    rho_ref_type=one)]
        D = SH._assemble([ep], obs, lab, ("v2",))
        gs = G.GenSet(D, a.plate)
        with torch.no_grad():
            b = gs.batch([0])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = G.forward(model, b)
            o = {k: v.float() for k, v in o.items()}
            s = M.posterior_summary(o)
        mk, mm = b["mask"], b["mod_mask"] > 0
        post = dict(p=s["p"][mk], ab=o["ab"][mk], nu_mu=s["nu_mu"][mm], nu_sd=s["nu_sd"][mm])
        legs_used = sorted({sg.leg for sg in segs})
        leg_names = tuple(cap.LEGS[l] for l in legs_used)
        from cfcap.t1 import reference
        refs = reference.rho_ref_many([c.asm for c in ep.cands], a.plate)          # (C, L) fixed T1 reference
        jobs, meta = [], []
        for k, (mu, loaded) in enumerate(keys):
            for c_i, cd in enumerate(ep.cands):
                mods = np.array([gs.D["asm_mod_off"][r] + kk for r, kk in cd.modules])
                widx = (4 * mods[:, None] + np.arange(4)).reshape(-1)
                lat = CO.sample_from_episode(ep, cd, post, widx, list(mods), a.env_S, rng, dr)
                lat["mu_mod"] = lat["mu_mod"] * float(mu)
                if loaded:
                    f = lambda v: torch.full((a.env_S,), float(v), device="cuda")
                    lat["payload"] = dict(mass=f(m.payload_kg), x=f(0.0), y=f(0.0))
                else:
                    lat["payload"] = None
                jobs.append((cd.asm, lat))
                meta.append((k, c_i))
        res = cap.capability(jobs, par, legs=leg_names)
        q = np.full((len(keys), C, L, 3), np.nan, np.float32)
        for (k, c_i), r in zip(meta, res):
            smp = r["rho"].cpu().numpy() / refs[c_i, legs_used]
            q[k, c_i, legs_used] = np.nanquantile(smp, (0.05, 0.5, 0.95), axis=0).T
        dt = time.perf_counter() - t0
        for c_i in range(C):
            for s_i, sg in enumerate(segs):
                k = int(ci[s_i])
                for key, v in (("family", B.FAM_CODE[fam]), ("mission", mi), ("building", m.building), ("cand", c_i),
                               ("seg", s_i), ("leg", sg.leg), ("cond", k), ("q05", q[k, c_i, sg.leg, 0]),
                               ("q50", q[k, c_i, sg.leg, 1]), ("q95", q[k, c_i, sg.leg, 2]), ("seconds", dt),
                               ("shard", ctx.runner.args.shard)):
                    cols[key].append(v)
        ctx.progress((j + 1) / len(idx), f"env {fam}")
    out = {k: np.asarray(v) for k, v in cols.items()}
    out = {k: (v.astype(np.float32) if v.dtype.kind == "f" else v.astype(np.int64)) for k, v in out.items()}
    n = len(out["mission"])
    out["status_conf"] = np.full(n, int(fam.startswith("conf-")), np.int8)
    ctx.meta.update(family=fam, missions=len(idx), model=a.env_model)
    return out


def summarize(records, a):
    m = {}
    for r in records:
        for k, v in r.get("meta", {}).items():
            if isinstance(v, (int, float)):
                m.setdefault(k, []).append(float(v))
    return {k: float(np.median(v)) for k, v in m.items()}


TASK = Task(name="models_eval2", units=units, run_unit=run_unit, add_args=add_args, summarize=summarize,
            requires=("numpy", "torch"), doc=__doc__)
