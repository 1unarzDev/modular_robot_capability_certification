"""Runner task `smc` (cfcap.runner contract): GPU nuisance-SMC baseline (role nuisance's sampler cfcap.nuisance_post,
wrapped by cfcap.baselines.bayes.NuisanceSMC) (and optionally the known-nuisance enumeration)
over T1 episodes generated from seeds on the box.

    python -m cfcap.runner smc --shard K --n-shards N --split test --episodes 500 [--per-unit 2] [--with-enum]
    python -m cfcap.runner smc ... --source pilot --split pilot --episodes 300     # the baselines pilot bank

Episodes (pure functions of (split, index), identical on any host):
  * --source gen (default): the generator's `cfcap.gen.episode.make_episode(plate, split, index)`, observed and
    labelled with the generator's own `simrun.probe_episodes` / `label_episodes` (baselines `episode.from_gen`);
    eid = the generator index, so these rows join every other method's rows for the same episode.
  * --source pilot: the baselines pilot bank (`episode.pilot_episode`: pair / square / 2x3 cycled).
One unit = `per_unit` consecutive episodes of the shard's slice; the finished episodes' rows are checkpointed after
every episode, and ctx.should_stop() is checked between episodes.

Rows: docs/board/paper.md 1a `dec` schema, one per (method, episode, candidate, leg): method, route, posterior,
train, test_plant, split, status, eid, shape, cand, kind, cand_size, has_fault, leg, truth, point, scale, lower,
alpha, seconds; plus ep_n_faults (parent faulted wheels: fault-rate shift analysis), cand_fault_class (0 healthy,
1 weak only, 2 coast/brake involved: per-fault-class analysis) and per-episode diagnostics repeated on the episode's
rows: sims, beta, stages, peak_vram_gb.
`lower` here is the method's raw posterior 5 % quantile (route "raw"); the simultaneous conformal bound needs the
cal split and is added offline by `scripts/baselines_pilot.py export` (route "simultaneous").
"""
from __future__ import annotations

import time

import numpy as np

from cfcap.runner import Stop, Task, chunk_units, parse_chunk, shard_range

ALPHA = 0.1


def _args(p):
    p.add_argument("--source", default="gen", choices=["gen", "pilot"])
    p.add_argument("--plate", default="plated")
    p.add_argument("--split", default="test")
    p.add_argument("--episodes", type=int, default=500, help="episodes in the split (all shards)")
    p.add_argument("--index0", type=int, default=0, help="first episode index of the split")
    p.add_argument("--ids", default="", help="explicit comma list of episode indices (overrides --episodes/--index0)")
    p.add_argument("--per-unit", type=int, default=2)
    p.add_argument("--status", default="exploratory", choices=["exploratory", "confirmatory"])
    p.add_argument("--particles", type=int, default=256, help="SMC particles (cfcap.nuisance_post.smc)")
    p.add_argument("--max-stages", type=int, default=150)
    p.add_argument("--wall-budget-s", type=float, default=None,
                   help="matched compute: nuisance-SMC wall seconds per decision (screen + SMC + propagation); the SMC "
                        "stage cap is set per episode to fit it (models' median end-to-end time, docs/board/models.md)")
    p.add_argument("--n-prop", type=int, default=32, help="particles pushed to the candidates")
    p.add_argument("--with-enum", action="store_true", help="also the known-nuisance exact enumeration")
    p.add_argument("--methods", default="nuisance_smc",
                   help="comma list of nuisance_smc (nuisance_smc_p<N>: N particles), enum_known, capxfer, rls_interval, nominal, oracle, detector_gated, "
                        "physics_gains, copy_parent, weakest_member")
    p.add_argument("--capxfer-dir", default=None,
                   help="dir with capxfer rung_v2 models/ and sim_probs_<plate>.csv (default: the local cache)")


def _id_list(a):
    return [int(x) for x in a.ids.split(",")] if a.ids else None


def _units(a, shard, n_shards):
    ids = _id_list(a)
    if ids is not None:                                # explicit episode ids: units are chunks of list positions
        r = shard_range(len(ids), shard, n_shards)
        return chunk_units(f"{a.split}-pos", r, a.per_unit)
    r = shard_range(a.episodes, shard, n_shards)
    return chunk_units(a.split, range(a.index0 + r.start, a.index0 + r.stop), a.per_unit)


def _episode_ids(a, unit):
    r = parse_chunk(unit)
    ids = _id_list(a)
    return [ids[j] for j in r] if ids is not None else list(r)


def episode(a, j):
    from cfcap.baselines import episode as E
    if a.source == "pilot":
        return E.pilot_episode(a.split, j)
    return E.from_gen(a.plate, a.split, j)


def dec_rows(ep, method, posterior, pred, seconds, split, status, extra=None):
    """dec-schema columns of one method on one episode."""
    C, L = ep.truth.shape
    n = C * L
    hf = ep.cand_has_fault()
    pm, _ = ep.faults()
    cls = []
    for cm, _ in ep.cand_true_modes():                       # 0 healthy, 1 weak only, 2 coast/brake involved
        cls.append(2 if np.isin(cm, (2, 3)).any() else (1 if (cm == 1).any() else 0))
    rows = dict(method=np.full(n, method), route=np.full(n, "raw"), posterior=np.full(n, posterior),
                train=np.full(n, "T1"), test_plant=np.full(n, "T1"), split=np.full(n, split),
                status=np.full(n, status), eid=np.full(n, ep.eid, np.int64), shape=np.full(n, ep.shape),
                cand=np.repeat(np.arange(C), L), kind=np.repeat([c.kind for c in ep.cands], L),
                cand_size=np.repeat([c.size for c in ep.cands], L), has_fault=np.repeat(hf.astype(np.int8), L),
                leg=np.tile(np.arange(L), C), truth=ep.truth.ravel().astype(np.float64),
                point=pred.point.ravel().astype(np.float64), scale=pred.scale.ravel().astype(np.float64),
                lower=pred.lower.ravel().astype(np.float64), alpha=np.full(n, ALPHA), seconds=np.full(n, seconds),
                ep_n_faults=np.full(n, int((pm != 0).sum()), np.int16),
                parent_n=np.full(n, len(ep.parent_asm.modules), np.int16),
                cand_fault_class=np.repeat(np.asarray(cls, np.int8), L))
    for k, v in (extra or {}).items():
        rows[k] = np.full(n, float(v))
    return rows


POSTERIOR = dict(nuisance_smc="smc", enum_known="enum", capxfer="none", rls_interval="none", nominal="none",
                 oracle="oracle", detector_gated="none", physics_gains="none", copy_parent="none",
                 weakest_member="none")
TRIVIAL = ("detector_gated", "physics_gains", "copy_parent", "weakest_member")
_B = {}


def _baseline(name, a):
    if name not in _B:
        if name == "nominal":
            from cfcap.baselines.simple import Nominal as K
            _B[name] = K()
        elif name == "oracle":
            from cfcap.baselines.simple import Oracle as K
            _B[name] = K()
        elif name == "rls_interval":
            from cfcap.baselines.rls import RLSInterval as K
            _B[name] = K(a.plate)
        elif name == "capxfer":
            import pathlib
            from cfcap.baselines import capxfer_cert as CX
            if a.capxfer_dir:
                CX.BASE = pathlib.Path(a.capxfer_dir)
            _B[name] = CX.CapxferCert(alpha=ALPHA, plate=a.plate)
        elif name == "enum_known":
            from cfcap.baselines.bayes import Enumeration as K
            _B[name] = K()
        elif name in TRIVIAL:
            from cfcap.baselines import trivial as Tv
            _B[name] = dict(detector_gated=Tv.DetectorGated, physics_gains=Tv.PhysicsGains,
                            copy_parent=Tv.CopyParent, weakest_member=Tv.WeakestMember)[name]()
        elif name == "nuisance_smc" or name.startswith("nuisance_smc_p"):
            from cfcap.baselines.bayes import NuisanceSMC as K
            n_p = int(name[len("nuisance_smc_p"):]) if name.startswith("nuisance_smc_p") else a.particles
            _B[name] = K(particles=n_p, max_stages=a.max_stages, n_prop=a.n_prop, wall_budget_s=a.wall_budget_s)
        else:
            raise ValueError(name)
    return _B[name]


def _one(ep, a):
    import torch
    from cfcap.runner import unit_seed
    methods = a.methods.split(",") + (["enum_known"] if a.with_enum and "enum_known" not in a.methods else [])
    parts = []
    for name in methods:
        b = _baseline(name, a)
        if name.startswith("nuisance_smc"):
            b.seed = unit_seed("smc", a.split, ep.eid) % (1 << 31)
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        p = b(ep)
        if cuda:
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        rows = dec_rows(ep, name, POSTERIOR.get(name, "smc" if name.startswith("nuisance_smc") else "none"), p, sec,
                        a.split, a.status,
                        dict(sims=p.info.get("n_sims", 0), beta=p.info.get("beta", np.nan),
                             stages=p.info.get("n_stages", 0), mass_true=p.info.get("mass_true", np.nan),
                             truth_in_support=p.info.get("truth_in_support", np.nan),
                             truth_retained=p.info.get("truth_retained", np.nan),
                             dropped_mass=p.info.get("dropped_mass", np.nan), abstain=p.info.get("abstain", 0),
                             stage_cap=p.info.get("stage_cap", np.nan),
                             peak_vram_gb=torch.cuda.max_memory_reserved() / 1e9 if cuda else 0.0))
        if name == "capxfer":
            rows["point"] = np.full_like(rows["point"], np.nan)     # capxfer has a bound only
            rows["scale"] = np.full_like(rows["scale"], np.nan)
        parts.append(rows)
    return {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}


def _run(unit, ctx):
    a = ctx.args
    js = _episode_ids(a, unit)
    st = ctx.load_state() or {"done": {}}
    for n_done, j in enumerate(js):
        if j in st["done"]:
            continue
        if ctx.should_stop():
            ctx.save_state(st)
            raise Stop()
        ctx.progress(n_done / max(len(js), 1), phase=f"smc episode {j}")
        rows = _one(episode(a, j), a)
        rows["shard"] = np.full(len(rows["eid"]), int(getattr(a, "shard", -1)), np.int32)
        st["done"][j] = rows
        ctx.save_state(st)
    parts = [st["done"][j] for j in sorted(st["done"])]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    meta = dict(episodes=len(parts), peak_vram_gb=float(out["peak_vram_gb"].max()))
    for meth in np.unique(out["method"]):
        sm = out["method"] == meth
        first = np.unique(out["eid"][sm], return_index=True)[1]
        meta[f"{meth}_s_mean"] = float(out["seconds"][sm][first].mean())
        meta[f"{meth}_sims_mean"] = float(out["sims"][sm][first].mean())
    ctx.meta.update(meta)
    return out


def _summary(recs, a):
    m = [r.get("meta", {}) for r in recs]
    n = sum(x.get("episodes", 0) for x in m)
    w = lambda k: sum(x.get(k, 0) * x.get("episodes", 0) for x in m) / max(n, 1)
    out = {"smc.episodes": n, "smc.peak_vram_gb": max([x.get("peak_vram_gb", 0) for x in m] or [0])}
    names = sorted({k[:-len("_s_mean")] for x in m for k in x if k.endswith("_s_mean")})
    for meth in names:
        if any(f"{meth}_s_mean" in x for x in m):
            out[f"smc.{meth}.s_per_episode_mean"] = w(f"{meth}_s_mean")
            out[f"smc.{meth}.sims_per_episode_mean"] = w(f"{meth}_sims_mean")
    return out


TASK = Task("smc", _units, _run, _args, _summary, ("numpy", "torch", "capxfer"),
            "nuisance-SMC (+ optional known-nuisance enumeration) baseline over seeded T1 episodes, dec-schema rows")
