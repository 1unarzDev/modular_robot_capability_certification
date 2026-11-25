"""Runner task `decide` (cfcap.runner contract; `--task-spec cfcap.decide.task:TASK`): T1 decision episodes with
nuisance-marginalised SMC posterior draws (cfcap/nuisance_post.py, the agreed interface), ready for the offline
decision evaluation (cfcap.decide.evaluate via scripts/decide_t1_eval.py).

    python -m cfcap.runner decide --task-spec cfcap.decide.task:TASK --shard K --n-shards N \
        --shapes pair,square,2x3 --episodes 96 --per-unit 8

Episodes. Unit = (shape, chunk of episode indices); episode e of a shape is nuisance_post's episode (seed 909 default,
`episode_rng(seed, shape, e)`), so truth, merges and v2 observation are the nuisance role's, on every host. Per batch:
  1. nuisance_post SMC on the v2 routine (particles 256), `draws` (64) posterior draws per episode;
  2. capability per candidate (connected children + 10 merges with the 1-2-module other assembly) from T1 closed-loop
     legs, label rho_dec = rho_abs / rho_ref(candidate, leg) (coordinator decision 2026-11-25: rho_ref = absolute rho
     of the nominal healthy body, point theta, no spread; truth uses the same reference);
  3. probes: clamp, tug, moddiff = the extra legs of nuisance_post.Probe(kind) after the v2 part (same bins, same sd,
     the posterior is the v2 one so only the new bins enter the likelihood), and a diagnostic split per child = the
     child's own v2 routine (nuisance_post.Probe(child, ())) with a fresh normal-load split; cost T_SPLIT_S + its v2
     duration. Every probe's true observation comes from the true latents with the robot's noise (observe_feats).
  4. tasks: problem.make_tasks over t1.capability.LEGS from unit_seed("cfcap-decide-tasks", shape, e).
Outputs: OUT/decide_eps/<unit>.pkl (episode dicts, float32 arrays) and columnar rows per (episode, candidate, leg):
eid, shape_id, cand, kind_id (0 child, 1 merge), cand_size, leg, truth (rho_dec), point, scale, q05, has_fault, wall
seconds per episode, peak VRAM.
"""
from __future__ import annotations

import pickle
import time

import numpy as np

from cfcap.runner import Stop, Task, atomic_bytes, parse_chunk, unit_seed

SHAPES = ("pair", "square", "2x3")
T_SPLIT_S = 30.0
EXTRA = ("clamp", "tug", "moddiff")


def _args(p):
    p.add_argument("--shapes", default=",".join(SHAPES))
    p.add_argument("--episodes", type=int, default=96, help="episodes per shape")
    p.add_argument("--per-unit", type=int, default=8)
    p.add_argument("--nseed", type=int, default=909, help="nuisance_post episode seed")
    p.add_argument("--particles", type=int, default=256)
    p.add_argument("--draws", type=int, default=64)
    p.add_argument("--max-stages", type=int, default=150)
    p.add_argument("--engine", default="graph")


def _units(a, shard, n_shards):
    us = []
    for s in a.shapes.split(","):
        us += [f"dec-{s}-{lo:05d}-{min(lo + a.per_unit, a.episodes):05d}" for lo in range(0, a.episodes, a.per_unit)]
    return us[shard::n_shards]


def _flat(f):
    """nuisance_post feats dict (B, nb, ...) -> (B, D) and its sd (D,)."""
    import torch
    from cfcap import nuisance_post as NP
    parts, sds = [], []
    for k in ("duty", "speed", "gz"):
        x = f[k].reshape(f[k].shape[0], -1)
        parts.append(x)
        sds.append(torch.full((x.shape[1],), NP.SD[k]))
    return torch.cat(parts, 1).double().cpu().numpy(), torch.cat(sds).numpy()


def _probe_sim(asm, par, kinds, lat_rows, lat_true, seed, engine, skip_s):
    """Simulate a nuisance_post routine; keep only the bins after skip_s (the v2 part already in the posterior)."""
    from cfcap import nuisance_post as NP
    from cfcap.decide import posterior as DP
    pr = NP.Probe(asm, par, kinds)
    simr = NP.Sim(asm, par, pr, engine, 8192)
    f = simr.run(lat_rows)
    R = simr.run(lat_true, raw=True)
    y = NP.observe_feats(R, lat_true, seed=seed)
    k0 = int(round(skip_s / (NP.BIN * 0.02)))
    cut = lambda d: {kk: v[:, k0:] for kk, v in d.items()}
    sig, sd = _flat(cut(f))
    yy, _ = _flat(cut(y))
    return DP.Probe(sig.astype(np.float32), np.broadcast_to(sd, sig.shape).astype(np.float32), pr.T - skip_s,
                    yy[0])


def rho_ref(cands, par):
    """Absolute rho of the nominal healthy body (point theta, no spread) per candidate -> (C, L)."""
    import torch
    from cfcap.t1 import capability as cap
    from cfcap.t1 import params as P
    from cfcap.t1 import plant
    th = P.load_theta("plated")
    rng = np.random.default_rng(0)
    targets = [(c.asm, plant.sample_latents(c.asm, 1, rng, plant.HET_OFF, domain=P.point_domain(th, 1, "cuda")))
               for c in cands]
    return np.stack([o["rho"][0].float().cpu().numpy() for o in cap.capability(targets, par)])


def decision_batch(shape, episodes, a, log=None):
    import torch
    from cfcap import nuisance_post as NP
    from cfcap.decide import posterior as DP
    from cfcap.decide import problem as Pb
    from cfcap.t0 import candidates as Cn
    from cfcap.t1 import capability as cap
    from cfcap.t1 import params as P
    from cfcap.t1 import plant
    from capxfer import probes as CP, sim as CS
    S_ = NP.setup(shape, "v2", a.engine)
    par, dr, asm, space, kids, simr = S_["par"], S_["dr"], S_["asm"], S_["space"], S_["kids"], S_["sim"]
    E = len(episodes)
    truths, others, merges, lab_seeds = [], [], [], []
    for ep in episodes:                                          # identical to nuisance_post.run_batch
        rng = NP.episode_rng(a.nseed, shape, ep)
        truths.append(NP.sample_truth(space, rng))
        other = P.named_assembly(str(rng.choice(["single", "pair"])), par)
        torch.manual_seed(int(rng.integers(2**31)))
        others.append(plant.sample_latents(other, 1, rng, NP.HET_OTHER, dr=dr))
        merges.append(Cn.sample_merges(asm, other, par, rng, n_merge=10))
        lab_seeds.append(int(rng.integers(2**31)))
    modes_t = torch.stack([t[0] for t in truths]); z_t = torch.stack([t[1] for t in truths])
    gen = torch.Generator(device=NP.DEV).manual_seed(int(NP.episode_rng(a.nseed + 1, shape + "v2", episodes[0])
                                                         .integers(2**62)))
    t0 = time.time()
    lat_t = space.to_lat(modes_t, z_t)
    R = simr.run(lat_t, raw=True)
    y = NP.observe_feats(R, lat_t, seed=lab_seeds[0] % (2**31 - 10))
    del R
    q = NP.Q(NP.screen(space, simr, y, E), asm.n)
    modes, z, ll, info = NP.smc(space, simr, y, q, E, a.particles, gen, max_stages=a.max_stages)
    torch.cuda.synchronize()
    t_smc = (time.time() - t0) / E
    t_v2 = CS.probe_duration(CP.Schedule(asm.char_length(), pids=("P1", "P2", "P5")))
    out = []
    for e, ep in enumerate(episodes):
        t1 = time.time()
        pick = torch.randperm(a.particles, device=NP.DEV, generator=gen)[:a.draws]
        Mr = torch.cat([modes_t[e:e + 1], modes[e, pick]], 0)
        Zr = torch.cat([z_t[e:e + 1], z[e, pick]], 0)
        lat = space.to_lat(Mr, Zr)                                   # row 0 = truth
        lg = torch.Generator(device=NP.DEV).manual_seed(lab_seeds[e])
        cands = kids + merges[e]
        targets = []
        for c in cands:
            if c.kind == "child":
                lc = NP.fresh_split(plant.subset(lat, c.modules), lg)
            else:
                lo_ = plant._index(others[e], torch.zeros(lat["mode"].shape[0], dtype=torch.long, device=NP.DEV))
                lc = NP.fresh_split(NP.concat(plant.subset(lat, c.modules), lo_), lg)
            targets.append((c.asm, lc))
        rr = cap.capability(targets, par)
        ref = rho_ref(cands, par)                                    # (C, L)
        rho = np.stack([o["rho"].float().cpu().numpy() for o in rr], 1) / ref[None]   # (1 + draws, C, L)
        draws_lat = plant._index(lat, torch.arange(1, lat["mode"].shape[0], device=NP.DEV))
        true_lat = plant._index(lat, torch.zeros(1, dtype=torch.long, device=NP.DEV))
        true_lat = dict(true_lat, gyro_bias=lat_t["gyro_bias"][e:e + 1], acc_bias=lat_t["acc_bias"][e:e + 1])
        probes = {}
        for kd in EXTRA:
            probes[kd] = _probe_sim(asm, par, (kd,), draws_lat, true_lat,
                                    unit_seed("decide-probe", shape, ep, kd) % (2**31 - 10), a.engine, t_v2)
        for ci, c in enumerate(cands):
            if c.kind != "child":
                continue
            lc = NP.fresh_split(plant.subset(draws_lat, c.modules), lg)
            ltc = NP.fresh_split(plant.subset(true_lat, c.modules), lg)
            probes[f"split:{ci}"] = _probe_sim(c.asm, par, (), lc, ltc,
                                               unit_seed("decide-split", shape, ep, ci) % (2**31 - 10), a.engine, 0.0)
            probes[f"split:{ci}"].t_s += T_SPLIT_S
        lat_id = np.unique(np.concatenate([Mr[1:].cpu().numpy(),
                                           np.round(lat["g"][1:].cpu().numpy() * 10)], 1), axis=0,
                           return_inverse=True)[1].reshape(-1)
        dp = DP.DecisionPosterior(np.ones(a.draws), rho[1:].astype(np.float32), lat_id, probes)
        inst = Pb.Instance.from_candidates(Cn.cells_of(asm), cands, include_parent=False)
        tasks = Pb.make_tasks(np.random.default_rng(unit_seed("cfcap-decide-tasks", shape, ep) % (1 << 62)), cap.LEGS)
        tm = modes_t[e].cpu().numpy()
        hf = np.array([bool((tm[c.src[c.src >= 0]] != 0).any()) for c in cands])
        eid = SHAPES.index(shape) * 100000 + int(ep) if shape in SHAPES else int(ep)
        meta = dict(seed=eid, episode=int(ep), shape=shape, parent_size=len(asm.modules), n_faults=int((tm != 0).sum()),
                    P=a.draws, smc_s=t_smc, probe_s=time.time() - t1, stages=info.get("stages"))
        out.append(dict(inst=inst, post=dp, rho_true=rho[0].astype(float), rho_nom=np.ones_like(rho[0], dtype=float),
                        has_fault=hf, tasks=tasks, fold=int(ep) % 3, meta=meta, legs=cap.LEGS,
                        kinds=[c.kind for c in cands], sizes=[int(c.size) for c in cands]))
        if log:
            log(f"{shape} ep {ep}: {len(cands)} cands, {len(probes)} probes, {time.time() - t1:.1f}s")
    return out


def _rows(e):
    C, L = e["rho_true"].shape
    pt, sc = e["post"].summary()
    q05 = e["post"].bayes_lower(0.05)
    rep = lambda v: np.full(C * L, float(v))
    return dict(eid=rep(e["meta"]["seed"]), shape_id=rep(SHAPES.index(e["meta"]["shape"])),
                cand=np.repeat(np.arange(C), L), kind_id=np.repeat([k == "merge" for k in e["kinds"]], L).astype(np.int8),
                cand_size=np.repeat(e["sizes"], L), leg=np.tile(np.arange(L), C), truth=e["rho_true"].ravel(),
                point=pt.ravel(), scale=sc.ravel(), q05=q05.ravel(),
                has_fault=np.repeat(e["has_fault"][:, None], L, 1).ravel().astype(np.int8),
                smc_s=rep(e["meta"]["smc_s"]), probe_s=rep(e["meta"]["probe_s"]),
                peak_vram_gb=rep(e["meta"].get("peak_vram_gb", np.nan)))


def _run(unit, ctx):
    import torch
    a = ctx.args
    _, shape, lo, hi = unit.split("-")
    ctx.progress(0.0, phase=f"decide {shape} {lo}-{hi}")
    if ctx.should_stop():
        raise Stop()
    torch.cuda.reset_peak_memory_stats()
    eps = decision_batch(shape, list(range(int(lo), int(hi))), a)
    vram = torch.cuda.max_memory_reserved() / 1e9
    for e in eps:
        e["meta"]["peak_vram_gb"] = vram
    atomic_bytes(ctx.out_dir / "decide_eps" / f"{unit}.pkl", pickle.dumps(eps))
    parts = [_rows(e) for e in eps]
    ctx.meta.update(episodes=len(eps), smc_s_mean=float(np.mean([e["meta"]["smc_s"] for e in eps])),
                    probe_s_mean=float(np.mean([e["meta"]["probe_s"] for e in eps])), peak_vram_gb=vram)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _summary(recs, a):
    m = [r.get("meta", {}) for r in recs]
    n = sum(x.get("episodes", 0) for x in m)
    w = lambda k: sum(x.get(k, 0) * x.get("episodes", 0) for x in m) / max(n, 1)
    return {"decide.episodes": n, "decide.smc_s_mean": w("smc_s_mean"), "decide.probe_s_mean": w("probe_s_mean"),
            "decide.peak_vram_gb": max([x.get("peak_vram_gb", 0) for x in m] or [0])}


TASK = Task("decide", _units, _run, _args, _summary, ("numpy", "torch", "capxfer"),
            "T1 decision episodes: nuisance_post SMC draws + clamp/tug/moddiff/diagnostic-split probe predictions")
