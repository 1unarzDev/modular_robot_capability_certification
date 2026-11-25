"""Runner task `train` (role models): one Stage B training run of the arch x diversity x seed grid.

  python -m cfcap.runner train --shard K --n-shards N --grid b1 [--chunks 8 --gen-s 90 --train-s 150 ...]

Shard K is grid entry K (`grid(name)[K]`; n_shards must equal the grid length). Data are the generator's episodes
(cfcap.gen: plated, heterogeneity HET_FULL, parents N 1-6 of the train families, children + merges, payload),
generated inside the job from their indices, so episode ids match every other role. Units, in order:
  train-00 .. train-{C-1}   one epoch chunk (<= 5 min on a 3090): generate --gen-s seconds of fresh train-split
                            episodes whose parent passes the run's diversity filter (cheap pre-filter on the parent
                            shape before any simulation; self + max-cand - 1 labelled candidates each), keep them in a
                            FIFO buffer of the last `--buffer` pools, train --train-s seconds.
                            Model / optimiser / step live in OUT/ckpt.pt (atomic) between units; a stop request
                            inside a unit is checkpointed with ctx.save_state. Rows: the loss log.
  Label: rho_dec = rho_abs / rho_ref (cfcap.t1.reference, coordinator decision 2026-11-25); rho_rel kept as
  `truth_rel`.
  eval-cal, eval-test, eval-H3, eval-H4
                            the generator's cal / test / H3 / H4 episodes [0, --eval-n) (identical for every run),
                            inferred; rows per (episode, candidate, leg): direct-head quantiles, compositional
                            quantiles (posterior samples -> T1 rollouts of the candidate; cal and test, first
                            --compose-n episodes), truth; and per wheel: mode posterior, Beta, truth. OUT/weights.pt.
summarize -> metrics.json. `scripts/models_export.py` turns pulled runs into results/models/paper/{dec_models,
diversity,latency_models}.csv (docs/board/paper.md contracts).
"""
from __future__ import annotations

import argparse
import io
import math
import time
from functools import lru_cache

import numpy as np

from cfcap.runner import Stop, Task, atomic_bytes, unit_seed

PLATE = "plated"
ARCHS = {
    "st": dict(h=96, layers=3),                  # set-transformer posterior + direct head (joint)
    "stl": dict(h=128, layers=4),                # larger set transformer
    "ds": dict(h=96, layers=0),                  # DeepSets ablation (capxfer-like pooled context)
    "bb": dict(h=96, layers=3, blackbox=True),   # black-box direct counterfactual only (cf loss only)
}
# Training diversity = parent filter on the generator's train split (max parent modules, allowed topologies).
DIVERSITY = {
    "n2": dict(max_n=2),
    "n4c": dict(max_n=4, topologies=("single", "chain", "grid")),
    "n4": dict(max_n=4),
    "n6c": dict(max_n=6, topologies=("single", "chain", "grid")),
    "n6": dict(max_n=6),
}
EVAL_SPLITS = ("cal", "test", "H3", "H4")
QCOLS = ("q05", "q25", "q50", "q75", "q95")


def grid(name):
    """Grid entries in priority order (shard index = entry index)."""
    if name == "b1":
        out = []
        for seed in (1, 2, 3):
            for div in ("n6", "n4", "n2", "n6c", "n4c"):
                for arch in ("st", "bb", "ds"):
                    out.append(dict(arch=arch, div=div, seed=seed))
            out.append(dict(arch="stl", div="n6", seed=seed))
        return out
    if name == "smoke":
        return [dict(arch="st", div="n4", seed=1), dict(arch="bb", div="n2", seed=1)]
    raise ValueError(f"unknown grid {name}")


def n_train_families(div):
    from cfcap.gen import families as F
    d = DIVERSITY[div]
    return sum(1 for n, _, topo, split, _, _ in F.family_table(6)
               if split == "train" and n <= d["max_n"] and (d.get("topologies") is None or topo in d["topologies"]))


def parent_ok(div, split, index):
    """Pre-filter on the parent shape only (replays make_episode's first draw; no simulation)."""
    from cfcap.gen import episode as E
    from cfcap.gen import families as F
    d = DIVERSITY[div]
    rule = E.RULES[split]
    sh = F.sample_shape(E.episode_rng(PLATE, split, index), rule.parent_size_w, rule.parent_split)
    return len(sh) <= d["max_n"] and (d.get("topologies") is None or F.topology(sh) in d["topologies"])


def train_indices(div, chunk, n, stride=10_000_000):
    """The first n train-split indices of chunk `chunk`'s index block that pass the diversity filter."""
    out, i = [], chunk * stride
    while len(out) < n and i < (chunk + 1) * stride:
        if parent_ok(div, "train", i):
            out.append(i)
        i += 1
    return out


# ------------------------------------------------------------------ args / units
def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--grid", default="b1")
    p.add_argument("--chunks", type=int, default=8, help="training units")
    p.add_argument("--gen-s", type=float, default=90.0, help="pool generation seconds per chunk (first run)")
    p.add_argument("--train-s", type=float, default=150.0, help="training seconds per chunk (compute-matched)")
    p.add_argument("--pool", type=int, default=6000, help="max fresh episodes per chunk")
    p.add_argument("--buffer", type=int, default=4, help="pools kept (FIFO)")
    p.add_argument("--wheels", type=int, default=1536, help="wheel sequences per step (whole episodes)")
    p.add_argument("--max-cand", type=int, default=8, help="candidates per episode per step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-n", type=int, default=300, help="episodes per eval split")
    p.add_argument("--compose-n", type=int, default=100, help="cal/test episodes with the compositional route")
    p.add_argument("--compose-S", type=int, default=16)
    p.add_argument("--smoke", action="store_true", help="one unit: a training chunk then every eval split (canary)")
    p.add_argument("--episodes", type=int, default=None, help="(plan bookkeeping only)")
    p.add_argument("--data", default="on-the-fly", help="(plan bookkeeping only)")


def units(args, shard, n_shards):
    g = grid(args.grid)
    if n_shards > len(g) or shard >= len(g):
        raise ValueError(f"grid {args.grid} has {len(g)} entries; shard {shard} / n_shards {n_shards} out of range")
    if args.smoke:
        return ["smoke"]
    return [f"train-{i:02d}" for i in range(args.chunks)] + [f"eval-{s}" for s in EVAL_SPLITS]


def _cfg(ctx):
    return grid(ctx.args.grid)[ctx.runner.args.shard]


def build_model(cfg):
    from cfcap.models import posterior as M
    a = ARCHS[cfg["arch"]]
    return M.MorphPosterior(h=a["h"], layers=a["layers"], direct=True)


def loss_weights(cfg):
    return dict(w_ce=0.0, w_g=0.0, w_nu=0.0, w_cf=5.0) if ARCHS[cfg["arch"]].get("blackbox") else {}


# ------------------------------------------------------------------ data
_POOLS = {}


def make_set(split, indices, log=None, cand_cap=None, time_budget=None):
    from cfcap.models import gendata as G
    D, _, st = G.episodes(PLATE, split, len(indices), indices=indices, log=log, cand_cap=cand_cap,
                          time_budget=time_budget)
    return G.GenSet(D, PLATE), st


def pool(div, chunk, n_max, cand_cap, n_exact=None, time_budget=None, log=None):
    """Chunk `chunk`'s training pool: the first episodes of its index block passing the diversity filter, generated
    until `time_budget` s (first time) or exactly `n_exact` episodes (a restart replays the recorded count)."""
    key = (div, chunk)
    if key not in _POOLS:
        idx = train_indices(div, chunk, n_exact or n_max)
        gs, st = make_set("train", idx, log, cand_cap, None if n_exact else time_budget)
        _POOLS[key] = gs
    return _POOLS[key]


# ------------------------------------------------------------------ checkpoint I/O
def _save(path, obj):
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    atomic_bytes(path, buf.getvalue())
    return len(buf.getvalue())


def _load(path):
    import torch
    return torch.load(path, map_location="cuda", weights_only=False) if path.exists() else None


# ------------------------------------------------------------------ units
def run_unit(unit, ctx):
    if unit == "smoke":                                     # canary: the whole pipeline at toy size in one unit
        ctx.args.chunks = 1
        m = dict(_train_chunk(0, ctx) and ctx.meta)
        out = None
        for s in EVAL_SPLITS:
            ctx.meta = {}
            out = _eval(s, ctx)
            m.update({f"{s}/{k}": v for k, v in ctx.meta.items()})
        ctx.meta = m
        return out
    if unit.startswith("train-"):
        return _train_chunk(int(unit.split("-")[1]), ctx)
    return _eval(unit.split("-", 1)[1], ctx)


def _train_chunk(ci, ctx):
    """Generate this chunk's pool for --gen-s seconds, then train for --train-s seconds on the FIFO buffer of the
    last --buffer pools. LR follows a cosine over the run's total training time (compute-matched across archs);
    steps and pool sizes are recorded in ckpt.pt so a restarted job replays the same pools."""
    import torch
    from cfcap.models import gendata as G
    a, cfg = ctx.args, _cfg(ctx)
    torch.manual_seed(cfg["seed"] * 1000 + ci)
    model = build_model(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    ck_path = ctx.out_dir / "ckpt.pt"
    ck = _load(ck_path)
    st = ctx.load_state()
    src = st if st is not None else ck
    step0, t_train0, pool_n = 0, 0.0, {}
    if src is not None:
        model.load_state_dict(src["model"])
        opt.load_state_dict(src["opt"])
        step0, t_train0, pool_n = int(src["step"]), float(src.get("t_train", 0.0)), dict(src.get("pool_n", {}))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ctx.progress(0.0, "gen")
    want = list(range(max(0, ci - a.buffer + 1), ci + 1))
    for k in [k for k in _POOLS if k[0] != cfg["div"] or k[1] not in want]:
        del _POOLS[k]
    pools = []
    for cj in want:
        pools.append(pool(cfg["div"], cj, a.pool, a.max_cand, n_exact=pool_n.get(cj), time_budget=a.gen_s))
        pool_n[cj] = pools[-1].N
        if ctx.should_stop():
            raise Stop("during pool generation")
    t_gen = time.time() - t0
    w = np.array([p.N for p in pools], float)
    w /= w.sum()
    rng = np.random.default_rng(unit_seed("cfcap-models-batch", cfg["arch"], cfg["div"], cfg["seed"], ci, step0))
    lw = loss_weights(cfg)
    rows = dict(step=[], loss=[], ce=[], beta=[], nuis=[], cf=[])
    model.train()
    step, total = step0, a.chunks * a.train_s
    budget_end = (ci + 1) * a.train_s                       # cumulative training seconds at the end of this chunk
    t_last = time.time()
    t_train = t_train0
    while t_train < budget_end:
        if ctx.should_stop():
            ctx.save_state(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, t_train=t_train,
                                pool_n=pool_n))
            raise Stop("train checkpoint")
        lr = a.lr * min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, t_train / total)))
        for g in opt.param_groups:
            g["lr"] = lr
        ps = pools[rng.choice(len(pools), p=w)]
        eps, nw = [], 0
        while nw < a.wheels:
            e = int(rng.integers(ps.N))
            eps.append(e)
            nw += int(ps.ep_wheels[e])
        b = ps.batch(eps, a.max_cand, rng)
        b["af"][..., 3:] += 0.01 * torch.randn(b["af"].shape[0], 1, 3, device="cuda")
        b["wf"][..., 2] *= 1 + 0.01 * torch.randn(b["wf"].shape[0], 1, b["wf"].shape[2], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = G.forward(model, b)
        out = {k: v.float() for k, v in out.items()}
        loss, parts = G.loss(out, b, **lw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        now = time.time()
        t_train += now - t_last
        t_last = now
        if step % 100 == 0:
            rows["step"].append(step)
            rows["loss"].append(float(loss.detach()))
            for k in ("ce", "beta", "nuis", "cf"):
                rows[k].append(float(parts[k].detach()) if k in parts else float("nan"))
            ctx.progress((t_train - ci * a.train_s) / a.train_s, "train")
    _save(ck_path, dict(model=model.state_dict(), opt=opt.state_dict(), step=step, t_train=t_train, pool_n=pool_n,
                        cfg=cfg))
    wall = time.time() - t0
    ctx.meta.update(gen_s=round(t_gen, 1), train_s=round(wall - t_gen, 1), wall_s=round(wall, 1),
                    steps=step - step0, steps_per_s=round((step - step0) / max(wall - t_gen, 1e-6), 2),
                    peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                    pool_new=int(pools[-1].N), pool_episodes=int(sum(p.N for p in pools)),
                    gen_s_per_new_episode=round(t_gen / max(1, pools[-1].N), 3))
    if not rows["step"]:
        rows = dict(step=[step], loss=[float("nan")], ce=[float("nan")], beta=[float("nan")], nuis=[float("nan")],
                    cf=[float("nan")])
    return rows


@lru_cache(maxsize=None)
def _eval_set(split, n):
    return make_set(split, list(range(n)))


def _eval(split, ctx):
    import torch
    from cfcap.models import gendata as G
    from cfcap.models import posterior as M
    a, cfg = ctx.args, _cfg(ctx)
    ck = _load(ctx.out_dir / "ckpt.pt")
    if ck is None:
        raise RuntimeError("no ckpt.pt: training units missing")
    model = build_model(cfg).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    if split == "cal":
        ctx.meta["weights_mb"] = round(_save(ctx.out_dir / "weights.pt",
                                             dict(state=model.state_dict(), cfg=cfg, arch=ARCHS[cfg["arch"]])) / 2 ** 20,
                                       2)
    ctx.progress(0.0, f"gen-{split}")
    t0 = time.time()
    gs, gst = _eval_set(split, a.eval_n)
    ctx.meta["gen_s"] = round(time.time() - t0, 1)
    D = gs.D
    # inference, episode batches
    torch.cuda.synchronize()
    t0 = time.time()
    P = dict(p=[], ab=[], nu_mu=[], nu_sd=[], cf=[], wrows=[], mrows=[], cands=[])
    with torch.no_grad():
        e = 0
        while e < gs.E:
            eps, nw = [], 0
            while e < gs.E and nw < 2048:
                eps.append(e)
                nw += int(gs.ep_wheels[e])
                e += 1
            b = gs.batch(eps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = G.forward(model, b)
            o = {k: v.float() for k, v in o.items()}
            s = M.posterior_summary(o)
            m, mm = b["mask"], b["mod_mask"] > 0
            P["p"].append(s["p"][m])
            P["ab"].append(o["ab"][m])
            P["nu_mu"].append(s["nu_mu"][mm])
            P["nu_sd"].append(s["nu_sd"][mm])
            P["cf"].append(o["cf"])
            P["cands"].append(b["cands"])
    torch.cuda.synchronize()
    t_inf = time.time() - t0
    post = {k: torch.cat(P[k]) for k in ("p", "ab", "nu_mu", "nu_sd", "cf")}   # wheels / modules in global row order
    cands = np.concatenate(P["cands"])
    assert (cands == np.arange(gs.C)).all()
    ctx.meta["infer_ms_per_episode"] = 1e3 * t_inf / gs.E
    # compositional route (posterior samples -> T1 rollouts of each candidate), cal/test, first compose_n episodes
    comp = np.full((gs.C, 10, 5), np.nan, np.float32)
    comp_s = np.full(gs.E, np.nan)
    if split in ("cal", "test") and a.compose_n > 0 and not ARCHS[cfg["arch"]].get("blackbox"):
        comp, comp_s = _compose(gs, post, a.compose_n, a.compose_S, ctx)
    # rows: one per (episode, candidate, leg)
    L = 10
    cf = post["cf"].cpu().numpy()
    ce = D["cand_ep"]
    cols = dict(eid=np.repeat(D["ep_index"][ce], L), ep_row=np.repeat(ce, L), cand=np.repeat(
        np.arange(gs.C) - D["ep_cand_off"][ce], L), kind=np.repeat(D["cand_kind"], L),
        cand_size=np.repeat(np.diff(D["cand_cm_off"]), L), has_fault=np.repeat((D["cand_n_fault"] > 0).astype(np.int8), L),
        cand_family=np.repeat(D["cand_family"], L), cand_fam_split=np.repeat(D["cand_fam_split"], L),
        parent_family=np.repeat(D["asm_family"][D["ep_asm_off"][:-1]][ce], L),
        parent_n=np.repeat(gs.asm_m[D["ep_asm_off"][:-1]][ce], L),
        parent_topo=np.repeat(D["asm_topo"][D["ep_asm_off"][:-1]][ce], L),
        leg=np.tile(np.arange(L), gs.C), truth=D["cand_rho_dec"].reshape(-1),
        truth_rel=D["cand_rho_rel"].reshape(-1), rho_ref=D["cand_rho_ref"].reshape(-1),
        comp_seconds=np.repeat(comp_s[ce], L))
    for k, q in enumerate(QCOLS):
        cols[q] = cf[..., k].reshape(-1)
        cols[f"c{q}"] = comp[..., k].reshape(-1)
    # per-wheel posterior (parent and other assemblies) as a sidecar npz (not row-aligned with the decision rows)
    pw = post["p"].cpu().numpy()
    wep = np.repeat(D["asm_ep"], gs.asm_n)
    side = dict(eid=D["ep_index"][wep], asm_role=np.repeat(D["asm_role"], gs.asm_n), mode=D["wheel_mode"],
                g=D["wheel_g"], p=pw.astype(np.float32), ab=post["ab"].cpu().numpy().astype(np.float32),
                parent_n=gs.asm_m[D["ep_asm_off"][:-1]][wep])
    buf = io.BytesIO()
    np.savez_compressed(buf, **side)
    atomic_bytes(ctx.out_dir / f"wheels_{split}.npz", buf.getvalue())
    ctx.meta.update({f"det/{k}": v for k, v in detection(side).items()})
    out = {}
    for k, v in cols.items():
        v = np.asarray(v)
        out[k] = v.astype(np.float32) if v.dtype.kind == "f" else v.astype(np.int64 if "family" in k else np.int32)
    return out


def _compose(gs, post, n_ep, S, ctx):
    import torch
    from cfcap.gen import episode as E
    from cfcap.models import compose as CO
    from cfcap.t1 import capability as cap
    D = gs.D
    par, dr = E.params(PLATE), E.domain_dr(PLATE)
    rng = np.random.default_rng(unit_seed("cfcap-models-compose", ctx.unit))
    comp = np.full((gs.C, 10, 5), np.nan, np.float32)
    secs = np.full(gs.E, np.nan)
    for e0 in range(0, min(n_ep, gs.E), 25):
        if ctx.should_stop():
            raise Stop("compose")
        jobs, idx = [], []
        t0 = time.time()
        for e in range(e0, min(e0 + 25, n_ep, gs.E)):
            for c in gs.ep_cand[e]:
                k0, k1 = D["cand_cm_off"][c:c + 2]
                mods = D["cm_mod"][k0:k1]
                ca = E.build("cand", D["cm_cell"][k0:k1], D["cm_yaw"][k0:k1], ["M1"] * len(mods), PLATE)
                widx = (4 * mods[:, None] + np.arange(4)).reshape(-1)
                lat = CO.sample_child_latents(ca, widx, post, S, rng, dr, mods=list(mods))
                pm = D["cand_payload"][c]
                if pm[0] > 0:
                    f = lambda v: torch.full((S,), float(v), device="cuda")
                    lat["payload"] = dict(mass=f(pm[0]), x=f(pm[1]), y=f(pm[2]))
                jobs.append((ca, lat))
                idx.append(c)
        res = CO.compositional_quantiles(jobs, par, ref=gs.ref[idx])
        dt = time.time() - t0
        for c, (q, _) in zip(idx, res):
            comp[c] = q
        es = np.unique(D["cand_ep"][idx])
        secs[es] = dt / max(len(es), 1)
        ctx.progress(min(1.0, (e0 + 25) / n_ep), "compose")
    return comp, secs


def detection(side, fa=0.01):
    """Parent wheels: episode score max(1 - p_ok); threshold at the healthy-parent 99th percentile; detection of
    parents with exactly one weak wheel, all g and g in [0.75, 0.9]; median |E g - g| on that wheel; ECE."""
    par = side["asm_role"] == 0
    eid, p, mode, g, ab = side["eid"][par], side["p"][par], side["mode"][par], side["g"][par], side["ab"][par]
    ue, inv = np.unique(eid, return_inverse=True)
    score = np.zeros(len(ue))
    np.maximum.at(score, inv, 1 - p[:, 0])
    nf = np.zeros(len(ue), int)
    np.add.at(nf, inv, (mode != 0).astype(int))
    nw = np.zeros(len(ue), int)
    np.add.at(nw, inv, (mode == 1).astype(int))
    out = {}
    if (nf == 0).sum() < 20:
        return out
    thr = float(np.quantile(score[nf == 0], 1 - fa))
    one_weak = (nf == 1) & (nw == 1)
    wsel = (mode == 1) & one_weak[inv]
    gt, gh = g[wsel], (ab[wsel, 0] / ab[wsel].sum(1))
    det = score[inv[wsel]] > thr
    mild = (gt >= 0.75) & (gt <= 0.9)
    out.update(det_all=float(det.mean()) if len(det) else np.nan,
               det_mild=float(det[mild].mean()) if mild.any() else np.nan,
               gerr=float(np.median(np.abs(gh - gt))) if len(gt) else np.nan,
               ece_fault=_ece(1 - p[:, 0], (mode != 0).astype(float)),
               ece_top=_ece(p.max(1), (p.argmax(1) == mode).astype(float)))
    return {k: v for k, v in out.items() if np.isfinite(v)}


def _ece(conf, correct, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - correct[s].mean())
    return float(e)


def _pin(q, y, taus=(0.05, 0.25, 0.5, 0.75, 0.95)):
    t = np.asarray(taus)
    d = y[..., None] - q
    return float(np.nanmean(np.maximum(t * d, (t - 1) * d)))


# ------------------------------------------------------------------ summary
def summarize(records, args):
    import os
    from cfcap.runner import resolve_out
    m = {}
    tr = [r for r in records if r["unit"].startswith("train-")]
    if tr:
        m["chunk_wall_s_mean"] = float(np.mean([r["meta"].get("wall_s", r["wall_s"]) for r in tr]))
        m["chunk_gen_s_mean"] = float(np.mean([r["meta"].get("gen_s", 0) for r in tr]))
        m["steps_per_s"] = float(np.mean([r["meta"].get("steps_per_s", 0) for r in tr]))
        m["peak_vram_gb"] = float(max(r["meta"].get("peak_vram_gb", 0) for r in tr))
    out = resolve_out(getattr(args, "out", None), "train", args.shard, os.environ)
    for r in records:
        if not r["unit"].startswith("eval-"):
            continue
        s = r["unit"][5:]
        for k, v in r["meta"].items():
            if isinstance(v, (int, float)):
                m[f"{s}/{k}"] = float(v)
        f = out / "units" / f"{r['unit']}.npz"
        if f.exists():
            with np.load(f) as z:
                q = np.stack([z[c] for c in QCOLS], -1)
                y = z["truth"]
                m[f"{s}/pinball"] = _pin(q, y)
                m[f"{s}/cov90"] = float(((y >= q[:, 0]) & (y <= q[:, -1])).mean())
                cq = np.stack([z["c" + c] for c in QCOLS], -1)
                ok = np.isfinite(cq).all(1)
                if ok.any():
                    m[f"{s}/comp_pinball"] = _pin(cq[ok], y[ok])
                    m[f"{s}/direct_pinball_on_comp_rows"] = _pin(q[ok], y[ok])
    return {k: v for k, v in m.items() if np.isfinite(v)}


TASK = Task(name="train", units=units, run_unit=run_unit, add_args=add_args, summarize=summarize,
            requires=("numpy", "torch"), doc=__doc__)
