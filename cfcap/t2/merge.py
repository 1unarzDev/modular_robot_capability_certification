"""Merges of never-co-assembled modules (E2 / N4 / confirmatory C-MERGE): pair generation, truths and certifiers.

One pair = two assemblies G1, G2 (1-3-module polyominoes, module yaws U{0..3}, <= 1 fault each, HET_FULL, a shared theta
draw from the v2 DR), probed separately with v2 (T1, no payload, cfcap.baselines.episode.observe_parent recipe), and
up to 6 face-touching merges of all of G2 against all of G1. Per merge (x 10 legs):
  truth T1      merged body, carried latents, the indeterminate load split redrawn (HET_FULL), the arm's payload;
  truth T2      cfcap.t2 `all` on the same merged latents;
  comp_oracle   T1 composition from the true separate latents (least-norm split, known payload): the delta_comp reference;
  enum / copy_parent / detector_gated   cfcap.baselines.compose on the two v2 probes (inferred latents).
Every label is rho_abs over the FIXED T1 nominal reference (cfcap.baselines.episode.ref_table). Margins are not applied
here (they are fitted on exploratory data and applied by the exporter).

Seeds: pair p of arm a uses np.random.default_rng(root + [a == "P5", p]); C-MERGE root = [92599, 21].
"""
from __future__ import annotations

import numpy as np

MAX_MERGES = 6
ARMS = ("P0", "P5")


def polyomino(rng, k):
    cells = [(0, 0)]
    while len(cells) < k:
        x, y = cells[int(rng.integers(len(cells)))]
        dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(rng.integers(4))]
        if (x + dx, y + dy) not in cells:
            cells.append((x + dx, y + dy))
    mx, my = min(c[0] for c in cells), min(c[1] for c in cells)
    return [(x - mx, y - my) for x, y in cells]


def one_fault(n, rng):
    mode = np.zeros((1, n), np.int64)
    g = np.ones((1, n), np.float32)
    if rng.uniform() < 0.5:
        w = int(rng.integers(n))
        mode[0, w] = int(rng.choice([1, 2, 3]))
        if mode[0, w] == 1:
            g[0, w] = rng.uniform(0.3, 0.95)
    return mode, g


def _clone(dom):
    import torch
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in dom.items()}


def merged_latents(l1, l2, load_z, payload):
    import torch
    cat = lambda k: torch.cat([l1[k], l2[k]], 1)                          # noqa: E731
    return dict(dom=_clone(l1["dom"]), mode=cat("mode"), g=cat("g"), stall_w=cat("stall_w"), noload_w=cat("noload_w"),
                yaw_err=cat("yaw_err"), vbatt_off=cat("vbatt_off"), mu_mod=cat("mu_mod"),
                load_z=torch.as_tensor(np.asarray(load_z)[None], dtype=torch.float32), gyro_bias=l1["gyro_bias"],
                acc_bias=l1["acc_bias"], drop_p=l1["drop_p"], payload=payload)


def gen_pair(root, arm, p, par, dr):
    """One pair: dict(pair, k1, k2, g1, g2, l1, l2, merges=[dict(asm, lat_truth, lat_comp, payload tuple)])."""
    import torch
    from capxfer import assembly as A, sim
    from cfcap.t0 import candidates as C
    from cfcap.t1 import params as P, plant
    rng = np.random.default_rng(list(root) + [int(arm == "P5"), int(p)])
    het = plant.HET_FULL
    while True:
        k1, k2 = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        c1, c2 = polyomino(rng, k1), polyomino(rng, k2)
        pl = C.merge_placements(c1, c2)
        if pl:
            break
    y1, y2 = rng.integers(0, 4, k1), rng.integers(0, 4, k2)
    g1 = A.build(f"p{p}g1", [(P.MODULE_IDS[q], i, j, int(y)) for q, ((i, j), y) in enumerate(zip(c1, y1))], par)
    g2 = A.build(f"p{p}g2", [(P.MODULE_IDS[k1 + q], i, j, int(y)) for q, ((i, j), y) in enumerate(zip(c2, y2))], par)
    dom = sim.sample_domain(1, rng, dr, device="cpu")
    l1 = plant.sample_latents(g1, 1, rng, het, domain=_clone(dom), faults=one_fault(g1.n, rng), device="cpu")
    l2 = plant.sample_latents(g2, 1, rng, het, domain=_clone(dom), faults=one_fault(g2.n, rng), device="cpu")
    sel = rng.choice(len(pl), size=min(MAX_MERGES, len(pl)), replace=False)
    host = int(rng.integers(k1))
    off = rng.uniform(-0.08, 0.08, 2)
    pm = float(rng.uniform(0.0, 5.0)) if arm == "P5" else 0.0
    merges = []
    for s in sel:
        asm = C.merge(g1, tuple(range(k1)), g2, pl[int(s)], par).asm
        pay, ptup = None, None
        if pm > 0:
            xy = np.asarray(asm.module_pos)[host] + off
            pay = dict(mass=torch.tensor([pm]), x=torch.tensor([float(xy[0])]), y=torch.tensor([float(xy[1])]))
            ptup = (pm, float(xy[0]), float(xy[1]))
        lt = merged_latents(l1, l2, rng.normal(size=asm.n) * het.load_split, pay)
        lc = merged_latents(l1, l2, np.zeros(asm.n), pay)
        merges.append(dict(asm=asm, lat_truth=lt, lat_comp=lc, payload=ptup))
    return dict(pair=int(p), k1=k1, k2=k2, g1=g1, g2=g2, l1=l1, l2=l2, merges=merges)


def probe_v2(g, lat, par, seed, device="cuda"):
    """The baselines' observe_parent recipe on one assembly: v2 (no jitter), T1, plant.observe, duty noise."""
    import torch
    from cfcap.baselines.episode import DUTY_NOISE, lat_to
    from cfcap.t1 import plant
    tab, T = plant.v2_probe(g, par, 1, jitter=False)
    ld = lat_to(dict(lat, payload=None), device)
    R = plant.simulate_bucket([dict(asm=g, lat=ld, cmd=tab[0])], par, T, device=device)[0]
    obs = plant.observe(R, ld, seed=seed)
    gen = torch.Generator(device=obs["pre_duty"].device).manual_seed(int(seed) + 1)
    obs["pre_duty"] = obs["pre_duty"] + DUTY_NOISE * torch.randn(obs["pre_duty"].shape, generator=gen,
                                                                 device=obs["pre_duty"].device)
    return tab[0], T, {k: v[0].cpu() for k, v in obs.items()}


def score_pair(rec, par, device="cuda", t2_seed=0, backend="auto"):
    """Rows (one per merge) of truths and certifier outputs for one pair: dict of arrays (M, L) / (M,)."""
    from cfcap.baselines import compose as CO
    from cfcap.baselines.episode import lat_to, ref_table
    from cfcap.t1 import capability as cap1
    from cfcap.t2 import build, capability as cap2, eval as ev
    backend = ev.resolve_backend(backend)
    ms = rec["merges"]
    mods = [(0, k) for k in range(rec["k1"])] + [(1, k) for k in range(rec["k2"])]
    cands = [CO.MergeCand(m["asm"], list(mods), None, m["payload"]) for m in ms]
    ref = ref_table(cands, "plated", device)                                    # (M, L) fixed T1 reference
    t1 = np.stack([o["rho"].cpu().numpy()[0] for o in cap1.capability([(m["asm"], lat_to(m["lat_truth"], device)) for m in ms], par,
                                                                       device=device)])
    tc = np.stack([o["rho"].cpu().numpy()[0] for o in cap1.capability([(m["asm"], lat_to(m["lat_comp"], device)) for m in ms], par,
                                                                       device=device)])
    t2 = np.stack([o["rho"][0] for o in cap2.capability([(m["asm"], m["lat_truth"]) for m in ms], par, build.ALL,
                                                         backend=backend, seed=t2_seed)])
    probed = []
    for j, (g, l) in enumerate(((rec["g1"], rec["l1"]), (rec["g2"], rec["l2"]))):
        cmd, T, obs = probe_v2(g, l, par, rec["pair"] * 7 + j, device)
        probed.append(CO.Probed(g, cmd, T, obs, dict(l, payload=None)))
    pe = CO.merge_enum_known(probed, cands, device=device)
    pc = CO.merge_copy_parent(probed, cands)
    pd = CO.merge_detector_gated(probed, cands)
    return dict(truth_T1=t1 / ref, truth_T2=t2 / ref, comp_oracle=tc / ref, enum_point=pe.point, enum_lower=pe.lower,
                enum_scale=pe.scale, copy_parent=pc.lower, detector_gated=pd.lower,
                n_mod=np.full(len(ms), rec["k1"] + rec["k2"]),
                n_fault=np.array([int((m["lat_truth"]["mode"] > 0).sum()) for m in ms]),
                merge=np.arange(len(ms)))
