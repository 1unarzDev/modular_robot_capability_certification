"""cfcap.runner task `t2_adv` (role stress): adversarial CEM search for confident-wrong certificates on T2.
EVALUATION ONLY: the found cases never enter training or calibration.

    python -m cfcap.runner t2_adv --shard K --n-shards N --restarts 240 --pop 256 --iters 25

One unit = one CEM restart on one morphology (weighted to singles and pairs, where A5 found the T1 -> T2 shifts:
single 0.45, pair 0.40, L3 0.10, square 0.05; deterministic per restart index). A member is a vector z in [0, 1]^d
mapped to:
  theta        every key of the v2 training DR box around the identified theta (cfcap.t1.params.theta_dr), cg offsets;
  heterogeneity per-wheel stall / no-load multipliers +-8 %, per-module v_batt +-0.3 V and mu factor +-10 % (HET_FULL);
  faults       0..2 faulted wheels (count, which wheels by score ranking, mode weak/coast/brake, weak g in [0.3, 0.95]);
  payload      0-10 kg at a point inside 0.8 x the wheel extent;
  T2 effects   per-wheel roller-friction factor [0.7, 1.3] (aniso_mu) and roller-drag factors per free direction
               log-uniform [1/4, 16] (aniso); the other T2 nuisances (compliance stiffnesses, COM heights) are the
               restart's seeded draws.
Score (higher = worse certificate), on rho_dec = rho / rho_ref of each plant (its own nominal reference):
    certifier(l) = rho_dec_T1(l)  (a T1 oracle with the latents known: the best any T1-trusting method can do), or
                   the lower bound of `--predictor mod:fn(asm, lat, params) -> (B, L)` (a real method);
    score = max over legs l and tau of min(certifier(l) - tau, tau - rho_dec_T2(l)),
positive iff the certifier says go with margin and T2's truth is below tau ("confident-wrong" by that margin).
CEM: Gaussian in z (clipped to [0, 1]), elite 10 %, smoothing 0.7, a state checkpoint after every iteration.
Rows per unit: the best member of every iteration and the final top 16 (score, margin, leg, tau, rho_dec per plant,
the decoded latents as columns, z as a JSON string). T1 runs on the CPU (torch), T2 on numba (backend auto).
"""
from __future__ import annotations

import json

import numpy as np

from cfcap.runner import Stop, Task, shard_range
from cfcap.t2 import deps  # noqa: F401

SHAPE_W = (("single", 0.45), ("pair", 0.40), ("L3", 0.10), ("square", 0.05))
TAUS = (0.7, 0.8)
DOM_SEARCH = ("mu", "s0", "c_roller", "crr", "tau_bd", "latency", "stall_scale", "noload_scale", "mass_scale",
              "izz_scale", "radius_scale", "v_batt", "jeff_scale")


def _args(p):
    p.add_argument("--restarts", type=int, default=240)
    p.add_argument("--pop", type=int, default=256)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--elite", type=float, default=0.1)
    p.add_argument("--plate", default="plated")
    p.add_argument("--predictor", default=None, help="module:function(asm, lat, params) -> lower (B, 10)")
    p.add_argument("--backend", default="auto")
    p.add_argument("--seed0", type=int, default=771000)


def _units(a, shard, n_shards):
    return [f"adv-{a.plate}-r{j:04d}" for j in shard_range(a.restarts, shard, n_shards)]


def restart_shape(j, seed0):
    r = np.random.default_rng([seed0, j, 1]).uniform()
    c = 0.0
    for s, w in SHAPE_W:
        c += w
        if r < c:
            return s
    return SHAPE_W[-1][0]


class Space:
    """z in [0, 1]^d <-> T1-format latents (+ T2 effect factors) for B members on one assembly."""

    def __init__(self, asm, dr):
        self.asm, self.dr = asm, dr
        self.n, self.m = asm.n, len(asm.modules)
        n, m = self.n, self.m
        sl, o = {}, 0
        for k, d in (("dom", len(DOM_SEARCH) + 2), ("stall", n), ("noload", n), ("vb", m), ("mum", m), ("kf", 1),
                     ("wsc", n), ("mode", 2), ("g", 2), ("pay", 3), ("muf", n), ("dpos", n), ("dneg", n)):
            sl[k] = slice(o, o + d)
            o += d
        self.sl, self.d = sl, o

    def decode(self, z, device="cpu"):
        import torch
        from capxfer import sim
        from cfcap.t1 import plant
        B, n, m = len(z), self.n, self.m
        z = np.clip(z, 0.0, 1.0)
        g_ = lambda k: z[:, self.sl[k]]                                       # noqa: E731
        lerp = lambda u, lo, hi: lo + (hi - lo) * u                           # noqa: E731
        dom = sim.sample_domain(B, np.random.default_rng(0), self.dr, device=device)
        zd = g_("dom")
        for i, k in enumerate(DOM_SEARCH):
            lo, hi = self.dr.get(k, (1.0, 1.0)) if isinstance(self.dr.get(k), tuple) else (1.0, 1.0)
            dom[k] = torch.tensor(lerp(zd[:, i], lo, hi), dtype=torch.float32)
        cg = float(self.dr.get("cg_off", 0.015))
        dom["cg_dx"] = torch.tensor(lerp(zd[:, -2], -2 * cg, 2 * cg), dtype=torch.float32)
        dom["cg_dy"] = torch.tensor(lerp(zd[:, -1], -2 * cg, 2 * cg), dtype=torch.float32)
        dom["wheel_yaw_err_std"] = 0.0
        lat = plant.sample_latents(self.asm, B, np.random.default_rng(1), plant.HET_OFF, domain=dom, device=device)
        f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)     # noqa: E731
        wm = np.asarray(self.asm.wheel_module)
        lat["stall_w"], lat["noload_w"] = f32(lerp(g_("stall"), 0.92, 1.08)), f32(lerp(g_("noload"), 0.92, 1.08))
        lat["vbatt_off"], lat["mu_mod"] = f32(lerp(g_("vb"), -0.3, 0.3)), f32(lerp(g_("mum"), 0.9, 1.1))
        k = np.minimum((g_("kf")[:, 0] * 3).astype(int), 2)                   # 0, 1 or 2 faulted wheels
        order = np.argsort(-g_("wsc"), axis=1)
        mode = np.zeros((B, n), np.int64)
        g = np.ones((B, n), np.float32)
        for b in range(B):
            for q in range(k[b]):
                w = order[b, q]
                mode[b, w] = 1 + min(int(g_("mode")[b, q] * 3), 2)
                if mode[b, w] == 1:
                    g[b, w] = lerp(g_("g")[b, q], 0.3, 0.95)
        lat["mode"], lat["g"] = torch.tensor(mode), torch.tensor(g)
        ext = np.abs(self.asm.wheel_pos).max(0) * 0.8
        p = g_("pay")
        lat["payload"] = dict(mass=f32(lerp(p[:, 0], 0.0, 10.0)), x=f32(lerp(p[:, 1], -ext[0], ext[0])),
                              y=f32(lerp(p[:, 2], -ext[1], ext[1])))
        lat["t2_mu_fac"] = lerp(g_("muf"), 0.7, 1.3)
        lat["t2_drag_pos"] = np.exp(lerp(g_("dpos"), np.log(0.25), np.log(16.0)))
        lat["t2_drag_neg"] = np.exp(lerp(g_("dneg"), np.log(0.25), np.log(16.0)))
        lat["_wm"] = wm
        return lat


def _refs(asm, par, plate, backend):
    from cfcap.t1 import capability as cap1, params as P, plant
    from cfcap.t2 import build, capability as cap2
    th = P.load_theta(plate)
    lat = plant.sample_latents(asm, 1, np.random.default_rng(0), plant.HET_OFF, domain=P.point_domain(th, 1, "cpu"),
                               device="cpu")
    r1 = cap1.capability([(asm, lat)], par, device="cpu")[0]["rho"].cpu().numpy()[0]
    r2 = cap2.rho_ref(asm, par, plate, build.ALL, n_draws=1, backend=backend)
    return r1, r2


def score_members(space, z, par, refs, seed, backend, predictor=None):
    from cfcap.t1 import capability as cap1
    from cfcap.t2 import build, capability as cap2
    lat = space.decode(z)
    lat1 = {k: v for k, v in lat.items() if not k.startswith("t2_") and k != "_wm"}
    r1 = cap1.capability([(space.asm, lat1)], par, device="cpu")[0]["rho"].cpu().numpy() / refs[0][None]
    r2 = cap2.capability([(space.asm, lat)], par, build.ALL, backend=backend, seed=seed)[0]["rho"] / refs[1][None]
    cert = r1 if predictor is None else np.asarray(predictor(space.asm, lat1, par), float)
    best = np.full(len(z), -np.inf)
    leg = np.zeros(len(z), int)
    tau = np.zeros(len(z))
    for t in TAUS:
        sc = np.minimum(cert - t, t - r2)                                     # (B, L)
        i = sc.argmax(1)
        v = sc[np.arange(len(z)), i]
        up = v > best
        best, leg, tau = np.where(up, v, best), np.where(up, i, leg), np.where(up, t, tau)
    return best, leg, tau, r1, r2, lat


def _row(space, z, b, sc, leg, tau, r1, r2, lat, it, kind):
    from cfcap.t1.capability import LEGS
    d = lat["dom"]
    r = dict(iter=it, kind=kind, score=float(sc[b]), leg=LEGS[int(leg[b])], tau=float(tau[b]),
             rho_dec_T1=float(r1[b, leg[b]]), rho_dec_T2=float(r2[b, leg[b]]),
             shift_max=float(np.abs(r2[b] - r1[b]).max()), n_fault=int((lat["mode"][b] > 0).sum()),
             faults=json.dumps([[int(w), int(lat["mode"][b, w]), round(float(lat["g"][b, w]), 3)]
                                for w in np.nonzero(lat["mode"][b].numpy())[0]]),
             payload_kg=float(lat["payload"]["mass"][b]), z=json.dumps(np.round(z[b], 4).tolist()))
    for k in DOM_SEARCH + ("cg_dx", "cg_dy"):
        r[f"dom_{k}"] = float(d[k][b])
    return r


def _run(unit, ctx):
    from cfcap.t1 import params as P
    from cfcap.t2 import eval as ev
    a = ctx.args
    backend = ev.resolve_backend(a.backend)
    j = int(unit.rsplit("-r", 1)[1])
    shape = restart_shape(j, a.seed0)
    par = P.base_params(a.plate)
    asm = P.named_assembly(shape, par)
    space = Space(asm, P.theta_dr(P.load_theta(a.plate)))
    refs = _refs(asm, par, a.plate, backend)
    pred = None
    if a.predictor:
        import importlib
        mod, _, fn = a.predictor.partition(":")
        pred = getattr(importlib.import_module(mod), fn)
    rng = np.random.default_rng([a.seed0, j, 2])
    st = ctx.load_state() or dict(it=0, mu=np.full(space.d, 0.5), sd=np.full(space.d, 0.3), rows=[], best=-np.inf)
    ne = max(2, int(a.elite * a.pop))
    seed = a.seed0 * 1000 + j
    for it in range(st["it"], a.iters):
        if ctx.should_stop():
            ctx.save_state(st)
            raise Stop("t2_adv checkpoint")
        rng_it = np.random.default_rng([a.seed0, j, 3, it])
        z = np.clip(st["mu"] + st["sd"] * rng_it.standard_normal((a.pop, space.d)), 0, 1)
        sc, leg, tau, r1, r2, lat = score_members(space, z, par, refs, seed, backend, pred)
        el = np.argsort(-sc)[:ne]
        st["mu"] = 0.7 * z[el].mean(0) + 0.3 * st["mu"]
        st["sd"] = np.maximum(0.7 * z[el].std(0) + 0.3 * st["sd"], 0.02)
        b = int(el[0])
        st["rows"].append(dict(_row(space, z, b, sc, leg, tau, r1, r2, lat, it, "iter_best"), shape=shape, restart=j))
        st["best"] = max(st["best"], float(sc[b]))
        if it == a.iters - 1:
            for b in el[:16]:
                st["rows"].append(dict(_row(space, z, int(b), sc, leg, tau, r1, r2, lat, it, "final_top"),
                                       shape=shape, restart=j))
        st["it"] = it + 1
        ctx.save_state(st)
        ctx.progress((it + 1) / a.iters, phase=f"cem {shape} it {it}")
    ctx.meta.update(shape=shape, best_score=st["best"], confident_wrong=int(st["best"] > 0), backend=backend)
    rows = st["rows"]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def _summary(recs, a):
    m = [r.get("meta", {}) for r in recs]
    out = {"adv.restarts": len(m), "adv.confident_wrong": sum(x.get("confident_wrong", 0) for x in m)}
    for s, _ in SHAPE_W:
        bs = [x["best_score"] for x in m if x.get("shape") == s]
        if bs:
            out[f"adv.best_score.{s}"] = float(max(bs))
            out[f"adv.cw_share.{s}"] = float(np.mean([b > 0 for b in bs]))
    return out


TASK = Task("t2_adv", _units, _run, _args, _summary, ("numpy", "torch"),
            "CEM adversarial search on T2 for confident-wrong T1 certificates (singles/pairs weighted)")
