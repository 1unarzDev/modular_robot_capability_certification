"""T1 dynamic plant: capxfer's batched simulator (read-only dependency) with the effects the design critique found
missing, for mixed-size batches of assemblies.

capxfer `sim._prepare` builds the per-trial constants C and `sim._integrate` runs the CUDA-graph integrator. This module
calls both unchanged and only edits C between them:
  * per-wheel and per-module motor spread: stall torque and no-load speed multipliers (B,n) (capxfer: one per trial);
  * per-module battery voltage (B,n via the wheel's module) and per-module friction factor;
  * normal loads: capxfer's least-norm split plus a random null-space component (statically indeterminate share);
  * per-wheel mounting-yaw error owned by the latents (so several legs of one episode share it);
  * payload + CoM offset (capxfer's payload dict and domain cg offsets);
  * sensor bias / dropout in `observe` (IMU bias, wheel-sample dropout with hold, IMU noise as capxfer selfmodel).
Per-wheel commanded speed or duty legs (tug-of-war probes) need the per-wheel reference inside the firmware step, so
`_step_pw` is capxfer `sim._step` rewritten from its own source at import time (inspect + two anchored insertions; it
fails loudly if capxfer's step changes): where a per-wheel table entry is finite it replaces the IK reference (speed
legs, rad/s) or the PI duty (duty legs). Without per-wheel tables the original `sim._step` runs.

Wheel order is capxfer's: module k's corners fl, fr, rl, rr are wheels 4k..4k+3, so per-wheel latents reshape to
(B, modules, 4) and transfer to children / merges by module (`subset`, `concat`).
"""
from __future__ import annotations

import contextlib
import inspect
import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from cfcap.t1 import params as P  # noqa: F401  (registers a yaml stub when PyYAML is missing)
from capxfer import probes, sim

OK, WEAK, COAST, BRAKE = sim.OK, sim.WEAK, sim.COAST, sim.BRAKE
DT_OBS = 0.02                       # robot-visible grid (50 Hz), as capxfer.corpus
IMU_NOISE = (0.002, 0.02, 0.02)     # capxfer.corpus.selfmodel.IMU_NOISE (gz, ax, ay per-bin sd)


# ----------------------------------------------------------------------------- heterogeneity spec and latents
@dataclass(frozen=True)
class Het:
    wheel_motor: float = 0.0     # half-width of per-wheel stall / no-load multipliers (0.08 = +-8 %)
    module_motor: float = 0.0    # half-width of per-module stall / no-load multipliers
    vbatt_module: float = 0.0    # half-width (V) of per-module battery offsets around the trial's v_batt
    mu_module: float = 0.0       # half-width of per-module friction factors
    load_split: float = 0.0      # sd of the null-space normal-load component, relative to the mean wheel load
    gyro_bias: float = 0.0       # sd of a constant gyro bias (rad/s)
    acc_bias: float = 0.0        # sd of constant accelerometer biases (m/s^2)
    drop_p: float = 0.0          # per-sample wheel-stream dropout probability (value held)
    payload: tuple | None = None  # (lo, hi) kg payload point mass at a random deck position, or None


HET_OFF = Het()
HET_WHEEL8 = Het(wheel_motor=0.08)
HET_FULL = Het(wheel_motor=0.08, module_motor=0.04, vbatt_module=0.3, mu_module=0.1, load_split=0.15,
               gyro_bias=0.003, acc_bias=0.03, drop_p=0.01)


def sample_latents(asm, B, rng, het=HET_OFF, dr=None, faults=None, device="cuda", domain=None):
    """Episode latents for B episodes on `asm`. dr: capxfer DR dict (sim.sample_domain); or pass `domain` directly.
    faults: (mode (B,n), g (B,n)) or None for healthy. Returns a dict of tensors on `device`."""
    n, m = asm.n, len(asm.modules)
    dom = domain if domain is not None else sim.sample_domain(B, rng, dr, device=device)
    f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    u = lambda h, *s: rng.uniform(-h, h, size=s) if h > 0 else np.zeros(s)
    mod_stall, mod_noload = 1 + u(het.module_motor, B, m), 1 + u(het.module_motor, B, m)
    wm = np.asarray(asm.wheel_module)
    lat = dict(
        dom=dom,
        mode=torch.zeros(B, n, dtype=torch.long, device=device), g=torch.ones(B, n, device=device),
        stall_w=f32((1 + u(het.wheel_motor, B, n)) * mod_stall[:, wm]),
        noload_w=f32((1 + u(het.wheel_motor, B, n)) * mod_noload[:, wm]),
        vbatt_off=f32(u(het.vbatt_module, B, m)), mu_mod=f32(1 + u(het.mu_module, B, m)),
        yaw_err=f32(rng.normal(0, float(dom["wheel_yaw_err_std"]), size=(B, n))),
        load_z=f32(rng.normal(size=(B, n)) * het.load_split),
        gyro_bias=f32(rng.normal(0, 1, B) * het.gyro_bias), acc_bias=f32(rng.normal(0, 1, (B, 2)) * het.acc_bias),
        drop_p=float(het.drop_p), payload=None)
    if faults is not None:
        lat["mode"] = torch.as_tensor(faults[0], device=device).long()
        lat["g"] = torch.as_tensor(faults[1], device=device).float()
    if het.payload is not None:
        lo, hi = het.payload
        ext = np.abs(asm.wheel_pos).max(0) * 0.8
        lat["payload"] = dict(mass=f32(rng.uniform(lo, hi, B)), x=f32(rng.uniform(-ext[0], ext[0], B)),
                              y=f32(rng.uniform(-ext[1], ext[1], B)))
    return lat


def _index(lat, idx):
    out = {}
    for k, v in lat.items():
        if k == "dom":
            out[k] = {kk: (vv[idx] if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        elif k == "payload":
            out[k] = None if v is None else {kk: vv[idx] for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def repeat(lat, r):
    """Each episode's latents repeated r times consecutively (B -> B*r rows)."""
    B = lat["mode"].shape[0]
    return _index(lat, torch.arange(B, device=lat["mode"].device).repeat_interleave(r))


def subset(lat, modules, asm_child=None):
    """Latents of a child made of the given parent module indices (in the child's module order). Assembly-level
    quantities (normal-load split, payload) do not transfer: the split is redrawn as zero and payload dropped."""
    mods = torch.as_tensor(list(modules), device=lat["mode"].device)
    widx = (mods[:, None] * 4 + torch.arange(4, device=mods.device)).reshape(-1)
    out = dict(lat)
    for k in ("mode", "g", "stall_w", "noload_w", "yaw_err"):
        out[k] = lat[k][:, widx]
    for k in ("vbatt_off", "mu_mod"):
        out[k] = lat[k][:, mods]
    out["load_z"] = torch.zeros_like(out["mode"], dtype=torch.float32)
    out["payload"] = None
    return out


# ----------------------------------------------------------------------------- C edits
def _apply_latents(C, asm, lat, nom):
    """Edit capxfer's prepared constants C in place for the latents (see module docstring)."""
    B, n = C["N"].shape
    dev = C["N"].device
    wm = torch.as_tensor(np.asarray(asm.wheel_module), device=dev)
    C["tau_s"] = (C["tau_s"] * lat["stall_w"]).contiguous()
    C["omega0"] = (C["omega0"] * lat["noload_w"]).contiguous()
    vb = (lat["dom"]["v_batt"][:, None] + lat["vbatt_off"][:, wm]) / nom["v0"]
    C["vb"] = vb.contiguous()
    C["mu"] = (C["mu"] * lat["mu_mod"][:, wm]).contiguous()
    # mounting yaw error (capxfer._prepare geometry, rotated by the latent-owned per-wheel error)
    yerr = lat.get("yaw_err")
    if yerr is not None and bool(yerr.abs().max() > 0):
        f32 = dict(dtype=torch.float32, device=dev)

        def rot(v):
            v = torch.tensor(v, **f32).expand(B, n, 2)
            c, s = torch.cos(yerr), torch.sin(yerr)
            return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)
        w_hat, r_hat = rot(asm.w_hat), rot(asm.r_hat)
        f_hat = torch.stack([-r_hat[..., 1], r_hat[..., 0]], -1)
        n_hat = torch.stack([-w_hat[..., 1], w_hat[..., 0]], -1)
        roller = C["fric2"]
        d1 = torch.where(roller[..., None], w_hat, r_hat)
        d2 = torch.where(roller[..., None], n_hat, f_hat)
        radius = C["radius"]
        pos = C["pos"]
        lever = lambda d: torch.stack([d[..., 0], d[..., 1], pos[..., 0] * d[..., 1] - pos[..., 1] * d[..., 0]], -1)
        C.update(d1=d1, d2=d2, c1=radius * (w_hat * d1).sum(-1), c2=radius * (w_hat * d2).sum(-1),
                 a1=lever(d1), a2=lever(d2))
    # normal loads: least-norm (capxfer) + random null-space component
    z = lat["load_z"]
    if z.abs().max() > 0 and n > 3:
        pos = C["pos"]
        A = torch.cat([torch.ones(B, 1, n, device=dev), pos.transpose(1, 2)], 1)
        Pn = torch.eye(n, device=dev) - A.transpose(1, 2) @ torch.linalg.solve(A @ A.transpose(1, 2), A)
        N = C["N"]
        tot = N.sum(1, keepdim=True)
        N = N + (Pn @ z[..., None])[..., 0] * tot / n
        N = torch.clamp(N, min=0.02 * sim.G_EARTH)
        C["N"] = (N * tot / N.sum(1, keepdim=True)).contiguous()
    return C


# ----------------------------------------------------------------------------- per-wheel command step
def _pw_lookup(table, k0, base):
    """table (B, K, n) per command tick (NaN = keep base); k0 (B,) tick index."""
    row = torch.gather(table, 1, k0[:, None, None].expand(-1, 1, table.shape[2]))[:, 0]
    return torch.where(torch.isnan(row), base, row)


def _build_pw_step():
    src = inspect.getsource(sim._step)
    a1 = '    om_ref = (vrx * C["r_nom"][..., 0] + vry * C["r_nom"][..., 1]) / C["denom"]\n'
    a2 = '    volt = u * C["gw"] * vb\n'
    if a1 not in src or a2 not in src or "k0 = " not in src:
        raise RuntimeError("capxfer sim._step changed: per-wheel command patch anchors not found")
    src = src.replace(a1, a1 + '    if "pw_ref" in C:\n        om_ref = _pw_lookup(C["pw_ref"], k0, om_ref)\n')
    src = src.replace(a2, '    if "pw_duty" in C:\n'
                          '        u = _pw_lookup(C["pw_duty"], k0, u)\n'
                          '        integ = torch.where(torch.isnan(torch.gather(C["pw_duty"], 1, k0[:, None, None]'
                          '.expand(-1, 1, u.shape[1]))[:, 0]), integ, torch.zeros_like(integ))\n' + a2)
    src = src.replace("def _step(", "def _step_pw(", 1)
    ns = dict(vars(sim))
    ns["_pw_lookup"] = _pw_lookup
    exec(compile(src, f"<cfcap patched {sim.__file__}:_step>", "exec"), ns)
    return ns["_step_pw"]


_step_pw = _build_pw_step()
_COMPILED = {}


@contextlib.contextmanager
def _use_step(fn, compile_):
    """Temporarily point capxfer's integrator at `fn` (sim._get_step is a module-level lookup)."""
    orig = sim._get_step
    if compile_:
        import torch._dynamo
        cfg = torch._dynamo.config                   # one specialisation per (B, n, table) shape; allow many
        for k in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit", "accumulated_cache_size_limit"):
            if hasattr(cfg, k):
                setattr(cfg, k, max(getattr(cfg, k), 1024))
        key = fn.__name__
        if key not in _COMPILED:
            _COMPILED[key] = torch.compile(fn, dynamic=False, fullgraph=True)
        fn_c = _COMPILED[key]
    else:
        fn_c = fn
    sim._get_step = lambda c: fn_c
    try:
        yield
    finally:
        sim._get_step = orig


# ----------------------------------------------------------------------------- command tables
class TickSchedule:
    """Duck-typed probes.Schedule of a given length; the command itself comes from per-trial tick tables."""

    def __init__(self, T):
        self.segments = []
        self.T = float(T)

    def twist(self, t):
        return np.zeros(np.shape(t) + (3,))

    def windows(self):
        return []


def n_ticks(T, params):
    return int(math.ceil(T * float(params["firmware"]["cmd_rate_hz"]) - 1e-6))


def schedule_ticks(sched, params, T=None):
    """Command tick table (K+1, 3) of a probes.Schedule exactly as capxfer._prepare builds it."""
    T = sim.probe_duration(sched) if T is None else T
    hz = float(params["firmware"]["cmd_rate_hz"])
    K = n_ticks(T, params)
    tab = sched.twist(np.arange(K + 1) / hz)
    tab[K:] = 0.0
    return tab


def v2_probe(asm, params, B, rng=None, jitter=True, T=None):
    """v2 probe routine (capxfer selfmodel.gen_routine: P1/P2/P5 legs, amplitude U(0.85,1.1), lead U(0,1) s, 1 s tail).
    Returns (table (B, K+1, 3) float32, T). Without jitter: the plain routine (12.5 s)."""
    sched = probes.Schedule(asm.char_length(), pids=("P1", "P2", "P5"))
    hz = float(params["firmware"]["cmd_rate_hz"])
    if not jitter:
        T = T or sim.probe_duration(sched)
        tab = schedule_ticks(sched, params, T)
        return np.broadcast_to(tab, (B,) + tab.shape).astype(np.float32), T
    T = T or (sched.T + 2.0)
    K = n_ticks(T, params)
    amp = rng.uniform(0.85, 1.1, B)
    lead = rng.uniform(0.0, 1.0, B)
    tt = np.arange(K + 1)[None, :] * (1.0 / hz) - lead[:, None]                       # gen_routine: 0.02 s grid
    tab = sched.twist(np.clip(tt, 0, None).reshape(-1)).reshape(B, K + 1, 3) * amp[:, None, None]
    end = (sched.T + lead + 1.0)                                                       # trial length per episode
    tab[(tt < 0) | (np.arange(K + 1)[None, :] / hz >= end[:, None])] = 0.0
    tab[:, K:] = 0.0
    return tab.astype(np.float32), T


# ----------------------------------------------------------------------------- simulation
def _prepare_one(asm, params, lat, cmd, T, dt, device, seed, pw_ref=None, pw_duty=None):
    B = lat["mode"].shape[0]
    dom = dict(lat["dom"])
    if lat.get("yaw_err") is not None:
        dom["wheel_yaw_err_std"] = 0.0                  # yaw error is owned by the latents (None: capxfer's own draw)
    C, steps, nom, mode, g = sim._prepare(asm, params, dom, lat["mode"], lat["g"], TickSchedule(T), dt,
                                          torch.device(device), lat["payload"], None, seed)
    _apply_latents(C, asm, lat, nom)
    cmd = torch.as_tensor(np.asarray(cmd), dtype=torch.float32, device=device)
    K1 = C["tw_table"].shape[0]
    if cmd.dim() == 2:
        cmd = cmd.expand(B, *cmd.shape)
    assert cmd.shape[1] == K1, (cmd.shape, K1)
    C["tw_table"] = cmd.contiguous()
    for key, tab in (("pw_ref", pw_ref), ("pw_duty", pw_duty)):
        if tab is not None:
            t = torch.as_tensor(np.asarray(tab), dtype=torch.float32, device=device)
            C[key] = (t.expand(B, *t.shape) if t.dim() == 2 else t).contiguous()
    for k in ("pos_nom", "r_nom", "denom"):
        C[k] = C[k].expand(B, *C[k].shape)
    return C, steps, nom


_SCALAR_KEYS = ("steps_f", "dt", "tick_per_step", "n_tick", "ctrl_every", "t_ctrl", "cnt_per_rad")


@torch.no_grad()
def simulate_bucket(jobs, params, T, dt=0.005, record_hz=50, device="cuda", engine="auto", seed=0,
                    store_device=None, block_records=50):
    """Equal-wheel-count jobs in one batched run. jobs: list of dict(asm, lat, cmd[, pw_ref, pw_duty]) where cmd is a
    (K+1,3) or (b,K+1,3) tick table for duration T. Returns one result dict per job (capxfer simulate format)."""
    n = jobs[0]["asm"].n
    assert all(j["asm"].n == n for j in jobs), "bucket needs equal wheel counts"
    device = torch.device(device)
    if engine == "auto":
        engine = "graph" if device.type == "cuda" else "eager"
    pw = any(j.get("pw_ref") is not None or j.get("pw_duty") is not None for j in jobs)
    Cs, sizes, noms = [], [], []
    steps = None
    for i, j in enumerate(jobs):
        C, st, nom = _prepare_one(j["asm"], params, j["lat"], j["cmd"], T, dt, device, seed + 7919 * i,
                                  j.get("pw_ref"), j.get("pw_duty"))
        if pw:                                         # every job of a per-wheel bucket carries both tables
            b = j["lat"]["mode"].shape[0]
            K1 = C["tw_table"].shape[1]
            for key in ("pw_ref", "pw_duty"):
                if key not in C:
                    C[key] = torch.full((b, K1, n), float("nan"), device=device)
        assert steps is None or st == steps
        steps = st
        Cs.append(C)
        sizes.append(j["lat"]["mode"].shape[0])
        noms.append(nom)
    if len(Cs) == 1:
        C = Cs[0]
    else:
        C = {}
        for k in Cs[0]:
            v = Cs[0][k]
            if torch.is_tensor(v) and v.dim() >= 1 and k not in _SCALAR_KEYS:
                C[k] = torch.cat([c[k] for c in Cs], 0).contiguous()
            else:
                C[k] = v
    B = sum(sizes)
    with _use_step(_step_pw if pw else sim._step, engine == "compile"):
        R, n_rec, rec_every = sim._integrate(C, steps, B, n, dt, record_hz, device, engine, block_records,
                                             store_device, torch.float32, None)
    sd = R["wheel_angle"].device
    t = torch.arange(n_rec, dtype=torch.float32, device=sd) * rec_every * dt
    out, o = [], 0
    for i, b in enumerate(sizes):
        Ri = {k: R[k][o:o + b] for k in sim.REC_KEYS}
        Ri["t"] = t
        Ri["mode"], Ri["g"] = jobs[i]["lat"]["mode"].to(sd), jobs[i]["lat"]["g"].to(sd)
        asm = jobs[i]["asm"]
        Ri["meta"] = dict(n=n, B=b, record_hz=record_hz, cpr=noms[i]["cpr"], assembly=asm, asm=asm.name,
                          char_length=asm.char_length(), dt=dt, engine=engine, seed=seed + 7919 * i)
        out.append(Ri)
        o += b
    return out


def simulate_mixed(jobs, params, T, max_wheels=1 << 17, **kw):
    """Jobs of mixed assembly sizes: bucketed on wheel count, each bucket split into runs of <= max_wheels B*n.
    Jobs larger than a run are split by rows and re-joined. Returns results aligned with jobs."""
    res = [None] * len(jobs)
    pieces = []                                        # (job index, row slice)
    for i, j in enumerate(jobs):
        b, n = j["lat"]["mode"].shape
        step = max(1, max_wheels // n)
        for r0 in range(0, b, step):
            pieces.append((i, slice(r0, min(b, r0 + step))))
    by_n = {}
    for p in pieces:
        by_n.setdefault(jobs[p[0]]["asm"].n, []).append(p)
    parts = {i: [] for i in range(len(jobs))}
    for n, ps in sorted(by_n.items()):
        run, w = [], 0
        for p in ps + [None]:
            rows = 0 if p is None else p[1].stop - p[1].start
            if p is None or (run and w + rows * n > max_wheels):
                sub = []
                for i, sl in run:
                    j = jobs[i]
                    idx = torch.arange(sl.start, sl.stop, device=j["lat"]["mode"].device)
                    sj = dict(j, lat=_index(j["lat"], idx))
                    for key in ("cmd", "pw_ref", "pw_duty"):
                        v = j.get(key)
                        if v is not None and np.ndim(v) == 3:
                            sj[key] = v[sl]
                    sub.append(sj)
                outs = simulate_bucket(sub, params, T, **kw)
                for (i, sl), o in zip(run, outs):
                    parts[i].append(o)
                run, w = [], 0
            if p is not None:
                run.append(p)
                w += rows * n
    for i, ps in parts.items():
        if len(ps) == 1:
            res[i] = ps[0]
        else:
            r = {k: torch.cat([p[k] for p in ps], 0) for k in sim.REC_KEYS + ("mode", "g")}
            r["t"], r["meta"] = ps[0]["t"], dict(ps[0]["meta"], B=sum(p["meta"]["B"] for p in ps))
            res[i] = r
    return res


# ----------------------------------------------------------------------------- robot-visible observations
@torch.no_grad()
def observe(R, lat, seed=0, cpr=P.CPR):
    """Robot-visible channels on the 50 Hz grid (R recorded at 50 Hz), as capxfer.corpus.replay.simulate_trial plus
    selfmodel's IMU noise: pre_duty (B,T,n), speed (B,T,n) from quantised count differences, gz/ax/ay (B,T); plus the
    latents' gyro/acc bias and wheel-sample dropout (a dropped sample repeats the previous one)."""
    assert abs(float(R["t"][1] - R["t"][0]) - DT_OBS) < 1e-6, "record at 50 Hz"
    dev = R["duty"].device
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    th = R["wheel_angle"].double()
    cnt = torch.floor(th / (2 * math.pi) * cpr)
    spd = torch.zeros_like(th)
    spd[:, 1:] = torch.diff(cnt, dim=1) / cpr * 2 * math.pi / DT_OBS
    spd = spd.float()
    duty = R["duty"].float()
    B, T, n = duty.shape
    if lat.get("drop_p", 0.0) > 0:
        drop = torch.rand(B, T, n, generator=gen, device=dev) < lat["drop_p"]
        drop[:, 0] = False
        idx = torch.where(drop, torch.zeros(1, dtype=torch.long, device=dev),
                          torch.arange(T, device=dev)[None, :, None].expand(B, T, n))
        idx = torch.cummax(idx, 1).values
        duty, spd = torch.gather(duty, 1, idx), torch.gather(spd, 1, idx)
    rn = lambda: torch.randn(B, T, generator=gen, device=dev)
    gb = lat["gyro_bias"].to(dev)[:, None]
    ab = lat["acc_bias"].to(dev)
    return dict(pre_duty=duty, speed=spd,
                gz=R["twist"][:, :, 2] + IMU_NOISE[0] * rn() + gb,
                ax=R["accel"][:, :, 0] + IMU_NOISE[1] * rn() + ab[:, 0:1],
                ay=R["accel"][:, :, 1] + IMU_NOISE[2] * rn() + ab[:, 1:2])


def weak_or_healthy(B, n, rng, p_healthy=0.5, g_range=(0.3, 0.9)):
    """Faults: healthy or one weak wheel with g ~ U(g_range)."""
    mode = np.zeros((B, n), np.int64)
    g = np.ones((B, n), np.float32)
    for b in np.nonzero(rng.uniform(size=B) >= p_healthy)[0]:
        w = rng.integers(n)
        mode[b, w] = WEAK
        g[b, w] = rng.uniform(*g_range)
    return mode, g


def random_faults(B, n, rng, k_max=2, p_none=0.25, modes=(WEAK, COAST, BRAKE), g_range=(0.3, 0.9)):
    """0..k_max faulted wheels among weak(g)/coast/brake on any wheel."""
    mode = np.zeros((B, n), np.int64)
    g = np.ones((B, n), np.float32)
    for b in range(B):
        if rng.uniform() < p_none:
            continue
        k = int(rng.integers(1, k_max + 1))
        for w in rng.choice(n, size=min(k, n), replace=False):
            mode[b, w] = int(rng.choice(modes))
            if mode[b, w] == WEAK:
                g[b, w] = rng.uniform(*g_range)
    return mode, g
