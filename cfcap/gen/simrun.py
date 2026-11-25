"""Batched T1 rollouts for Stage B episodes: probe observations of every assembly and closed-loop labels of every
candidate.

`run_jobs` is `cfcap.t1.plant.simulate_bucket` with three changes for throughput and reproducibility only (same
physics, same capxfer `_prepare` / `_integrate`):
  * every job's constants are prepared on the CPU (capxfer `_prepare` is ~200 small tensor ops; 0.6 ms on the CPU vs
    2 ms on the GPU) and moved to the GPU once per run;
  * jobs are grouped by wheel count and packed into runs of <= max_wheels B*n, each padded to a fixed row count
    (full runs) or the next power of two (the last run), so torch.compile sees few shapes;
  * only the record channels a consumer needs come back to the host.
T1 is deterministic given the latents (the capxfer seed only draws mounting-yaw errors, which the latents own), and
observation noise is drawn per assembly from the episode's own RNG stream, so an episode does not depend on the shard
or batch it was generated in (up to float reassociation inside batched kernels).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from cfcap.gen import episode as E
from cfcap.gen import rho_ref as RR
from cfcap.t1 import capability as cap
from cfcap.t1 import params as P
from cfcap.t1 import plant
from capxfer import probes, sim

DT, REC_HZ = 0.005, 50
PROBE_KINDS = ("v2", "clamp", "tug")
TUG_V, TUG_HOLD = 0.40, 1.0


# ----------------------------------------------------------------------------- generic runner
_COMPILE_BROKEN = False


def _set_compile_broken():
    global _COMPILE_BROKEN
    _COMPILE_BROKEN = True


def _pad_rows(B, cap_rows, min_rows=16):
    if B >= cap_rows:
        return cap_rows
    return min(cap_rows, max(min_rows, 1 << int(math.ceil(math.log2(B)))))


@torch.no_grad()
def run_jobs(jobs, params, T, keys, engine="compile", max_wheels=1 << 17, device="cuda", stats=None, reduce=None):
    """jobs: dict(asm, lat (CPU tensors, b rows), cmd (b,K+1,3) | (K+1,3)[, pw_ref (b,K+1,n) | (K+1,n)]); all jobs of
    one call share T and either all or none carry pw_ref. Returns per job {key: CPU tensor (b, n_rec, c)}, or with
    reduce (a function of the run's {key: GPU tensor (B, n_rec, c)} -> {name: (B, ...)}) the reduced rows."""
    device = torch.device(device)
    out = [None] * len(jobs)
    by_n = {}
    for i, j in enumerate(jobs):
        by_n.setdefault(j["asm"].n, []).append(i)
    pw = any(j.get("pw_ref") is not None for j in jobs)
    for n, idx in sorted(by_n.items()):
        cap_rows = max(1, max_wheels // n)
        runs, cur, rows = [], [], 0
        for i in idx:
            b = jobs[i]["lat"]["mode"].shape[0]
            assert b <= cap_rows, (b, cap_rows)
            if cur and rows + b > cap_rows:
                runs.append(cur)
                cur, rows = [], 0
            cur.append(i)
            rows += b
        if cur:
            runs.append(cur)
        for run in runs:
            t0 = time.perf_counter()
            Cs, sizes, steps = [], [], None
            for i in run:
                j = jobs[i]
                C, st, _ = plant._prepare_one(j["asm"], params, j["lat"], j["cmd"], T, DT, "cpu", 0,
                                              j.get("pw_ref"), None)
                steps = st if steps is None else steps
                assert st == steps
                Cs.append(C)
                sizes.append(j["lat"]["mode"].shape[0])
            B = sum(sizes)
            Bp = _pad_rows(B, cap_rows)
            pad = torch.cat([torch.arange(B), torch.zeros(Bp - B, dtype=torch.long)])
            C = {}
            for k, v in Cs[0].items():
                if torch.is_tensor(v) and v.dim() >= 1 and k not in plant._SCALAR_KEYS:
                    C[k] = torch.cat([c[k] for c in Cs], 0)[pad].contiguous().to(device, non_blocking=True)
                else:
                    C[k] = v.to(device) if torch.is_tensor(v) else v
            t1 = time.perf_counter()
            eng = "graph" if (engine == "compile" and _COMPILE_BROKEN) else engine
            try:
                with plant._use_step(plant._step_pw if pw else sim._step, eng == "compile"):
                    R, n_rec, _ = sim._integrate(C, steps, Bp, n, DT, REC_HZ, device, eng, 50, None, torch.float32,
                                                 None)
            except Exception as ex:                    # no working inductor/triton toolchain on the box: CUDA graphs
                if eng != "compile":
                    raise
                print(f"[gen] torch.compile failed ({type(ex).__name__}: {ex}); falling back to engine=graph",
                      flush=True)
                _set_compile_broken()
                with plant._use_step(plant._step_pw if pw else sim._step, False):
                    R, n_rec, _ = sim._integrate(C, steps, Bp, n, DT, REC_HZ, device, "graph", 50, None,
                                                 torch.float32, None)
            Rh = {k: R[k][:B] for k in keys}
            Rh = {k: v.cpu() for k, v in (reduce(Rh) if reduce is not None else Rh).items()}
            t2 = time.perf_counter()
            if stats is not None:
                stats["prep_s"] = stats.get("prep_s", 0.0) + t1 - t0
                stats["sim_s"] = stats.get("sim_s", 0.0) + t2 - t1
                stats["rows"] = stats.get("rows", 0) + B
                stats["rows_padded"] = stats.get("rows_padded", 0) + Bp
                stats["wheel_seconds"] = stats.get("wheel_seconds", 0.0) + B * n * T
            o = 0
            for i, b in zip(run, sizes):
                out[i] = {k: v[o:o + b] for k, v in Rh.items()}
                o += b
            del R, C
    return out


# ----------------------------------------------------------------------------- probes
def _profile(K1, hz, legs, t_start):
    """(K1, L) ramp-hold-ramp weights of consecutive legs [(hold s)] starting at t_start, capxfer leg timing."""
    t = np.arange(K1) / hz
    W = np.zeros((K1, len(legs)))
    s = t_start
    for li, hold in enumerate(legs):
        up = np.clip((t - s) / probes.RAMP, 0, 1)
        h1 = s + probes.RAMP + hold
        down = np.clip((h1 + probes.RAMP - t) / probes.RAMP, 0, 1)
        m = (t >= s) & (t < h1 + probes.RAMP)
        W[m, li] = np.minimum(up, down)[m]
        s += probes.leg_duration(probes.Primitive("x", (0, 0, 0), hold))
    return W, s


def probe_T(kind):
    if kind == "v2":
        return probes.Schedule(1.0, pids=("P1", "P2", "P5")).T + 2.0
    if kind == "clamp":
        return probes.Schedule(1.0, pids=("P6x", "P6y")).T + 2.0
    if kind == "tug":
        return probes.SETTLE + 4 * probes.leg_duration(probes.Primitive("x", (0, 0, 0), TUG_HOLD)) + 1.2
    raise ValueError(kind)


def probe_tables(kind, asm, params, rng):
    """Command of one probe trial: (cmd (K+1,3), pw_ref (K+1,n) or None, info dict).
    v2 / clamp: capxfer selfmodel.gen_routine jitter (amplitude U(0.85,1.1), lead U(0,1) s) on the P1/P2/P5 resp.
    P6x/P6y schedule. tug: body twist 0; per module the diagonal pair (fl, rr) [then (fr, rl)] is commanded opposite
    wheel speeds +-0.40 m/s surface (cfcap.t0.probesets 'tug'), other wheels zero, 4 legs of 1.0 s hold."""
    hz = float(params["firmware"]["cmd_rate_hz"])
    T = probe_T(kind)
    K = plant.n_ticks(T, params)
    if kind in ("v2", "clamp"):
        pids = ("P1", "P2", "P5") if kind == "v2" else ("P6x", "P6y")
        sched = probes.Schedule(asm.char_length(), pids=pids)
        amp, lead = float(rng.uniform(0.85, 1.1)), float(rng.uniform(0.0, 1.0))
        tt = np.arange(K + 1) / hz - lead
        tab = sched.twist(np.clip(tt, 0, None)) * amp
        tab[(tt < 0) | (np.arange(K + 1) / hz >= sched.T + lead + 1.0)] = 0.0
        tab[K:] = 0.0
        return tab.astype(np.float32), None, dict(amp=amp, lead=lead)
    corner = np.array([c.split(":")[1] for c in asm.wheel_corner])
    wv = TUG_V / float(params["wheel"]["radius_m"])
    W, _ = _profile(K + 1, hz, [TUG_HOLD] * 4, probes.SETTLE)
    ref = np.zeros((K + 1, asm.n))
    li = 0
    for a, b in (("fl", "rr"), ("fr", "rl")):
        for s in (+1, -1):
            om = np.zeros(asm.n)
            om[corner == a], om[corner == b] = s * wv, -s * wv
            ref += W[:, li:li + 1] * om[None]
            li += 1
    ref[K:] = 0.0
    return np.zeros((K + 1, 3), np.float32), ref.astype(np.float32), dict(amp=1.0, lead=0.0)


def observe_np(R, lat, rng, cpr=P.CPR):
    """Robot-visible channels as cfcap.t1.plant.observe, noise from a numpy RNG: pre_duty (T,n) float, encoder count
    increments per 20 ms bin dcnt (T,n) int (speed = dcnt / cpr * 2 pi / 0.02), imu (T,3) = gz, ax, ay with the
    assembly's biases and capxfer selfmodel IMU noise; wheel-sample dropout holds the previous sample."""
    th = R["wheel_angle"][0].double().numpy()
    cnt = np.floor(th / (2 * math.pi) * cpr)
    dcnt = np.zeros_like(cnt)
    dcnt[1:] = np.diff(cnt, axis=0)
    duty = R["duty"][0].numpy()
    T, n = duty.shape
    p = float(lat["drop_p"])
    if p > 0:
        drop = rng.uniform(size=(T, n)) < p
        drop[0] = False
        idx = np.where(drop, 0, np.arange(T)[:, None])
        idx = np.maximum.accumulate(idx, axis=0)
        duty = np.take_along_axis(duty, idx, 0)
        dcnt = np.take_along_axis(dcnt, idx, 0)
    gb = float(lat["gyro_bias"][0])
    ab = lat["acc_bias"][0].numpy()
    imu = np.stack([R["twist"][0, :, 2].numpy() + plant.IMU_NOISE[0] * rng.normal(size=T) + gb,
                    R["accel"][0, :, 0].numpy() + plant.IMU_NOISE[1] * rng.normal(size=T) + ab[0],
                    R["accel"][0, :, 1].numpy() + plant.IMU_NOISE[2] * rng.normal(size=T) + ab[1]], -1)
    return duty, dcnt.astype(np.int32), imu


def probe_episodes(eps, kinds, engine="compile", max_wheels=1 << 15, stats=None):
    """Probe every assembly of every episode with each probe kind. Returns {kind: [per episode [per assembly
    dict(duty, dcnt, imu, cmd, ref, amp, lead)]]}."""
    out = {}
    for kind in kinds:
        T = probe_T(kind)
        jobs, meta = [], []
        for ei, ep in enumerate(eps):
            par = E.params(ep.plate)
            for a in ep.asms:
                rng = E.episode_rng(ep.plate, ep.split, ep.index, stream=1 + 16 * a.role + PROBE_KINDS.index(kind))
                cmd, ref, info = probe_tables(kind, a.asm, par, rng)
                pay = E.parent_payload(ep) if a.role == 0 else None
                lat = E.plant_latents(ep, [(a.role, k) for k in range(a.m)], a.load_z, pay,
                                      gyro=(a.gyro_bias, a.acc_bias))
                jobs.append(dict(asm=a.asm, lat=lat, cmd=cmd, pw_ref=ref, plate=ep.plate))
                meta.append((ei, a.role, rng, info, cmd, ref))
        res = _run_by_plate(jobs, T, ("wheel_angle", "duty", "twist", "accel"), engine, max_wheels, stats)
        per = [[None] * len(ep.asms) for ep in eps]
        for (ei, role, rng, info, cmd, ref), j, R in zip(meta, jobs, res):
            duty, dcnt, imu = observe_np(R, j["lat"], rng)
            nr = duty.shape[0]
            per[ei][role] = dict(duty=duty, dcnt=dcnt, imu=imu, cmd=cmd[:nr], ref=None if ref is None else ref[:nr],
                                 **info)
        out[kind] = per
    return out


def _run_by_plate(jobs, T, keys, engine, max_wheels, stats, reduce=None):
    res = [None] * len(jobs)
    for plate in E.PLATES:
        idx = [i for i, j in enumerate(jobs) if j["plate"] == plate]
        if idx:
            r = run_jobs([jobs[i] for i in idx], E.params(plate), T, keys, engine, max_wheels, stats=stats,
                         reduce=reduce)
            for i, ri in zip(idx, r):
                res[i] = ri
    return res


# ----------------------------------------------------------------------------- labels
def leg_window_mask(T=cap.T_LEG, rec_hz=REC_HZ):
    n_rec = int(round(T / DT)) // int(round(1.0 / (rec_hz * DT)))
    t = torch.arange(n_rec, dtype=torch.float32) * (1.0 / rec_hz)
    a, b = cap.window(T)
    return (t >= a) & (t <= b)


def window_mean(R):
    m = leg_window_mask().to(R["twist"].device)
    return dict(xr=R["twist"][:, m].float().mean(1))


def rho_rows(xr, tw, Lc):
    """xr (B,3) window-mean body twist of leg rows (`window_mean`), tw (B,3) commanded hold twists, Lc (B,) -> rho,
    drift (B,) exactly as capxfer sim.targets_from_twist over the cfcap.t1.capability leg window."""
    w = torch.stack([torch.ones_like(Lc), torch.ones_like(Lc), Lc * Lc], -1)
    nc = (w * tw * tw).sum(-1)
    r = (w * xr * tw).sum(-1) / nc
    res = xr - r[:, None] * tw
    return r, torch.sqrt((w * res * res).sum(-1) / nc)


def label_episodes(eps, engine="compile", max_wheels=1 << 17, skip_healthy_twin=True, stats=None):
    """Closed-loop rho on the 10 decision legs for every candidate, with its paired fault-free rollout (D17 relative
    capability). A candidate without a faulted wheel has rho_healthy = rho exactly, so its twin is skipped unless
    skip_healthy_twin=False. Decision label rho_dec = rho / rho_ref(shape, leg), the nominal healthy reference
    (cfcap.gen.rho_ref; keys not yet cached are simulated in the same runs). Returns per episode a dict of (n_cand, L)
    float32 arrays rho, rho_healthy, rho_rel, rho_ref, rho_dec, drift."""
    L = len(cap.LEGS)
    jobs, meta = [], []
    for ei, ep in enumerate(eps):
        par = E.params(ep.plate)
        for ci, c in enumerate(ep.cands):
            lat = E.plant_latents(ep, c.modules, c.load_z, c.payload, B=L)
            faulted = bool((lat["mode"][0] != 0).any())
            twin = faulted or not skip_healthy_twin
            if twin:
                lat = E.cat_latents([lat, E.plant_latents(ep, c.modules, c.load_z, c.payload, B=L, healthy=True)])
            tabs, tw = cap.leg_tables(c.asm, par)
            reps = 2 if twin else 1
            jobs.append(dict(asm=c.asm, lat=lat, cmd=np.tile(tabs, (reps, 1, 1)), plate=ep.plate))
            meta.append((ei, ci, twin, np.tile(tw, (reps, 1)), c.asm.char_length()))
    n_cand_jobs = len(jobs)
    ref_keys = []
    for plate in E.PLATES:
        asms = [c.asm for ep in eps if ep.plate == plate for c in ep.cands]
        rj, rk, rtw, rlc = RR.jobs_for(plate, asms, E.params(plate))
        jobs += rj
        ref_keys += [(plate, k) for k in rk]
        meta += [(None, None, False, tw, lc) for tw, lc in zip(rtw, rlc)]
    res = _run_by_plate(jobs, cap.T_LEG, ("twist",), engine, max_wheels, stats, reduce=window_mean)
    out = [dict(rho=np.zeros((len(ep.cands), L), np.float32), rho_healthy=np.zeros((len(ep.cands), L), np.float32),
                drift=np.zeros((len(ep.cands), L), np.float32)) for ep in eps]
    if res:
        tw_all = torch.as_tensor(np.concatenate([m[3] for m in meta]), dtype=torch.float32)
        Lc_all = torch.as_tensor(np.concatenate([np.full(len(m[3]), m[4]) for m in meta]), dtype=torch.float32)
        rho, drift = rho_rows(torch.cat([r["xr"] for r in res]), tw_all, Lc_all)
        rho, drift = rho.numpy(), drift.numpy()
        o = 0
        for ji, (ei, ci, twin, tw, _) in enumerate(meta):
            if ji >= n_cand_jobs:
                RR.put(*ref_keys[ji - n_cand_jobs], rho[o:o + L])
                o += len(tw)
                continue
            out[ei]["rho"][ci] = rho[o:o + L]
            out[ei]["drift"][ci] = drift[o:o + L]
            out[ei]["rho_healthy"][ci] = rho[o + L:o + 2 * L] if twin else rho[o:o + L]
            o += len(tw)
    for ep, d in zip(eps, out):
        d["rho_rel"] = d["rho"] / d["rho_healthy"]
        d["rho_ref"] = np.stack([RR.get(ep.plate, RR.key(c.asm)) for c in ep.cands]).astype(np.float32) \
            if ep.cands else np.zeros((0, L), np.float32)
        d["rho_dec"] = d["rho"] / d["rho_ref"]
    return out
