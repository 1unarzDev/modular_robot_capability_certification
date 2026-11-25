"""T2 kernel: crane's modular mecanum kernel (read-only import) driven by the robot firmware, plus three plant effects.

The physics of one plant step is crane `engine/kernels/mecanum.py` (the port of the modular truth model, capplan
`truth/model.py`): tanh roller-axis slip on a slip RATIO, Coulomb roller-bearing drag along the free direction,
friction-circle cap, massless wheels with a bracketed root for each wheel speed, a finite-difference twist Jacobian
and a linearly-implicit Euler step, exact SE(2) pose update. `step_cmd` is `mecanum.step` with the per-wheel speed
command passed in (mecanum.step computes it from the body twist by its ideal velocity-loop firmware); it calls the
kernel's own `_wheels/_state/_contact/_bias/_solve3` and reproduces `mecanum.step` exactly when given
`mecanum.firmware(w, u)` (tests/test_t2.py).

Drive (`firmware='robot'`, the measured firmware, capxfer params): per-wheel PI on ideal-IK references at 50 Hz,
duty = clamp(kff w_ref + kp e + ki int e, -1, 1) with conditional-integration anti-windup, the wheel speed measured
from encoder-count differences, duty held between loop ticks (ZOH), command ticks at 50 Hz behind a continuous
actuation latency. Duty drives crane's motor as a DC motor: with the kernel's velocity-loop gain column K set to
tau_stall / w_nl and the speed command cmd = duty g v_b w_nl, crane's clip(K (cmd - w), +-lim) is
tau_stall (g v_b duty - w / w_nl), i.e. the linear torque-speed curve (lim never binds while |duty v_b| <= 1 except
in regenerative braking beyond tau_stall). `firmware='crane'` is mecanum_run.replay's drive (IK + velocity loop,
integer delay in plant steps): used by the reproduction test.

Added effects (per env arrays, each with its own on/off flag; built by cfcap.t2.build):
  load transfer   N = N0 + dN/dcom . (-(h/g) a_com): the inertial force at COM height h shifts the support, using
                  the previous step's acceleration (capplan truth refreshes loads from the last acceleration too);
                  clamped at 0.02 g per wheel, total weight kept.
  compliance      per-module yaw deflection of the wheel directions (capplan truth item 4): planar spring network
                  of the latched faces (k_axial, k_lat, k_yaw) under each module's wheel forces + inertial loads,
                  deflection relative to the mass-weighted mean yaw, applied to the next step (one-step lag); hook
                  play adds +-theta_play/2 per module in the direction of its elastic yaw (smoothed sign,
                  tanh(rot / 2e-4 rad)). The firmware keeps the nominal geometry.
  anisotropy      roller-bearing drag coefficient direction-dependent: mu_f = mu_f+ (v_f > 0) or mu_f- (v_f < 0),
                  blended by tanh(v_f_prev / V_EPS) of the previous step's free-direction slip.
"""
from __future__ import annotations

import math

import numpy as np

from cfcap.t2 import deps  # noqa: F401  (crane + numba/jax paths)
from engine.jit import numba
from engine.kernels import mecanum as K

G0 = 9.81
NSUB = 4                         # plant steps per 50 Hz tick at dt = 5 ms
DT = 0.005
ROT_SMOOTH = 2e-4                # rad: hook-play sign smoothing
MOTOR_TAU_TH = 30.0              # s: thermal time constant of the misspecified motor (build.MOTOR)
IWc = K.IW
I_EX, I_EY, I_TX, I_TY, I_MUT, I_MUF, I_N = (IWc[k] for k in ("ex", "ey", "tx", "ty", "mu_t", "mu_f", "N"))
I_IMX, I_IMY = K.IB['imu_x'], K.IB['imu_y']


# ----------------------------------------------------------------------------------------------- array form
def step_cmd(x, cmd, p, dt, xp=np):
    """mecanum.step with the per-wheel speed command cmd [E, W] given. Returns (x_new, om [E, W])."""
    w = K._wheels(p, xp)
    b = {k: p[:, i] for k, i in K.IB.items()}
    E, W = x.shape[0], (p.shape[-1] - K.G) // K.Q
    t = (x[:, 9] + K.ONSET_EPS) * dt
    f = K._state(w, t, xp)
    xi = (x[:, 3], x[:, 4], x[:, 5])
    pert = [xi, (xi[0] + K.H_FD, xi[1] + 0.0, xi[2] + 0.0), (xi[0] + 0.0, xi[1] + K.H_FD, xi[2] + 0.0),
            (xi[0] + 0.0, xi[1] + 0.0, xi[2] + K.H_FD)]
    Wr, om = [], None
    for q in pert:
        wr, o = K._contact(w, f, *q, cmd, xp)
        bs = K._bias(b, *q)
        Wr.append(tuple(a - c for a, c in zip(wr, bs)))
        om = o if om is None else om
    J = [[(Wr[i + 1][r] - Wr[0][r]) / K.H_FD for i in range(3)] for r in range(3)]
    m, c0, c1, I = b['mass'], b['xc'], b['yc'], b['inertia']
    zero = 0.0 * m
    M = [[m, zero, -m * c1], [zero, m, m * c0], [-m * c1, m * c0, I]]
    A = [[M[r][c] - dt * J[r][c] for c in range(3)] for r in range(3)]
    d = K._solve3(A, [dt * Wr[0][r] for r in range(3)], xp)
    v = [xi[i] + d[i] for i in range(3)]
    acc = [d[i] / dt for i in range(3)]
    vx, vy, wz = v[0] * dt, v[1] * dt, v[2] * dt
    small = xp.abs(wz) < 1e-9
    ws = xp.where(small, 1.0, wz)
    s, c = xp.sin(ws), xp.cos(ws)
    dx = xp.where(small, vx, (s * vx - (1 - c) * vy) / ws)
    dy = xp.where(small, vy, ((1 - c) * vx + s * vy) / ws)
    cs, sn = xp.cos(x[:, 2]), xp.sin(x[:, 2])
    pose = [x[:, 0] + cs * dx - sn * dy, x[:, 1] + sn * dx + cs * dy, x[:, 2] + wz]
    tcmd = w['K'] * (cmd - om)
    lim = w['tau_stall'] * xp.where(tcmd * om > 0, xp.minimum(xp.maximum(1.0 - xp.abs(om) / w['w_nl'], 0.0), 1.0), 1.0)
    tm = xp.minimum(xp.maximum(tcmd, -lim), lim)
    duty = xp.minimum(xp.maximum(tm / w['tau_stall'] + om / w['w_nl'], -1.0), 1.0)
    duty = xp.where((w['kind'] != K.BALL) & (w['active'] > 0), duty, 0.0)
    wx = x[:, K.S0:].reshape(E, W, K.PER)
    ang = wx[..., 0] + om * dt
    wheels = xp.stack([ang, om, duty], axis=-1).reshape(E, W * K.PER)
    body = xp.stack(pose + v + acc + [x[:, 9] + 1.0], axis=-1)
    return xp.concatenate([body, wheels], axis=-1), om


def _rot2(cx, cy, r, xp):
    c, s = xp.cos(r), xp.sin(r)
    return c * cx - s * cy, s * cx + c * cy


def _effective_p(p, S, st, xp):
    """p with the effects applied for this step: normal loads (load transfer), wheel directions (compliance),
    roller drag (anisotropy). st: the T2 state dict (rot, vf, x)."""
    E = p.shape[0]
    W = (p.shape[-1] - K.G) // K.Q
    wp = p[:, K.G:].reshape(E, W, K.Q)
    cols = {}
    x = st["x"]
    if S["lt_on"]:
        vx, vy, wz, dvx, dvy, dwz = (x[:, i] for i in range(3, 9))
        cx, cy = p[:, K.IB['xc']], p[:, K.IB['yc']]
        ax = dvx - wz * vy - dwz * cy - wz * wz * cx
        ay = dvy + wz * vx + dwz * cx - wz * wz * cy
        sc = -S["h"] / G0
        N = S["N0"] + S["GN"][..., 0] * (sc * ax)[:, None] + S["GN"][..., 1] * (sc * ay)[:, None]
        act = wp[..., IWc['active']] > 0
        N = xp.where(act, xp.maximum(N, 0.02 * G0), 0.0)
        N = N * (S["N0"].sum(1) / N.sum(1))[:, None]
        cols[I_N] = N
    if S["cp_on"]:
        r = st["rot"]
        ex, ey = _rot2(S["ex0"], S["ey0"], r, xp)
        tx, ty = _rot2(S["tx0"], S["ty0"], r, xp)
        cols.update({I_EX: ex, I_EY: ey, I_TX: tx, I_TY: ty})
    if S["an_on"]:
        s = K.tanh(st["vf"] / K.V_EPS, xp)
        cols[I_MUF] = 0.5 * (S["mufp"] + S["mufn"]) + 0.5 * (S["mufp"] - S["mufn"]) * s
    if not cols:
        return p
    parts = [cols.get(q, wp[..., q]) for q in range(K.Q)]
    return xp.concatenate([p[:, :K.G], xp.stack(parts, -1).reshape(E, W * K.Q)], axis=-1)


def _wheel_forces(pe, x, om, xp):
    """Per-wheel traction (Fx, Fy) [E, W] and free-direction slip v_f at the pre-step twist."""
    w = K._wheels(pe, xp)
    px, py = w['px'], w['py']
    vcx = x[:, 3:4] + x[:, 5:6] * (-py)
    vcy = x[:, 4:5] + x[:, 5:6] * px
    Fx, Fy, _ = K._traction(w, vcx, vcy, om, xp)
    live = (w['N'] > 0) & (w['active'] > 0) & (w['kind'] != K.BALL)
    rw = w['r'] * om
    vf = -w['ty'] * (vcx - rw * w['ex']) + w['tx'] * (vcy - rw * w['ey'])
    return xp.where(live, Fx, 0.0), xp.where(live, Fy, 0.0), vf


def _firmware(S, st, u_tw, xp):
    """Robot firmware at a 50 Hz loop tick: new duty (and integrator) from the command twist and encoder counts."""
    x = st["x"]
    E = x.shape[0]
    W = S["px0"].shape[1]
    vcx = u_tw[:, 0:1] - u_tw[:, 2:3] * S["py0"]
    vcy = u_tw[:, 1:2] + u_tw[:, 2:3] * S["px0"]
    te = S["ftx"] * S["fex"] + S["fty"] * S["fey"]
    wref = (S["ftx"] * vcx + S["fty"] * vcy) / (S["r_nom"][:, None] * xp.where(xp.abs(te) > 1e-6, te, 1.0))
    ang = x[:, K.S0:].reshape(E, W, K.PER)[..., 0]
    k = (S["cpr"] / (2 * math.pi))[:, None]
    meas = (xp.floor(ang * k) - xp.floor(st["thc"] * k)) / k / (NSUB * DT)
    e = wref - meas
    raw = S["kff"][:, None] * wref + S["kp"][:, None] * e + S["ki"][:, None] * st["integ"]
    unsat = (xp.abs(raw) < 1) | (xp.sign(e) != xp.sign(raw))
    integ = xp.where(unsat & (S["driven"] > 0), st["integ"] + e * (NSUB * DT), st["integ"])
    u = xp.minimum(xp.maximum(raw, -1.0), 1.0)
    return u, integ, ang


def _command(S, tables, which, k, xp):
    """Body-twist command of plant step k [E, 3]: robot = 50 Hz tick behind the continuous latency (capxfer
    convention); crane = mecanum_run's tick floor((k - delay) / nsub)."""
    nt = tables.shape[1]
    if S["fw_robot"]:
        tk = k * (1.0 / NSUB) - S["lat_ticks"]
        ok = tk >= -1e-4
        k0 = xp.clip(xp.floor(tk + 1e-4), 0, nt - 1).astype(np.int32 if xp is np else 'int32')
    else:
        idx = (k - S["delay"]) // NSUB
        ok = (idx >= 0) & (idx < nt)
        k0 = xp.clip(idx, 0, nt - 1)
    u = tables[which, k0]
    return xp.where(ok[:, None], u, 0.0)


def init_state(S, xp=np):
    E, W = S["px0"].shape
    z = xp.zeros((E, W))
    return dict(x=xp.zeros((E, K.S0 + K.PER * W)), integ=z, thc=z, u=z, rot=z, vf=z, tau=z,
                eth=xp.zeros((E, W)) + S["eth0"])


def _motor_cmd(S, st, xp):
    """Misspecified motor (build.MOTOR): this step's true torque at the previous step's wheel speed and torque, as the
    crane speed command cmd = w + tau / K (the kernel's local slope -K around w is kept)."""
    E, W = S["px0"].shape
    om = st["x"][:, K.S0:].reshape(E, W, K.PER)[..., 1]
    ts, wnl = S["ts"], S["wnl"]
    Kg = ts / wnl
    load = (S["Amod"] * (xp.abs(st["tau"]) / ts)[:, None, :]).sum(-1)
    vb = S["vb"] * xp.maximum(1.0 - S["ksag"] * load, 0.5)
    if xp is np:        # theory's shared knee law, cfcap/t0/envelope.py (the numba / JAX paths inline the same formula)
        from cfcap.t0 import envelope as ENV
        env = dict(f_w=S["omk"] / (S["vb"] * wnl), c_w=S["cw"], f_t=S["tauk"] / ts, c_t=S["ct"])
        fs, fn = ENV.knee_factors(env, om, st["tau"], S["vb"] * wnl, ts)
        nl, se = wnl * fn, ts * fs * xp.maximum(0.4, 1.0 - S["cth"] * st["eth"])
    else:
        nl = wnl * xp.maximum(0.4, 1.0 - S["cw"] * xp.maximum(xp.abs(om) / S["omk"] - 1.0, 0.0))
        se = ts * xp.maximum(0.4, 1.0 - S["ct"] * xp.maximum(xp.abs(st["tau"]) / S["tauk"] - 1.0, 0.0)) \
            * xp.maximum(0.4, 1.0 - S["cth"] * st["eth"])
    tau = se * (st["u"] * S["gw"] * vb - om / nl)
    tau = xp.minimum(xp.maximum(tau, -S["ilim"]), S["ilim"])
    return om + tau / Kg


def _motor_after(S, st, cmd, om_new, xp):
    Kg = S["ts"] / S["wnl"]
    tau = xp.minimum(xp.maximum(Kg * (cmd - om_new), -S["ilim"]), S["ilim"])
    tau = xp.where(S["driven"] > 0, tau, 0.0)
    eth = st["eth"] + DT * ((tau / S["ts"]) ** 2 - st["eth"] / MOTOR_TAU_TH)
    return tau, eth


def tick(S, st, tables, which, j, xp=np):
    """One 50 Hz tick (NSUB plant steps) of every env. Returns the new state dict."""
    p = S["p"]
    for s in range(NSUB):
        k = j * NSUB + s
        u_tw = _command(S, tables, which, k, xp)
        if S["fw_robot"]:
            if s == 0:
                u, integ, thc = _firmware(S, st, u_tw, xp)
                st = dict(st, u=u, integ=integ, thc=thc)
            cmd = _motor_cmd(S, st, xp) if S["mm_on"] else st["u"] * S["gw"] * S["vb"] * S["wnl"]
        else:
            cmd = K.firmware(K._wheels(p, xp), u_tw, xp)
        pe = _effective_p(p, S, st, xp)
        x_new, om = step_cmd(st["x"], cmd, pe, DT, xp)
        rot, vf = st["rot"], st["vf"]
        if S["cp_on"] or S["an_on"]:
            Fx, Fy, vf_n = _wheel_forces(pe, st["x"], om, xp)
            if S["an_on"]:
                vf = vf_n
            if S["cp_on"]:
                vx, vy, wz, dvx, dvy, dwz = (x_new[:, i] for i in range(3, 9))
                z = xp.concatenate([Fx, Fy, (dvx - wz * vy)[:, None], (dvy + wz * vx)[:, None], dwz[:, None],
                                    (wz * wz)[:, None]], axis=1)
                rel = (S["Mrot"] * z[:, None, :]).sum(-1)
                rot = rel + S["play"][:, None] * xp.tanh(rel / ROT_SMOOTH)
        st = dict(st, x=x_new, rot=rot, vf=vf)
        if S["mm_on"]:
            tau, eth = _motor_after(S, st, cmd, om, xp)
            st = dict(st, tau=tau, eth=eth)
    return st


CH = ("twist", "pose", "duty", "speed", "angle", "gz", "ax", "ay")


def _record(S, st, xp):
    x = st["x"]
    E, W = S["px0"].shape
    wx = x[:, K.S0:].reshape(E, W, K.PER)
    gz, ax, ay = K.imu(x, S["p"], xp)
    duty = st["u"] if S["fw_robot"] else wx[..., 2]
    return dict(twist=x[:, 3:6], pose=x[:, 0:3], duty=duty, speed=wx[..., 1], angle=wx[..., 0], gz=gz, ax=ax, ay=ay)


def run_numpy(S, tables, which, n_ticks):
    """Reference backend (float64, python loop over ticks). Records at the end of every tick."""
    st = init_state(S)
    rec = {c: [] for c in CH}
    for j in range(n_ticks):
        st = tick(S, st, tables, which, j)
        for c, v in _record(S, st, np).items():
            rec[c].append(v)
    return {c: np.stack(v, 1) for c, v in rec.items()}


_JAX_CACHE = {}


def run_jax(S, tables, which, n_ticks, dtype="float32"):
    """JAX backend: lax.scan over ticks (one compile per shape and effect flags)."""
    from engine.jaxconf import jax, jnp
    f = jnp.float32 if dtype == "float32" else jnp.float64
    flags = ("lt_on", "cp_on", "an_on", "fw_robot", "mm_on")
    arr = {k: v for k, v in S.items() if k not in flags}
    fl = {k: bool(S[k]) for k in flags}
    key = (tuple(sorted(fl.items())), dtype, n_ticks)

    def conv(v):
        v = np.asarray(v)
        return jnp.asarray(v, f) if v.dtype.kind == "f" else jnp.asarray(v)
    A = {k: conv(v) for k, v in arr.items()}
    tb, wh = jnp.asarray(tables, f), jnp.asarray(which)
    if key not in _JAX_CACHE:
        def go(A, tb, wh):
            SS = dict(A, **fl)
            st0 = {k: v.astype(f) for k, v in init_state(SS, jnp).items()}

            def body(st, j):
                st = tick(SS, st, tb, wh, j, jnp)
                return st, _record(SS, st, jnp)
            _, ys = jax.lax.scan(body, st0, jnp.arange(n_ticks))
            return {c: jnp.moveaxis(y, 0, 1) for c, y in ys.items()}
        _JAX_CACHE[key] = jax.jit(go)
    out = _JAX_CACHE[key](A, tb, wh)
    return {c: np.asarray(v, np.float64) for c, v in out.items()}


# ----------------------------------------------------------------------------------------------- numba form
@numba.njit(cache=True, inline='always')
def _nb_step_cmd(x, p, dt, cmd, om, scratch, Wr, A, r):
    """mecanum.nb_step from the per-wheel command cmd (given) on one env's row, in place."""
    W = (p.shape[0] - K.G) // K.Q
    t = (x[9] + K.ONSET_EPS) * dt
    m, c0, c1, I = p[0], p[1], p[2], p[3]
    xi0, xi1, xi2 = x[3], x[4], x[5]
    for q in range(4):
        a0 = xi0 + (K.H_FD if q == 1 else 0.0)
        a1 = xi1 + (K.H_FD if q == 2 else 0.0)
        a2 = xi2 + (K.H_FD if q == 3 else 0.0)
        sx, sy, sz = K._nb_contact(p, W, t, a0, a1, a2, cmd, scratch[q])
        vcx = a0 - a2 * c1
        vcy = a1 + a2 * c0
        bl0 = m * (-a2 * vcy)
        bl1 = m * (a2 * vcx)
        Wr[q, 0], Wr[q, 1], Wr[q, 2] = sx - bl0, sy - bl1, sz - (c0 * bl1 - c1 * bl0)
    A[0, 0], A[0, 1], A[0, 2] = m, 0.0, -m * c1
    A[1, 0], A[1, 1], A[1, 2] = 0.0, m, m * c0
    A[2, 0], A[2, 1], A[2, 2] = -m * c1, m * c0, I
    for rr in range(3):
        for cc in range(3):
            A[rr, cc] = A[rr, cc] - dt * ((Wr[cc + 1, rr] - Wr[0, rr]) / K.H_FD)
        r[rr] = dt * Wr[0, rr]
    d0, d1, d2 = K._nb_solve3(A, r)
    v0, v1, v2 = xi0 + d0, xi1 + d1, xi2 + d2
    vx, vy, wz = v0 * dt, v1 * dt, v2 * dt
    if abs(wz) < 1e-9:
        dx, dy = vx, vy
    else:
        s, c = math.sin(wz), math.cos(wz)
        dx = (s * vx - (1 - c) * vy) / wz
        dy = ((1 - c) * vx + s * vy) / wz
    cs, sn = math.cos(x[2]), math.sin(x[2])
    x[0], x[1], x[2] = x[0] + cs * dx - sn * dy, x[1] + sn * dx + cs * dy, x[2] + wz
    x[3], x[4], x[5] = v0, v1, v2
    x[6], x[7], x[8] = d0 / dt, d1 / dt, d2 / dt
    x[9] = x[9] + 1.0
    for j in range(W):
        o = K.G + j * K.Q
        w = scratch[0, j]
        om[j] = w
        k = K.S0 + K.PER * j
        x[k] = x[k] + w * dt
        x[k + 1] = w
        if p[o + 6] == K.BALL or p[o + 22] <= 0:
            x[k + 2] = 0.0
            continue
        tcmd = p[o + 15] * (cmd[j] - w)
        ts, wnl = p[o + 16], p[o + 17]
        lim = ts * (min(max(1.0 - abs(w) / wnl, 0.0), 1.0) if tcmd * w > 0 else 1.0)
        tm = min(max(tcmd, -lim), lim)
        x[k + 2] = min(max(tm / ts + w / wnl, -1.0), 1.0)


@numba.njit(cache=True, parallel=True)
def _nb_run(p, px0, py0, ex0, ey0, tx0, ty0, fex, fey, ftx, fty, r_nom, kff, kp, ki, cpr, vb, gw, wnl, driven, lat_ticks, delay,
            fw_robot, lt_on, N0, GN, h, cp_on, Mrot, play, an_on, mufp, mufn, mm_on, ts, ilim, omk, tauk, cw, ct,
            cth, eth0, ksag, Amod, tables, which, n_ticks,
            o_twist, o_pose, o_duty, o_speed, o_angle, o_gz, o_ax, o_ay):
    E = p.shape[0]
    W = (p.shape[1] - K.G) // K.Q
    nt = tables.shape[1]
    nz = 2 * W + 4
    for e in numba.prange(E):
        pe = p[e].copy()
        x = np.zeros(K.S0 + K.PER * W)
        cmd, om, scratch, Wr, A, r = np.zeros(W), np.zeros(W), np.zeros((4, W)), np.zeros((4, 3)), \
            np.zeros((3, 3)), np.zeros(3)
        integ, thc, u = np.zeros(W), np.zeros(W), np.zeros(W)
        rot, vf, z = np.zeros(W), np.zeros(W), np.zeros(nz)
        taum, eth = np.zeros(W), eth0[e].copy()
        m = which[e]
        kc = cpr[e] / (2 * math.pi)
        Ntot = 0.0
        for jw in range(W):
            Ntot += N0[e, jw]
        for jt in range(n_ticks):
            for s in range(NSUB):
                k = jt * NSUB + s
                # command
                if fw_robot:
                    tk = k * (1.0 / NSUB) - lat_ticks[e]
                    ok = tk >= -1e-4
                    k0 = int(min(max(math.floor(tk + 1e-4), 0.0), nt - 1.0))
                else:
                    idx = (k - delay[e]) // NSUB
                    ok = idx >= 0 and idx < nt
                    k0 = min(max(idx, 0), nt - 1)
                u0 = tables[m, k0, 0] if ok else 0.0
                u1 = tables[m, k0, 1] if ok else 0.0
                u2 = tables[m, k0, 2] if ok else 0.0
                # firmware -> per-wheel speed command (crane motor units)
                for jw in range(W):
                    o = K.G + jw * K.Q
                    if fw_robot:
                        if s == 0:
                            vcx = u0 - u2 * py0[e, jw]
                            vcy = u1 + u2 * px0[e, jw]
                            te = ftx[e, jw] * fex[e, jw] + fty[e, jw] * fey[e, jw]
                            wref = (ftx[e, jw] * vcx + fty[e, jw] * vcy) / (r_nom[e] * (te if abs(te) > 1e-6 else 1.0))
                            ang = x[K.S0 + K.PER * jw]
                            meas = (math.floor(ang * kc) - math.floor(thc[jw] * kc)) / kc / (NSUB * DT)
                            er = wref - meas
                            raw = kff[e] * wref + kp[e] * er + ki[e] * integ[jw]
                            se = 1.0 if er > 0 else (-1.0 if er < 0 else 0.0)
                            sr = 1.0 if raw > 0 else (-1.0 if raw < 0 else 0.0)
                            if (abs(raw) < 1 or se != sr) and driven[e, jw] > 0:
                                integ[jw] = integ[jw] + er * (NSUB * DT)
                            u[jw] = min(max(raw, -1.0), 1.0)
                            thc[jw] = ang
                        if mm_on:
                            omw = x[K.S0 + K.PER * jw + 1]
                            ld = 0.0
                            for q in range(W):
                                ld += Amod[e, jw, q] * abs(taum[q]) / ts[e, q]
                            vbe = vb[e, jw] * max(1.0 - ksag[e, jw] * ld, 0.5)
                            nle = wnl[e, jw] * max(0.4, 1.0 - cw[e, jw] * max(abs(omw) / omk[e, jw] - 1.0, 0.0))
                            se = ts[e, jw] * max(0.4, 1.0 - ct[e, jw] * max(abs(taum[jw]) / tauk[e, jw] - 1.0, 0.0)) \
                                * max(0.4, 1.0 - cth[e, jw] * eth[jw])
                            tq = se * (u[jw] * gw[e, jw] * vbe - omw / nle)
                            tq = min(max(tq, -ilim[e, jw]), ilim[e, jw])
                            cmd[jw] = omw + tq / (ts[e, jw] / wnl[e, jw])
                        else:
                            cmd[jw] = u[jw] * gw[e, jw] * vb[e, jw] * wnl[e, jw]
                    else:
                        pb = p[e]
                        if pb[o + 6] == K.BALL or pb[o + 22] <= 0:
                            cmd[jw] = 0.0
                        else:
                            vcx = u0 + u2 * (-pb[o + 1])
                            vcy = u1 + u2 * pb[o]
                            te = pb[o + 4] * pb[o + 2] + pb[o + 5] * pb[o + 3]
                            c = (pb[o + 4] * vcx + pb[o + 5] * vcy) / (pb[o + 8] * (te if abs(te) > 1e-6 else 1.0))
                            cmd[jw] = min(max(c, -pb[o + 21]), pb[o + 21])
                # effects -> this step's row
                if lt_on:
                    vx, vy, wz, dvx, dvy, dwz = x[3], x[4], x[5], x[6], x[7], x[8]
                    cx, cy = pe[1], pe[2]
                    ax = dvx - wz * vy - dwz * cy - wz * wz * cx
                    ay = dvy + wz * vx + dwz * cx - wz * wz * cy
                    sc = -h[e] / G0
                    tot = 0.0
                    for jw in range(W):
                        o = K.G + jw * K.Q
                        nn = N0[e, jw] + GN[e, jw, 0] * sc * ax + GN[e, jw, 1] * sc * ay
                        nn = max(nn, 0.02 * G0) if pe[o + 22] > 0 else 0.0
                        pe[o + I_N] = nn
                        tot += nn
                    for jw in range(W):
                        pe[K.G + jw * K.Q + I_N] *= Ntot / tot
                if cp_on:
                    for jw in range(W):
                        o = K.G + jw * K.Q
                        c, sn = math.cos(rot[jw]), math.sin(rot[jw])
                        pe[o + I_EX] = c * ex0[e, jw] - sn * ey0[e, jw]
                        pe[o + I_EY] = sn * ex0[e, jw] + c * ey0[e, jw]
                        pe[o + I_TX] = c * tx0[e, jw] - sn * ty0[e, jw]
                        pe[o + I_TY] = sn * tx0[e, jw] + c * ty0[e, jw]
                if an_on:
                    for jw in range(W):
                        sg = math.tanh(vf[jw] / K.V_EPS)
                        pe[K.G + jw * K.Q + I_MUF] = 0.5 * (mufp[e, jw] + mufn[e, jw]) + \
                            0.5 * (mufp[e, jw] - mufn[e, jw]) * sg
                xv0, xv1, xv2 = x[3], x[4], x[5]
                _nb_step_cmd(x, pe, DT, cmd, om, scratch, Wr, A, r)
                if mm_on:
                    for jw in range(W):
                        kg = ts[e, jw] / wnl[e, jw]
                        tq = min(max(kg * (cmd[jw] - om[jw]), -ilim[e, jw]), ilim[e, jw])
                        taum[jw] = tq if driven[e, jw] > 0 else 0.0
                        eth[jw] = eth[jw] + DT * ((taum[jw] / ts[e, jw]) ** 2 - eth[jw] / MOTOR_TAU_TH)
                if cp_on or an_on:
                    for jw in range(W):
                        o = K.G + jw * K.Q
                        vcx = xv0 + xv2 * (-pe[o + 1])
                        vcy = xv1 + xv2 * pe[o]
                        live = pe[o + I_N] > 0 and pe[o + 22] > 0 and pe[o + 6] != K.BALL
                        Fx, Fy, _ = K._nb_traction(pe, o, vcx, vcy, om[jw])
                        rw = pe[o + 7] * om[jw]
                        vf[jw] = -pe[o + I_TY] * (vcx - rw * pe[o + I_EX]) + pe[o + I_TX] * (vcy - rw * pe[o + I_EY])
                        z[jw] = Fx if live else 0.0
                        z[W + jw] = Fy if live else 0.0
                    if cp_on:
                        z[2 * W] = x[6] - x[5] * x[4]
                        z[2 * W + 1] = x[7] + x[5] * x[3]
                        z[2 * W + 2] = x[8]
                        z[2 * W + 3] = x[5] * x[5]
                        for jw in range(W):
                            acc = 0.0
                            for q in range(nz):
                                acc += Mrot[e, jw, q] * z[q]
                            rot[jw] = acc + play[e] * math.tanh(acc / ROT_SMOOTH)
            # record at the end of the tick
            for c3 in range(3):
                o_pose[e, jt, c3] = x[c3]
                o_twist[e, jt, c3] = x[3 + c3]
            for jw in range(W):
                b = K.S0 + K.PER * jw
                o_angle[e, jt, jw] = x[b]
                o_speed[e, jt, jw] = x[b + 1]
                o_duty[e, jt, jw] = u[jw] if fw_robot else x[b + 2]
            vx, vy, wz, dvx, dvy, dwz = x[3], x[4], x[5], x[6], x[7], x[8]
            rx, ry = pe[I_IMX], pe[I_IMY]
            o_gz[e, jt] = wz
            o_ax[e, jt] = dvx - wz * vy - dwz * ry - wz * wz * rx
            o_ay[e, jt] = dvy + wz * vx + dwz * rx - wz * wz * ry


def run_numba(S, tables, which, n_ticks):
    E, W = S["px0"].shape
    out = dict(twist=np.zeros((E, n_ticks, 3)), pose=np.zeros((E, n_ticks, 3)), duty=np.zeros((E, n_ticks, W)),
               speed=np.zeros((E, n_ticks, W)), angle=np.zeros((E, n_ticks, W)), gz=np.zeros((E, n_ticks)),
               ax=np.zeros((E, n_ticks)), ay=np.zeros((E, n_ticks)))
    f = lambda k: np.ascontiguousarray(S[k], dtype=np.float64)      # noqa: E731
    _nb_run(f("p"), f("px0"), f("py0"), f("ex0"), f("ey0"), f("tx0"), f("ty0"), f("fex"), f("fey"), f("ftx"),
            f("fty"), f("r_nom"), f("kff"), f("kp"),
            f("ki"), f("cpr"), f("vb"), f("gw"), f("wnl"), f("driven"), f("lat_ticks"),
            np.ascontiguousarray(S["delay"], np.int64), bool(S["fw_robot"]), bool(S["lt_on"]), f("N0"), f("GN"),
            f("h"), bool(S["cp_on"]), f("Mrot"), f("play"), bool(S["an_on"]), f("mufp"), f("mufn"),
            bool(S.get("mm_on", False)), f("ts"), f("ilim"), f("omk"), f("tauk"), f("cw"), f("ct"), f("cth"), f("eth0"),
            f("ksag"), f("Amod"),
            np.ascontiguousarray(tables, np.float64), np.ascontiguousarray(which, np.int64), int(n_ticks),
            out["twist"], out["pose"], out["duty"], out["speed"], out["angle"], out["gz"], out["ax"], out["ay"])
    return out


def run(S, tables, which, n_ticks, backend="numba", dtype="float32"):
    if backend == "numba":
        return run_numba(S, tables, which, n_ticks)
    if backend == "numpy":
        return run_numpy(S, tables, which, n_ticks)
    if backend == "jax":
        return run_jax(S, tables, which, n_ticks, dtype)
    raise ValueError(f"backend {backend!r}: numba, numpy or jax")
