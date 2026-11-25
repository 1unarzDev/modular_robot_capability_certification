"""T2 inputs: crane kernel rows + firmware + effect arrays from a capxfer assembly and T1-format episode latents.

Geometry (wheel positions, rolling and roller directions, module centres) is the shared robot spec, taken from the
capxfer Assembly so that T1 and T2 see the same morphology; every physical number goes through the mapping below.

theta / T1 latent -> crane kernel (and how the two plants define it differently)
  mu (x mu_mod per module)  mu_t (roller_mu_f = 1)          same: tanh friction along the roller axis, cap mu N
  s0 [m/s]                  s_x0 = s0 / V_REF               DIFFERENT: capxfer slip VELOCITY scale; crane slip RATIO
                                                            over V = max(|r w|, |v_c|, 0.02); equal at |v| = V_REF
  c_roller [N s/m]          mu_f = c_roller V_REF / N0_j    DIFFERENT: capxfer viscous drag; crane Coulomb
                                                            mu_f N tanh(v_f / 0.01); equal force at |v_f| = V_REF
  crr                       c_rr                            DIFFERENT: capxfer torque crr N r tanh(w/0.5) at the
                                                            wheel; crane force c_rr N tanh(e.v/0.01) on the body
  latency [s]               continuous tick delay           same (capxfer convention, 50 Hz ticks)
  stall/noload_scale (x per-wheel stall_w/noload_w)  tau_stall, w_nl     same linear torque-speed curve
  v_batt (+ vbatt_off)      duty -> v_b = V / 12 in the firmware        same
  tau_bd                    tau_bd                          DIFFERENT smoothing: tanh(w/0.05) vs tanh(w/0.3)
  jeff_scale                -- (none)                       DIFFERENT: crane wheels are massless (algebraic speed)
  izz_scale, mass_scale     inertia I_o = izz + m |c|^2     DIFFERENT: crane couples the COM offset in M
                                                            (capxfer: diag(m, m, izz) about the origin)
  radius_scale              r (physics); r_fw nominal       same
  kp/ki_scale               firmware gains                  same
  cg_dx/dy, payload         body COM, mass, inertia         same point-mass payload
  yaw_err (per wheel)       physics e, t rotated; firmware nominal       same
  normal loads              least-norm + T1 null-space load_z           same at rest (crane equal springs =
                                                                         least-norm); load transfer adds dynamics
  faults weak/coast/brake   mode 1/2/3 (weak: crane scales tau_stall and w_nl by g = capxfer duty x g)
  capxfer mode 4 (roller-locked) has no crane counterpart (crane 4 = wheel locked) and is refused.

Effects (drawn per episode from `seed`, shared by every leg of the episode and its healthy twin):
  compliance      k_axial U(2e4, 2e5) N/m, k_lat U(5e3, 5e4) N/m, k_yaw U(50, 1000) N m/rad per latched face
                  (capplan truth COUPLING_SPEC), hook play 1.1 mm -> theta_play = 2 play / 0.12 m hook spacing.
  load transfer   COM height per module U(0.04, 0.11) m (capxfer params cg_height 0.041; truth prior 0.06-0.11);
                  payload at 0.20 m.
  anisotropy      (capxfer I4 stress recipe) per-wheel roller drag x logU(1/4, 4), one free direction x U(1, 4);
                  `aniso_mu`: per-wheel roller friction x U(0.7, 1.3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from cfcap.t2 import deps  # noqa: F401
from cfcap.t2.kernel import G0, NSUB, DT
from engine.kernels import mecanum as K

V_REF = 0.30                 # m/s: speed at which the slip-velocity and slip-ratio (and viscous/Coulomb) forms agree
HOOK_SPACING = 0.12          # m, capplan profile docking.hook_lateral_spacing
CMD_HZ = 50.0


@dataclass(frozen=True)
class Effects:
    compliance: bool = False
    load_transfer: bool = False
    aniso: bool = False
    aniso_mu: bool = False
    hook_play: float = 1.1e-3
    firmware: str = "robot"       # robot | crane (crane only for the reproduction test)
    motor: bool = False           # misspecified motor law, the stress superset (OFF by default; see MOTOR below)
    motor_kind: str = "none"      # "knee": theory's shared wrong law (cfcap/t0/envelope.py), severity knee_severity
    knee_severity: int = 0

    def name(self):
        on = [k for k in ("compliance", "load_transfer", "aniso", "aniso_mu", "motor") if getattr(self, k)]
        if self.motor_kind == "knee":
            on.append(f"knee{self.knee_severity}")
        return "+".join(on) or "base"

    @property
    def motor_on(self):
        return self.motor or (self.motor_kind == "knee" and self.knee_severity > 0)


NONE = Effects()
ALL = Effects(True, True, True, True)
ABLATIONS = {"base": NONE, "compliance": Effects(compliance=True), "load_transfer": Effects(load_transfer=True),
             "aniso": Effects(aniso=True), "aniso_mu": Effects(aniso_mu=True), "all": ALL,
             "all-aniso_mu": Effects(True, True, True, False),
             # misspecified motor law (flagged; not part of `all`)
             "motor": Effects(motor=True), "all+motor": Effects(True, True, True, True, motor=True)}
for _sv in (1, 2, 3):           # theory's shared knee law (E1), alone and with the A5 effects
    ABLATIONS[f"knee{_sv}"] = Effects(motor_kind="knee", knee_severity=_sv)
    ABLATIONS[f"all+knee{_sv}"] = Effects(True, True, True, True, motor_kind="knee", knee_severity=_sv)

# Misspecified motor (Effects.motor; T1 and the identified theta assume the linear DC law tau = ts (g v_b u - w/w_nl)).
# Per wheel (draws from their own stream, so the other effects' draws are unchanged), each relative to that wheel's
# nominal tau_s / w_nl:
#   current limit   |tau| <= c_I tau_s, c_I ~ U(0.5, 0.9)               (driver current limiting below stall)
#   speed knee      w_nl_eff = w_nl max(0.4, 1 - c_w (|w|/w_k - 1)_+), w_k = U(0.55, 0.95) v_b w_nl, c_w ~ U(0.5, 1.5)
#   torque knee     tau_s_eff = tau_s max(0.4, 1 - c_t (|tau|/tau_k - 1)_+), tau_k = U(0.1, 0.4) tau_s, c_t ~ U(0.5,1.5)
#                   (the theory ENVELOPE form, cfcap/theory_task.py)
#   thermal derate  tau_s_eff x max(0.4, 1 - c_th E), dE/dt = (tau/tau_s)^2 - E / 30 s, E0 ~ U(0, 1) (prior use),
#                   c_th ~ U(0.1, 0.3)
#   voltage sag     v_b_eff = v_b (1 - k_sag mean_module(|tau|/tau_s)), k_sag ~ U(0.05, 0.2) per module
MOTOR_TAU_TH = 30.0


def _np(v):
    try:
        import torch
        if torch.is_tensor(v):
            return v.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(v)


def _least_norm(pos, cg, mass):
    """capxfer _prepare's normal loads and their COM gradient dN/dcg (n, 2)."""
    n = len(pos)
    A = np.vstack([np.ones(n), pos[:, 0], pos[:, 1]])
    P = A.T @ np.linalg.inv(A @ A.T)
    N = P @ (mass * G0 * np.array([1.0, cg[0], cg[1]]))
    GN = P[:, 1:] * mass * G0
    return N, GN


def _couplings(asm):
    """Latched faces: 4-neighbour module pairs on the docking grid -> (a, b, face point, unit normal a->b)."""
    c = np.asarray(asm.module_pos, float)
    pitch = float(asm.meta.get("pitch", 0.32))
    out = []
    for a in range(len(c)):
        for b in range(a + 1, len(c)):
            d = c[b] - c[a]
            if abs(np.linalg.norm(d) - pitch) < 1e-6:
                out.append((a, b, 0.5 * (c[a] + c[b]), d / np.linalg.norm(d)))
    return out


def compliance_map(asm, wm, pos, mmod, Imod, ks, n_wheels):
    """Linear map (n, 2n + 4): z = [Fx (n), Fy (n), a0x, a0y, alpha, wz^2] -> per-wheel deflection yaw (capplan truth
    body_loads + inertial loads, residual removed, planar_deflection, _rel_rot)."""
    cps = _couplings(asm)
    nb = len(asm.module_pos)
    if nb < 2 or not cps:
        return np.zeros((n_wheels, 2 * n_wheels + 4))
    bc = np.asarray(asm.module_pos, float)
    nd = 3 * nb
    Kp = np.zeros((nd, nd))
    perp = lambda v: np.array([-v[1], v[0]])                         # noqa: E731
    for (a, b, pt, nrm), (kax, klat, kyaw) in zip(cps, ks):
        for ax, k in ((nrm, kax), (perp(nrm), klat)):
            r = np.zeros(nd)
            for bi, sg in ((a, 1.0), (b, -1.0)):
                d = pt - bc[bi]
                r[3 * bi: 3 * bi + 2] += sg * ax
                r[3 * bi + 2] += sg * (d[0] * ax[1] - d[1] * ax[0])
            Kp += k * np.outer(r, r)
        r = np.zeros(nd)
        r[3 * a + 2], r[3 * b + 2] = 1.0, -1.0
        Kp += kyaw * np.outer(r, r)
    Kpinv = np.linalg.pinv(Kp, rcond=1e-12)
    n = len(wm)
    cols = []
    for q in range(2 * n + 4):
        z = np.zeros(2 * n + 4)
        z[q] = 1.0
        F = np.stack([z[:n], z[n:2 * n]], -1)
        a0, al, w2 = z[2 * n:2 * n + 2], z[2 * n + 2], z[2 * n + 3]
        L = np.zeros((nb, 3))
        d = pos - bc[wm]
        for j in range(n):
            L[wm[j], :2] += F[j]
            L[wm[j], 2] += d[j, 0] * F[j, 1] - d[j, 1] * F[j, 0]
        for i in range(nb):
            ai = a0 + al * perp(bc[i]) - w2 * bc[i]
            L[i, :2] += -mmod[i] * ai
            L[i, 2] += -Imod[i] * al
        L[:, :2] -= L[:, :2].sum(0) / nb
        th = (Kpinv @ L.reshape(-1)).reshape(nb, 3)[:, 2]
        th = th - (th * mmod).sum() / mmod.sum()
        cols.append(th[wm])
    M = np.zeros((n_wheels, 2 * n_wheels + 4))
    Mq = np.stack(cols, 1)                                            # (n, 2n+4)
    M[:n, :n], M[:n, n_wheels:n_wheels + n] = Mq[:, :n], Mq[:, n:2 * n]
    M[:n, 2 * n_wheels:] = Mq[:, 2 * n:]
    return M


def _ep(lat, k, b):
    v = lat[k]
    return None if v is None else _np(v)[b]


def episode_specs(asm, lat, params, effects=NONE, seed=0, clear_faults=False):
    """Spec rows (one per episode of `lat`) for the T2 kernel. lat: T1 latents (cfcap.t1.plant.sample_latents,
    torch or numpy). clear_faults: the same latents with every fault cleared (capxfer D17 healthy twin)."""
    from capxfer import sim
    nom = sim.nominal(params)
    dom = {k: (_np(v) if not isinstance(v, float) else v) for k, v in lat["dom"].items()}
    mode, g = _np(lat["mode"]).astype(np.int64), _np(lat["g"]).astype(np.float64)
    B, n = mode.shape
    if clear_faults:
        mode, g = np.zeros_like(mode), np.ones_like(g)
    if (mode == 4).any():
        raise ValueError("capxfer roller-locked faults (mode 4) have no crane counterpart")
    wm = np.asarray(asm.wheel_module)
    pos = np.asarray(asm.wheel_pos, float)
    w_hat, r_hat = np.asarray(asm.w_hat, float), np.asarray(asm.r_hat, float)
    m_mod = np.asarray(asm.module_mass, float)
    side = float(asm.side)
    rows, fw = [], {k: [] for k in ("px0", "py0", "ex0", "ey0", "tx0", "ty0", "fex", "fey", "ftx", "fty", "vb", "gw", "wnl",
                                     "driven", "N0", "ts", "ilim", "omk", "tauk", "cw", "ct", "cth", "eth0", "ksag", "Amod",
                                     "GN", "Mrot", "mufp", "mufn")}
    sc = {k: [] for k in ("r_nom", "kff", "kp", "ki", "cpr", "lat_ticks", "h", "play", "delay")}
    for b in range(B):
        rng = np.random.default_rng([int(seed), b, 7177])
        ga = lambda k, d=1.0: float(dom[k][b]) if k in dom else d     # noqa: E731
        yerr = lat.get("yaw_err")
        ye = np.zeros(n) if yerr is None else _np(yerr)[b].astype(float)
        rot = lambda v: np.stack([np.cos(ye) * v[:, 0] - np.sin(ye) * v[:, 1],     # noqa: E731
                                  np.sin(ye) * v[:, 0] + np.cos(ye) * v[:, 1]], -1)
        e_ph, t_ph = rot(w_hat), rot(r_hat)
        mass = float(m_mod.sum()) * ga("mass_scale")
        izz = asm.izz() * ga("izz_scale") * ga("mass_scale")
        cg = asm.cg() + np.array([ga("cg_dx", 0.0), ga("cg_dy", 0.0)])
        mm = m_mod * ga("mass_scale")
        h_mod = rng.uniform(0.04, 0.11, len(m_mod))
        h_num, h_den = float((mm * h_mod).sum()), float(mm.sum())
        pl = lat.get("payload")
        if pl is not None:
            pm, pp = float(_np(pl["mass"])[b]), np.array([float(_np(pl["x"])[b]), float(_np(pl["y"])[b])])
            izz = izz + pm * float(((pp - cg) ** 2).sum()) * mass / (mass + pm)
            cg = (cg * mass + pp * pm) / (mass + pm)
            mass = mass + pm
            h_num, h_den = h_num + pm * 0.20, h_den + pm
        N, GN = _least_norm(pos, cg, mass)
        N = np.maximum(N, 0.02 * G0)
        N = N * (mass * G0 / N.sum())
        z = _ep(lat, "load_z", b)
        if z is not None and np.abs(z).max() > 0 and n > 3:
            A = np.vstack([np.ones(n), pos[:, 0], pos[:, 1]])
            Pn = np.eye(n) - A.T @ np.linalg.solve(A @ A.T, A)
            tot = N.sum()
            N = np.maximum(N + Pn @ z * tot / n, 0.02 * G0)
            N = N * tot / N.sum()
        mu = ga("mu") * _ep(lat, "mu_mod", b)[wm]
        ts = nom["tau_s"] * ga("stall_scale") * _ep(lat, "stall_w", b)
        wnl = nom["omega0"] * ga("noload_scale") * _ep(lat, "noload_w", b)
        muf = ga("c_roller") * V_REF / N
        mufp = mufn = muf
        mut = mu.copy()
        if effects.aniso:
            f = np.exp(rng.uniform(np.log(0.25), np.log(4.0), n))
            asym = rng.uniform(1.0, 4.0, n)
            flip = rng.uniform(size=n) < 0.5
            mufp = muf * f * np.where(flip, asym, 1.0)
            mufn = muf * f * np.where(flip, 1.0, asym)
        else:
            rng.uniform(size=3 * n)                                     # keep later draws aligned across ablations
        if effects.aniso_mu:
            mut = mu * rng.uniform(0.7, 1.3, n)
        else:
            rng.uniform(size=n)
        # adversarial search (cfcap.t2.adv): explicit per-wheel effect factors replace the draws (rng stays aligned)
        if effects.aniso and lat.get("t2_drag_pos") is not None:
            mufp, mufn = muf * _np(lat["t2_drag_pos"])[b], muf * _np(lat["t2_drag_neg"])[b]
        if effects.aniso_mu and lat.get("t2_mu_fac") is not None:
            mut = mu * _np(lat["t2_mu_fac"])[b]
        radius = nom["radius"] * ga("radius_scale")
        wheels = []
        for j in range(n):
            wheels.append(dict(px=pos[j, 0], py=pos[j, 1], ex=e_ph[j, 0], ey=e_ph[j, 1], tx=t_ph[j, 0], ty=t_ph[j, 1],
                               kind=K.MEC, r=radius, r_fw=nom["radius"], mu_t=mut[j], mu_f=0.5 * (mufp[j] + mufn[j]),
                               s_x0=ga("s0") / V_REF, alpha0=1.0, c_rr=ga("crr"), N=N[j], K=ts[j] / wnl[j],
                               tau_stall=ts[j], w_nl=wnl[j], b=0.0, tau_c=0.0, tau_bd=ga("tau_bd", 0.75),
                               phimax=1e3))
        body = dict(mass=mass, xc=cg[0], yc=cg[1], inertia=izz + mass * float(cg @ cg), imu_x=0.0, imu_y=0.0,
                    cpr=nom["cpr"])
        rows.append(K.pack(wheels, body, dict(mode=list(mode[b]), gain=list(g[b]), onset=[0.0] * n)))
        fw["px0"].append(pos[:, 0]); fw["py0"].append(pos[:, 1])
        fw["ex0"].append(e_ph[:, 0]); fw["ey0"].append(e_ph[:, 1])
        fw["tx0"].append(t_ph[:, 0]); fw["ty0"].append(t_ph[:, 1])
        fw["fex"].append(w_hat[:, 0]); fw["fey"].append(w_hat[:, 1])          # firmware: nominal geometry
        fw["ftx"].append(r_hat[:, 0]); fw["fty"].append(r_hat[:, 1])
        fw["vb"].append((ga("v_batt") + _ep(lat, "vbatt_off", b)[wm]) / nom["v0"])
        fw["gw"].append(np.where(mode[b] == K.WEAK, g[b], 1.0))
        fw["wnl"].append(wnl)
        fw["driven"].append(((mode[b] != K.COAST) & (mode[b] != K.BRAKE)).astype(float))
        fw["N0"].append(N); fw["GN"].append(GN)
        fw["mufp"].append(mufp); fw["mufn"].append(mufn)
        fw["ts"].append(ts)
        mr = np.random.default_rng([int(seed), b, 7178])
        mv = dict(ilim=mr.uniform(0.5, 0.9, n) * ts, omk=mr.uniform(0.55, 0.95, n), tauk=mr.uniform(0.1, 0.4, n) * ts,
                  cw=mr.uniform(0.5, 1.5, n), ct=mr.uniform(0.5, 1.5, n), cth=mr.uniform(0.1, 0.3, n),
                  eth0=mr.uniform(0.0, 1.0, n), ksag=mr.uniform(0.05, 0.2, len(m_mod))[wm])
        mv["omk"] = mv["omk"] * fw["vb"][-1] * wnl
        if effects.motor_kind == "knee":
            if effects.motor:
                raise ValueError("Effects: motor (superset) and motor_kind='knee' are separate options; pick one")
            from cfcap.t0 import envelope as ENV
            env = ENV.motor_envelope("knee", effects.knee_severity, np.random.default_rng([int(seed), b, 7179]), n)
            mv.update(ilim=np.full(n, np.inf), cth=np.zeros(n), eth0=np.zeros(n), ksag=np.zeros(n),
                      omk=env["f_w"] * fw["vb"][-1] * wnl, tauk=env["f_t"] * ts, cw=env["c_w"], ct=env["c_t"])
        for k, v in mv.items():
            fw[k].append(v)
        Am = (wm[:, None] == wm[None, :]).astype(float)
        fw["Amod"].append(Am / Am.sum(1, keepdims=True))
        ks = [(rng.uniform(2e4, 2e5), rng.uniform(5e3, 5e4), rng.uniform(50.0, 1000.0)) for _ in _couplings(asm)]
        if effects.compliance:
            Imod = mm * (2 * side * side) / 12.0
            fw["Mrot"].append(compliance_map(asm, wm, pos, mm, Imod, ks, n))
        else:
            fw["Mrot"].append(np.zeros((n, 2 * n + 4)))
        sc["r_nom"].append(nom["radius"])
        sc["kff"].append(nom["ff_duty_per_radps"] * ga("ff_scale"))
        sc["kp"].append(nom["kp"] * ga("kp_scale"))
        sc["ki"].append(nom["ki"] * ga("ki_scale"))
        sc["cpr"].append(nom["cpr"])
        sc["lat_ticks"].append(ga("latency") * CMD_HZ)
        sc["delay"].append(int(round(ga("latency") / DT)))
        sc["h"].append(h_num / h_den)
        multi = len(m_mod) > 1
        sc["play"].append(0.5 * 2 * effects.hook_play / HOOK_SPACING if (effects.compliance and multi) else 0.0)
    S = {k: np.stack(v) for k, v in fw.items()}
    S.update({k: np.asarray(v, np.int64 if k == "delay" else np.float64) for k, v in sc.items()})
    S["p"] = np.stack(rows)
    S.update(lt_on=effects.load_transfer, cp_on=effects.compliance, an_on=effects.aniso,
             fw_robot=effects.firmware == "robot", mm_on=effects.motor_on)
    return S


FLAGS = ("lt_on", "cp_on", "an_on", "fw_robot", "mm_on")


def repeat(S, r):
    """Each row repeated r times consecutively (episode-major rows b*r + l)."""
    return {k: (v if k in FLAGS else np.repeat(v, r, axis=0)) for k, v in S.items()}


def concat(specs):
    """Rows of equal wheel count and equal flags as one spec."""
    S0 = specs[0]
    for S in specs[1:]:
        assert all(S[k] == S0[k] for k in FLAGS) and S["px0"].shape[1] == S0["px0"].shape[1]
    return {k: (S0[k] if k in FLAGS else np.concatenate([S[k] for S in specs], 0)) for k in S0}


def with_effects(S, effects):
    return dict(S, lt_on=effects.load_transfer, cp_on=effects.compliance, an_on=effects.aniso, mm_on=effects.motor_on)
