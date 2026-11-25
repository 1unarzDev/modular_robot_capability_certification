"""A5m: A5 rerun with the misspecified-motor mode (build.Effects.motor: current limit, speed / torque knee, thermal and
voltage derating), with per-row features for every candidate including singles. Exploratory, small, local.

Same design as A5 (scripts/stress_A5.py, targets in docs/board/stress.md) except: seeds 906000+, singles included in
every analysis, label rho_dec only (each plant's own nominal reference; T2 references average 8 nuisance draws),
T2 conditions `all` (A5's plant), `motor` (motor law only) and `all+motor`. For every episode the child is also probed
with v2 and clamp under T1 and under each T2 condition (cfcap.gen.simrun.probe_tables, same latents and effect draws);
misfit D as scripts/stress_A5_clamp.py (latents known, IMU from the 50 Hz twist, duty with a 0.01 floor).

    scripts/heavy.sh $PY scripts/stress_A5m.py run --n 100      # -> $CFCAP_OUT/stress/A5m_rows.npz
    $PY scripts/stress_A5m.py analyze                           # -> results/stress/A5m_rows.csv, A5m_flips.csv, A5m_clamp.csv

Rows: one per (shape, episode, leg): leg, leg_type, n_fault, faults (JSON [wheel, mode, g]), min_g, fault-mode
counts, payload_kg, per-wheel roller-friction factor (mu_fac JSON, min, max) and roller-drag factors (drag_pos/neg
JSON), motor draws (min current-limit / tau_s, min knee speed / (v_b w_nl), max initial heat E0), rho_dec per plant,
margin |rho_dec_T1 - tau| and flip per tau and condition, v2 / clamp misfit (IMU, duty) per condition.
"""
import argparse
import csv
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stress_A5 as A5            # noqa: E402
import stress_A5_clamp as CL      # noqa: E402

CONDS = ("all", "motor", "all+motor")
SEED0 = 906000


def _spec_features(S, n):
    """Per-row effect draws recovered from a T2 spec (rows = episodes)."""
    from engine.kernels import mecanum as K
    E = S["p"].shape[0]
    wp = S["p"][:, K.G:].reshape(E, -1, K.Q)
    return wp[..., K.IW["mu_t"]][:, :n], S["mufp"], S["mufn"]


def run(n):
    import torch
    from cfcap.gen import simrun as SR
    from cfcap.t1 import capability as cap1, params as P, plant
    from cfcap.t2 import build, capability as cap2
    par = P.base_params("plated")
    th = P.load_theta("plated")
    dr = P.theta_dr(th)
    rng = np.random.default_rng(SEED0)
    targets = []
    for s in A5.SHAPES:
        asm = P.named_assembly(s, par)
        targets.append((asm, plant.sample_latents(asm, n, rng, plant.HET_FULL, dr=dr, faults=A5.faults_k(n, asm.n, rng),
                                                  device="cpu")))
    out = {}
    t0 = time.time()
    o1 = cap1.capability(targets, par, device="cpu")
    ref1 = {s: cap1.capability([(asm, plant.sample_latents(asm, 1, np.random.default_rng(0), plant.HET_OFF,
                                                             domain=P.point_domain(th, 1, "cpu"), device="cpu"))],
                               par, device="cpu")[0]["rho"].cpu().numpy()[0] for s, (asm, _) in zip(A5.SHAPES, targets)}
    for s, o in zip(A5.SHAPES, o1):
        out[f"{s}/T1/rho_dec"] = o["rho"].cpu().numpy() / ref1[s][None]
    for c in CONDS:
        eff = build.ABLATIONS[c]
        o2 = cap2.capability(targets, par, eff, seed=SEED0)
        for i, (s, (asm, lat), o) in enumerate(zip(A5.SHAPES, targets, o2)):
            out[f"{s}/{c}/rho_dec"] = o["rho"] / cap2.rho_ref(asm, par, "plated", eff)[None]
        print(c, f"{time.time() - t0:.0f}s", flush=True)
    # features + probe misfits
    for si, (s, (asm, lat)) in enumerate(zip(A5.SHAPES, targets)):
        S = build.episode_specs(asm, lat, par, build.ABLATIONS["all+motor"], SEED0 + 1000003 * si)
        base_mu = (lat["dom"]["mu"].numpy()[:, None] * lat["mu_mod"].numpy()[:, np.asarray(asm.wheel_module)])
        mut, mfp, mfn = _spec_features(S, asm.n)
        muf0 = 0.5 * (mfp + mfn)
        out[f"{s}/mu_fac"] = mut / base_mu
        cr = lat["dom"]["c_roller"].numpy()[:, None] * build.V_REF / S["N0"]
        out[f"{s}/drag_pos"], out[f"{s}/drag_neg"] = mfp / cr, mfn / cr
        out[f"{s}/ilim_frac"] = S["ilim"] / S["ts"]
        out[f"{s}/knee_frac"] = S["omk"] / (S["vb"] * S["wnl"])
        out[f"{s}/eth0"] = S["eth0"]
        out[f"{s}/mode"], out[f"{s}/g"] = lat["mode"].numpy(), lat["g"].numpy()
        lt = dict(lat, payload=None)
        for kind in ("v2", "clamp"):
            prng = np.random.default_rng([SEED0, si, 17 + ("v2", "clamp").index(kind)])
            tabs = np.stack([SR.probe_tables(kind, asm, par, prng)[0] for _ in range(n)])
            T = SR.probe_T(kind)
            R1 = plant.simulate_bucket([dict(asm=asm, lat=lt, cmd=tabs)], par, T, device="cpu")[0]
            for c in CONDS:
                R2 = cap2.simulate(asm, lat, par, tabs, T, build.ABLATIONS[c], seed=SEED0 + 1000003 * si)
                di, dd, _, _ = CL.misfit(R1["twist"].numpy().astype(float), R2["twist"],
                                         R1["duty"].numpy().astype(float), R2["duty"])
                out[f"{s}/{c}/D_imu_{kind}"], out[f"{s}/{c}/D_duty_{kind}"] = di, dd
        print(s, "probes", f"{time.time() - t0:.0f}s", flush=True)
    A5.OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(A5.OUT / "A5m_rows.npz", **out)


def _auc(a, b):
    return float((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean()) if len(a) and len(b) else None


def analyze():
    from cfcap.t1.capability import LEGS
    z = np.load(A5.OUT / "A5m_rows.npz")
    ltype = ["yaw" if l.startswith("yaw") else "diag" if l.startswith("diag") else "axis" for l in LEGS]
    rows = []
    for s in A5.SHAPES:
        mode, g = z[f"{s}/mode"], z[f"{s}/g"]
        r1 = z[f"{s}/T1/rho_dec"]
        for b in range(len(mode)):
            fw = np.nonzero(mode[b])[0]
            base = dict(shape=s, ep=b, n_fault=len(fw),
                        faults=json.dumps([[int(w), int(mode[b, w]), round(float(g[b, w]), 3)] for w in fw]),
                        min_g=float(g[b].min()), n_weak=int((mode[b] == 1).sum()), n_coast=int((mode[b] == 2).sum()),
                        n_brake=int((mode[b] == 3).sum()), payload_kg=0.0,
                        mu_fac=json.dumps(np.round(z[f"{s}/mu_fac"][b], 3).tolist()),
                        mu_fac_min=float(z[f"{s}/mu_fac"][b].min()), mu_fac_max=float(z[f"{s}/mu_fac"][b].max()),
                        drag_pos=json.dumps(np.round(z[f"{s}/drag_pos"][b], 3).tolist()),
                        drag_neg=json.dumps(np.round(z[f"{s}/drag_neg"][b], 3).tolist()),
                        ilim_frac_min=float(z[f"{s}/ilim_frac"][b].min()),
                        knee_frac_min=float(z[f"{s}/knee_frac"][b].min()), eth0_max=float(z[f"{s}/eth0"][b].max()))
            for c in CONDS:
                for k in ("D_imu_v2", "D_imu_clamp", "D_duty_v2", "D_duty_clamp"):
                    base[f"{c}.{k}"] = float(z[f"{s}/{c}/{k}"][b])
            for l, leg in enumerate(LEGS):
                r = dict(base, leg=leg, leg_type=ltype[l], rho_dec_T1=float(r1[b, l]))
                for c in CONDS:
                    r2 = float(z[f"{s}/{c}/rho_dec"][b, l])
                    r[f"{c}.rho_dec"] = r2
                    for t in A5.TAUS:
                        r[f"{c}.flip_{t}"] = int((r1[b, l] >= t) != (r2 >= t))
                for t in A5.TAUS:
                    r[f"margin_{t}"] = float(abs(r1[b, l] - t))
                rows.append(r)
    A5.RES.mkdir(parents=True, exist_ok=True)
    with open(A5.RES / "A5m_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # flips by shape x leg type x condition (decisions = rows x 2 taus)
    summ = []
    for c in CONDS:
        for s in A5.SHAPES + ("children",):
            for lt in ("axis", "diag", "yaw", "any"):
                sel = [r for r in rows if (r["shape"] == s or (s == "children" and r["shape"] != "single"))
                       and (lt == "any" or r["leg_type"] == lt)]
                fl = np.array([[r[f"{c}.flip_{t}"] for t in A5.TAUS] for r in sel])
                d = np.array([r[f"{c}.rho_dec"] - r["rho_dec_T1"] for r in sel])
                faulted = np.array([r["n_fault"] > 0 for r in sel])
                summ.append(dict(cond=c, shape=s, leg_type=lt, decisions=fl.size, flips=int(fl.sum()),
                                 flip=float(fl.mean()), flip_faulted=float(fl[faulted].mean()) if faulted.any() else 0,
                                 flip_healthy=float(fl[~faulted].mean()) if (~faulted).any() else 0,
                                 share_d05=float((np.abs(d) > 0.05).mean()), d_q01=float(np.percentile(d, 1)),
                                 d_q50=float(np.median(d)), d_q99=float(np.percentile(d, 99))))
    with open(A5.RES / "A5m_flips.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    # clamp / v2 misfit: flipped vs faulted non-flipped episodes, per shape group
    cl = []
    for c in CONDS:
        for grp, shapes in (("single", ("single",)), ("pair", ("pair",)), ("3+mod", ("L3", "square", "2x3")),
                            ("all", A5.SHAPES)):
            eps = {}
            for r in rows:
                if r["shape"] in shapes:
                    key = (r["shape"], r["ep"])
                    e = eps.setdefault(key, dict(flip=0, nf=r["n_fault"], **{k: r[f"{c}.{k}"] for k in
                                                ("D_imu_v2", "D_imu_clamp", "D_duty_v2", "D_duty_clamp")}))
                    e["flip"] |= max(r[f"{c}.flip_{t}"] for t in A5.TAUS)
            E = list(eps.values())
            fl = np.array([e["flip"] for e in E], bool)
            nf = np.array([e["nf"] > 0 for e in E])
            row = dict(cond=c, group=grp, episodes=len(E), flipped=int(fl.sum()))
            for k in ("D_imu_v2", "D_imu_clamp", "D_duty_v2", "D_duty_clamp"):
                d = np.array([e[k] for e in E])
                row[f"auc_{k}"] = _auc(d[fl], d[~fl & nf])
            cl.append(row)
    with open(A5.RES / "A5m_clamp.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cl[0]))
        w.writeheader()
        w.writerows(cl)
    for r in summ:
        if r["leg_type"] == "any":
            print(r["cond"], r["shape"], r["flips"], f"{r['flip']:.4f}", f"faulted {r['flip_faulted']:.4f}",
                  f"healthy {r['flip_healthy']:.4f}", f"d05 {r['share_d05']:.4f}")
    for r in cl:
        print(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "analyze"))
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    run(a.n) if a.cmd == "run" else analyze()
