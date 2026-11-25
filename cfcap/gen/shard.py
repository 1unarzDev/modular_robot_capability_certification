"""Stage B shards: columnar npz (np.savez_compressed) of episodes, assemblies, modules, wheels, candidates and
candidate modules, plus per-shard JSON sidecars and a split-level manifest.json. Schema: docs/board/generator.md.

Layout: <out>/<plate>/<split>/shard_<k:05d>.npz (+ .json sidecar); manifest.json is rebuilt from the sidecars after
every shard, so shards from many boxes can be merged by copying the directories together. A shard whose sidecar
exists and whose npz checksum matches is skipped (resume). Episode indices of shard k are [k*n, (k+1)*n) with n the
split's fixed episodes_per_shard (a mismatch is refused).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import time

import numpy as np

from cfcap.gen import episode as E
from cfcap.gen import families as F
from cfcap.gen import simrun as S
from cfcap.t1 import capability as cap

SCHEMA = "cfcap.gen/v1"


def config(kinds):
    return dict(schema=SCHEMA, seed0=E.SEED0, het=E.HET.__dict__, max_child=E.MAX_CHILD, n_merge=E.N_MERGE,
                fault_modes=["weak", "coast", "brake"], fault_mode_p=E.FAULT_MODE_P, g_range=E.G_RANGE,
                p_k_parent=E.P_K_PARENT, p_k_other=E.P_K_OTHER, parent_size_w=E.PARENT_SIZE_W,
                other_size_w=E.OTHER_SIZE_W, rules={k: dict(v.__dict__) for k, v in E.RULES.items()},
                dom_keys=E.DOM_KEYS, legs=cap.LEGS, t_leg=cap.T_LEG, leg_window=cap.window(cap.T_LEG),
                topologies=F.TOPOLOGIES, fam_names=F.FAM_NAMES, kinds=list(kinds),
                probe_T={k: S.probe_T(k) for k in kinds}, obs_hz=S.REC_HZ, cpr=S.P.CPR, splits=E.SPLITS,
                plates=E.PLATES, kind_codes=E.KIND)


def _assemble(eps, obs, labs, kinds):
    """Columnar arrays of a list of episodes (see the schema)."""
    c = {k: [] for k in (
        "ep_index", "ep_split", "ep_plate", "ep_dom", "ep_payload", "ep_payload_host", "ep_n_asm", "ep_n_cand",
        "asm_ep", "asm_role", "asm_family", "asm_topo", "asm_fam_split", "asm_n_mod", "asm_gyro_bias", "asm_acc_bias",
        "asm_char_length", "mod_asm", "mod_cell", "mod_yaw", "mod_vbatt_off", "mod_mu",
        "wheel_mode", "wheel_g", "wheel_stall", "wheel_noload", "wheel_yaw_err", "wheel_load_z",
        "cand_ep", "cand_kind", "cand_family", "cand_topo", "cand_fam_split", "cand_n_mod", "cand_payload",
        "cand_n_fault", "cand_char_length", "cand_rho", "cand_rho_healthy", "cand_rho_rel", "cand_drift",
        "cm_mod", "cm_cell", "cm_yaw", "cw_load_z")}
    for k in kinds:
        for f in ("imu", "cmd", "amp", "lead", "duty", "dcnt") + (("ref",) if k == "tug" else ()):
            c[f"obs_{k}_{f}"] = []
    a_row = m_row = 0
    for ei, ep in enumerate(eps):
        e_row = len(c["ep_index"])
        c["ep_index"].append(ep.index)
        c["ep_split"].append(E.SPLITS[ep.split])
        c["ep_plate"].append(E.PLATES.index(ep.plate))
        c["ep_dom"].append([ep.dom[k] for k in E.DOM_KEYS])
        pp = E.parent_payload(ep)
        c["ep_payload"].append(pp if pp is not None else (0.0, 0.0, 0.0))
        c["ep_payload_host"].append(ep.payload[1] if ep.payload is not None else -1)
        c["ep_n_asm"].append(len(ep.asms))
        c["ep_n_cand"].append(len(ep.cands))
        mod_row0 = {}
        for a in ep.asms:
            mod_row0[a.role] = m_row
            c["asm_ep"].append(e_row)
            c["asm_role"].append(a.role)
            c["asm_family"].append(a.family)
            c["asm_topo"].append(F.TOPOLOGIES.index(a.topo))
            c["asm_fam_split"].append(a.fam_split)
            c["asm_n_mod"].append(a.m)
            c["asm_gyro_bias"].append(a.gyro_bias)
            c["asm_acc_bias"].append(a.acc_bias)
            c["asm_char_length"].append(a.asm.char_length())
            for k, (_, i, j, y) in enumerate(a.asm.modules):
                c["mod_asm"].append(a_row)
                c["mod_cell"].append((i, j))
                c["mod_yaw"].append(y)
            c["mod_vbatt_off"].append(a.vbatt_off)
            c["mod_mu"].append(a.mu_mod)
            for f, key in (("wheel_mode", "mode"), ("wheel_g", "g"), ("wheel_stall", "stall_w"),
                           ("wheel_noload", "noload_w"), ("wheel_yaw_err", "yaw_err"), ("wheel_load_z", "load_z")):
                c[f].append(getattr(a, key))
            for k in kinds:
                o = obs[k][ei][a.role]
                c[f"obs_{k}_imu"].append(o["imu"])
                c[f"obs_{k}_cmd"].append(o["cmd"])
                c[f"obs_{k}_amp"].append(o["amp"])
                c[f"obs_{k}_lead"].append(o["lead"])
                c[f"obs_{k}_duty"].append(o["duty"].T)
                c[f"obs_{k}_dcnt"].append(o["dcnt"].T)
                if k == "tug":
                    c["obs_tug_ref"].append(o["ref"].T)
            a_row += 1
            m_row += a.m
        lab = labs[ei]
        for ci, cd in enumerate(ep.cands):
            c["cand_ep"].append(e_row)
            c["cand_kind"].append(cd.kind)
            c["cand_family"].append(cd.family)
            c["cand_topo"].append(F.TOPOLOGIES.index(cd.topo))
            c["cand_fam_split"].append(cd.fam_split)
            c["cand_n_mod"].append(len(cd.modules))
            c["cand_payload"].append(cd.payload if cd.payload is not None else (0.0, 0.0, 0.0))
            nf = sum(int((ep.asms[r].mode[4 * k:4 * k + 4] != 0).sum()) for r, k in cd.modules)
            c["cand_n_fault"].append(nf)
            c["cand_char_length"].append(cd.asm.char_length())
            for f in ("rho", "rho_healthy", "rho_rel", "drift"):
                c[f"cand_{f}"].append(lab[f][ci])
            for (r, k), (_, i, j, y) in zip(cd.modules, cd.asm.modules):
                c["cm_mod"].append(mod_row0[r] + k)
                c["cm_cell"].append((i, j))
                c["cm_yaw"].append(y)
            c["cw_load_z"].append(cd.load_z)
    D = dict(
        ep_index=np.array(c["ep_index"], np.int64), ep_split=np.array(c["ep_split"], np.int8),
        ep_plate=np.array(c["ep_plate"], np.int8), ep_dom=np.array(c["ep_dom"], np.float32),
        ep_payload=np.array(c["ep_payload"], np.float32), ep_payload_host=np.array(c["ep_payload_host"], np.int8),
        asm_ep=np.array(c["asm_ep"], np.int32), asm_role=np.array(c["asm_role"], np.int8),
        asm_family=np.array(c["asm_family"], np.int64), asm_topo=np.array(c["asm_topo"], np.int8),
        asm_fam_split=np.array(c["asm_fam_split"], np.int8), asm_gyro_bias=np.array(c["asm_gyro_bias"], np.float32),
        asm_acc_bias=np.array(c["asm_acc_bias"], np.float32),
        asm_char_length=np.array(c["asm_char_length"], np.float32),
        mod_asm=np.array(c["mod_asm"], np.int32), mod_cell=np.array(c["mod_cell"], np.int8),
        mod_yaw=np.array(c["mod_yaw"], np.int8),
        mod_vbatt_off=np.concatenate(c["mod_vbatt_off"]).astype(np.float32),
        mod_mu=np.concatenate(c["mod_mu"]).astype(np.float32),
        wheel_mode=np.concatenate(c["wheel_mode"]).astype(np.int8),
        cand_ep=np.array(c["cand_ep"], np.int32), cand_kind=np.array(c["cand_kind"], np.int8),
        cand_family=np.array(c["cand_family"], np.int64), cand_topo=np.array(c["cand_topo"], np.int8),
        cand_fam_split=np.array(c["cand_fam_split"], np.int8),
        cand_payload=np.array(c["cand_payload"], np.float32).reshape(-1, 3),
        cand_n_fault=np.array(c["cand_n_fault"], np.int8),
        cand_char_length=np.array(c["cand_char_length"], np.float32),
        cm_mod=np.array(c["cm_mod"], np.int32), cm_cell=np.array(c["cm_cell"], np.int8).reshape(-1, 2),
        cm_yaw=np.array(c["cm_yaw"], np.int8), cw_load_z=np.concatenate(c["cw_load_z"]).astype(np.float32))
    for f in ("wheel_g", "wheel_stall", "wheel_noload", "wheel_yaw_err", "wheel_load_z"):
        D[f] = np.concatenate(c[f]).astype(np.float32)
    for f in ("rho", "rho_healthy", "rho_rel", "drift"):
        D[f"cand_{f}"] = np.array(c[f"cand_{f}"], np.float32).reshape(-1, len(cap.LEGS))
    off = lambda cnt: np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
    D["ep_asm_off"] = off(c["ep_n_asm"])
    D["ep_cand_off"] = off(c["ep_n_cand"])
    D["asm_mod_off"] = off(c["asm_n_mod"])
    D["cand_cm_off"] = off(c["cand_n_mod"])
    for k in kinds:
        D[f"obs_{k}_imu"] = np.stack(c[f"obs_{k}_imu"]).astype(np.float16)
        D[f"obs_{k}_cmd"] = np.stack(c[f"obs_{k}_cmd"]).astype(np.float16)
        D[f"obs_{k}_amp"] = np.array(c[f"obs_{k}_amp"], np.float32)
        D[f"obs_{k}_lead"] = np.array(c[f"obs_{k}_lead"], np.float32)
        D[f"obs_{k}_duty"] = np.concatenate(c[f"obs_{k}_duty"]).astype(np.float16)
        dc = np.concatenate(c[f"obs_{k}_dcnt"])
        assert np.abs(dc).max() < 32767
        D[f"obs_{k}_dcnt"] = dc.astype(np.int16)
        if k == "tug":
            D["obs_tug_ref"] = np.concatenate(c["obs_tug_ref"]).astype(np.float16)
    return D


def generate(plate, split, index0, n, kinds=("v2",), engine="compile", block=256, max_wheels=1 << 17,
             skip_healthy_twin=True, log=print):
    """Columnar arrays + timing stats for episodes [index0, index0 + n)."""
    parts, st = [], dict(make_s=0.0, probe_s=0.0, label_s=0.0, assemble_s=0.0)
    pst, lst = {}, {}
    for b0 in range(index0, index0 + n, block):
        t0 = time.perf_counter()
        eps = [E.make_episode(plate, split, i) for i in range(b0, min(index0 + n, b0 + block))]
        t1 = time.perf_counter()
        obs = S.probe_episodes(eps, kinds, engine=engine, stats=pst)
        t2 = time.perf_counter()
        labs = S.label_episodes(eps, engine=engine, max_wheels=max_wheels, skip_healthy_twin=skip_healthy_twin,
                                stats=lst)
        t3 = time.perf_counter()
        parts.append(_assemble(eps, obs, labs, kinds))
        t4 = time.perf_counter()
        st["make_s"] += t1 - t0
        st["probe_s"] += t2 - t1
        st["label_s"] += t3 - t2
        st["assemble_s"] += t4 - t3
        if log:
            log(f"[gen] {plate}/{split} episodes {b0}..{b0 + len(eps) - 1}: make {t1 - t0:.1f}s probe {t2 - t1:.1f}s "
                f"label {t3 - t2:.1f}s")
    st["probe"], st["label"] = pst, lst
    return concat(parts), st


_INDEX_COLS = dict(asm_ep="ep", mod_asm="asm", cand_ep="ep", cm_mod="mod")
_OFF_COLS = dict(ep_asm_off="asm", ep_cand_off="cand", asm_mod_off="mod", cand_cm_off="cm")


def concat(parts):
    """Concatenate columnar parts, shifting row indices and offsets."""
    if len(parts) == 1:
        return parts[0]
    out = {}
    base = dict(ep=0, asm=0, mod=0, cand=0, cm=0)
    rows = lambda p, t: len(p[{"ep": "ep_index", "asm": "asm_ep", "mod": "mod_asm", "cand": "cand_ep",
                                "cm": "cm_mod"}[t]])
    acc = {k: [] for k in parts[0]}
    for p in parts:
        for k, v in p.items():
            if k in _INDEX_COLS:
                acc[k].append(v + base[_INDEX_COLS[k]])
            elif k in _OFF_COLS:
                acc[k].append((v[1:] if acc[k] else v) + base[_OFF_COLS[k]])
            else:
                acc[k].append(v)
        for t in base:
            base[t] += rows(p, t)
    for k, v in acc.items():
        out[k] = np.concatenate(v)
    return out


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def shard_paths(out, plate, split, k):
    d = pathlib.Path(out) / plate / split
    return d, d / f"shard_{k:05d}.npz", d / f"shard_{k:05d}.json"


def is_done(out, plate, split, k):
    _, npz, side = shard_paths(out, plate, split, k)
    if not (npz.exists() and side.exists()):
        return False
    try:
        return json.loads(side.read_text())["sha256"] == _sha(npz)
    except Exception:
        return False


def write_manifest(d, kinds):
    sides = sorted(d.glob("shard_*.json"))
    shards = [json.loads(s.read_text()) for s in sides]
    eps = {s["episodes"] for s in shards}
    man = dict(schema=SCHEMA, plate=shards[0]["plate"], split=shards[0]["split"],
               episodes_per_shard=shards[0]["episodes"] if len(eps) == 1 else sorted(eps),
               n_shards=len(shards), n_episodes=sum(s["episodes"] for s in shards),
               n_candidates=sum(s["n_cand"] for s in shards), bytes=sum(s["bytes"] for s in shards),
               config=config(kinds), shards=shards)
    tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(man, indent=1, default=str))
    os.replace(tmp, d / "manifest.json")
    return man


def write_shard(out, plate, split, k, n, kinds=("v2",), engine="compile", block=256, force=False, log=print, **kw):
    """Generate and write shard k (episodes [k n, (k+1) n)); skip when already complete. Returns the sidecar dict."""
    d, npz, side = shard_paths(out, plate, split, k)
    d.mkdir(parents=True, exist_ok=True)
    for s in d.glob("shard_*.json"):
        prev = json.loads(s.read_text())
        if prev["episodes"] != n or prev["kinds"] != list(kinds):
            raise ValueError(f"{s}: existing shards use episodes={prev['episodes']} kinds={prev['kinds']}")
    if not force and is_done(out, plate, split, k):
        if log:
            log(f"[gen] skip {npz} (done)")
        return json.loads(side.read_text())
    t0 = time.perf_counter()
    D, st = generate(plate, split, k * n, n, kinds, engine, block, log=log, **kw)
    t1 = time.perf_counter()
    tmp = npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **D)
    os.replace(tmp, npz)
    t2 = time.perf_counter()
    import torch
    info = dict(file=npz.name, plate=plate, split=split, shard=k, index0=k * n, episodes=n, kinds=list(kinds),
                n_cand=int(len(D["cand_ep"])), n_asm=int(len(D["asm_ep"])), n_wheel=int(len(D["wheel_mode"])),
                bytes=npz.stat().st_size, sha256=_sha(npz), gen_seconds=round(t1 - t0, 2),
                write_seconds=round(t2 - t1, 2), stats=st, engine=engine, host=platform.node(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                job=os.environ.get("CRANE_JOB_ID"), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = side.with_suffix(".tmp")
    tmp.write_text(json.dumps(info, indent=1, default=float))
    os.replace(tmp, side)
    write_manifest(d, kinds)
    return info


def load(path):
    """All arrays of a shard as a dict (np.load, eager)."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def episode(D, e):
    """Row views of episode e of a loaded shard: dict(ep, asms=[...], cands=[...]) with plain numpy slices."""
    a0, a1 = D["ep_asm_off"][e:e + 2]
    c0, c1 = D["ep_cand_off"][e:e + 2]
    asms = []
    for a in range(a0, a1):
        m0, m1 = D["asm_mod_off"][a:a + 2]
        asms.append(dict(role=int(D["asm_role"][a]), modules=slice(m0, m1), wheels=slice(4 * m0, 4 * m1),
                         cells=D["mod_cell"][m0:m1], yaws=D["mod_yaw"][m0:m1]))
    cands = []
    for c in range(c0, c1):
        k0, k1 = D["cand_cm_off"][c:c + 2]
        mods = D["cm_mod"][k0:k1]
        cands.append(dict(kind=int(D["cand_kind"][c]), mod_rows=mods,
                          wheel_rows=(4 * mods[:, None] + np.arange(4)).reshape(-1), cells=D["cm_cell"][k0:k1],
                          rho_rel=D["cand_rho_rel"][c]))
    return dict(index=int(D["ep_index"][e]), asms=asms, cands=cands)
