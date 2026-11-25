"""Stage B episodes: parent assembly, other assemblies, latents, payload and the candidate set, as a pure function of
(plate, split, index). Everything here is numpy on the CPU; `simrun` turns episodes into rollouts.

Episode (one decision situation):
  * plate state (plated | plateless): params mass and the identified theta of that plate state (never pooled);
  * parent: a fixed polyomino drawn uniformly within the split's family set (`families`), module yaws U{0..3};
  * 1-2 other assemblies (1-3 modules, train families) whose latents are sampled too (merge partners);
  * domain: one capxfer DR draw around theta (P.theta_dr: floor mu, slip s0, roller drag, latency, motor scales,
    v_batt, back-drive torque tau_bd, cg offset, ...), shared by every assembly of the episode (same floor, firmware);
  * heterogeneity HET_FULL (cfcap.t1.plant): per-wheel +-8 % and per-module +-4 % stall / no-load multipliers,
    per-module v_batt offset +-0.3 V and mu factor +-10 %, per-wheel mounting-yaw error, random null-space normal-load
    share (redrawn for every candidate: the indeterminate split changes with geometry), IMU biases, 1 % wheel-sample
    dropout;
  * faults per wheel: ok / weak(g ~ U(0.3, 0.95)) / coast / brake; parent 0/1/2/3 faults w.p. 0.2/0.4/0.3/0.1 and
    others 0/1/2 w.p. 0.5/0.4/0.1, at most one faulted wheel per module (H6 breaks exactly this);
  * payload: none w.p. 0.5, else U(0.5, 2.5) kg on a random parent module's deck (offset U(-0.08, 0.08) m per axis);
    H7: U(3, 6) kg always. A candidate carries it iff it keeps the host module;
  * candidates: the parent itself ("self"), every connected child (capped at MAX_CHILD, random subset), and up to
    N_MERGE merges per other assembly (piece = the parent or one of its connected children, the whole other, a random
    face-touching placement), all restricted to the split's allowed candidate families and sizes.

Splits (episode stream, parent family set, allowed candidate families):
  train / cal / test  FAM_TRAIN parents N 1-6, candidates FAM_TRAIN N <= 6       (disjoint seed streams; cal ~ test)
  H3                  FAM_H3 parents N 4-6, candidates any family N <= 6       capplan H3 (unseen topology)
  H4                  FAM_H4 parents N 7-10, candidates any family N <= 10     capplan H4a (larger assemblies)
  H6                  train morphology rules; parent faults: >= 2 wheels of one module (+ maybe 1 elsewhere)
                      capplan H6 (unseen failure combinations)
  H7                  train morphology rules; payload regime U(3, 6) kg                capplan H7 (task family)
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import torch

from cfcap.gen import families as F
from cfcap.t0 import candidates as Cand
from cfcap.t1 import params as P
from cfcap.t1 import plant
from capxfer import assembly as A
from capxfer import sim

SEED0 = 92511
PLATES = ("plated", "plateless")
SPLITS = {"train": 0, "cal": 1, "test": 2, "H3": 3, "H4": 4, "H6": 6, "H7": 7}
SPLIT_NAMES = {v: k for k, v in SPLITS.items()}
KIND = {"self": 0, "child": 1, "merge": 2}
DOM_KEYS = ("mu", "s0", "c_roller", "crr", "tau_bd", "latency", "stall_scale", "noload_scale", "mass_scale",
            "izz_scale", "radius_scale", "kp_scale", "ki_scale", "v_batt", "jeff_scale", "cg_dx", "cg_dy")
HET = plant.HET_FULL
MAX_CHILD = 40
N_MERGE = 4
FAULT_MODES = (sim.WEAK, sim.COAST, sim.BRAKE)
FAULT_MODE_P = (0.5, 0.25, 0.25)
G_RANGE = (0.3, 0.95)
P_K_PARENT = (0.2, 0.4, 0.3, 0.1)
P_K_OTHER = (0.5, 0.4, 0.1)
PARENT_SIZE_W = {1: 0.05, 2: 0.1, 3: 0.15, 4: 0.2, 5: 0.2, 6: 0.3}
OTHER_SIZE_W = {1: 0.5, 2: 0.3, 3: 0.2}


@dataclass(frozen=True)
class SplitRule:
    parent_split: int
    parent_size_w: dict
    cand_splits: tuple
    cand_max_n: int
    faults: str = "train"         # "train" | "h6"
    payload: str = "train"        # "train" | "h7"


RULES = {
    "train": SplitRule(F.FAM_TRAIN, PARENT_SIZE_W, (F.FAM_TRAIN,), 6),
    "H3": SplitRule(F.FAM_H3, {4: 0.2, 5: 0.3, 6: 0.5}, (F.FAM_TRAIN, F.FAM_H3), 6),
    "H4": SplitRule(F.FAM_H4, {7: 0.3, 8: 0.3, 9: 0.2, 10: 0.2}, (F.FAM_TRAIN, F.FAM_H3, F.FAM_H4), 10),
    "H6": SplitRule(F.FAM_TRAIN, {k: w for k, w in PARENT_SIZE_W.items() if k >= 2}, (F.FAM_TRAIN,), 6, faults="h6"),
    "H7": SplitRule(F.FAM_TRAIN, PARENT_SIZE_W, (F.FAM_TRAIN,), 6, payload="h7"),
}
RULES["cal"] = RULES["test"] = RULES["train"]


@functools.lru_cache(maxsize=None)
def params(plate):
    return P.base_params(plate)


@functools.lru_cache(maxsize=None)
def domain_dr(plate):
    return P.theta_dr(P.load_theta(plate))


@dataclass
class AsmSpec:
    role: int                  # 0 parent, 1.. other assemblies
    asm: object                # capxfer Assembly
    family: int
    topo: str
    fam_split: int
    vbatt_off: np.ndarray      # (m,)
    mu_mod: np.ndarray         # (m,)
    mode: np.ndarray           # (n,) int
    g: np.ndarray              # (n,)
    stall_w: np.ndarray        # (n,)
    noload_w: np.ndarray       # (n,)
    yaw_err: np.ndarray        # (n,)
    load_z: np.ndarray         # (n,) null-space load share during this assembly's own probes
    gyro_bias: float
    acc_bias: np.ndarray       # (2,)

    @property
    def m(self):
        return len(self.asm.modules)


@dataclass
class CandSpec:
    kind: int
    asm: object
    modules: list              # [(asm role, module index in that assembly)] in the candidate's module order
    family: int
    topo: str
    fam_split: int
    load_z: np.ndarray         # (n,)
    payload: tuple | None      # (mass, x, y) in the candidate frame, or None


@dataclass
class Episode:
    plate: str
    split: str
    index: int
    dom: dict                  # DOM_KEYS -> float
    asms: list                 # [AsmSpec], parent first
    payload: tuple | None      # (mass, host parent module, off_x, off_y)
    cands: list = field(default_factory=list)

    @property
    def parent(self):
        return self.asms[0]


def episode_rng(plate, split, index, stream=0):
    return np.random.default_rng([SEED0, PLATES.index(plate), SPLITS[split], int(index), int(stream)])


def cells(asm):
    return [(int(i), int(j)) for _, i, j, _ in asm.modules]


def build(name, cells_, yaws, ids, plate):
    a = A.build(name, [(ids[k], int(c[0]), int(c[1]), int(yaws[k])) for k, c in enumerate(cells_)], params(plate))
    return a


def sample_faults(rng, m, p_k, h6=False):
    """Per-wheel mode, g for an m-module assembly: at most one faulted wheel per module (train prior); h6: one module
    with 2 faulted wheels plus, w.p. 0.5, one more faulted wheel on another module."""
    n = 4 * m
    mode = np.zeros(n, np.int64)
    g = np.ones(n, np.float32)
    wheels = []
    if h6:
        k0 = int(rng.integers(m))
        wheels += [4 * k0 + int(c) for c in rng.choice(4, 2, replace=False)]
        if m > 1 and rng.uniform() < 0.5:
            k1 = int(rng.choice([k for k in range(m) if k != k0]))
            wheels.append(4 * k1 + int(rng.integers(4)))
    else:
        k = min(int(rng.choice(len(p_k), p=p_k)), m)
        for mod in rng.choice(m, k, replace=False):
            wheels.append(4 * int(mod) + int(rng.integers(4)))
    for w in wheels:
        mode[w] = FAULT_MODES[int(rng.choice(3, p=FAULT_MODE_P))]
        if mode[w] == sim.WEAK:
            g[w] = rng.uniform(*G_RANGE)
    return mode, g


def _asm_spec(role, asm, rng, dom_t, p_k, h6=False):
    lat = plant.sample_latents(asm, 1, rng, HET, domain=dom_t, device="cpu")
    mode, g = sample_faults(rng, len(asm.modules), p_k, h6)
    c = cells(asm)
    t = lambda k: lat[k][0].numpy().astype(np.float32)
    return AsmSpec(role=role, asm=asm, family=F.family_hash(c), topo=F.topology(c), fam_split=F.family_split(c),
                   vbatt_off=t("vbatt_off"), mu_mod=t("mu_mod"), mode=mode, g=g, stall_w=t("stall_w"),
                   noload_w=t("noload_w"), yaw_err=t("yaw_err"), load_z=t("load_z"),
                   gyro_bias=float(lat["gyro_bias"][0]), acc_bias=t("acc_bias"))


def _random_asm(rng, size_w, split, plate, name):
    sh = F.sample_shape(rng, size_w, split)
    m = len(sh)
    yaws = rng.integers(4, size=m)
    ids = [str(x) for x in rng.choice(("M1", "M2", "M3", "M4"), m)]
    return build(name, sh, yaws, ids, plate)


def _cand(kind, asm, modules, origin_cells, rng, ep):
    """CandSpec with a fresh null-space load share and the payload moved into the candidate frame."""
    c = cells(asm)
    pitch = float(asm.meta["pitch"])
    pay = None
    if ep.payload is not None:
        mass, host, ox, oy = ep.payload
        if (0, host) in modules:
            hc = np.array(cells(ep.parent.asm)[host], float) * pitch + (ox, oy)
            origin = np.array(origin_cells, float).mean(0) * pitch
            pay = (float(mass), float(hc[0] - origin[0]), float(hc[1] - origin[1]))
    lz = (rng.normal(size=asm.n) * HET.load_split).astype(np.float32)
    return CandSpec(kind, asm, list(modules), F.family_hash(c), F.topology(c), F.family_split(c), lz, pay)


def make_episode(plate, split, index, n_other=None):
    rule = RULES[split]
    rng = episode_rng(plate, split, index)
    par = params(plate)
    parent = _random_asm(rng, rule.parent_size_w, rule.parent_split, plate, "parent")
    k_other = int(rng.integers(1, 3)) if n_other is None else int(n_other)
    others = [_random_asm(rng, OTHER_SIZE_W, F.FAM_TRAIN, plate, f"other{r}") for r in range(1, k_other + 1)]
    dom_t = sim.sample_domain(1, rng, domain_dr(plate), device="cpu")
    dom = {k: float(dom_t[k][0]) for k in DOM_KEYS}
    asms = [_asm_spec(0, parent, rng, dom_t, P_K_PARENT, h6=(rule.faults == "h6"))]
    asms += [_asm_spec(r, a, rng, dom_t, P_K_OTHER) for r, a in enumerate(others, 1)]
    payload = None
    if rule.payload == "h7" or rng.uniform() < 0.5:
        mass = rng.uniform(3.0, 6.0) if rule.payload == "h7" else rng.uniform(0.5, 2.5)
        host = int(rng.integers(len(parent.modules)))
        payload = (float(mass), host, float(rng.uniform(-0.08, 0.08)), float(rng.uniform(-0.08, 0.08)))
    ep = Episode(plate, split, int(index), dom, asms, payload)
    ep.cands = candidates(ep, rng, rule, par)
    return ep


def _allowed(c, rule):
    return len(c) <= rule.cand_max_n and F.family_split(c) in rule.cand_splits


def candidates(ep, rng, rule, par):
    pa = ep.parent.asm
    pc = cells(pa)
    m = len(pc)
    out = [_cand(KIND["self"], pa, [(0, k) for k in range(m)], pc, rng, ep)] if _allowed(pc, rule) else []
    subs = Cand.connected_subsets(pc)
    kids = [s for s in subs if _allowed([pc[k] for k in s], rule)]
    if len(kids) > MAX_CHILD:
        keep = np.sort(rng.choice(len(kids), MAX_CHILD, replace=False))
        kids = [kids[i] for i in keep]
    for s in kids:
        asm = A.build(f"{pa.name}/c{'-'.join(map(str, s))}", [pa.modules[k] for k in s], par)
        out.append(_cand(KIND["child"], asm, [(0, k) for k in s], [pc[k] for k in s], rng, ep))
    pieces = [tuple(range(m))] + subs
    seen = set()
    for o in ep.asms[1:]:
        oc = cells(o.asm)
        got, tries = 0, 0
        while got < N_MERGE and tries < 8 * N_MERGE:
            tries += 1
            piece = pieces[int(rng.integers(len(pieces)))]
            if len(piece) + len(oc) > rule.cand_max_n:
                continue
            pl = Cand.merge_placements([pc[k] for k in piece], oc)
            if not pl:
                continue
            t = pl[int(rng.integers(len(pl)))]
            mc = [pc[k] for k in piece] + [(x + t[0], y + t[1]) for x, y in oc]
            key = (o.role, piece, t)
            if key in seen or not _allowed(mc, rule):
                continue
            seen.add(key)
            mc_ = Cand.merge(pa, piece, o.asm, t, par)
            mods = [(0, k) for k in piece] + [(o.role, j) for j in range(o.m)]
            out.append(_cand(KIND["merge"], mc_.asm, mods, mc, rng, ep))
            got += 1
    return out


# ----------------------------------------------------------------------------- latents for the plant
def domain_tensors(dom, B=1):
    d = {k: torch.full((B,), float(v), dtype=torch.float32) for k, v in dom.items()}
    d["wheel_yaw_err_std"] = 0.0
    return d


def plant_latents(ep, modules, load_z, payload, gyro=(0.0, (0.0, 0.0)), B=1, healthy=False):
    """cfcap.t1.plant latent dict (CPU tensors, B identical rows) of a morphology made of `modules`
    [(asm role, module index)] of episode `ep`."""
    w = np.concatenate([np.arange(4 * k, 4 * k + 4) for _, k in modules])
    rw = np.repeat([r for r, _ in modules], 4)
    pick = lambda key: np.array([getattr(ep.asms[r], key)[i] for r, i in zip(rw, w)])
    pickm = lambda key: np.array([getattr(ep.asms[r], key)[k] for r, k in modules])
    T = lambda a, dt=torch.float32: torch.as_tensor(np.broadcast_to(a, (B,) + np.shape(a)).copy(), dtype=dt)
    mode, g = pick("mode"), pick("g")
    if healthy:
        mode, g = np.zeros_like(mode), np.ones_like(g)
    lat = dict(dom=domain_tensors(ep.dom, B), mode=T(mode, torch.long), g=T(g), stall_w=T(pick("stall_w")),
               noload_w=T(pick("noload_w")), vbatt_off=T(pickm("vbatt_off")), mu_mod=T(pickm("mu_mod")),
               yaw_err=T(pick("yaw_err")), load_z=T(load_z), gyro_bias=T(np.float32(gyro[0])),
               acc_bias=T(np.asarray(gyro[1], np.float32)), drop_p=float(HET.drop_p), payload=None)
    if payload is not None:
        lat["payload"] = dict(mass=T(np.float32(payload[0])), x=T(np.float32(payload[1])),
                              y=T(np.float32(payload[2])))
    return lat


def cat_latents(lats):
    """Row-concatenate plant latent dicts of the same morphology (payload all-or-none)."""
    out = {}
    for k, v in lats[0].items():
        if k == "dom":
            out[k] = {kk: (torch.cat([l[k][kk] for l in lats]) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        elif k == "payload":
            out[k] = None if v is None else {kk: torch.cat([l[k][kk] for l in lats]) for kk in v}
        elif torch.is_tensor(v):
            out[k] = torch.cat([l[k] for l in lats])
        else:
            out[k] = v
    return out


def parent_payload(ep):
    """Payload (mass, x, y) in the parent frame, or None."""
    if ep.payload is None:
        return None
    mass, host, ox, oy = ep.payload
    pa = ep.parent.asm
    pitch = float(pa.meta["pitch"])
    c = np.array(cells(pa), float) * pitch
    p = c[host] + (ox, oy) - c.mean(0)
    return (mass, float(p[0]), float(p[1]))
