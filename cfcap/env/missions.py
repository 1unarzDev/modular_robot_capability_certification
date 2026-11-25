"""Missions: a route through a procedural building (`buildings`) plus loads, turned into per-segment requirements on
the 10 decision legs (`cfcap.t1.capability.LEGS`), footprint and load feasibility, and closed-loop T1 truth labels.

Segments (in route order; each has a leg, an absolute speed requirement, the floor-mu factor and payload state at its
arc position, the mission phase, and a door width or None):
  * straight (axis leg) -> surge+ (or surge- for a reversal instead of a 180-deg rotation, or sway+/- when the body
    strafes a 90-deg corner instead of rotating);
  * diagonal leg (45 deg off the body heading) -> diag++ / diag+- / diag-+ / diag--, no rotation;
  * turn (body rotation at a corner, radius r) -> yaw+/yaw- at the turn rate omega = v_turn / r;
  * door (every door, including the narrow passage) -> a short sway+/- alignment, and the footprint check.
Requirement units: an absolute speed (m/s or rad/s); in rho_dec units (the decision label rho_dec = rho_abs / rho_ref,
the fixed nominal T1 reference `cfcap.gen.rho_ref`) the requirement of body c on leg l is
    req[c, s] = v_req[s] / (rho_ref[c, l] * |cmd[c, l]|),
i.e. the fraction of the nominal healthy body's achieved leg speed (|cmd| = planar speed or |wz| of the T1 leg command, `capability.leg_twists`:
0.30 m/s surge/sway, 0.21/0.21 m/s diagonals, yaw W_NOM capped by WHEEL_SPEED_CAP / char_length, so large bodies have
a larger yaw requirement for the same turn). Truth: segment ok iff rho_dec[c, cond(s), l] >= req[c, s].

Footprint: assembly bounding width = (min cell extent - 1) * pitch + housing side (capxfer assembly: 0.24 m housings on a
0.32 m pitch); a door is passable iff width + CLEARANCE <= door width (the mecanum body may strafe through sideways, so
the narrower bounding side counts). Load: payload <= LOAD_PER_MODULE_KG * modules (a mission-spec assumption: deck
rating), else the body cannot carry it after pickup.

Conditions: segments are labelled under (floor-mu factor rounded to MU_Q, payload on/off); every distinct condition of
a mission is one T1 label run of all 10 legs of every candidate, with the episode floor mu scaled by the factor and the
payload point mass at the candidate centre. The candidate bodies are the generator's episode candidates (self /
children / merges, `cfcap.gen.episode.make_episode`) for a fixed body-episode stream per family (index offset
BODY_OFFSET, disjoint from the gen reference splits).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from cfcap.env import buildings as B
from cfcap.t1.capability import LEGS

L = len(LEGS)
LEG = {n: i for i, n in enumerate(LEGS)}
CLEARANCE = 0.10                 # m, total (0.05 per side)
LOAD_PER_MODULE_KG = 2.0
T_RECONF = 30.0                  # s per split / merge (= decide T_SPLIT_S; scripts/env_score.py)
MU_Q = 0.05
V_REF = 0.27                     # m/s: speeds are drawn as fractions of ~ the nominal healthy achieved surge speed
SIGMA_STRAIGHT = (0.3, 1.0)       # codex task library surge r: q05 0.22, median 0.55, q95 0.97
SIGMA_DIAG = (0.2, 0.98)          # library diag r: q05 0.15, median 0.32, q95 0.96
SIGMA_SWAY = (0.05, 0.5)          # library sway r: q05 0.02, median 0.11, q95 0.34 (door alignment)
SIGMA_TURN = (0.2, 0.6)
P_STRAFE = 0.3
P_REVERSE = 0.5
BODY_SPLIT = {"train": "train", "cal": "cal", "test": "test", "E-H1": "test", "E-H2": "test", "E-H3": "test",
              "E-H4": "test", "conf-ID": "test", "conf-E-H1": "test", "conf-E-H2": "test", "conf-E-H3": "test",
              "conf-E-H4": "test"}
BODY_OFFSET = 3_000_000          # body episode index = BODY_OFFSET + 100_000 * family code + mission index
KIND_NAMES = ("straight", "diag", "turn", "door")


@dataclasses.dataclass
class Segment:
    kind: str
    leg: int
    v_req: float                 # m/s, or rad/s for yaw
    s: float
    length: float
    mu: float
    loaded: bool
    phase: int                   # 0 before the reconfiguration site, 1 after
    door_w: float | None = None
    narrow: bool = False


def segments(m: B.Mission, rng=None):
    rng = np.random.default_rng([B.SEED0, 7, B.FAM_CODE[m.family], m.index]) if rng is None else rng
    cuts = sorted({0.0, m.route_m, m.station_s, *(x for p in m.patches for x in p[:2] if 0 < x < m.route_m),
                   *([m.site] if m.site is not None else [])})
    out = []

    def cond(s):
        return (round(m.mu_at(s) / MU_Q) * MU_Q, m.payload_kg > 0 and s >= m.station_s,
                int(m.site is not None and s >= m.site))

    def straight(kind, leg, a, b, sigma):
        v = float(rng.uniform(*sigma)) * V_REF
        pts = [a] + [c for c in cuts if a < c < b] + [b]
        for x, y in zip(pts, pts[1:]):
            if y - x > 1e-6:
                mu, ld, ph = cond(0.5 * (x + y))
                out.append(Segment(kind, leg, v, x, y - x, mu, ld, ph))

    body = m.legs[0][1] - (m.legs[0][1] % 2)
    s = 0.0
    doors = list(m.doors)
    for i, (l, k) in enumerate(m.legs):
        if i > 0:
            dk, r = m.turns[i - 1]
            if k % 2 == 0 and k != body:
                d = ((k - body + 4) % 8) - 4
                if abs(d) == 4 and rng.uniform() < P_REVERSE:
                    pass                                   # reverse: drive surge- below
                elif abs(d) == 2 and rng.uniform() < P_STRAFE:
                    pass                                   # strafe the corner: sway below
                else:
                    w = float(rng.uniform(*SIGMA_TURN)) * V_REF / r
                    mu, ld, ph = cond(s)
                    out.append(Segment("turn", LEG["yaw+"] if d > 0 or (abs(d) == 4 and rng.uniform() < 0.5)
                                       else LEG["yaw-"], w, s, 0.0, mu, ld, ph))
                    body = k
        off = (k - body) % 8
        if off == 0:
            name = "surge+"
        elif off == 4:
            name = "surge-"
        elif off in (2, 6):
            name = "sway+" if off == 2 else "sway-"
        else:
            name = {1: "diag++", 7: "diag+-", 3: "diag-+", 5: "diag--"}[off]
        kind = "diag" if off % 2 else "straight"
        a, b = s, s + l
        # doors on this leg split it
        here = [d for d in doors if a - 1e-9 <= d.s < b + (1e-9 if i == len(m.legs) - 1 else 0)]
        pts = [a] + [d.s for d in here] + [b]
        for j, (x, y) in enumerate(zip(pts, pts[1:])):
            if j > 0:
                d = here[j - 1]
                mu, ld, ph = cond(d.s)
                out.append(Segment("door", LEG["sway+"] if rng.uniform() < 0.5 else LEG["sway-"],
                                   float(rng.uniform(*SIGMA_SWAY)) * V_REF, d.s, 0.0, mu, ld, ph, d.width, d.narrow))
            straight(kind, LEG[name], x, y, SIGMA_DIAG if kind == "diag" else SIGMA_STRAIGHT)
        s = b
    return out


# ----------------------------------------------------------------------------- bodies
def footprint_width(asm):
    c = np.array([(i, j) for _, i, j, _ in asm.modules], float)
    ext = c.max(0) - c.min(0) + 1
    pitch, side = float(asm.meta["pitch"]), float(getattr(asm, "side", 0.24))
    return float((ext.min() - 1) * pitch + side)


def body_episode(m: B.Mission, plate="plated"):
    from cfcap.gen import episode as E
    return E.make_episode(plate, BODY_SPLIT[m.family], BODY_OFFSET + 100_000 * B.FAM_CODE[m.family] + m.index)


def conditions(segs, payload_kg):
    keys = sorted({(sg.mu, sg.loaded) for sg in segs})
    return keys, np.array([keys.index((sg.mu, sg.loaded)) for sg in segs], np.int16)


def label_missions(ms, plate="plated", engine="compile", max_wheels=1 << 17, stats=None):
    """Per mission a record dict (see `record`): segments, candidates, conditions and T1 truth rho_dec (C, K, L).
    Labels: the generator's `simrun.label_conditions` (mu factor on every module's friction, payload point mass at
    the candidate centre; rho_ref the fixed nominal table)."""
    from cfcap.gen import simrun as SR
    from cfcap.t1 import capability as cap
    pre, items = [], []
    for m in ms:
        segs = segments(m)
        ep = body_episode(m, plate)
        keys, ci = conditions(segs, m.payload_kg)
        k0 = len(items)
        for mu, ld in keys:
            pay = (float(m.payload_kg), 0.0, 0.0) if ld else None
            items += [dict(ep=ep, cand=c, mu=float(mu), payload=pay) for c in ep.cands]
        pre.append((m, segs, ep, keys, ci, k0))
    lab = SR.label_conditions(items, engine=engine, max_wheels=max_wheels, stats=stats) if items else None
    out = []
    for m, segs, ep, keys, ci, k0 in pre:
        C, K = len(ep.cands), len(keys)
        rho_dec = np.asarray(lab["rho_dec"][k0:k0 + C * K]).reshape(K, C, L).transpose(1, 0, 2)    # (C, K, L)
        rho_ref = np.asarray(lab["rho_ref"][k0:k0 + C])                                             # (C, L)
        cmd = np.stack([cmd_speed(cap.leg_twists(c.asm)) for c in ep.cands])                 # (C, L) |cmd| per leg
        out.append(record(m, segs, ep, keys, ci, rho_dec, rho_ref, cmd))
    return out


def cmd_speed(tw):
    """|commanded leg speed|: planar speed (m/s) for translation legs, |wz| (rad/s) for yaw legs."""
    tw = np.asarray(tw, float)
    return np.hypot(tw[:, 0], tw[:, 1]) + np.abs(tw[:, 2])


def record(m, segs, ep, keys, ci, rho_dec, rho_ref, cmd):
    """Mission decision record. Candidates c, segments s, conditions k:
    cand_kind (C) 0 self / 1 child / 2 merge, cand_n_mod, cand_width, cand_load_ok, cand_footprint_ok (narrowest door);
    seg_leg, seg_kind, seg_cond, seg_phase, seg_v_req, seg_door_w (nan if no door), seg_narrow;
    req (C, S) in rho_dec units; fp_ok (C, S) door passable (True off doors); load_ok (C, S) (True before pickup);
    rho_dec (C, K, L) truth under each condition; seg_ok (C, S) = fp_ok & load_ok & rho_dec[c, cond, leg] >= req;
    mission_ok (C) = all segments ok (a single-body plan); phase_ok (C, 2) per phase (a split/merge plan at the site
    uses body a for phase 0 and body b for phase 1)."""
    C, S = len(ep.cands), len(segs)
    leg = np.array([sg.leg for sg in segs], np.int16)
    vreq = np.array([sg.v_req for sg in segs], np.float32)
    req = vreq[None, :] / np.maximum(rho_ref[:, leg] * cmd[:, leg], 1e-6)
    width = np.array([footprint_width(c.asm) for c in ep.cands], np.float32)
    nmod = np.array([len(c.modules) for c in ep.cands], np.int16)
    door = np.array([np.nan if sg.door_w is None else sg.door_w for sg in segs], np.float32)
    fp = np.where(np.isnan(door)[None, :], True, (width[:, None] + CLEARANCE) <= door[None, :])
    loaded = np.array([sg.loaded for sg in segs], bool)
    lok_c = m.payload_kg <= LOAD_PER_MODULE_KG * nmod
    lok = np.where(loaded[None, :], lok_c[:, None], True)
    truth = rho_dec[:, ci, leg] if S else np.zeros((C, 0), np.float32)
    ok = fp & lok & (truth >= req)
    phase = np.array([sg.phase for sg in segs], np.int8)
    phase_ok = np.stack([ok[:, phase == p].all(1) for p in (0, 1)], 1)
    return dict(
        family=m.family, mission=m.index, building=m.building, level=m.level, route_m=m.route_m,
        narrowest=m.narrowest, payload_kg=m.payload_kg, site=-1.0 if m.site is None else m.site,
        body_split=ep.split, body_index=ep.index, plate=ep.plate,
        cand_kind=np.array([c.kind for c in ep.cands], np.int8), cand_n_mod=nmod, cand_width=width,
        cand_mask=np.array([sum(1 << (16 * r + k) for r, k in c.modules) for c in ep.cands], np.int64),
        cand_load_ok=lok_c, cand_footprint_ok=(width + CLEARANCE) <= m.narrowest,
        seg_leg=leg, seg_kind=np.array([KIND_NAMES.index(sg.kind) for sg in segs], np.int8),
        seg_cond=ci, seg_phase=phase, seg_v_req=vreq, seg_door_w=door,
        seg_narrow=np.array([sg.narrow for sg in segs], bool), seg_mu=np.array([sg.mu for sg in segs], np.float32),
        seg_loaded=loaded, cond_mu=np.array([k[0] for k in keys], np.float32),
        cond_loaded=np.array([k[1] for k in keys], bool),
        req=req.astype(np.float32), fp_ok=fp, load_ok=lok, rho_dec=rho_dec.astype(np.float32),
        rho_ref=rho_ref.astype(np.float32), seg_truth=truth.astype(np.float32), seg_ok=ok,
        mission_ok=ok.all(1), phase_ok=phase_ok)


def relevant(rec, band=0.15):
    """Decision-relevant (candidate, segment) pairs: footprint- and load-feasible, with truth within `band` of the
    requirement (where the certificate, not geometry, decides)."""
    return rec["fp_ok"] & rec["load_ok"] & (np.abs(rec["seg_truth"] - rec["req"]) <= band)


# ----------------------------------------------------------------------------- decision interface (decide role)
def plan_options(rec):
    """Mission plans [(a, b)]: body a drives phase 0 (start -> reconfiguration site), body b phase 1 (site -> goal).
    a == b: no reconfiguration. a != b: b is a split of a (b's modules a subset of a's) or a merge containing a
    (a's modules a subset of b's, the partner joins at the site). Without a site every plan is (a, a)."""
    C = len(rec["cand_kind"])
    out = [(a, a) for a in range(C)]
    if rec["site"] < 0:
        return out
    mk = rec["cand_mask"]
    for a in range(C):
        for b in range(C):
            if a != b and ((mk[b] & ~mk[a]) == 0 or (mk[a] & ~mk[b]) == 0):
                out.append((a, b))
    return out


def certified(rec, lb):
    """(C, S) certified segment mask from per-(candidate, condition, leg) lower bounds lb (C, K, L) in rho_dec units
    (NaN certifies nothing): footprint and load hold (geometry, known) and lb >= req."""
    lbs = lb[:, rec["seg_cond"], rec["seg_leg"]]
    return rec["fp_ok"] & rec["load_ok"] & (np.nan_to_num(lbs, nan=-np.inf) >= rec["req"])


def plan_ok(rec, ok, plan):
    """Whether plan (a, b) succeeds under a (C, S) segment mask (truth `seg_ok` or a certificate)."""
    a, b = plan
    ph = rec["seg_phase"]
    return bool(ok[a, ph == 0].all() and ok[b, ph == 1].all())
