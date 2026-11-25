"""Procedural buildings for cfcap missions, calibrated to the HM3D-derived mission statistics (`hm3d_stats`).

A building (pure function of (family, building index)) has `levels` levels (HM3D pmf) and MISSIONS_PER_BUILDING
missions (as the MRMN real-buildings draw: 3 per building), each on one level. A level is a graph of spaces (rooms
and corridors: nodes with a size) joined by doors (edges with a width). Missions are drawn route-first, as in the
HM3D draw (start -> narrow passage -> goal): the route's knobs are a draw from the Gaussian KDE of the HM3D joint
route features (log route length, log tortuosity, metres per lattice step, passage position, station distance),
then a polyline on the 8-heading lattice is chosen among PROPOSALS random proposals (number of legs, turns, leg
lengths, diagonal legs) to realize those knobs, and the spaces are laid along it: every leg lies in one space, a
change of space happens through a door, and the narrow passage (the door that forces the shape change) sits where
the route reaches the drawn passage position. Calibration (`calibrate`) recomputes the HM3D statistics from the
realized geometry (not from the knobs) and compares them with the HM3D records.

Mission geometry (all in metres, route arc length s):
  legs        [(length, heading k in 0..7, i.e. k*45 deg)] (odd k = a diagonal leg);
  turns       between legs: delta heading (45 deg units, signed) and radius (0.3-1.5 m, <= half the space width);
  doors       [(s, width, space_from, space_to, narrow)]; widths 0.45-1.4 m;
  site        s of the reconfiguration site (HM3D: one per mission w.p. 0.976), just before the narrow passage;
  station     s of the payload station (pickup) and its mass, 0-5 kg (family E-H4: 5-7 kg);
  mu patches  [(s0, s1, factor)] floor-friction factors 0.5-1.1 on the episode floor mu (the rest of the route has the
              building's base factor).

Families (`FAMILIES`; splits fixed by building seed, disjoint seed streams):
  train / cal / test  the calibrated rule;
  E-H1  narrow-door tail: narrow passage width from the left tail (0.45-0.62 m) and narrower ordinary doors;
  E-H2  high friction variance: every leg is a patch, factor U(0.5, 1.1), plus mid-leg patches;
  E-H3  long multi-turn routes: knobs drawn from the HM3D upper tail (route_m >= q75 and tortuosity >= q50);
  E-H4  heavy payload: station mass U(5, 7) kg always.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cfcap.env import hm3d_stats as HS

SEED0 = 90925
FAMILIES = ("train", "cal", "test", "E-H1", "E-H2", "E-H3", "E-H4")
CONF_FAMILIES = ("conf-ID", "conf-E-H1", "conf-E-H2", "conf-E-H3", "conf-E-H4")
FAM_CODE = {f: i for i, f in enumerate(FAMILIES + CONF_FAMILIES)}
MISSIONS_PER_BUILDING = 3
PROPOSALS = 48
RES = HS.RESOLUTION
MIN_LEG = 0.2
DOOR_W = (0.45, 1.4)
RADIUS = (0.3, 1.5)
MU_RANGE = (0.5, 1.1)
PAYLOAD_MAX = 5.0


@dataclass(frozen=True)
class Knobs:
    """Generator knobs (tuned against HM3D, results/env/calibration.csv)."""
    p_diag: float = 0.30              # probability that a leg is a 45-deg diagonal
    p_door_turn: float = 0.6          # a turn between legs changes space through a door
    narrow_beta: tuple = (2.0, 2.0)   # narrow passage width = 0.45 + 0.95 * Beta(a, b)
    door_w: tuple = (0.75, 1.4)       # ordinary doors
    mu_base: tuple = (0.9, 1.05)
    p_patch: float = 0.25             # per leg
    patch_mu: tuple = (0.6, 1.1)
    p_payload: float = 0.7
    payload: tuple = (0.5, PAYLOAD_MAX)
    tail: bool = False                # E-H3: KDE restricted to the HM3D upper tail
    mid_patches: int = 0


KNOBS = {
    "train": Knobs(), "cal": Knobs(), "test": Knobs(),
    "E-H1": Knobs(narrow_beta=(1.0, 6.0), door_w=(0.6, 0.95)),
    "E-H2": Knobs(p_patch=1.0, patch_mu=MU_RANGE, mu_base=(0.75, 1.1), mid_patches=2),
    "E-H3": Knobs(tail=True),
    "E-H4": Knobs(p_payload=1.0, payload=(5.0, 7.0)),
}
E_H1_NARROW_MAX = 0.62
KDE_BW_SCALE = 0.5                # KDE bandwidth = 0.5 x Scott (less smoothing bias in the tails)
HS_STEPS = (20, 250)              # MRMN real_buildings ROUTE_STEPS
# Confirmatory C-ENV stream (docs/CONFIRMATORY.md §0.1, §4): buildings [92599, 23, b]; 300 ID + 150 per E-H family
CONF_ROOT, CONF_CLAIM = 92599, 23
CONF_MISSIONS_PER_BUILDING = 5
CONF_BLOCKS = {"conf-ID": (0, 300), "conf-E-H1": (300, 150), "conf-E-H2": (450, 150), "conf-E-H3": (600, 150),
               "conf-E-H4": (750, 150)}
CONF_KNOBS = {"conf-ID": "test", "conf-E-H1": "E-H1", "conf-E-H2": "E-H2", "conf-E-H3": "E-H3", "conf-E-H4": "E-H4"}


@dataclass
class Space:
    kind: str                 # "room" | "corridor"
    width: float              # m (corridor width or room short side)
    length: float


@dataclass
class Door:
    s: float
    width: float
    a: int                    # space index before
    b: int                    # space index after
    narrow: bool = False


@dataclass
class Mission:
    family: str
    building: int
    index: int                # mission index within the family (building * 3 + j)
    level: int
    legs: list                # [(length, heading k)]
    turns: list               # [(dk, radius)] between legs (len(legs) - 1)
    leg_space: list           # space index per leg
    spaces: list              # [Space]
    doors: list               # [Door] sorted by s
    passage: int              # index in doors of the narrow passage
    site: float | None        # s of the reconfiguration site, or None
    station_s: float
    payload_kg: float
    mu_base: float
    patches: list             # [(s0, s1, factor)]
    knobs: dict = field(default_factory=dict)

    @property
    def route_m(self):
        return float(sum(l for l, _ in self.legs))

    def points(self):
        p = [np.zeros(2)]
        for l, k in self.legs:
            a = k * math.pi / 4
            p.append(p[-1] + l * np.array([math.cos(a), math.sin(a)]))
        return np.array(p)

    def at(self, s):
        P, acc = self.points(), 0.0
        for i, (l, _) in enumerate(self.legs):
            if s <= acc + l or i == len(self.legs) - 1:
                f = min(max((s - acc) / max(l, 1e-9), 0.0), 1.0)
                return P[i] + f * (P[i + 1] - P[i])
            acc += l
        return P[-1]

    def mu_at(self, s):
        f = self.mu_base
        for a, b, v in self.patches:
            if a <= s < b:
                f = v
        return f

    @property
    def narrowest(self):
        return min(d.width for d in self.doors)


# ----------------------------------------------------------------------------- HM3D-calibrated knob sampler
_FIT = None


def fitted():
    global _FIT
    if _FIT is None:
        _FIT = HS.load_fitted()
        X = np.asarray(_FIT["kde"]["points"], float)
        _FIT["_X"] = X
        rm, tor = np.exp(X[:, 0]), np.exp(X[:, 1])
        q = lambda k, p: _FIT["mission"][k]["quantiles"][p]
        _FIT["_tail"] = np.nonzero((rm >= q("route_m", 75)) & (tor >= q("tortuosity", 50)))[0]
    return _FIT


def sample_knobs(rng, tail=False):
    """(route_m, tortuosity, m_per_step, passage_pos, station_m) from the KDE (smoothed bootstrap)."""
    f = fitted()
    X, bw = f["_X"], np.asarray(f["kde"]["bandwidth"])
    pool = f["_tail"] if tail else np.arange(len(X))
    x0 = X[pool[int(rng.integers(len(pool)))]]
    x = x0 + rng.normal(size=X.shape[1]) * bw * KDE_BW_SCALE
    route = float(np.clip(math.exp(x[0]), 1.2, 30.0))
    tort = float(np.clip(math.exp(x[1]), 1.0, 25.0))
    mps = float(np.clip(x[2], 0.05, 0.14))
    ppos = float(np.clip(x[3], 0.0, 1.0))
    st = float(math.exp(x0[4]))                  # station distance: the record's own value (bounded, grid-discrete)
    return route, tort, mps, ppos, st


def _rot_steps(legs):
    return sum(min(abs(b - a) % 8, 8 - abs(b - a) % 8) for (_, a), (_, b) in zip(legs, legs[1:]))


def route_stats(legs, n_sites=1):
    """HM3D statistics of a polyline: route_m, steps (lattice moves + 45-deg rotation steps + transitions),
    tortuosity."""
    route = sum(l for l, _ in legs)
    cells = sum(l / (RES * (math.sqrt(2) if k % 2 else 1.0)) for l, k in legs)
    steps = int(round(cells)) + _rot_steps(legs) + n_sites
    P = [np.zeros(2)]
    for l, k in legs:
        a = k * math.pi / 4
        P.append(P[-1] + l * np.array([math.cos(a), math.sin(a)]))
    eu = float(np.linalg.norm(P[-1]))
    return route, steps, route / max(eu, RES)


def _propose(rng, route, kn):
    n_leg = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8], p=[.22, .22, .18, .13, .1, .07, .05, .03]))
    w = rng.dirichlet(np.full(n_leg, 1.5))
    lens = np.maximum(w * route, MIN_LEG)
    lens *= route / lens.sum()
    k = int(rng.integers(4)) * 2
    legs = []
    for i, l in enumerate(lens):
        if i > 0:
            dk = int(rng.choice([2, -2, 4, 1, -1, 3, -3], p=[.32, .32, .12, .07, .07, .05, .05]))
            k = (k + dk) % 8
        kk = k
        if k % 2 == 0 and rng.uniform() < kn.p_diag * 0.5:
            kk = (k + int(rng.choice([1, -1]))) % 8
        legs.append((float(l), kk))
        k = kk
    return legs


def _route(rng, route, tort, mps, kn):
    best, err = None, np.inf
    n_sites = 1
    for _ in range(PROPOSALS):
        legs = _propose(rng, route, kn)
        r, steps, t = route_stats(legs, n_sites)
        e = abs(math.log(t / tort)) + 4.0 * abs(math.log((r / steps) / mps))
        if e < err:
            best, err = legs, e
    return best


def seeds(family, index):
    """(knob family, building number, building seed, mission seed). Exploratory families: root [SEED0, code, b],
    3 missions per building. Confirmatory C-ENV families conf-*: root [92599, 23, b] (docs/CONFIRMATORY.md), b in
    the family's building block (CONF_BLOCKS), 5 missions per building."""
    if family in CONF_BLOCKS:
        b0, nb = CONF_BLOCKS[family]
        b, j = divmod(int(index), CONF_MISSIONS_PER_BUILDING)
        if b >= nb:
            raise ValueError(f"{family}: mission {index} beyond the {nb} pre-registered buildings")
        b += b0
        return CONF_KNOBS[family], b, [CONF_ROOT, CONF_CLAIM, b], [CONF_ROOT, CONF_CLAIM, b, 1 + j]
    b, j = divmod(int(index), MISSIONS_PER_BUILDING)
    return family, b, [SEED0, FAM_CODE[family], b], [SEED0, FAM_CODE[family], b, 1 + j]


def make_mission(family, index):
    kfam, b, bseed, mseed = seeds(family, index)
    kn = KNOBS[kfam]
    brng = np.random.default_rng(bseed)
    f = fitted()
    lv = f["pmf"]["levels"]
    n_lev = int(brng.choice([int(k) for k in lv], p=np.asarray(list(lv.values())) / sum(lv.values())))
    rng = np.random.default_rng(mseed)
    level = int(rng.integers(n_lev))
    for _ in range(50):                          # HM3D draws keep routes of 20-250 lattice steps
        route, tort, mps, ppos, st = sample_knobs(rng, kn.tail)
        legs = _route(rng, route, tort, mps, kn)
        if HS_STEPS[0] <= route_stats(legs, 1)[1] <= HS_STEPS[1]:
            break
    turns = [(((b2 - a2 + 4) % 8) - 4, float(rng.uniform(*RADIUS))) for (_, a2), (_, b2) in zip(legs, legs[1:])]
    # spaces along the route: a turn changes space through a door w.p. p_door_turn; long legs may cross one
    spaces = [Space("room", float(rng.uniform(2.5, 6.0)), 0.0)]
    leg_space, doors, s = [], [], 0.0
    for i, (l, k) in enumerate(legs):
        if i > 0 and rng.uniform() < kn.p_door_turn:
            kind = "corridor" if spaces[-1].kind == "room" else str(rng.choice(["room", "corridor"]))
            spaces.append(Space(kind, float(rng.uniform(0.9, 2.0) if kind == "corridor" else rng.uniform(2.5, 6.0)),
                                0.0))
            doors.append(Door(s, float(rng.uniform(*kn.door_w)), len(spaces) - 2, len(spaces) - 1))
        spaces[-1].length += l
        leg_space.append(len(spaces) - 1)
        s += l
    turns = [(dk, min(r, 0.5 * spaces[leg_space[i + 1]].width + 0.3)) for i, (dk, r) in enumerate(turns)]
    # narrow passage: the route point whose straight-line position start->p->goal is closest to ppos
    P, goal = mission_points(legs), None
    goal = P[-1]
    ss = np.linspace(0.0, route, max(int(route / RES), 2) + 1)
    pts = np.array([_at(legs, P, x) for x in ss])
    sp = np.linalg.norm(pts - P[0], axis=1)
    pg = np.linalg.norm(pts - goal, axis=1)
    pos = np.where(sp + pg > 0, sp / np.maximum(sp + pg, 1e-9), 1.0)
    s_pass = float(ss[int(np.argmin(np.abs(pos - ppos)))])
    a, bb = narrow_width(rng, kn, kfam)
    sp_i = int(np.searchsorted(np.cumsum([l for l, _ in legs]), s_pass, side="left"))
    sp_i = min(sp_i, len(legs) - 1)
    spaces.append(Space("corridor", float(rng.uniform(0.6, 1.2)), 0.0))
    doors.append(Door(s_pass, a, leg_space[sp_i], len(spaces) - 1, narrow=True))
    doors.sort(key=lambda d: d.s)
    passage = next(i for i, d in enumerate(doors) if d.narrow)
    n_sites = int(rng.uniform() < f["pmf"]["n_sites"].get("1", 1.0))
    site = max(0.0, s_pass - float(rng.uniform(0.3, 1.0))) if n_sites else None
    station_s = max(0.0, s_pass - st) if site is None else max(0.0, min(site, s_pass - st))
    mass = float(rng.uniform(*kn.payload)) if rng.uniform() < kn.p_payload else 0.0
    mu_base = float(rng.uniform(*kn.mu_base))
    patches, acc = [], 0.0
    for l, _ in legs:
        if rng.uniform() < kn.p_patch:
            patches.append((acc, acc + l, float(rng.uniform(*kn.patch_mu))))
        for _ in range(kn.mid_patches):
            c = acc + float(rng.uniform(0, l))
            w = float(rng.uniform(0.2, 1.0))
            patches.append((c - w / 2, c + w / 2, float(rng.uniform(*kn.patch_mu))))
        acc += l
    return Mission(family, b, int(index), level, legs, turns, leg_space, spaces, doors, passage, site, station_s,
                   mass, mu_base, patches, knobs=dict(route=route, tort=tort, mps=mps, ppos=ppos, station_m=st))


def narrow_width(rng, kn, family):
    lo, hi = DOOR_W
    if family == "E-H1":
        return float(lo + (E_H1_NARROW_MAX - lo) * rng.beta(*kn.narrow_beta)), None
    return float(lo + (hi - lo) * rng.beta(*kn.narrow_beta)), None


def mission_points(legs):
    P = [np.zeros(2)]
    for l, k in legs:
        a = k * math.pi / 4
        P.append(P[-1] + l * np.array([math.cos(a), math.sin(a)]))
    return np.array(P)


def _at(legs, P, s):
    acc = 0.0
    for i, (l, _) in enumerate(legs):
        if s <= acc + l or i == len(legs) - 1:
            f = min(max((s - acc) / max(l, 1e-9), 0.0), 1.0)
            return P[i] + f * (P[i + 1] - P[i])
        acc += l
    return P[-1]


# ----------------------------------------------------------------------------- calibration against HM3D
CAL_STATS = ("route_m", "route_steps", "m_per_step", "tortuosity", "passage_pos", "station_m", "n_sites")


def mission_stats(m: Mission):
    """The HM3D statistics recomputed from the realized geometry."""
    n_sites = 0 if m.site is None else 1
    route, steps, tort = route_stats(m.legs, n_sites)
    P = m.points()
    p = m.at(m.doors[m.passage].s)
    sp, pg = float(np.linalg.norm(p - P[0])), float(np.linalg.norm(P[-1] - p))
    st = m.knobs["station_m"]
    return dict(route_m=route, route_steps=steps, m_per_step=route / max(steps, 1), tortuosity=tort,
                passage_pos=sp / (sp + pg) if sp + pg > 0 else 1.0, station_m=st, n_sites=n_sites)


def ks_2samp(x, y):
    """Two-sample KS statistic and asymptotic p (ties handled by the joint ECDF)."""
    x, y = np.sort(x), np.sort(y)
    z = np.concatenate([x, y])
    d = float(np.max(np.abs(np.searchsorted(x, z, "right") / x.size - np.searchsorted(y, z, "right") / y.size)))
    ne = x.size * y.size / (x.size + y.size)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def cliffs_delta(x, y):
    x, y = np.asarray(x, float), np.sort(np.asarray(y, float))
    gt = np.searchsorted(y, x, "left").sum()
    lt = (y.size - np.searchsorted(y, x, "right")).sum()
    return float((gt - lt) / (x.size * y.size))


def calibrate(n=3000, family="train"):
    """Rows per statistic: n_hm3d, n_gen, means, KS D and p, Cliff's delta, Cohen's d, pass (KS p > 0.05 or
    KS D < 0.1: the effect size of docs/CONFIRMATORY.md §4)."""
    ms, _ = HS.load_records()
    g = [mission_stats(make_mission(family, i)) for i in range(n)]
    rows = []
    for k in CAL_STATS:
        x, y = ms[k], np.array([r[k] for r in g], float)
        d, p = ks_2samp(x, y)
        cd = cliffs_delta(y, x)
        sd = math.sqrt(0.5 * (x.var(ddof=1) + y.var(ddof=1))) or 1.0
        rows.append(dict(stat=k, family=family, bar="shifted by design (HM3D upper tail)" if family in ("E-H3", "conf-E-H3")
                         and k in ("route_m", "route_steps", "m_per_step", "tortuosity") else "match HM3D",
                         n_hm3d=x.size, n_gen=y.size, mean_hm3d=float(x.mean()),
                         mean_gen=float(y.mean()), median_hm3d=float(np.median(x)), median_gen=float(np.median(y)),
                         ks_d=d, ks_p=p, cliffs_delta=cd, cohens_d=float((y.mean() - x.mean()) / sd),
                         passed=bool(p > 0.05 or d < 0.1)))
    return rows


def main():
    import csv
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rows = []
    for fam in FAMILIES:
        rows += calibrate(n if fam in ("train", "test") else n // 3, fam)
    out = HS.OUT.parent / "calibration.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['family']:6s} {r['stat']:12s} hm3d {r['mean_hm3d']:7.3f} gen {r['mean_gen']:7.3f} "
              f"KS D {r['ks_d']:.3f} p {r['ks_p']:.3g} cliff {r['cliffs_delta']:+.3f} {'PASS' if r['passed'] else 'fail'}")


if __name__ == "__main__":
    main()
