"""Baseline episodes: one parent assembly observed under the v2 probe, its candidate set, and closed-loop T1 labels.

An `Episode` is what every baseline consumes:
  * public (robot-visible): parent morphology `parent_asm`, the v2 command table `cmd` (duration T), observations
    `obs` (pre_duty, quantised encoder speed, gz/ax/ay with IMU noise and bias, wheel-sample dropout) and the
    candidate set `cands` (list of `Cand`);
  * privileged (only the oracle-type baselines may read it): the parent's true latents `lat` (faults and every
    nuisance) via `Episode.privileged()`;
  * candidate context treated as known by every baseline: a merge partner's modules (`Cand.other`, their true
    latents: the partner assembly is taken as already certified), a candidate's own normal-load split and payload;
  * labels (evaluation only): `truth` = `rho_dec` (C, L) = rho_abs / rho_ref(shape, leg), the primary decision label
    (rho_ref: nominal healthy reference, cfcap.t1.reference); secondary `rho_rel` (C, L) and absolute `rho` (C, L).

Two sources, one contract:
  * `generate(shape, B, seed)` / `pilot_episode(split, j)`: the baselines pilot bank (named parent shapes pair /
    square / 2x3, module yaws 0, HET_FULL, 0-2 faults, children only, labels from cfcap.t1.capability);
  * `from_gen(plate, split, index)`: the generator's Stage B episode `cfcap.gen.episode.make_episode` with the
    generator's own v2 observation (`simrun.probe_episodes`) and labels (`simrun.label_episodes`), so `eid` = the
    generator index and rows join every other method's rows (docs/board/paper.md 1a). Candidates: self, children
    and merges.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from cfcap.t0 import candidates as Cands
from cfcap.t1 import capability as cap
from cfcap.t1 import params as P
from cfcap.t1 import plant

LEGS = cap.LEGS
DUTY_NOISE = 0.005
PILOT_SHAPES = ("pair", "square", "2x3")


@dataclass
class Cand:
    name: str
    kind: str                          # self | child | merge
    asm: object
    pmods: tuple                       # parent module indices, in the candidate's module order (they come first)
    src: np.ndarray                    # (n_c,) parent wheel index, or -1 for a merge partner's wheel
    other: dict | None = None          # known latents (B = 1, cpu) of the partner modules (merge), else None
    load_z: np.ndarray | None = None   # (n_c,) candidate normal-load null-space share (None = 0)
    payload: tuple | None = None       # (mass, x, y) in the candidate frame

    @property
    def size(self):
        return len(self.asm.modules)

    @property
    def modules(self):
        return self.pmods


@dataclass
class Episode:
    eid: int
    shape: str
    plate: str
    cmd: np.ndarray                    # (K+1, 3) v2 command tick table (50 Hz)
    T: float
    obs: dict                          # pre_duty/speed (T, n), gz/ax/ay (T,) cpu tensors
    cand_modules: list                 # parent module tuples (kept for the pilot bank)
    lat: dict = field(repr=False, default=None)        # PRIVILEGED true parent latents (B = 1, cpu)
    rho_rel: np.ndarray = field(repr=False, default=None)  # LABEL (C, L)
    rho: np.ndarray = field(repr=False, default=None)      # LABEL (C, L)
    meta: dict = field(default_factory=dict)
    parent_asm: object = field(repr=False, default=None)
    cands: list = field(repr=False, default=None)
    split: str = "pilot"
    rho_dec: np.ndarray = field(repr=False, default=None)  # LABEL (C, L): rho / rho_ref (primary decision label)

    def __post_init__(self):
        if self.parent_asm is None and self.shape in P.SHAPES:
            _, self.parent_asm, cl = context(self.shape, self.plate)
            if self.cands is None:
                self.cands = [Cand(c.name, c.kind, c.asm, tuple(c.modules), np.asarray(c.src)) for c in cl
                              if tuple(c.modules) in set(map(tuple, self.cand_modules))]

    @property
    def truth(self):
        """Primary decision label rho_dec = rho_abs / rho_ref(shape, leg) (docs/board/README.md)."""
        if self.rho_dec is None:
            self.rho_dec = self.rho / ref_table(self.cands, self.plate)
        return self.rho_dec

    @property
    def n_cand(self):
        return len(self.cands) if self.cands is not None else len(self.cand_modules)

    @property
    def params(self):
        return P.base_params(self.plate) if self.plate not in _PAR else _PAR[self.plate]

    def privileged(self):
        return self.lat

    def faults(self):
        return self.lat["mode"][0].numpy(), self.lat["g"][0].numpy()

    def cand_has_fault(self):
        m = self.lat["mode"][0].numpy()
        out = []
        for c in self.cands:
            f = bool((m[c.src[c.src >= 0]] != 0).any())
            if c.other is not None:
                f = f or bool((c.other["mode"] != 0).any())
            out.append(f)
        return np.array(out, bool)

    def cand_true_modes(self):
        """Per candidate (mode, g) arrays (n_c,) of the truth (oracle / known-partner use)."""
        m, g = self.faults()
        out = []
        for c in self.cands:
            cm, cg = np.zeros(len(c.src), np.int64), np.ones(len(c.src), np.float32)
            p = c.src >= 0
            cm[p], cg[p] = m[c.src[p]], g[c.src[p]]
            if c.other is not None:
                cm[~p], cg[~p] = c.other["mode"][0].numpy(), c.other["g"][0].numpy()
            out.append((cm, cg))
        return out


_CACHE, _PAR = {}, {}


def context(shape, plate="plated"):
    """(params, parent asm, candidate list) for a named parent shape (cached)."""
    key = (shape, plate)
    if key not in _CACHE:
        par = P.base_params(plate)
        _PAR.setdefault(plate, par)
        asm = P.named_assembly(shape, par)
        cands = Cands.children(asm, par)
        _CACHE[key] = (par, asm, cands)
    return _CACHE[key]


def lat_to(lat, device):
    out = {}
    for k, v in lat.items():
        if k == "dom":
            out[k] = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        elif k == "payload":
            out[k] = None if v is None else {kk: vv.to(device) for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


_WHEEL_KEYS = ("mode", "g", "stall_w", "noload_w", "yaw_err")
_MOD_KEYS = ("vbatt_off", "mu_mod")


def cand_latents(rows, c):
    """Latents (B rows, rows' device) of candidate c built from parent latent rows: the parent modules' per-wheel and
    per-module latents (plant.subset), the merge partner's known latents appended in candidate order, the
    candidate's own load split and payload."""
    B = rows["mode"].shape[0]
    dev = rows["mode"].device
    out = plant.subset(rows, c.pmods)
    if c.other is not None:
        o = lat_to(c.other, dev)
        for k in _WHEEL_KEYS + _MOD_KEYS:
            out[k] = torch.cat([out[k], o[k].expand(B, -1).to(out[k].dtype)], 1)
        if "wtype" in out or "wtype" in o:              # module wheel types (cfcap.t1.plant): 0 = mecanum default
            n_p = len(c.pmods) * 4
            a = out["wtype"] if "wtype" in out else torch.zeros(B, n_p, dtype=torch.long, device=dev)
            b = o["wtype"].expand(B, -1) if "wtype" in o else torch.zeros(B, out["mode"].shape[1] - n_p,
                                                                          dtype=torch.long, device=dev)
            out["wtype"] = torch.cat([a, b.to(torch.long)], 1)
    n = out["mode"].shape[1]
    if c.load_z is not None:
        out["load_z"] = torch.as_tensor(np.asarray(c.load_z), dtype=torch.float32, device=dev).expand(B, n).contiguous()
    else:
        out["load_z"] = torch.zeros(B, n, device=dev)
    if c.payload is not None:
        f = lambda v: torch.full((B,), float(v), device=dev)
        out["payload"] = dict(mass=f(c.payload[0]), x=f(c.payload[1]), y=f(c.payload[2]))
    return out


def ref_table(cands, plate="plated", device=None):
    """(C, L) nominal healthy reference rho of each candidate's shape (cfcap.t1.reference)."""
    from cfcap.t1.reference import rho_ref_many
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return rho_ref_many([c.asm for c in cands], plate, device)


def candidate_rho(par, cands, rows, device="cuda", relative=True, plate="plated"):
    """rho_rel (B, C, L), rho (B, C, L), rho_dec (B, C, L) of every row of parent latents on every candidate (T1).
    relative=False skips the fault-free twin rollouts (rho_rel then NaN)."""
    rows = lat_to(rows, device)
    targets = [(c.asm, cand_latents(rows, c)) for c in cands]
    outs = cap.capability(targets, par, relative=relative, device=device)
    rho = np.stack([o["rho"].cpu().numpy() for o in outs], 1)
    rr = np.stack([o["rho_rel"].cpu().numpy() for o in outs], 1) if relative else np.full_like(rho, np.nan)
    return rr, rho, rho / ref_table(cands, plate, device)[None]


# ----------------------------------------------------------------------------- pilot bank
def candidate_truth(shape, lat, plate="plated", device="cuda"):
    par, asm, cands = context(shape, plate)
    cv = [Cand(c.name, c.kind, c.asm, tuple(c.modules), np.asarray(c.src)) for c in cands]
    return candidate_rho(par, cv, lat, device, plate=plate)


def observe_parent(shape, lat, plate="plated", device="cuda", seed=0, duty_noise=DUTY_NOISE):
    par, asm, _ = context(shape, plate)
    tab, T = plant.v2_probe(asm, par, 1, jitter=False)
    R = plant.simulate_bucket([dict(asm=asm, lat=lat_to(lat, device), cmd=tab[0])], par, T, device=device)[0]
    obs = plant.observe(R, lat_to(lat, device), seed=seed)
    gen = torch.Generator(device=obs["pre_duty"].device).manual_seed(int(seed) + 1)
    obs["pre_duty"] = obs["pre_duty"] + duty_noise * torch.randn(obs["pre_duty"].shape, generator=gen,
                                                                 device=obs["pre_duty"].device)
    return tab[0], T, {k: v.cpu() for k, v in obs.items()}


def generate(shape, B, seed, plate="plated", het=plant.HET_FULL, p_none=0.2, k_max=2, device="cuda", eid0=0,
             split="pilot"):
    """B pilot-bank episodes of one named parent shape."""
    t0 = time.time()
    par, asm, cands = context(shape, plate)
    rng = np.random.default_rng(seed)
    dr = P.theta_dr(P.load_theta(plate))
    faults = plant.random_faults(B, asm.n, rng, k_max=k_max, p_none=p_none)
    lat = plant.sample_latents(asm, B, rng, het, dr=dr, faults=faults, device=device)
    tab, T, obs = observe_parent(shape, lat, plate, device, seed=seed % (1 << 62))
    rr, r, rd = candidate_truth(shape, lat, plate, device)
    lat_c = lat_to(lat, "cpu")
    eps = []
    for b in range(B):
        lb = plant._index(lat_c, torch.tensor([b]))
        eps.append(Episode(eid=eid0 + b, shape=shape, plate=plate, cmd=tab, T=T,
                           obs={k: v[b] for k, v in obs.items()},
                           cand_modules=[tuple(c.modules) for c in cands], lat=lb, rho_rel=rr[b], rho=r[b],
                           meta=dict(seed=seed, gen_s=(time.time() - t0) / B), split=split, rho_dec=rd[b]))
    return eps


def pilot_episode(split, j, shapes=PILOT_SHAPES, device=None):
    """Pilot-bank episode j of a split: a pure function of (split, j)."""
    from cfcap.runner import unit_seed
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    shape = shapes[j % len(shapes)]
    seed = unit_seed("cfcap-baselines-episode", split, j) % (1 << 62)
    return generate(shape, 1, seed=seed, device=device, eid0=j, split=split)[0]


# ----------------------------------------------------------------------------- generator episodes
def from_gen(plate, split, index, engine="compile"):
    """The generator's episode (plate, split, index) as a baselines Episode (eid = index)."""
    from cfcap.gen import episode as GE
    from cfcap.gen import simrun as S
    gep = GE.make_episode(plate, split, index)
    par = GE.params(plate)
    pa = gep.parent
    o = S.probe_episodes([gep], ("v2",), engine=engine)["v2"][0][0]
    rng = GE.episode_rng(plate, split, index, stream=1 + S.PROBE_KINDS.index("v2"))
    cmd, _, _ = S.probe_tables("v2", pa.asm, par, rng)          # the same draw probe_episodes used (full table)
    speed = o["dcnt"].astype(np.float32) / P.CPR * 2 * np.pi / plant.DT_OBS
    obs = dict(pre_duty=torch.as_tensor(o["duty"], dtype=torch.float32), speed=torch.as_tensor(speed),
               gz=torch.as_tensor(o["imu"][:, 0], dtype=torch.float32),
               ax=torch.as_tensor(o["imu"][:, 1], dtype=torch.float32),
               ay=torch.as_tensor(o["imu"][:, 2], dtype=torch.float32))
    lat = GE.plant_latents(gep, [(0, k) for k in range(pa.m)], pa.load_z, GE.parent_payload(gep),
                           gyro=(pa.gyro_bias, pa.acc_bias))
    kinds = {v: k for k, v in GE.KIND.items()}
    cands = []
    for ci, c in enumerate(gep.cands):
        pm = tuple(k for r, k in c.modules if r == 0)
        om = [(r, k) for r, k in c.modules if r != 0]
        assert all(r == 0 for r, _ in c.modules[:len(pm)]), "parent modules come first"
        src = np.concatenate([np.arange(4 * k, 4 * k + 4) for k in pm] + [np.full(4 * len(om), -1)]).astype(int)
        other = None
        if om:
            ol = GE.plant_latents(gep, om, np.zeros(4 * len(om), np.float32), None)
            other = {k: ol[k] for k in _WHEEL_KEYS + _MOD_KEYS + ("wtype",) if k in ol}
        cands.append(Cand(f"{split}{index}/c{ci}", kinds[c.kind], c.asm, pm, src, other, c.load_z, c.payload))
    lab = S.label_episodes([gep], engine=engine)[0]
    T = S.probe_T("v2")
    rho_dec = lab["rho_dec"] if "rho_dec" in lab else lab["rho"] / ref_table(cands, plate)
    return Episode(rho_dec=rho_dec, eid=int(index), shape=f"{pa.m}m:{pa.topo}", plate=plate, cmd=cmd, T=T, obs=obs,
                   cand_modules=[c.pmods for c in cands], lat=lat, rho_rel=lab["rho_rel"], rho=lab["rho"],
                   meta=dict(family=[c.family for c in gep.cands]), parent_asm=pa.asm, cands=cands, split=split)


def batch_obs(eps):
    return {k: torch.stack([e.obs[k] for e in eps]) for k in eps[0].obs}


def save(eps, path):
    torch.save([e.__dict__ for e in eps], path)


def load(path):
    return [Episode(**d) for d in torch.load(path, weights_only=False)]
