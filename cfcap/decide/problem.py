"""Decision problem of PLAN §3: choose a partition / merge of the current assemblies and an assignment of pieces to
tasks, or abstain, from per-(candidate, leg) lower bounds. Pure numpy; no simulator.

Tasks (`make_tasks`). A task is a requirement vector r over the decision legs (NaN = leg not required) plus a minimum
piece size (payload tasks need >= 2 modules) and a value V (seconds of useful work, the same unit as time costs).
  route   1-3 legs (a leg pair such as sway+/sway- or yaw+/yaw-, a single leg, or a diagonal pair) at levels drawn from
          LEVELS, e.g. sway >= 0.8 and yaw >= 0.6;
  payload surge+-, sway+- >= 0.7 and >= 2 modules.
Tasks are drawn from their own seed stream, independent of the episode's truth.

Instance. Candidates are the pieces a decision may use: the whole parent (kind "parent"), connected children
("child") and merges of a parent piece with the other assembly ("merge"; `t0.candidates`). Each carries a module
bitmask over the parent (+ one bit for the other assembly), so the options below are disjoint by construction.

Options. Every injective partial map tasks -> pieces with pairwise disjoint pieces (the other assembly used at most
once). Time cost of an option = T_SPLIT_S x (parts of the resulting partition of the parent - 1) + T_SPLIT_S per merge
piece, where the parts are the used pieces plus the connected components of the leftover modules (a split is one
undock of one part). The all-(-1) option is "abstain" (cost 0).

Decision (`decide`). Commit only certified pairs: certified(c, t) iff every required leg has a finite lower bound
>= r and the piece is large enough. The chosen option maximises V x (tasks assigned) - time among options whose every
assigned pair is certified; NaN / empty bounds certify nothing, so the decision is abstain.

Evaluation (`outcome`). Under the true capability: served = assigned and truly capable; unsafe = assigned and truly
below r on some required leg. Utility U = V x served - C_UNSAFE x unsafe - time - probe time. The oracle decision is
`decide` on the true capability (no probing). Regret = U_oracle - U_policy.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

T_SPLIT_S = 30.0          # = t0.probesets.T_SPLIT_S (ESTIMATE until 04's split/combine timing)
V_TASK = 120.0            # value of one served task, seconds of useful work (ASSUMPTION)
C_UNSAFE = 600.0          # cost of one unsafe commit (failed task + recovery), seconds (ASSUMPTION)
LEVELS = (0.6, 0.7, 0.8, 0.9)


# ----------------------------------------------------------------------------- tasks
@dataclass
class Task:
    r: np.ndarray                 # (L,) requirement per leg, NaN = not required
    min_modules: int = 1
    value: float = V_TASK
    kind: str = "route"

    @property
    def legs(self):
        return np.flatnonzero(np.isfinite(self.r))


def make_tasks(rng, legs, n_tasks=None, p_payload=0.25, levels=LEVELS):
    """Random task set (1-3 tasks) over the leg names `legs` (order of the predictor's L axis)."""
    legs = list(legs)
    idx = {n: i for i, n in enumerate(legs)}
    pairs = [("surge+", "surge-"), ("sway+", "sway-"), ("yaw+", "yaw-"), ("diag++", "diag--"), ("diag+-", "diag-+")]
    T = int(rng.integers(1, 4)) if n_tasks is None else int(n_tasks)
    out = []
    for _ in range(T):
        r = np.full(len(legs), np.nan)
        if rng.random() < p_payload:
            for n in ("surge+", "surge-", "sway+", "sway-"):
                r[idx[n]] = 0.7
            out.append(Task(r, 2, V_TASK, "payload"))
            continue
        k = int(rng.integers(1, 3))                      # 1 or 2 requirement groups
        for g in rng.choice(len(pairs), k, replace=False):
            lv = float(rng.choice(levels))
            grp = pairs[int(g)] if rng.random() < 0.7 else (pairs[int(g)][int(rng.integers(2))],)
            for n in grp:
                r[idx[n]] = lv
        out.append(Task(r, 1, V_TASK, "route"))
    return out


def task_arrays(tasks, L):
    R = np.stack([t.r for t in tasks]) if tasks else np.zeros((0, L))
    return R, np.array([t.min_modules for t in tasks], int), np.array([t.value for t in tasks], float)


# ----------------------------------------------------------------------------- instance and options
@dataclass
class Instance:
    """kinds (C,) in {parent, child, merge}; masks (C,) int bitmask over parent modules (+ bit m for the other
    assembly); sizes (C,) modules per piece; cells: parent module cells (for leftover components)."""
    kinds: list
    masks: np.ndarray
    sizes: np.ndarray
    cells: list
    names: list = field(default_factory=list)
    _opts: object = field(default=None, repr=False)

    @property
    def C(self):
        return len(self.kinds)

    @classmethod
    def from_candidates(cls, parent_cells, cands, include_parent=True):
        """cands: t0.candidates.Candidate list (children and merges). Prepends the whole parent if include_parent."""
        m = len(parent_cells)
        kinds, masks, sizes, names = [], [], [], []
        if include_parent:
            kinds.append("parent"); masks.append((1 << m) - 1); sizes.append(m); names.append("parent")
        for c in cands:
            mk = sum(1 << int(k) for k in c.modules)
            if c.kind == "merge":
                mk |= 1 << m
            kinds.append(c.kind); masks.append(mk); sizes.append(c.size); names.append(c.name)
        return cls(kinds, np.array(masks, np.int64), np.array(sizes, int), list(parent_cells), names)

    def options(self, T):
        """(O, T) piece index per task (-1 none) and (O,) time cost; cached per T."""
        if self._opts is not None and self._opts[0] == T:
            return self._opts[1], self._opts[2]
        m = len(self.cells)
        full = (1 << m) - 1
        rows = []
        for combo in itertools.product(range(-1, self.C), repeat=T):
            used = [c for c in combo if c >= 0]
            if len(set(used)) != len(used):
                continue
            acc, ok = 0, True
            for c in used:
                if acc & int(self.masks[c]):
                    ok = False
                    break
                acc |= int(self.masks[c])
            if ok:
                rows.append(combo)
        P = np.array(rows, np.int64).reshape(-1, T)
        cost = np.array([self._time(r) for r in P])
        self._opts = (T, P, cost)
        return P, cost

    def _time(self, combo):
        m = len(self.cells)
        full = (1 << m) - 1
        used = [int(c) for c in combo if c >= 0]
        if not used:
            return 0.0
        n_merge = sum(self.kinds[c] == "merge" for c in used)
        pm = [int(self.masks[c]) & full for c in used]
        if len(used) == 1 and pm[0] == full:          # the whole parent (or the whole parent merged)
            return T_SPLIT_S * n_merge
        left = full
        for x in pm:
            left &= ~x
        parts = len(used) + _components([self.cells[k] for k in range(m) if left >> k & 1])
        return T_SPLIT_S * (parts - 1 + n_merge)


def _components(cells):
    cells = set(cells)
    n = 0
    while cells:
        n += 1
        stack = [cells.pop()]
        while stack:
            x, y = stack.pop()
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c = (x + d[0], y + d[1])
                if c in cells:
                    cells.remove(c)
                    stack.append(c)
    return n


# ----------------------------------------------------------------------------- certification and decisions
def capable(rho, R, min_mod, sizes):
    """rho (..., C, L) -> (..., C, T) bool: every required leg >= r (NaN rho fails) and size >= min_modules."""
    rho = np.asarray(rho, float)
    req = np.isfinite(R)                                              # (T, L)
    ge = rho[..., :, None, :] >= np.where(req, R, -np.inf)[None]     # (..., C, T, L); NaN rho -> False
    ok = np.where(req[None], ge & np.isfinite(rho[..., :, None, :]), True).all(-1)
    return ok & (np.asarray(sizes)[:, None] >= np.asarray(min_mod)[None, :])


def certified(lower, R, min_mod, sizes):
    """Certified pairs (C, T) from lower bounds (C, L); empty or non-finite -> nothing certified."""
    lower = np.asarray(lower, float)
    if lower.size == 0 or lower.shape[0] != len(sizes):
        return np.zeros((len(sizes), len(R)), bool)
    return capable(lower, R, min_mod, sizes)


def best_option(cert, P, cost, values):
    """cert (C, T) -> index of the best option (ties: lower time, then the earlier option, abstain is P row 0)."""
    C, T = cert.shape
    ext = np.vstack([cert, np.ones((1, T), bool)])                    # row C = "no piece"
    Pi = np.where(P < 0, C, P)
    ok = ext[Pi, np.arange(T)[None]].all(1)
    val = ((P >= 0) * values[None]).sum(1) - cost
    val = np.where(ok, val, -np.inf)
    best = np.max(val)
    cand = np.flatnonzero(val >= best - 1e-9)
    return int(cand[np.argmin(cost[cand])])


def decide(inst, tasks, lower):
    """-> (option row (T,), time cost). Abstain = all -1."""
    T = len(tasks)
    if T == 0:
        return np.zeros(0, np.int64), 0.0
    R, mm, V = task_arrays(tasks, np.asarray(lower).shape[-1] if np.asarray(lower).ndim else 0)
    P, cost = inst.options(T)
    cert = certified(lower, R, mm, inst.sizes)
    o = best_option(cert, P, cost, V)
    return P[o], float(cost[o])


def outcome(inst, tasks, choice, time_cost, rho_true, probe_time=0.0):
    """Realised result of a committed option under the true capability rho_true (C, L)."""
    R, mm, V = task_arrays(tasks, rho_true.shape[-1])
    cap = capable(rho_true, R, mm, inst.sizes)
    a = choice >= 0
    tr = np.array([cap[c, t] if c >= 0 else False for t, c in enumerate(choice)], bool)
    served = a & tr
    unsafe = a & ~tr
    U = float((V * served).sum() - C_UNSAFE * unsafe.sum() - (time_cost if a.any() else 0.0) - probe_time)
    return dict(n_tasks=len(tasks), n_assigned=int(a.sum()), n_served=int(served.sum()), n_unsafe=int(unsafe.sum()),
                abstain=int(not a.any()), time_s=float(time_cost if a.any() else 0.0), probe_s=float(probe_time), U=U)


def oracle(inst, tasks, rho_true):
    ch, tc = decide(inst, tasks, rho_true)
    return ch, tc, outcome(inst, tasks, ch, tc, rho_true)


def expected_utility(inst, tasks, choice, time_cost, rho_samples, w):
    """E_w[U] of a fixed option under posterior samples rho (S, C, L)."""
    R, mm, V = task_arrays(tasks, rho_samples.shape[-1])
    if not (choice >= 0).any():
        return 0.0
    cap = capable(rho_samples, R, mm, inst.sizes)                     # (S, C, T)
    u = 0.0
    for t, c in enumerate(choice):
        if c >= 0:
            p = float(np.dot(w, cap[:, c, t]))
            u += V[t] * p - C_UNSAFE * (1 - p)
    return u - time_cost
