"""Candidate morphologies of a parent assembly: connected children (splits at module boundaries) and merges.

Module identity is tracked through `src`: for every wheel of a candidate, the index of that wheel in the parent
(>= 0) or -(1 + index) in the other assembly (< 0). A candidate's plant keeps each wheel's latents (fault mode, gain,
motor spread, mu factor) through this map; geometry, origin, CG and normal loads are recomputed (capxfer
`assembly.build`, `normal_loads`).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from capxfer import assembly as A


@dataclass
class Candidate:
    name: str
    kind: str              # "child" | "merge"
    asm: object
    src: np.ndarray        # (n_c,) parent wheel index (>= 0) or -(1 + other-assembly wheel index)
    modules: tuple         # parent module indices kept (merge: those of the parent piece)
    n_other: int = 0       # modules from the other assembly

    @property
    def size(self):
        return len(self.asm.modules)


def cells_of(asm):
    return [(int(i), int(j)) for _, i, j, _ in asm.modules]


def _connected(cells):
    cells = set(cells)
    if not cells:
        return False
    start = next(iter(cells))
    seen, stack = {start}, [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = (x + dx, y + dy)
            if c in cells and c not in seen:
                seen.add(c)
                stack.append(c)
    return len(seen) == len(cells)


def connected_subsets(cells, include_full=False):
    """All connected proper subsets (as sorted index tuples) of a polyomino's cells."""
    m = len(cells)
    out = []
    for k in range(1, m + (1 if include_full else 0)):
        for sub in itertools.combinations(range(m), k):
            if _connected([cells[i] for i in sub]):
                out.append(sub)
    return out


def wheel_index(asm, module_idx):
    return np.concatenate([np.flatnonzero(asm.wheel_module == k) for k in module_idx])


def children(parent, params, include_full=False):
    cells = cells_of(parent)
    out = []
    for sub in connected_subsets(cells, include_full):
        mods = [parent.modules[k] for k in sub]
        asm = A.build(f"{parent.name}/c{'-'.join(map(str, sub))}", mods, params)
        out.append(Candidate(asm.name, "child", asm, wheel_index(parent, sub), tuple(sub)))
    return out


def merge_placements(cells_a, cells_b):
    """Translations t of cells_b such that cells_a and cells_b + t do not overlap and touch on a face."""
    A_, B_ = set(cells_a), list(cells_b)
    out = []
    xa = [c[0] for c in cells_a]; ya = [c[1] for c in cells_a]
    xb = [c[0] for c in B_]; yb = [c[1] for c in B_]
    for tx in range(min(xa) - max(xb) - 1, max(xa) - min(xb) + 2):
        for ty in range(min(ya) - max(yb) - 1, max(ya) - min(yb) + 2):
            moved = [(x + tx, y + ty) for x, y in B_]
            if any(c in A_ for c in moved):
                continue
            if _connected(list(A_) + moved):
                out.append((tx, ty))
    return out


def merge(parent, piece_modules, other, t, params):
    """Merge parent modules `piece_modules` with the whole `other` assembly translated by t (grid cells)."""
    mods = [parent.modules[k] for k in piece_modules]
    omods = [(mid, i + t[0], j + t[1], yk) for (mid, i, j, yk) in other.modules]
    asm = A.build(f"{parent.name}/m{'-'.join(map(str, piece_modules))}+o{t}", mods + omods, params)
    src = np.concatenate([wheel_index(parent, piece_modules), -(1 + np.arange(other.n))])
    return Candidate(asm.name, "merge", asm, src, tuple(piece_modules), len(other.modules))


def sample_merges(parent, other, params, rng, n_merge=10):
    """Random merges: piece = the whole parent or one of its connected children; random valid placement."""
    cells = cells_of(parent)
    pieces = [tuple(range(len(cells)))] + connected_subsets(cells)
    ocells = cells_of(other)
    out, tries = [], 0
    while len(out) < n_merge and tries < 20 * n_merge:
        tries += 1
        piece = pieces[int(rng.integers(len(pieces)))]
        pl = merge_placements([cells[k] for k in piece], ocells)
        if not pl:
            continue
        t = pl[int(rng.integers(len(pl)))]
        c = merge(parent, piece, other, t, params)
        if all(c.name != o.name for o in out):
            out.append(c)
    return out


def carry(values_parent, values_other, src):
    """Per-wheel latent array of a candidate from parent / other per-wheel arrays (last axis = wheels)."""
    vp = np.asarray(values_parent)
    vo = np.asarray(values_other) if values_other is not None else None
    par = src >= 0
    out_shape = vp.shape[:-1] + (len(src),)
    out = np.empty(out_shape, dtype=np.result_type(vp, vo) if vo is not None else vp.dtype)
    out[..., par] = vp[..., src[par]]
    if (~par).any():
        out[..., ~par] = np.broadcast_to(vo, vp.shape[:-1] + vo.shape[-1:])[..., -(src[~par] + 1)]
    return out
