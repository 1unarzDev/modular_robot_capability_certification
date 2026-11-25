"""Morphology families, topology classes and the frozen split rule of the Stage B generator.

A *family* is a free polyomino: the module cell set up to translation, rotation and reflection (module yaws are
within-family variation). Its hash is blake2b-64 of the canonical sorted cell tuple. The split a family belongs to is
a fixed rule of its canonical form (size and topology), so every morphology, parent or candidate, lands in exactly one
family split whatever episode it appears in:

  FAM_TRAIN  N <= 6, topology chain / L / path (zigzag) / cyclic / grid (single included)   -> train, cal, test
  FAM_H3     N <= 6, a junction tree: T, star (plus) or branched (several junctions)        -> capplan H3 (topology)
  FAM_H4     N >= 7, any topology                                                          -> capplan H4a (scale)

Grid (square, 2x3) stays in train because it holds the field shapes; T/star/branched are the unseen topology classes.
Topology classes follow capplan `configs.classify_topology` on the cell adjacency graph (chain and side merge into
"chain" because families are rotation-free).
"""
from __future__ import annotations

import functools
import hashlib

import numpy as np

from capxfer import assembly as A

FAM_TRAIN, FAM_H3, FAM_H4 = 0, 3, 4
FAM_NAMES = {FAM_TRAIN: "train", FAM_H3: "H3", FAM_H4: "H4"}
TOPOLOGIES = ("single", "chain", "L", "path", "cyclic", "grid", "T", "star", "branched")
JUNCTION = ("T", "star", "branched")
MAX_TRAIN_N = 6
_SYM = ((1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0),
        (-1, 0, 0, 1), (1, 0, 0, -1), (0, 1, 1, 0), (0, -1, -1, 0))


def _norm(cells):
    mx = min(c[0] for c in cells)
    my = min(c[1] for c in cells)
    return tuple(sorted((int(x - mx), int(y - my)) for x, y in cells))


def canonical(cells):
    """Free-polyomino canonical form: the lexicographically smallest normalised image under the 8 symmetries."""
    return min(_norm([(a * x + b * y, c * x + d * y) for x, y in cells]) for a, b, c, d in _SYM)


def family_hash(cells):
    """Signed 64-bit family id (stored as int64)."""
    key = ";".join(f"{x},{y}" for x, y in canonical(cells)).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little", signed=True)


def topology(cells):
    cells = [tuple(c) for c in cells]
    n = len(cells)
    if n == 1:
        return "single"
    S = set(cells)
    nb = {c: [(c[0] + dx, c[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (c[0] + dx, c[1] + dy) in S]
          for c in cells}
    edges = sum(len(v) for v in nb.values()) // 2
    if edges >= n:
        xs, ys = [c[0] for c in cells], [c[1] for c in cells]
        return "grid" if (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) == n else "cyclic"
    deg = {c: len(v) for c, v in nb.items()}
    md = max(deg.values())
    if md <= 2:
        end = next(c for c, d in deg.items() if d == 1)
        path, prev = [end], None
        while len(path) < n:
            nxt = [c for c in nb[path[-1]] if c != prev][0]
            prev = path[-1]
            path.append(nxt)
        ax = ["x" if a[0] != b[0] else "y" for a, b in zip(path, path[1:])]
        bends = sum(p != q for p, q in zip(ax, ax[1:]))
        return "chain" if bends == 0 else ("L" if bends == 1 else "path")
    if md >= 4:
        return "star"
    return "T" if sum(d == 3 for d in deg.values()) == 1 else "branched"


def family_split(cells):
    if len(cells) > MAX_TRAIN_N:
        return FAM_H4
    return FAM_H3 if topology(cells) in JUNCTION else FAM_TRAIN


@functools.lru_cache(maxsize=None)
def fixed_shapes(k):
    """Fixed polyominoes of size k (capxfer assembly._polyominoes), cached."""
    return tuple(A._polyominoes(k))


@functools.lru_cache(maxsize=None)
def shapes_by_split(k):
    """{family split: tuple of fixed shapes of size k in it}."""
    out = {}
    for s in fixed_shapes(k):
        out.setdefault(family_split(s), []).append(s)
    return {k_: tuple(v) for k_, v in out.items()}


def family_table(max_n=6):
    """Rows (n, family_hash, topology, split, canonical cells, n_fixed) for every free polyomino with n <= max_n."""
    rows = {}
    for k in range(1, max_n + 1):
        for s in fixed_shapes(k):
            c = canonical(s)
            if c not in rows:
                rows[c] = [k, family_hash(c), topology(c), FAM_NAMES[family_split(c)], c, 0]
            rows[c][5] += 1
    return sorted(rows.values(), key=lambda r: (r[0], r[2], r[4]))


def sample_shape(rng, size_w, split):
    """A uniform fixed polyomino inside family split `split`, size drawn from size_w {size: weight} (sizes without a
    shape in the split are dropped)."""
    sizes = [k for k, w in sorted(size_w.items()) if w > 0 and split in shapes_by_split(k)]
    w = np.array([size_w[k] for k in sizes], float)
    k = int(rng.choice(sizes, p=w / w.sum()))
    sh = shapes_by_split(k)[split]
    return sh[int(rng.integers(len(sh)))]
