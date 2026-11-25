"""Amortized posterior: shapes, permutation equivariance over modules, quantile monotonicity, losses (CPU)."""
import numpy as np
import pytest
import torch

from cfcap.models import data as D
from cfcap.models import posterior as M
from cfcap.t1 import params as P


@pytest.fixture(scope="module")
def square():
    par = P.base_params("plated")
    return P.named_assembly("square", par), par


def _inputs(asm, B=3, T=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    n, m = asm.n, len(asm.modules)
    return dict(wf=torch.randn(B, T, n, 3, generator=g), af=torch.randn(B, T, 6, generator=g),
                geom=torch.as_tensor(M.wheel_geometry(asm)).expand(B, -1, -1),
                wm=torch.as_tensor(np.asarray(asm.wheel_module)).long().expand(B, -1),
                mask=torch.ones(B, n, dtype=torch.bool), n_mod=m)


def _model():
    torch.manual_seed(0)
    return M.MorphPosterior(h=32, layers=2, heads=4).eval()


def test_shapes(square):
    asm, _ = square
    x = _inputs(asm)
    C = 2
    member = torch.zeros(3, C, asm.n, dtype=torch.bool)
    member[:, 0, :8] = True
    member[:, 1] = True
    cgeom = torch.randn(3, C, asm.n, M.GEOM_DIM)
    out = _model()(**x, member=member, cgeom=cgeom)
    assert out["logits"].shape == (3, 16, 4)
    assert out["ab"].shape == (3, 16, 2) and bool((out["ab"] >= 1).all())
    assert out["nuis"].shape == (3, 4, 2 * len(M.NUIS))
    assert out["cf"].shape == (3, C, M.N_LEGS, len(M.QUANTILES))
    assert bool((out["cf"].diff(dim=-1) >= 0).all())


def test_module_permutation_equivariance(square):
    """Relabelling modules (moving whole 4-wheel blocks with their geometry) permutes the per-wheel and per-module
    outputs the same way and leaves the counterfactual output of the same physical child unchanged."""
    asm, _ = square
    x = _inputs(asm)
    member = torch.zeros(3, 1, asm.n, dtype=torch.bool)
    member[:, 0, 4:12] = True                                  # modules 1 and 2
    cgeom = torch.randn(3, 1, asm.n, M.GEOM_DIM)
    model = _model()
    perm_m = torch.tensor([2, 0, 3, 1])                        # new module k = old module perm_m[k]
    pw = (perm_m[:, None] * 4 + torch.arange(4)).reshape(-1)   # wheel permutation
    inv = torch.empty_like(perm_m)
    inv[perm_m] = torch.arange(4)
    xp = dict(x, wf=x["wf"][:, :, pw], geom=x["geom"][:, pw], wm=inv[x["wm"][:, pw]])
    with torch.no_grad():
        o = model(**x, member=member, cgeom=cgeom)
        op = model(**xp, member=member[..., pw], cgeom=cgeom[:, :, pw])
    torch.testing.assert_close(op["logits"], o["logits"][:, pw], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(op["ab"], o["ab"][:, pw], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(op["nuis"][:, inv], o["nuis"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(op["cf"], o["cf"], atol=1e-4, rtol=1e-4)


def test_padding_invariance(square):
    """Padded wheels (mask False) do not change the outputs of real wheels."""
    asm, _ = square
    x = _inputs(asm, B=2)
    pad = 4
    xp = dict(x, wf=torch.cat([x["wf"], torch.randn(2, x["wf"].shape[1], pad, 3)], 2),
              geom=torch.cat([x["geom"], torch.randn(2, pad, M.GEOM_DIM)], 1),
              wm=torch.cat([x["wm"], torch.full((2, pad), 3)], 1),
              mask=torch.cat([x["mask"], torch.zeros(2, pad, dtype=torch.bool)], 1))
    model = _model()
    with torch.no_grad():
        o, op = model(**x), model(**xp)
    torch.testing.assert_close(op["logits"][:, :16], o["logits"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(op["nuis"], o["nuis"], atol=1e-4, rtol=1e-4)


def test_merge_call(square):
    """The direct head accepts a concatenated token set (merge of two parents)."""
    asm, _ = square
    model = _model()
    with torch.no_grad():
        z1, _ = model.encode(**{k: v for k, v in _inputs(asm, seed=1).items() if k not in ("wm", "n_mod")})
        z2, _ = model.encode(**{k: v for k, v in _inputs(asm, seed=2).items() if k not in ("wm", "n_mod")})
        z = torch.cat([z1, z2], 1)
        member = torch.ones(3, 1, 32, dtype=torch.bool)
        q = model.counterfactual(z, member, torch.randn(3, 1, 32, M.GEOM_DIM))
    assert q.shape == (3, 1, M.N_LEGS, len(M.QUANTILES))


def test_losses_finite(square):
    asm, _ = square
    x = _inputs(asm)
    member = torch.ones(3, 2, asm.n, dtype=torch.bool)
    cgeom = torch.randn(3, 2, asm.n, M.GEOM_DIM)
    out = _model().train()(**x, member=member, cgeom=cgeom)
    mode = torch.zeros(3, 16, dtype=torch.long)
    mode[0, 3], mode[1, 7] = 1, 2
    g = torch.ones(3, 16)
    g[0, 3] = 0.8
    batch = dict(mask=x["mask"], mode=mode, g=g, nuis=torch.randn(3, 4, 3), mod_mask=torch.ones(3, 4),
                 rho_rel=torch.rand(3, 2, M.N_LEGS))
    loss, parts = M.loss_fn(out, batch)
    assert torch.isfinite(loss) and set(parts) == {"ce", "beta", "nuis", "cf"}
    loss.backward()


def test_pinball_minimised_at_quantile():
    y = torch.linspace(0, 1, 1001)
    losses = [float(M.pinball(torch.full((1001, 1), c), y, (0.9,))) for c in (0.8, 0.9, 1.0)]
    assert losses[1] < losses[0] and losses[1] < losses[2]


def test_connected_subsets():
    assert len(D.connected_subsets(P.SHAPES["pair"])) == 3
    assert len(D.connected_subsets(P.SHAPES["square"])) == 13       # 4 + 4 + 4 + 1
    assert (0, 3) not in D.connected_subsets(P.SHAPES["square"])    # diagonal cells are not connected


def test_deepsets_equivariance(square):
    asm, _ = square
    x = _inputs(asm)
    torch.manual_seed(0)
    model = M.MorphPosterior(h=32, layers=0, heads=4).eval()
    pw = (torch.tensor([3, 1, 0, 2])[:, None] * 4 + torch.arange(4)).reshape(-1)
    inv = torch.empty(4, dtype=torch.long)
    inv[torch.tensor([3, 1, 0, 2])] = torch.arange(4)
    xp = dict(x, wf=x["wf"][:, :, pw], geom=x["geom"][:, pw], wm=inv[x["wm"][:, pw]])
    with torch.no_grad():
        o, op = model(**x), model(**xp)
    torch.testing.assert_close(op["logits"], o["logits"][:, pw], atol=1e-4, rtol=1e-4)


def test_train_task_grid_and_filters():
    import argparse
    from cfcap.gen import episode as E
    from cfcap.models import train as TR
    g = TR.grid("b1")
    assert len(g) == len({(c["arch"], c["div"], c["seed"]) for c in g})          # no duplicate runs
    assert {c["arch"] for c in g} == set(TR.ARCHS) and {c["div"] for c in g} == set(TR.DIVERSITY)
    p = argparse.ArgumentParser()
    TR.add_args(p)
    a = p.parse_args([])
    us = TR.units(a, 0, len(g))
    assert us[-4:] == [f"eval-{s}" for s in TR.EVAL_SPLITS] and len(us) == a.chunks + 4
    with pytest.raises(ValueError):
        TR.units(a, 0, len(g) + 1)
    # the parent pre-filter agrees with the generator's own episode (no simulation needed)
    idx = TR.train_indices("n4c", 0, 15)
    for i in idx[:5]:
        ep = E.make_episode("plated", "train", i)
        assert ep.parent.m <= 4 and ep.parent.topo in ("single", "chain", "grid")
    assert TR.n_train_families("n2") == 2 and TR.n_train_families("n6") > TR.n_train_families("n4")


def test_build_morph_matches_named(square):
    asm, par = square
    b = D.build_morph("square", par)
    np.testing.assert_allclose(b.wheel_pos, asm.wheel_pos)
    child = D.build_morph([(0, 0, 1), (1, 0, 3)], par)
    assert child.n == 8 and not np.allclose(child.r_hat, D.build_morph("pair", par).r_hat)
