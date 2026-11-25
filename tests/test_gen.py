"""Stage B generator tests: determinism, frozen morphology splits, label range, healthy candidates rho_rel = 1."""
import numpy as np
import pytest
import torch

from cfcap.gen import episode as E
from cfcap.gen import families as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _ep_arrays(ep):
    out = [ep.dom[k] for k in E.DOM_KEYS]
    for a in ep.asms:
        out += [a.mode, a.g, a.stall_w, a.noload_w, a.yaw_err, a.load_z, a.vbatt_off, a.mu_mod,
                np.array(a.asm.modules, dtype=object)]
    for c in ep.cands:
        out += [c.asm.name, c.modules, c.load_z, c.payload]
    return out


@pytest.mark.parametrize("split", ["train", "H3", "H4", "H6", "H7"])
def test_episode_deterministic(split):
    a, b = E.make_episode("plated", split, 17), E.make_episode("plated", split, 17)
    for x, y in zip(_ep_arrays(a), _ep_arrays(b)):
        if isinstance(x, np.ndarray) and x.dtype != object:
            np.testing.assert_array_equal(x, y)
        else:
            assert repr(x) == repr(y)
    c = E.make_episode("plated", split, 18)
    assert repr(_ep_arrays(a)) != repr(_ep_arrays(c))


def test_streams_differ_by_split_and_plate():
    eps = [E.make_episode(p, s, 3) for p in E.PLATES for s in ("train", "cal", "test")]
    doms = {tuple(e.dom.values()) for e in eps}
    assert len(doms) == len(eps)


def test_family_canonical_invariance():
    t = [(0, 0), (1, 0), (2, 0), (1, 1)]
    rot = [(-y, x) for x, y in t]
    ref = [(-x, y) for x, y in t]
    assert F.family_hash(t) == F.family_hash(rot) == F.family_hash(ref)
    assert F.topology(t) == "T" and F.family_split(t) == F.FAM_H3
    assert F.topology([(0, 0), (1, 0), (0, 1), (1, 1)]) == "grid"
    assert F.family_split([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]) == F.FAM_TRAIN
    assert F.family_split([(i, 0) for i in range(7)]) == F.FAM_H4


def test_family_splits_disjoint_and_frozen():
    rows = F.family_table(6)
    by = {}
    for n, h, topo, split, cells, _ in rows:
        by.setdefault(split, set()).add(h)
    assert not (by["train"] & by["H3"])
    assert len({r[1] for r in rows}) == len(rows)            # no hash collision among the 56 free polyominoes n <= 6
    assert len(by["train"]) == 37 and len(by["H3"]) == 19


@pytest.mark.parametrize("split", ["train", "cal", "test", "H6", "H7"])
def test_train_like_episodes_never_touch_heldout_families(split):
    for i in range(60):
        ep = E.make_episode("plated", split, i)
        assert ep.parent.fam_split == F.FAM_TRAIN and ep.parent.m <= 6
        for c in ep.cands:
            assert c.fam_split == F.FAM_TRAIN and len(c.modules) <= 6


def test_heldout_parents():
    for i in range(30):
        assert E.make_episode("plated", "H3", i).parent.fam_split == F.FAM_H3
        h4 = E.make_episode("plated", "H4", i)
        assert 7 <= h4.parent.m <= 10 and all(len(c.modules) <= 10 for c in h4.cands)
        h6 = E.make_episode("plated", "H6", i).parent
        per_mod = (h6.mode.reshape(-1, 4) != 0).sum(1)
        assert per_mod.max() >= 2
        tr = E.make_episode("plated", "train", i)
        for a in tr.asms:
            assert (a.mode.reshape(-1, 4) != 0).sum(1).max() <= 1   # H6 combos never in train
        h7 = E.make_episode("plated", "H7", i)
        assert h7.payload is not None and h7.payload[0] >= 3.0
        if tr.payload is not None:
            assert tr.payload[0] <= 2.5


def test_candidates_connected_and_carry_latents():
    from cfcap.t0.candidates import _connected
    for i in range(20):
        ep = E.make_episode("plated", "train", i)
        assert ep.cands[0].kind == E.KIND["self"]
        for c in ep.cands:
            assert _connected([(m[1], m[2]) for m in c.asm.modules])
            lat = E.plant_latents(ep, c.modules, c.load_z, c.payload)
            assert lat["mode"].shape == (1, c.asm.n)
            r, k = c.modules[0]
            np.testing.assert_array_equal(lat["stall_w"][0, :4].numpy(), ep.asms[r].stall_w[4 * k:4 * k + 4])


@cuda
def test_labels_range_and_healthy_twin():
    from cfcap.gen import simrun as S
    eps = [E.make_episode("plated", "train", i) for i in range(6)]
    lab = S.label_episodes(eps, engine="graph", skip_healthy_twin=False)
    n_healthy = 0
    for ep, l in zip(eps, lab):
        assert np.isfinite(l["rho"]).all() and np.isfinite(l["rho_rel"]).all()
        assert (l["rho_healthy"] > 0.5).all() and (l["rho_healthy"] < 1.1).all()
        assert (l["rho_rel"] > -1.0).all() and (l["rho_rel"] < 1.3).all()
        for c, rr in zip(ep.cands, l["rho_rel"]):
            nf = sum(int((ep.asms[r].mode[4 * k:4 * k + 4] != 0).sum()) for r, k in c.modules)
            if nf == 0:
                np.testing.assert_allclose(rr, 1.0, atol=1e-5)
                n_healthy += 1
    assert n_healthy > 10
    lab2 = S.label_episodes(eps, engine="graph")                # twin skipped for healthy candidates
    for a, b in zip(lab, lab2):
        np.testing.assert_allclose(a["rho"], b["rho"], atol=1e-5)
        np.testing.assert_allclose(a["rho_rel"], b["rho_rel"], atol=1e-5)


@cuda
def test_rho_matches_capability_module():
    from cfcap.gen import simrun as S
    from cfcap.t1 import capability as cap
    ep = E.make_episode("plated", "train", 5)
    lab = S.label_episodes([ep], engine="graph")[0]
    for ci in (0, len(ep.cands) - 1):
        c = ep.cands[ci]
        lat = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in E.plant_latents(ep, c.modules, c.load_z,
                                                                                         c.payload).items()}
        lat["dom"] = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in lat["dom"].items()}
        if lat["payload"] is not None:
            lat["payload"] = {k: v.cuda() for k, v in lat["payload"].items()}
        ref = cap.capability([(c.asm, lat)], E.params("plated"), engine="graph")[0]["rho"].cpu().numpy()[0]
        np.testing.assert_allclose(lab["rho"][ci], ref, atol=2e-4)


@cuda
def test_shard_deterministic_and_resumable(tmp_path):
    from cfcap.gen import shard as SH
    i1 = SH.write_shard(tmp_path / "a", "plated", "train", 1, 6, ("v2",), engine="graph", log=None)
    i2 = SH.write_shard(tmp_path / "b", "plated", "train", 1, 6, ("v2",), engine="graph", log=None)
    A, B = SH.load(tmp_path / "a/plated/train/shard_00001.npz"), SH.load(tmp_path / "b/plated/train/shard_00001.npz")
    assert A.keys() == B.keys()
    for k in A:
        np.testing.assert_array_equal(A[k], B[k], err_msg=k)
    assert list(A["ep_index"]) == list(range(6, 12))
    skip = SH.write_shard(tmp_path / "a", "plated", "train", 1, 6, ("v2",), engine="graph", log=None)
    assert skip["sha256"] == i1["sha256"] and skip["created"] == i1["created"]
    with pytest.raises(ValueError):
        SH.write_shard(tmp_path / "a", "plated", "train", 2, 7, ("v2",), engine="graph", log=None)
    # same episode in a differently composed batch: identical latents, labels equal to float tolerance
    D = SH.concat([A])
    C = SH.load(SH.write_shard(tmp_path / "c", "plated", "train", 0, 12, ("v2",), engine="graph", block=5,
                               log=None) and tmp_path / "c/plated/train/shard_00000.npz")
    e = 7
    a0, a1 = D["ep_cand_off"][e - 6:e - 6 + 2]
    c0, c1 = C["ep_cand_off"][e:e + 2]
    np.testing.assert_allclose(D["cand_rho_rel"][a0:a1], C["cand_rho_rel"][c0:c1], atol=1e-4)
    np.testing.assert_array_equal(D["cand_family"][a0:a1], C["cand_family"][c0:c1])


@cuda
def test_rho_ref_nominal_and_rho_dec():
    from cfcap.gen import rho_ref as RR
    from cfcap.gen import simrun as S
    from cfcap.t1 import capability as cap
    eps = [E.make_episode("plated", "train", i) for i in range(3)]
    lab = S.label_episodes(eps, engine="graph")
    ep, l = eps[0], lab[0]
    np.testing.assert_allclose(l["rho_dec"], l["rho"] / l["rho_ref"], rtol=1e-6)
    assert (l["rho_ref"] > 0.5).all() and (l["rho_ref"] < 1.1).all()
    c = ep.cands[-1]
    lat = RR.nominal_latents(c.asm, "plated", 1)
    lat = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in lat.items()}
    lat["dom"] = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in lat["dom"].items()}
    ref = cap.capability([(c.asm, lat)], E.params("plated"), engine="graph")[0]["rho"].cpu().numpy()[0]
    np.testing.assert_allclose(l["rho_ref"][-1], ref, atol=2e-4)
    # the same shape key in another episode gets the identical (cached) reference
    k = RR.key(c.asm)
    np.testing.assert_array_equal(RR.get("plated", k), l["rho_ref"][-1])


def test_new_splits_and_types():
    from cfcap.t1 import plant as PL
    for i in range(6):
        a, b = E.make_episode("plated", "H4b", i), E.make_episode("plated", "H4b", i)
        assert 11 <= a.parent.m <= 16 and repr(_ep_arrays(a)) == repr(_ep_arrays(b))
        assert all(len(c.modules) <= 16 for c in a.cands) and len(a.cands) > 10
    for i in range(30):
        h = E.make_episode("plated", "het_train", i)
        for c in h.cands:
            assert len(set(c.types.tolist())) == 1 and PL.OMNI not in c.types
        assert (E.make_episode("plated", "H1", i).parent.mtype == PL.OMNI).all()
        p2 = E.make_episode("plated", "H2", i).parent.mtype
        assert set(p2.tolist()) == {PL.MECANUM, PL.CONVENTIONAL}
        assert (E.make_episode("plated", "train", i).parent.mtype == 0).all()
    # confirmatory twins: same rule, fresh streams
    a, b = E.make_episode("plated", "test", 10**7), E.make_episode("plated", "conf_test", 10**7)   # far outside any scored range
    assert a.dom != b.dom and E.RULES["conf_test"] is E.RULES["test"]
    asm = E.build("x", [(0, 0), (1, 0)], [0, 1], ["M1", "M1"], "plated", types=[1, 0])
    assert list(asm.meta["wheel_type"]) == [1] * 4 + [0] * 4


def test_mecanum_episode_unchanged_by_type_support():
    ep = E.make_episode("plated", "train", 5)
    lat = E.plant_latents(ep, ep.cands[0].modules, ep.cands[0].load_z, ep.cands[0].payload)
    assert "wtype" not in lat


@cuda
def test_conventional_module_cannot_sway():
    from cfcap.gen import rho_ref as RR
    from cfcap.t1 import capability as cap
    asm = E.build("single", [(0, 0)], [0], ["M1"], "plated")
    out = {}
    for t in (0, 1, 2):
        lat = RR.nominal_latents(asm, "plated", 1, types=[t] if t else None)
        lat = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in lat.items()}
        lat["dom"] = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in lat["dom"].items()}
        out[t] = cap.capability([(asm, lat)], E.params("plated"), engine="graph")[0]["rho"].cpu().numpy()[0]
    sway = [cap.LEGS.index("sway+"), cap.LEGS.index("sway-")]
    surge = [cap.LEGS.index("surge+"), cap.LEGS.index("surge-")]
    assert (out[0][sway] > 0.8).all()
    assert (np.abs(out[1][sway]) < 0.1).all()                  # skid module: ~zero lone sway capability
    assert (out[1][surge] > 0.5).all()
    assert not np.allclose(out[2], out[0], atol=1e-3)           # omni-like differs from mecanum
