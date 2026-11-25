"""cfcap.env: HM3D calibration bar on a sample, footprint hand cases, determinism by seed, segment and label ranges."""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from cfcap.env import buildings as B
from cfcap.env import hm3d_stats as HS
from cfcap.env import missions as M

HAVE_STATS = HS.OUT.exists()
needs_stats = pytest.mark.skipif(not HAVE_STATS, reason="run python -m cfcap.env.hm3d_stats first")


@needs_stats
def test_calibration_bar_on_sample():
    if not HS.SRC.exists():
        pytest.skip("HM3D-derived records not on this host")
    for r in B.calibrate(600, "train"):
        assert r["passed"], r


@needs_stats
def test_determinism_by_seed():
    a, b = B.make_mission("test", 17), B.make_mission("test", 17)
    assert a.legs == b.legs and a.turns == b.turns and a.payload_kg == b.payload_kg
    assert [(d.s, d.width) for d in a.doors] == [(d.s, d.width) for d in b.doors]
    sa, sb = M.segments(a), M.segments(b)
    assert [(s.leg, s.v_req, s.mu) for s in sa] == [(s.leg, s.v_req, s.mu) for s in sb]
    c = B.make_mission("cal", 17)
    assert c.legs != a.legs                                   # disjoint seed streams per family


@needs_stats
def test_mission_ranges_and_families():
    for fam in B.FAMILIES:
        for i in range(30):
            m = B.make_mission(fam, i)
            assert all(B.DOOR_W[0] - 1e-9 <= d.width <= B.DOOR_W[1] + 1e-9 for d in m.doors)
            assert sum(d.narrow for d in m.doors) == 1
            assert all(B.RADIUS[0] - 1e-9 <= r <= B.RADIUS[1] for _, r in m.turns)
            assert all(B.MU_RANGE[0] - 1e-9 <= f <= B.MU_RANGE[1] + 1e-9 for *_, f in m.patches)
            if fam == "E-H1":
                assert m.doors[m.passage].width <= B.E_H1_NARROW_MAX
            if fam == "E-H4":
                assert 5.0 <= m.payload_kg <= 7.0
            else:
                assert 0.0 <= m.payload_kg <= B.PAYLOAD_MAX
            segs = M.segments(m)
            assert segs and all(0 <= s.leg < M.L and s.v_req > 0 for s in segs)
            assert sum(s.kind == "door" for s in segs) == len(m.doors)
            assert sum(s.length for s in segs) == pytest.approx(m.route_m, rel=1e-6)


class _Asm:
    def __init__(self, cells):
        self.modules = [("M1", i, j, 0) for i, j in cells]
        self.meta = {"pitch": 0.32}
        self.side = 0.24


def test_footprint_hand_cases():
    w = M.footprint_width
    assert w(_Asm([(0, 0)])) == pytest.approx(0.24)
    assert w(_Asm([(0, 0), (1, 0), (2, 0)])) == pytest.approx(0.24)       # a line strafes through sideways
    assert w(_Asm([(0, 0), (1, 0), (0, 1), (1, 1)])) == pytest.approx(0.56)
    assert w(_Asm([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)])) == pytest.approx(0.56)
    assert w(_Asm([(0, 0), (1, 1), (2, 2)])) == pytest.approx(0.88)       # bounding box, 3 x 3
    # door 0.60: a single or a line passes (0.34), a 2x2 does not (0.66)
    assert w(_Asm([(0, 0)])) + M.CLEARANCE <= 0.60
    assert w(_Asm([(0, 0), (1, 0), (0, 1), (1, 1)])) + M.CLEARANCE > 0.60


def test_cmd_speed():
    tw = np.array([(0.3, 0, 0), (0.21, 0.21, 0), (0, 0, 0.8)])
    assert M.cmd_speed(tw) == pytest.approx([0.3, math.hypot(0.21, 0.21), 0.8])


@needs_stats
def test_labels_range_and_record():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    ms = [B.make_mission("test", i) for i in range(2)]
    for r in M.label_missions(ms, engine="eager"):
        C, S = r["req"].shape
        assert r["rho_dec"].shape == (C, len(r["cond_mu"]), M.L)
        assert np.isfinite(r["rho_dec"]).all()
        assert (r["rho_dec"] > -0.5).all() and (r["rho_dec"] < 1.6).all()
        assert (r["req"] > 0).all()
        ok = r["fp_ok"] & r["load_ok"] & (r["seg_truth"] >= r["req"])
        assert (ok == r["seg_ok"]).all()
        assert (r["mission_ok"] == r["seg_ok"].all(1)).all()
        assert ((r["cand_width"] + M.CLEARANCE <= r["narrowest"]) == r["cand_footprint_ok"]).all()


@needs_stats
def test_record_and_flatten_with_stub_labels(monkeypatch):
    """Record / interface logic on stub labels (CPU): rho_dec = mu factor (0.5-1.1) per item, rho_ref = 0.9."""
    from cfcap.env import task as T
    from cfcap.gen import simrun as SR

    def stub(items, **kw):
        n = len(items)
        mu = np.array([it["mu"] for it in items], np.float32)[:, None]
        rho = np.repeat(mu * 0.9, M.L, 1)
        return dict(rho=rho, rho_ref=np.full((n, M.L), 0.9, np.float32), rho_dec=rho / 0.9)

    monkeypatch.setattr(SR, "label_conditions", stub)
    ms = [B.make_mission("test", i) for i in range(4)] + [B.make_mission("E-H1", 0)]
    recs = M.label_missions(ms)
    for r in recs:
        C, S = r["req"].shape
        K = len(r["cond_mu"])
        assert r["rho_dec"].shape == (C, K, M.L)
        assert np.allclose(r["seg_truth"], r["seg_mu"][None, :])        # truth looked up at the segment's condition
        assert (r["seg_ok"] == (r["fp_ok"] & r["load_ok"] & (r["seg_truth"] >= r["req"]))).all()
        lb = r["rho_dec"]                                                 # a perfect bound certifies exactly the truth
        assert (M.certified(r, lb) == r["seg_ok"]).all()
        assert not M.certified(r, np.full_like(lb, np.nan)).any()
        opts = M.plan_options(r)
        assert (0, 0) in opts and all(0 <= a < C and 0 <= b < C for a, b in opts)
        assert all(M.plan_ok(r, r["seg_ok"], (c, c)) == bool(r["mission_ok"][c]) for c in range(C))
        # footprint: loaded segments need the load rating, doors need width + clearance
        door = ~np.isnan(r["seg_door_w"])
        assert (r["fp_ok"][:, ~door]).all()
    rows = T.flatten(recs)
    n = sum(r["req"].size for r in recs)
    assert all(len(v) == n for v in rows.values())


def test_spare_shard0_units():
    import argparse
    from cfcap.env import task as T
    a = argparse.Namespace(families="conf-ID", missions=1500, per_unit=250, spare_shard0=True)
    u0 = T._units(a, 0, 4)
    assert len(u0) == 1 and u0[0].startswith("env-test-900000")
    rest = sum((T._units(a, k, 4) for k in range(1, 4)), [])
    assert sorted(rest) == sorted(T._units(argparse.Namespace(**{**vars(a), "spare_shard0": False}), 0, 1))


def test_conf_block_bounds():
    """Confirmatory missions beyond the pre-registered buildings are refused (no building is drawn)."""
    with pytest.raises(ValueError):
        B.seeds("conf-ID", 300 * B.CONF_MISSIONS_PER_BUILDING)
    kf, b, bs, ms = B.seeds("conf-E-H2", 0)
    assert kf == "E-H2" and b == 450 and bs == [92599, 23, 450]


@needs_stats
def test_env_score_carries_runner_shard(tmp_path, monkeypatch):
    """env_score writes the runner shard on every mission row: from the row column, else the run manifest; never -1."""
    import importlib.util
    import json
    import pandas as pd
    from cfcap.env import task as T
    from cfcap.gen import simrun as SR

    def stub(items, **kw):
        n = len(items)
        mu = np.array([it["mu"] for it in items], np.float32)[:, None]
        return dict(rho=np.repeat(mu, M.L, 1), rho_ref=np.ones((n, M.L), np.float32), rho_dec=np.repeat(mu, M.L, 1))

    monkeypatch.setattr(SR, "label_conditions", stub)
    spec = importlib.util.spec_from_file_location("env_score", pathlib.Path(__file__).parents[1] / "scripts/env_score.py")
    ES = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ES)
    for k, (fam, with_col) in enumerate([("test", True), ("E-H1", False)]):
        rows = T.flatten(M.label_missions([B.make_mission(fam, i) for i in range(2)]))
        shard = 3 + k
        if with_col:
            rows["shard"] = np.full(rows["req"].size, shard, np.int16)
        u = tmp_path / f"run{k}" / "units"
        u.mkdir(parents=True)
        np.savez(u / f"env-{fam}-000000-000002.npz", **rows)
        (tmp_path / f"run{k}" / "manifest.json").write_text(json.dumps({"shard": shard}))
    out = tmp_path / "m.csv"
    ES.main([str(tmp_path / "run0"), str(tmp_path / "run1"), "--out", str(out)])
    df = pd.read_csv(out)
    assert len(df) and (df.shard >= 0).all()
    assert set(df[df.family == "test"].shard) == {3} and set(df[df.family == "E-H1"].shard) == {4}
    # a unit without a shard column or manifest is refused
    bad = tmp_path / "bad" / "units"
    bad.mkdir(parents=True)
    rows.pop("shard", None)
    np.savez(bad / "env-test-000000-000002.npz", **rows)
    with pytest.raises(SystemExit):
        ES.load_rows([str(tmp_path / "bad")])
