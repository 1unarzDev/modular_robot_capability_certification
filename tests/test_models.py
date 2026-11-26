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
    pytest.importorskip("cfcap.gen.episode")
    from cfcap.gen import episode as E
    from cfcap.models import train as TR
    g = TR.grid("b1")
    assert len(g) == len({(c["arch"], c["div"], c["seed"]) for c in g})          # no duplicate runs
    assert {c["arch"] for c in g} == {"st", "stl", "bb", "ds"} and {c["div"] for c in g} == set(TR.DIVERSITY) - {"het"}
    gh = TR.grid("het")
    assert {c["div"] for c in gh} == {"het"} and TR.units(argparse.Namespace(grid="het", chunks=2, smoke=False), 0,
                                                       len(gh))[-4:] == ["eval-het_cal", "eval-het_test", "eval-H1",
                                                                         "eval-H2"]
    assert TR.train_split("het") == "het_train" and TR.parent_ok("het", "train", 0) in (True, False)
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


def _fake_eval(rng, n_ep=30, eid0=0):
    rows = []
    for e in range(n_ep):
        for c in range(3):
            for leg in range(10):
                y = 1.0 if rng.uniform() < 0.7 else rng.uniform(0.4, 1.0)
                rows.append((eid0 + e, e, c, c % 3, 1 + c, int(y < 1), leg, y))
    a = np.array(rows, float)
    n = len(a)
    q50 = a[:, 7] + rng.normal(0, 0.03, n)
    d = dict(eid=a[:, 0].astype(np.int32), ep_row=a[:, 1].astype(np.int32), cand=a[:, 2].astype(np.int32),
             kind=a[:, 3].astype(np.int32), cand_size=a[:, 4].astype(np.int32), has_fault=a[:, 5].astype(np.int32),
             cand_family=np.zeros(n, np.int64), cand_fam_split=np.zeros(n, np.int32), parent_family=np.zeros(n, np.int64),
             parent_n=np.full(n, 4, np.int32), parent_topo=np.full(n, 5, np.int32), leg=a[:, 6].astype(np.int32),
             truth=a[:, 7].astype(np.float32), truth_rel=a[:, 7].astype(np.float32), rho_ref=np.ones(n, np.float32),
             comp_seconds=np.full(n, 0.5, np.float32))
    for k, z in zip(("q05", "q25", "q50", "q75", "q95"), (-0.08, -0.03, 0, 0.03, 0.08)):
        d[k] = (q50 + z).astype(np.float32)
        d["c" + k] = (q50 + 1.2 * z).astype(np.float32)
    return d


def test_export_and_summarize(tmp_path):
    import json
    import subprocess
    import sys
    from cfcap.models import train as TR
    rng = np.random.default_rng(0)
    for shard in (16, 1):                                          # grid b1: 16 = st n6 s2 (A-SEED headline), 1 = bb n6 s1
        d = tmp_path / f"run{shard}"
        (d / "units").mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(dict(task="train", shard=shard, task_argv=["--grid", "b1"])))
        for s in TR.EVAL_SPLITS:
            np.savez(d / "units" / f"eval-{s}.npz", **_fake_eval(rng))
            (d / "units" / f"eval-{s}.json").write_text(json.dumps(dict(unit=f"eval-{s}",
                                                                        meta=dict(infer_ms_per_episode=3.0))))
    assert TR.grid("b1")[16] == dict(arch="st", div="n6", seed=2) and TR.grid("b1")[1]["arch"] == "bb"
    out = tmp_path / "paper"
    r = subprocess.run([sys.executable, "scripts/models_export.py", str(tmp_path), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import pandas as pd
    dec = pd.read_csv(out / "dec_models.csv.gz")
    assert {"amortized_marg", "amortized_sim", "amortized_comp", "blackbox_gnn"} <= set(dec.method)
    for c in ("method", "route", "posterior", "train", "test_plant", "split", "status", "eid", "shape", "cand", "kind",
              "cand_size", "has_fault", "leg", "truth", "point", "scale", "lower"):
        assert c in dec.columns
    t = dec[(dec.method == "amortized_sim") & (dec.split == "test")]
    assert (t.lower <= t.point + 1e-6).all()
    div = pd.read_csv(out / "diversity.csv")
    assert set(div.metric) >= {"cert_frac", "unsafe_commit", "pinball", "cov90"}
    assert (tmp_path / "strata" / "diversity_relevant.csv").exists()
    assert (out / "latency_models.csv").exists()


def test_export_eval2_end_to_end(tmp_path):
    import json
    import subprocess
    import sys
    import pandas as pd
    rng = np.random.default_rng(1)
    d = tmp_path / "e2"
    (d / "units").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(dict(task="models_eval2", shard=0, task_argv=[])))
    for split in ("cal", "test"):
        parts = []
        for mi in (0, 1):
            f = _fake_eval(rng)
            n = len(f["eid"])
            f.update(model=np.full(n, mi), sec_direct=np.full(n, 0.004, np.float32),
                     sec_comp=np.full(n, 0.9, np.float32), sec_gen=np.full(n, 0.1, np.float32))
            parts.append(f)
        z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        u = f"{split}-0000000-0000030"
        np.savez(d / "units" / f"{u}.npz", **z)
        (d / "units" / f"{u}.json").write_text(json.dumps(dict(meta=dict(models="st-n6-2,bb-n6-1", hardware="X"))))
    out, cal = tmp_path / "paper", tmp_path / "calib"
    r = subprocess.run([sys.executable, "scripts/models_export.py", str(tmp_path), "--out", str(out), "--calib-out",
                        str(cal)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    dec = pd.read_csv(out / "dec_models.csv.gz")
    assert {"amortized_sim", "amortized_comp", "blackbox_gnn"} <= set(dec.method)
    assert np.allclose(dec[dec.method == "amortized_comp"].seconds, 0.9, atol=1e-5)
    lat = pd.read_csv(out / "latency_models.csv")
    assert (lat[lat.method == "amortized_comp"].sec_per_decision > 0.5).all()
    c = json.loads((cal / "calib_st-n6-2.json").read_text())
    assert {"q_sim", "q_marg", "q_comp", "alpha"} <= set(c)


def test_scaling_curve(tmp_path):
    import json
    import subprocess
    import sys
    import pandas as pd
    rng = np.random.default_rng(2)
    d = tmp_path / "e2"
    (d / "units").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(dict(task="models_eval2", shard=0, task_argv=[])))
    for split, n in (("cal", 4), ("test", 4), ("H4b", 12)):
        f = _fake_eval(rng)
        k = len(f["eid"])
        f.update(model=np.zeros(k, int), parent_n=np.full(k, n, np.int32), sec_direct=np.full(k, 0.01, np.float32),
                 sec_comp=np.full(k, 1.0, np.float32), sec_gen=np.zeros(k, np.float32))
        np.savez(d / "units" / f"{split}-0000000-0000030.npz", **f)
        (d / "units" / f"{split}-0000000-0000030.json").write_text(json.dumps(dict(meta=dict(models="st-n6-2"))))
    out = tmp_path / "scaling.csv"
    r = subprocess.run([sys.executable, "scripts/models_scaling.py", str(tmp_path), "--tags", "st-n6-2", "--B", "20",
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    s = pd.read_csv(out)
    assert set(s.route) == {"direct_sim", "compositional"} and {"test", "H4b"} <= set(s.split)
    assert ((s.lo <= s.value + 1e-9) & (s.value <= s.hi + 1e-9)).all()
    assert "12" in set(s.test_n.astype(str))


def test_confirmatory_export_guard_and_columns(tmp_path):
    import json
    import subprocess
    import sys
    import pandas as pd
    rng = np.random.default_rng(3)
    d = tmp_path / "e2"
    (d / "units").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(dict(task="models_eval2", shard=1, task_argv=[])))
    for split, eid0 in (("cal", 0), ("conf_test", 30), ("conf_H4b", 0)):
        f = _fake_eval(rng, eid0=eid0)
        k = len(f["eid"])
        f.update(model=np.zeros(k, int), sec_direct=np.full(k, 0.01, np.float32), sec_comp=np.ones(k, np.float32),
                 sec_gen=np.zeros(k, np.float32), shard=np.ones(k, np.int64))
        for q in ("q05", "q25", "q50", "q75", "q95"):
            f["b" + q] = f["c" + q]
        f2 = dict(f, model=np.ones(k, int))                       # st-n6-1: descriptive seed variation only
        f = {kk: np.concatenate([f[kk], f2[kk]]) for kk in f}
        u = f"{split}-0000000-0000030"
        np.savez(d / "units" / f"{u}.npz", **f)
        (d / "units" / f"{u}.json").write_text(json.dumps(dict(meta=dict(models="st-n6-2,st-n6-1"))))
        (d / "wheels").mkdir(exist_ok=True)
        ne = 30
        np.savez(d / "wheels" / f"{u}.npz", split=np.array(split), model=np.array("st-n6-2"),
                 eid=np.repeat(np.arange(eid0, eid0 + ne), 8), asm_role=np.zeros(8 * ne, int),
                 wheel=np.tile(np.arange(8), ne), ep_wheel=np.tile(np.arange(8), ne),
                 p=np.full((8 * ne, 4), 0.25, np.float32), g_mean=np.ones(8 * ne, np.float32))
    cmd = [sys.executable, "scripts/models_export.py", str(tmp_path), "--out", str(tmp_path / "p"), "--calib-out",
           str(tmp_path / "c"), "--confirmatory", "--conf-out", str(tmp_path / "conf.csv.gz")]
    import datetime as dt
    if dt.datetime.now(dt.timezone.utc) < dt.datetime(2026, 11, 26, 1, 45, tzinfo=dt.timezone.utc):
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode != 0 and "closed" in (r.stderr + r.stdout)
    r = subprocess.run(cmd + ["--allow-early"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    c = pd.read_csv(tmp_path / "conf.csv.gz")
    for col in ("claim", "shard", "parent_n", "truth", "lower", "lower_marg", "method", "split", "status", "eid"):
        assert col in c.columns
    assert set(c.claim) == {"C-MAIN", "C-SCALE"}
    main = c[(c.claim == "C-MAIN") & (c.split != "cal")]
    assert main.eid.min() >= 40 and (main.status == "confirmatory").all()
    assert (c[c.split == "cal"].status == "exploratory").all()
    assert c.variant.str.startswith("st-n6-2").all() or (c.method == "blackbox_gnn").any()
    assert not c.variant.str.startswith("st-n6-1").any()
    sv = pd.read_csv(tmp_path / "seedvar_conf_models.csv.gz")
    assert sv.variant.str.startswith("st-n6-1").all() and sv.method.str.endswith("_seedvar").all()
    w = np.load(tmp_path / "wheels_conf_models.npz")
    assert w["p"].shape[1] == 4 and w["eid"].min() >= 40 and "mode" not in w.files
    p = pd.read_csv(tmp_path / "p" / "latency_models.csv") if (tmp_path / "p" / "latency_models.csv").exists() else None


def test_env_lower_export(tmp_path):
    import json
    import subprocess
    import sys
    import pandas as pd
    d = tmp_path / "e2"
    (d / "units").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(dict(task="models_eval2", shard=1, task_argv=[])))
    n = 40
    rows = dict(family=np.full(n, 2), mission=np.repeat(np.arange(4), 10), building=np.zeros(n, int),
                cand=np.tile(np.arange(5), 8), seg=np.tile(np.repeat(np.arange(2), 5), 4), leg=np.zeros(n, int),
                cond=np.zeros(n, int), q05=np.full(n, 0.8, np.float32), q50=np.full(n, 0.9, np.float32),
                q95=np.full(n, 0.95, np.float32), seconds=np.ones(n, np.float32), shard=np.ones(n, int),
                status_conf=np.zeros(n, np.int8))
    np.savez(d / "units" / "env-test-0000000-0000004.npz", **rows)
    (d / "units" / "env-test-0000000-0000004.json").write_text(json.dumps(dict(meta={})))
    cal = tmp_path / "calib"
    cal.mkdir()
    (cal / "calib_st-n6-2.json").write_text(json.dumps(dict(alpha=0.1, q_sim=1.0, q_marg=0.8, q_comp=2.0)))
    r = subprocess.run([sys.executable, "scripts/models_export.py", str(tmp_path), "--out", str(tmp_path / "p"),
                        "--calib-out", str(cal), "--env-out", str(tmp_path / "m")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    root = tmp_path / "m" / "env"
    comp = pd.read_csv(root / "env_lower_amortized_comp.csv.gz")
    raw = pd.read_csv(root / "env_lower_raw.csv.gz")
    assert {"family", "mission", "cand", "seg", "lower"} <= set(comp.columns)
    assert np.allclose(raw.lower, 0.9) and np.allclose(comp.lower, 0.9 - 2.0 * 0.1, atol=1e-5)
    assert (raw.bound_kind == "plugin_median").all()
    # envcal: truth rows of the exploratory ID missions -> q_env, a third file
    t = tmp_path / "envrun" / "units"
    t.mkdir(parents=True)
    np.savez(t / "env-test-000000-000004.npz", family=rows["family"], mission=rows["mission"], cand=rows["cand"],
             seg=rows["seg"], truth=np.full(n, 0.85, np.float32), fp_ok=np.ones(n, bool), load_ok=np.ones(n, bool))
    r = subprocess.run([sys.executable, "scripts/models_export.py", str(tmp_path / "e2"), "--out", str(tmp_path / "p"),
                        "--calib-out", str(cal), "--env-out", str(tmp_path / "m"), "--env-truth",
                        str(tmp_path / "envrun")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ec = pd.read_csv(root / "env_lower_amortized_comp_envcal.csv.gz")
    assert np.isfinite(ec.lower).all() or np.isinf(ec.lower).all()


def test_cand_index_is_generator_candidate_list_index():
    """dec_conf_models `cand` = position in make_episode(...).cands (C-EXPLAIN / confirmatory truth regeneration)."""
    from cfcap.gen import episode as E
    from cfcap.gen import shard as SH
    from cfcap.models import eval2 as V
    eps = [E.make_episode("plated", "test", i) for i in (5, 6, 7)]
    T = 20
    obs = {"v2": [[dict(duty=np.zeros((T, a.asm.n), np.float32), dcnt=np.zeros((T, a.asm.n), np.int32),
                        imu=np.zeros((T, 3), np.float32), cmd=np.zeros((T, 3), np.float32), amp=1.0, lead=0.0)
                   for a in ep.asms] for ep in eps]}
    labs = []
    for ep in eps:
        one = np.ones((len(ep.cands), 10), np.float32)
        labs.append(dict(rho=one, rho_healthy=one, rho_rel=one, drift=0 * one, rho_ref=one, rho_dec=one,
                         rho_ref_type=one))
    D = SH._assemble(eps, obs, labs, ("v2",))
    ci = V.cand_index(D)
    mod0 = D["asm_mod_off"]
    for row, (e, c) in enumerate(zip(D["cand_ep"], ci)):
        cd = eps[e].cands[c]
        k0, k1 = D["cand_cm_off"][row:row + 2]
        got = [int(x) for x in D["cm_mod"][k0:k1]]
        want = [int(mod0[D["ep_asm_off"][e] + r] + k) for r, k in cd.modules]
        assert got == want and D["cand_kind"][row] == cd.kind
