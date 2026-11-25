import math

import numpy as np
import pytest

from cfcap.decide import evaluate as Ev
from cfcap.decide import policies as Pol
from cfcap.decide import posterior as DP
from cfcap.decide import problem as Pb

L = 10
LEGS = ("surge+", "surge-", "sway+", "sway-", "diag++", "diag--", "diag+-", "diag-+", "yaw+", "yaw-")


class _C:                                   # minimal t0.candidates.Candidate stand-in
    def __init__(self, kind, modules, size):
        self.kind, self.modules, self.size, self.name = kind, modules, size, f"{kind}{modules}"


def line_instance(m=3, merges=1):
    """Parent = m modules in a row; children = connected proper sub-intervals; merges of the whole parent."""
    cells = [(i, 0) for i in range(m)]
    cands = [_C("child", tuple(range(a, b)), b - a) for a in range(m) for b in range(a + 1, m + 1) if b - a < m]
    cands += [_C("merge", tuple(range(m)), m + 1) for _ in range(merges)]
    return Pb.Instance.from_candidates(cells, cands)


def synth_episode(seed, faulted=True, m=3):
    rng = np.random.default_rng(seed)
    inst = line_instance(m)
    C = inst.C
    rho_nom = np.ones((C, L))
    rho_true = rng.uniform(0.85, 1.0, (C, L))
    if faulted:
        bad = rng.choice(C, max(1, C // 2), replace=False)
        rho_true[bad] *= rng.uniform(0.3, 0.8, (len(bad), L))
    tasks = Pb.make_tasks(rng, LEGS)
    # a posterior with the truth among a few particles
    P = 6
    rho = np.clip(rho_true[None] + rng.normal(0, 0.05, (P, C, L)), 0, 1.2)
    rho[0] = rho_true
    post = DP.DecisionPosterior(np.ones(P), rho, np.arange(P) % 3)
    return dict(inst=inst, post=post, rho_true=rho_true, rho_nom=rho_nom, tasks=tasks, fold=seed % 2,
                meta=dict(seed=seed, parent_size=m, n_faults=int(faulted)))


# ----------------------------------------------------------------------------- problem
def test_option_time_costs():
    inst = line_instance(2, merges=1)          # parent, child(0), child(1), merge(parent)
    P, cost = inst.options(2)
    rows = {tuple(r): c for r, c in zip(P.tolist(), cost)}
    assert rows[(-1, -1)] == 0.0
    assert rows[(0, -1)] == 0.0                # whole parent, no split
    assert rows[(1, -1)] == Pb.T_SPLIT_S       # one child off: one split
    assert (1, 0) not in rows                  # overlapping pieces never co-occur
    assert rows[(3, -1)] == Pb.T_SPLIT_S       # merge of the whole parent: one merge
    assert rows[(1, 2)] == Pb.T_SPLIT_S        # both singles: one split


def test_three_line_leftover_components():
    inst = line_instance(3, merges=0)
    P, cost = inst.options(1)
    # child (1,) is the middle module: leaves two components -> 3 parts -> 2 splits
    mid = [i for i, mk in enumerate(inst.masks) if mk == 0b010][0]
    assert cost[[i for i, r in enumerate(P) if r[0] == mid][0]] == 2 * Pb.T_SPLIT_S


def test_make_tasks_shapes():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ts = Pb.make_tasks(rng, LEGS)
        assert 1 <= len(ts) <= 3
        for t in ts:
            assert t.r.shape == (L,) and np.isfinite(t.r).any()
            if t.kind == "payload":
                assert t.min_modules == 2


@pytest.mark.parametrize("seed", range(30))
def test_oracle_zero_unsafe_zero_regret(seed):
    e = synth_episode(seed)
    ch, tc, o = Pb.oracle(e["inst"], e["tasks"], e["rho_true"])
    assert o["n_unsafe"] == 0
    ch2, tc2 = Pb.decide(e["inst"], e["tasks"], e["rho_true"])
    o2 = Pb.outcome(e["inst"], e["tasks"], ch2, tc2, e["rho_true"])
    assert o2["U"] == o["U"] and o2["n_unsafe"] == 0


def test_oracle_predictor_end_to_end():
    eps = [synth_episode(s) for s in range(40)]
    rows = Ev.run_policies(eps, ["v2"], alpha=0.1, alpha_crc=(0.1,))
    orc = [r for r in rows if r["predictor"] == "oracle"]
    assert orc and all(r["n_unsafe"] == 0 for r in orc)
    assert all(abs(r["regret"]) < 1e-9 for r in orc)


def test_nominal_on_faulted_parents_unsafe():
    eps = [synth_episode(s, faulted=True) for s in range(40)]
    rows = Ev.run_policies(eps, [], alpha=0.1)
    nom = [r for r in rows if r["predictor"] == "nominal" and r["cert"] == "raw"]
    assert sum(r["n_unsafe"] for r in nom) > 0


def test_nominal_unsafe_on_real_t0_episode():
    pytest.importorskip("capxfer")
    from cfcap.decide import t0data as T
    e = T.episode(104, faulted=True)
    gap = e["rho_nom"] - e["rho_true"]
    c, l = np.unravel_index(np.nanargmax(gap), gap.shape)
    assert gap[c, l] > 0.05
    r = np.full(L, np.nan)
    r[l] = e["rho_nom"][c, l] - 0.02
    task = Pb.Task(r, 1)
    ch, tc = Pb.decide(e["inst"], [task], e["rho_nom"])
    o = Pb.outcome(e["inst"], [task], ch, tc, e["rho_true"])
    # the nominal commits somewhere; whenever it picks the faulted piece it is unsafe
    cap = Pb.capable(e["rho_nom"], task.r[None], np.array([1]), e["inst"].sizes)[:, 0]
    truth = Pb.capable(e["rho_true"], task.r[None], np.array([1]), e["inst"].sizes)[:, 0]
    assert cap[c] and not truth[c]
    if ch[0] == c:
        assert o["n_unsafe"] == 1


# ----------------------------------------------------------------------------- refusal
def test_nan_and_empty_bounds_abstain():
    e = synth_episode(3)
    inst, tasks = e["inst"], e["tasks"]
    for lo in (np.full((inst.C, L), np.nan), np.full((inst.C, L), -np.inf), np.zeros((0, L)), np.array([])):
        ch, tc = Pb.decide(inst, tasks, lo)
        assert (ch < 0).all() and tc == 0.0
    cert = Pol.Certifier(q=math.inf)
    ch, tc = Pb.decide(inst, tasks, cert.lower(e["post"]))
    assert (ch < 0).all()


def test_partial_nan_never_certifies_that_entry():
    inst = line_instance(2, merges=0)
    lo = np.ones((inst.C, L))
    lo[:, 2] = np.nan
    r = np.full(L, np.nan); r[2] = 0.5
    ch, _ = Pb.decide(inst, [Pb.Task(r)], lo)
    assert (ch < 0).all()


# ----------------------------------------------------------------------------- posterior and policies
def test_reweight_concentrates_on_truth():
    rng = np.random.default_rng(0)
    P, D = 5, 4
    sig = rng.normal(0, 1, (P, D))
    pr = DP.Probe(sig, np.full((P, D), 0.05), 8.0, sig[2])
    post = DP.DecisionPosterior(np.ones(P), np.zeros((P, 1, L)), np.arange(P), {"x": pr})
    post.observe("x")
    assert post.w[2] > 0.99 and "x" not in post.probes


def test_degenerate_posterior_never_probes():
    e = synth_episode(5)
    post = DP.degenerate(e["rho_true"])
    post.probes["x"] = DP.Probe(np.zeros((1, 3)), np.ones((1, 3)), 5.0, np.zeros(3))
    for pol in ("evoi", "eig", "epig", "random"):
        ch, tc, tp, used, _ = Pol.run(pol, e["inst"], e["tasks"], post, Pol.Certifier(0.0), np.random.default_rng(0))
        assert tp == 0.0 and not used


def test_evoi_probes_when_it_decides():
    """Two latent states; a cheap probe separates them; the task is servable only in one state."""
    inst = line_instance(2, merges=0)
    C = inst.C
    good, bad = np.full((C, L), 1.0), np.full((C, L), 1.0)
    bad[0] = 0.4                                         # the whole parent is incapable in the bad state
    rho = np.stack([good, bad])
    r = np.full(L, np.nan); r[0] = 0.8
    sig = np.array([[0.0], [1.0]])
    post = DP.DecisionPosterior(np.array([0.5, 0.5]), rho, np.array([0, 1]),
                                {"p": DP.Probe(sig, np.full((2, 1), 0.05), 5.0, sig[0])})
    tasks = [Pb.Task(r, 2)]                              # needs both modules: only the parent qualifies
    ch0, _ = Pb.decide(inst, tasks, Pol.Certifier(0.0).lower(post))
    assert (ch0 < 0).all()                               # mean 0.7 < 0.8: abstain without the probe
    ch, tc, tp, used, post2 = Pol.run("evoi", inst, tasks, post, Pol.Certifier(0.0), np.random.default_rng(0))
    assert used == ["p"] and ch[0] == 0 and tp == 5.0
    o = Pb.outcome(inst, tasks, ch, tc, good, tp)
    assert o["n_served"] == 1 and o["n_unsafe"] == 0


def test_eig_and_epig_pick_informative_probe():
    inst = line_instance(2, merges=0)
    C = inst.C
    good, bad = np.full((C, L), 1.0), np.full((C, L), 1.0)
    bad[0] = 0.4
    rho = np.stack([good, bad])
    sig_inf = np.array([[0.0], [1.0]]); sig_none = np.array([[0.0], [0.0]])
    probes = {"inf": DP.Probe(sig_inf, np.full((2, 1), 0.05), 5.0, sig_inf[0]),
              "none": DP.Probe(sig_none, np.full((2, 1), 0.05), 5.0, sig_none[0])}
    post = DP.DecisionPosterior(np.array([0.5, 0.5]), rho, np.array([0, 1]), probes)
    r = np.full(L, np.nan); r[0] = 0.8
    for pol in ("eig", "epig"):
        sc, _ = Pol.scores(pol, inst, [Pb.Task(r, 2)], post, Pol.Certifier(0.0), np.random.default_rng(1))
        assert sc["inf"] > sc["none"] + 0.1


def test_crc_loss_curve_monotone():
    e = synth_episode(7)
    pt, sc = e["post"].summary()
    cur = Ev.episode_loss_curve(e, pt, sc)
    assert np.all(np.diff(cur) <= 0)
