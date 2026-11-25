"""Probe-selection policies and the sequential decision loop (PLAN §2 contribution 3, §6 D).

Certifier. lower = point - q x scale from a DecisionPosterior's (point, scale) summary, with q from
`conformal.core.SimultaneousLower` (q = 0 means the raw posterior summary; a degenerate posterior plus q = 0 is the
plug-in predictor). Optional extra CRC margin lambda: lower - lambda.

Loop (`run`). After v2 (already in the posterior) the policy repeatedly picks one remaining probe or stops; after each
probe the posterior is reweighted with the probe's actual observation and its time is charged. Then `decide` commits
the certified option (or abstains). Policies:
  v2      never probes (fixed v2);
  all     every library probe except diagnostic splits (fixed v2+clamp+moddiff+tug);
  fix:<a>+<b>  a fixed probe set (e.g. fix:clamp, fix:clamp+moddiff: the nested sets of the monotonicity check);
  random  a uniformly random remaining probe while the current decision leaves a task unserved (<= max_probes);
  eig     argmax expected information gain about the discrete latent (fault mode vector), same stop rule;
  epig    argmax expected predictive information gain about the decision-relevant capability events
          {rho[c, legs(t)] >= r_t} over all (candidate, task) pairs, same stop rule;
  evoi    decision-EVOI greedy: EVOI(u) = E_y[ max-decision expected utility after y ] - current expected utility,
          utility in seconds (V_TASK per served task, -C_UNSAFE per unsafe commit, -split/merge time); picks
          argmax EVOI(u)/t_u among probes with EVOI(u) > t_u and stops when none qualifies.
  always_split  one diagnostic split of the child with the largest summed binary entropy of P(capable for task t)
          over the tasks (ties: the larger child), then decide.
Every expectation over y uses S predictive draws (common random numbers across probes).
"""
from __future__ import annotations

import numpy as np

from . import problem as Pb

POLICIES = ("v2", "all", "random", "eig", "epig", "evoi", "always_split")
FIXED_ALL = ("clamp", "moddiff", "tug")


def policy_id(pol):
    """Stable small int per policy name (rng stream); fixed nested sets are 'fix:<k1>+<k2>...'."""
    if pol in POLICIES:
        return POLICIES.index(pol)
    return 100 + sum(ord(c) * (i + 1) for i, c in enumerate(pol)) % 100000


class Certifier:
    """lower = point - q x max(scale, floor) - lam."""

    def __init__(self, q=0.0, lam=0.0, floor=0.0):
        self.q, self.lam, self.floor = float(q), float(lam), float(floor)

    def lower(self, post, w=None):
        m, sc = post.summary(w)
        lo = m - self.q * np.maximum(sc, self.floor) - self.lam
        return np.where(np.isfinite(lo), lo, -np.inf)


def _ent(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1)


def _hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _lat_ent(w, lat_inv, n_lat):
    return _ent(np.bincount(lat_inv, weights=w, minlength=n_lat))


def decision_value(inst, tasks, post, cert, w):
    """Expected utility (posterior w) of the certified decision taken under w."""
    ch, tc = Pb.decide(inst, tasks, cert.lower(post, w))
    return Pb.expected_utility(inst, tasks, ch, tc, post.rho, w), ch


def scores(policy, inst, tasks, post, cert, rng, S=16, names=None):
    """Score of every remaining probe (higher is better) and, for evoi, the net value EVOI - t_u."""
    names = list(post.probes) if names is None else names
    out, net = {}, {}
    if policy == "eig":
        _, inv = np.unique(post.latent, return_inverse=True)
        nl = inv.max() + 1
        h0 = _lat_ent(post.w, inv, nl)
    if policy in ("epig", "evoi"):
        R, mm, _ = Pb.task_arrays(tasks, post.rho.shape[-1])
        capS = Pb.capable(post.rho, R, mm, inst.sizes).reshape(post.P, -1).astype(float)   # (P, C*T)
        h0b = _hb(post.w @ capS).sum()
    if policy == "evoi":
        v0, _ = decision_value(inst, tasks, post, cert, post.w)
    for n in names:
        ys = post.predictive(n, rng, S)
        acc = 0.0
        for y in ys:
            w1 = post.reweight(n, y)
            if policy == "eig":
                acc += h0 - _lat_ent(w1, inv, nl)
            elif policy == "epig":
                acc += h0b - _hb(w1 @ capS).sum()
            elif policy == "evoi":
                acc += decision_value(inst, tasks, post, cert, w1)[0]
        acc /= S
        if policy == "evoi":
            ev = acc - v0
            out[n] = ev / post.probes[n].t_s
            net[n] = ev - post.probes[n].t_s
        else:
            out[n] = acc
    return out, net


def _unserved(inst, tasks, post, cert):
    ch, _ = Pb.decide(inst, tasks, cert.lower(post))
    return bool((ch < 0).any())


def run(policy, inst, tasks, post, cert, rng, max_probes=2, S=16, eps=1e-3, allow_split=True):
    """Run one policy on one episode. post is copied. Returns (choice, time_cost, probe_time, probes_used, post)."""
    post = post.copy()
    if not allow_split:
        post.probes = {k: v for k, v in post.probes.items() if not k.startswith("split:")}
    used, t_probe = [], 0.0
    if post.unknown:                                   # pruned support: no bound, no probe can restore it
        return np.full(len(tasks), -1, np.int64), 0.0, 0.0, used, post
    if policy == "all" or policy.startswith("fix:"):
        for n in (FIXED_ALL if policy == "all" else policy[4:].split("+")):
            if n in post.probes:
                t_probe += post.observe(n)
                used.append(n)
    elif policy == "always_split":
        sp = [k for k in post.probes if k.startswith("split:")]
        if sp and post.P > 1:
            R, mm, _ = Pb.task_arrays(tasks, post.rho.shape[-1])
            pc = np.tensordot(post.w, Pb.capable(post.rho, R, mm, inst.sizes).astype(float), 1)   # (C, T)
            h = _hb(pc).sum(1)
            n = max(sp, key=lambda k: (h[int(k.split(":")[1])], inst.sizes[int(k.split(":")[1])]))
            t_probe += post.observe(n)
            used.append(n)
    elif policy != "v2":
        for _ in range(max_probes):
            if not post.probes or post.P <= 1 or post.unknown:
                break
            if policy != "evoi" and not _unserved(inst, tasks, post, cert):
                break
            if policy == "random":
                names = list(post.probes)
                n = names[int(rng.integers(len(names)))]
            else:
                sc, net = scores(policy, inst, tasks, post, cert, rng, S)
                if policy == "evoi":
                    ok = {k: v for k, v in sc.items() if net[k] > 0}
                    if not ok:
                        break
                    n = max(ok, key=ok.get)
                else:
                    n = max(sc, key=sc.get)
                    if sc[n] <= eps:
                        break
            t_probe += post.observe(n)
            used.append(n)
    ch, tc = Pb.decide(inst, tasks, cert.lower(post))
    return ch, tc, t_probe, used, post
