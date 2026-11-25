"""Stage A6 (exploratory): cost of an exact sparse-enumeration likelihood posterior, N = 6 modules (2x3, 24 wheels).

Hypothesis space: every wheel-mode vector with <= 2 faulted wheels, fault states weak(g) for g on the 0.05 grid
0.30..0.90 (13) plus coast and brake: 1 + 24*15 + C(24,2)*15^2 = 62,461 hypotheses.

Per episode: nuisance parameters drawn from the v2 self-model DR (known to the posterior: an oracle-nuisance exact
posterior; heterogeneity off, no mounting-yaw error), true faults drawn from the same <= 2-fault space with g
continuous U(0.3, 0.9), the v2 probe (12.5 s, no jitter) simulated and observed (pre_duty, quantised encoder speed, gz;
IMU noise + N(0, 0.005) duty noise). Every hypothesis is then simulated on the GPU at the episode's nuisance and scored
with an independent Gaussian likelihood (sd: duty 0.02, speed 0.10 rad/s, gz 0.005 rad/s; tempered above the actual
noise). Uniform prior over hypotheses.

Reports hypotheses/s, seconds per episode, and posterior accuracy: MAP fault set (which wheels, which mode) correct;
MAP weak gains within 0.05 of the truth; posterior mass on the true fault set.
Output: results/stageA/A6_posterior_cost.csv (per episode) and A6_summary.csv.
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import time

import numpy as np
import pandas as pd
import torch

from cfcap.t1 import params as P
from cfcap.t1 import plant

RES = pathlib.Path(__file__).resolve().parents[1] / "results/stageA"
G_GRID = np.round(np.arange(0.30, 0.9001, 0.05), 2)
SD = dict(duty=0.02, speed=0.10, gz=0.005)


def hypotheses(n):
    states = [(plant.WEAK, float(g)) for g in G_GRID] + [(plant.COAST, 1.0), (plant.BRAKE, 1.0)]
    H = 1 + n * len(states) + (n * (n - 1) // 2) * len(states) ** 2
    mode = np.zeros((H, n), np.int64)
    g = np.ones((H, n), np.float32)
    h = 1
    for w in range(n):
        for m, gg in states:
            mode[h, w], g[h, w] = m, gg
            h += 1
    for w1, w2 in itertools.combinations(range(n), 2):
        for (m1, g1), (m2, g2) in itertools.product(states, states):
            mode[h, w1], g[h, w1], mode[h, w2], g[h, w2] = m1, g1, m2, g2
            h += 1
    assert h == H
    return mode, g


def signals(R, cpr=P.CPR):
    th = R["wheel_angle"].double()
    cnt = torch.floor(th / (2 * np.pi) * cpr)
    spd = torch.zeros_like(th)
    spd[:, 1:] = torch.diff(cnt, dim=1) / cpr * 2 * np.pi / plant.DT_OBS
    return R["duty"], spd.float(), R["twist"][:, :, 2]


def one_domain(dom, b, B):
    return {k: (v[b:b + 1].expand(B).contiguous() if torch.is_tensor(v) else v) for k, v in dom.items()}


def run(n_ep, chunk, seed, engine):
    par = P.base_params("plated")
    th = P.load_theta("plated")
    dr = P.theta_dr(th)
    dr["wheel_yaw_err"] = 0.0
    asm = P.named_assembly("2x3", par)
    n = asm.n
    HM, HG = hypotheses(n)
    H = len(HM)
    rng = np.random.default_rng(seed)
    dom_all = plant.sim.sample_domain(n_ep, rng, dr, device="cuda")
    tm, tg = plant.random_faults(n_ep, n, rng, k_max=2, p_none=0.2)
    tab, T = plant.v2_probe(asm, par, 1, jitter=False)
    tab = tab[0]
    rows = []
    # warm-up compile at the chunk shape
    lat_w = plant.sample_latents(asm, chunk, rng, plant.HET_OFF, domain=one_domain(dom_all, 0, chunk))
    plant.simulate_bucket([dict(asm=asm, lat=lat_w, cmd=tab)], par, T, engine=engine)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for e in range(n_ep):
        # observed episode
        lat = plant.sample_latents(asm, 1, rng, plant.HET_OFF, domain=one_domain(dom_all, e, 1),
                                   faults=(tm[e:e + 1], tg[e:e + 1]))
        R = plant.simulate_bucket([dict(asm=asm, lat=lat, cmd=tab)], par, T, engine="graph")[0]
        obs = plant.observe(R, lat, seed=1000 + e)
        y_d = obs["pre_duty"] + 0.005 * torch.randn_like(obs["pre_duty"])
        y_s, y_g = obs["speed"], obs["gz"]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ll = torch.empty(H, dtype=torch.float64)
        lat_h = plant.sample_latents(asm, chunk, rng, plant.HET_OFF, domain=one_domain(dom_all, e, chunk))
        for h0 in range(0, H, chunk):
            h1 = min(H, h0 + chunk)
            m = np.zeros((chunk, n), np.int64)
            gg = np.ones((chunk, n), np.float32)
            m[:h1 - h0], gg[:h1 - h0] = HM[h0:h1], HG[h0:h1]               # pad the last chunk (fixed shape)
            lat_h["mode"] = torch.as_tensor(m, device="cuda")
            lat_h["g"] = torch.as_tensor(gg, device="cuda")
            Rh = plant.simulate_bucket([dict(asm=asm, lat=lat_h, cmd=tab)], par, T, engine=engine)[0]
            d, s, gz = signals(Rh)
            l = -0.5 * (((d - y_d) / SD["duty"]) ** 2).sum((1, 2)) - 0.5 * (((s - y_s) / SD["speed"]) ** 2).sum((1, 2)) \
                - 0.5 * (((gz - y_g) / SD["gz"]) ** 2).sum(1)
            ll[h0:h1] = l[:h1 - h0].double().cpu()
            del Rh, d, s, gz
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        post = torch.softmax(ll, 0).numpy()
        hm = int(np.argmax(post))
        true_set = {(w, int(tm[e, w])) for w in range(n) if tm[e, w] != 0}
        map_set = {(w, int(HM[hm, w])) for w in range(n) if HM[hm, w] != 0}
        set_ok = true_set == map_set
        g_ok = set_ok and all(abs(HG[hm, w] - tg[e, w]) <= 0.05 + 1e-6 for w, mm in true_set if mm == plant.WEAK)
        same = np.array([{(w, int(HM[h, w])) for w in np.nonzero(HM[h])[0]} == true_set for h in range(H)]) \
            if len(true_set) <= 2 else np.zeros(H, bool)
        rows.append(dict(episode=e, n_faults=len(true_set),
                         true=";".join(f"{w}:{m}:{tg[e, w]:.2f}" for w, m in sorted(true_set)),
                         map=";".join(f"{w}:{m}:{HG[hm, w]:.2f}" for w, m in sorted(map_set)),
                         map_set_correct=int(set_ok), map_g_within_0p05=int(g_ok),
                         post_mass_true_set=round(float(post[same].sum()), 4), map_post=round(float(post[hm]), 4),
                         hypotheses=H, seconds=round(sec, 2), hyp_per_s=round(H / sec, 1),
                         peak_vram_gb=round(torch.cuda.max_memory_reserved() / 1e9, 3)))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=606)
    ap.add_argument("--engine", default="compile")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    df = run(a.episodes, a.chunk, a.seed, a.engine)
    df.to_csv(RES / "A6_posterior_cost.csv", index=False)
    s = dict(gpu=torch.cuda.get_device_name(0), episodes=len(df), hypotheses=int(df.hypotheses[0]),
             hyp_per_s_median=float(df.hyp_per_s.median()), sec_per_episode_median=float(df.seconds.median()),
             map_set_acc=float(df.map_set_correct.mean()), map_set_and_g_acc=float(df.map_g_within_0p05.mean()),
             post_mass_true_set_median=float(df.post_mass_true_set.median()),
             peak_vram_gb=float(df.peak_vram_gb.max()), chunk=a.chunk, engine=a.engine)
    pd.DataFrame([s]).to_csv(RES / "A6_summary.csv", index=False)
    print(s)


if __name__ == "__main__":
    main()
