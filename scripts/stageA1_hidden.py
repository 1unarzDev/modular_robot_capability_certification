"""Stage A1 (exploratory): is a weak wheel hidden at all? Targets: scripts/stageA_targets.md "A1".

Simulates the v2 probe on pair / square / 2x3 with one weak wheel or healthy, heterogeneity OFF and ON (+-8 % per-wheel
motor), and scores two detectors on robot-visible channels:
  D1 heuristic g_hat_i = sibling-median(|u|)/|u_i| over cruise windows (u = pre_duty);
  D2 capxfer self-model v2 (3 seeds, plated).
Writes per-episode rows to $CFCAP_OUT/A1_episodes.csv and the summary to results/stageA/A1_summary.csv.

  python scripts/stageA1_hidden.py            # simulate + score + summarise
  python scripts/stageA1_hidden.py --summary  # re-summarise existing episodes
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cfcap.t1 import params as P
from cfcap.t1 import plant

OUT = pathlib.Path(os.environ.get("CFCAP_OUT", pathlib.Path.home() / ".cache/modular-runs/2026-11/cfcap"))
RES = pathlib.Path(__file__).resolve().parents[1] / "results/stageA"
MODELS = pathlib.Path.home() / ".cache/modular-runs/2026-11/capxfer/rung_v2/models"
SHAPES = ("pair", "square", "2x3")
HETS = {"off": plant.HET_OFF, "on": plant.HET_WHEEL8}
G_BINS = (0.3, 0.45, 0.6, 0.75, 0.9001)


# ----------------------------------------------------------------------------- detector 1
def cruise_windows(cmd, hold=10):
    """cmd (T,3) on the 50 Hz grid -> list of (start, end) bin ranges where a translational command (wz = 0, |v| > 0)
    has been unchanged for the previous `hold` bins (0.2 s)."""
    T = len(cmd)
    same = np.zeros(T, bool)
    for k in range(hold, T):
        same[k] = np.all(np.abs(cmd[k - hold:k + 1] - cmd[k]) < 1e-6)
    ok = same & (np.abs(cmd[:, 2]) < 1e-9) & (np.abs(cmd[:, :2]).sum(1) > 0.05)
    out, k = [], 0
    while k < T:
        if ok[k]:
            j = k
            while j < T and ok[j] and np.all(np.abs(cmd[j] - cmd[k]) < 1e-6):
                j += 1
            if j - k >= 10:
                out.append((k, j))
            k = j
        else:
            k += 1
    return out


def heuristic_ghat(duty, cmd, wheel_module):
    """duty (B,T,n) numpy, cmd (B,T,3) -> g_hat (B,n): median over cruise windows of sibling-median(|u|)/|u_i|."""
    B, T, n = duty.shape
    wm = np.asarray(wheel_module)
    sib = [np.nonzero((wm == wm[i]) & (np.arange(n) != i))[0] for i in range(n)]
    gh = np.ones((B, n), np.float32)
    for b in range(B):
        ws = cruise_windows(cmd[b])
        if not ws:
            continue
        mu = np.stack([np.abs(duty[b, a:e]).mean(0) for a, e in ws])          # (W, n)
        r = np.stack([np.median(mu[:, sib[i]], 1) / np.maximum(mu[:, i], 1e-6) for i in range(n)], 1)
        gh[b] = np.median(r, 0)
    return gh


# ----------------------------------------------------------------------------- detector 2 (capxfer self-model v2)
class SelfModelV2:
    def __init__(self, plate="plated", device="cuda"):
        from capxfer.corpus import selfmodel as S
        self.S = S
        self.models = []
        for s in (1, 2, 3):
            m = S.SelfModel().to(device)
            m.load_state_dict(torch.load(MODELS / f"selfmodel_{plate}_s{s}.pt", map_location=device))
            m.eval()
            self.models.append(m)
        self.device = device

    @torch.no_grad()
    def __call__(self, asm, cmd, obs, bs=256):
        """cmd (B,T,3) numpy; obs from plant.observe -> p (B,n,4), g_hat (B,n). Features as capxfer build_features;
        trial probabilities = mean log-probability over motion windows (capxfer trial_probs), averaged over seeds."""
        S = self.S
        dev = self.device
        B, T, n = obs["pre_duty"].shape
        ref = torch.tensor(np.stack([S.wheel_ref(asm, c) for c in cmd]), device=dev)          # (B,T,n)
        cmd_t = torch.tensor(cmd, device=dev)
        wf = torch.stack([ref / 5.0, obs["pre_duty"].to(dev), obs["speed"].to(dev) / 5.0], -1)
        wf = wf.half().float()                                                                  # SimSet stores fp16
        af = torch.stack([cmd_t[..., 0] / 0.3, cmd_t[..., 1] / 0.3, cmd_t[..., 2], obs["gz"].to(dev),
                          obs["ax"].to(dev), obs["ay"].to(dev)], -1)
        geom = torch.tensor(S.geometry(asm), device=dev)
        idx = [(b, s) for b in range(B) for s in S.motion_windows(cmd[b], np.ones(T, bool))]
        lp = torch.zeros(B, n, 4, device=dev)
        gs = torch.zeros(B, n, device=dev)
        cnt = torch.zeros(B, device=dev)
        ar = torch.arange(S.W, device=dev)
        for i in range(0, len(idx), bs):
            chunk = idx[i:i + bs]
            bb = torch.tensor([c[0] for c in chunk], device=dev)
            ss = torch.tensor([c[1] for c in chunk], device=dev)
            tt = ss[:, None] + ar[None, :]
            w = wf[bb[:, None], tt]                                                               # (k,W,n,3)
            a = af[bb[:, None], tt]
            g = geom.expand(len(chunk), n, 5)
            mask = torch.ones(len(chunk), n, dtype=torch.bool, device=dev)
            l_acc, g_acc = 0, 0
            for m in self.models:
                logits, gh = m(w, a, g, mask)
                l_acc = l_acc + F.log_softmax(logits, -1) / len(self.models)
                g_acc = g_acc + gh / len(self.models)
            lp.index_add_(0, bb, l_acc)
            gs.index_add_(0, bb, g_acc)
            cnt.index_add_(0, bb, torch.ones(len(chunk), device=dev))
        c = cnt.clamp(min=1)[:, None]
        return torch.softmax(lp / c[..., None], -1).cpu().numpy(), (gs / c).cpu().numpy()


# ----------------------------------------------------------------------------- simulation
def run(n_total, batch, seed, engine, use_sm):
    par = P.base_params("plated")
    th = P.load_theta("plated")
    dr = P.theta_dr(th)
    rng = np.random.default_rng(seed)
    sm = SelfModelV2() if use_sm else None
    per_cell = int(np.ceil(n_total / (len(SHAPES) * len(HETS))))
    rows = []
    t0 = time.time()
    for shape in SHAPES:
        asm = P.named_assembly(shape, par)
        n = asm.n
        for hname, het in HETS.items():
            done = 0
            while done < per_cell:
                B = min(batch, per_cell - done)
                mode, g = plant.weak_or_healthy(B, n, rng, p_healthy=0.5)
                lat = plant.sample_latents(asm, B, rng, het, dr=dr, faults=(mode, g))
                tab, T = plant.v2_probe(asm, par, B, rng)
                R = plant.simulate_bucket([dict(asm=asm, lat=lat, cmd=tab)], par, T, engine=engine,
                                          seed=int(rng.integers(1 << 30)))[0]
                obs = plant.observe(R, lat, seed=int(rng.integers(1 << 30)))
                Tn = obs["pre_duty"].shape[1]
                cmd = tab[:, :Tn]                       # published command per 50 Hz bin (tick k = bin k)
                gh1 = heuristic_ghat(obs["pre_duty"].cpu().numpy(), cmd, asm.wheel_module)
                if sm is not None:
                    p2, gh2 = sm(asm, cmd, obs)
                for b in range(B):
                    w = int(np.argmax(mode[b])) if mode[b].any() else -1
                    r = dict(shape=shape, het=hname, weak=int(w >= 0), wheel=w, g=float(g[b, w]) if w >= 0 else 1.0,
                             d1_score=float((1 / np.maximum(gh1[b], 1e-6)).max()), d1_argmin=int(np.argmin(gh1[b])),
                             d1_ghat_true=float(gh1[b, w]) if w >= 0 else np.nan, d1_ghat_min=float(gh1[b].min()))
                    if sm is not None:
                        pw = 1 - p2[b, :, 0]
                        r.update(d2_score=float(pw.max()), d2_argmax=int(np.argmax(pw)),
                                 d2_ghat_true=float(gh2[b, w]) if w >= 0 else np.nan,
                                 d2_pweak_true=float(p2[b, w, 1]) if w >= 0 else np.nan)
                    rows.append(r)
                done += B
                print(f"{shape} {hname} {done}/{per_cell} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "A1_episodes.csv", index=False)
    return df


# ----------------------------------------------------------------------------- summary
def summarise(df):
    out = []
    dets = [d for d in ("d1", "d2") if f"{d}_score" in df]
    for (shape, het), c in df.groupby(["shape", "het"]):
        h = c[c.weak == 0]
        wk = c[c.weak == 1]
        for d in dets:
            thr = float(np.quantile(h[f"{d}_score"], 0.99))
            fa = float((h[f"{d}_score"] > thr).mean())
            loc_col = f"{d}_argmin" if d == "d1" else f"{d}_argmax"
            bins = [("all", wk)] + [(f"{lo:.2f}-{min(hi, 0.9):.2f}", wk[(wk.g >= lo) & (wk.g < hi)])
                                    for lo, hi in zip(G_BINS[:-1], G_BINS[1:])]
            for bname, s in bins:
                err = (s[f"{d}_ghat_true"] - s.g).abs()
                out.append(dict(shape=shape, het=het, detector=d, g_bin=bname, n_weak=len(s), n_healthy=len(h),
                                thr_1pct=round(thr, 4), false_alarm=round(fa, 4),
                                detection=round(float((s[f"{d}_score"] > thr).mean()), 4),
                                localisation=round(float((s[loc_col] == s.wheel).mean()), 4),
                                ghat_err_median=round(float(err.median()), 4),
                                ghat_err_le_0p05=round(float((err <= 0.05).mean()), 4)))
    return pd.DataFrame(out)


def verdict(sm):
    a = sm[(sm.detector == "d1") & (sm.g_bin == "all")].set_index(["shape", "het"])
    lines = []
    trivial = all(a.loc[(s, "off"), "detection"] >= 0.95 and a.loc[(s, "off"), "ghat_err_median"] <= 0.05
                  for s in ("pair", "square"))
    drops = {s: a.loc[(s, "off"), "detection"] - a.loc[(s, "on"), "detection"] for s in ("pair", "square")}
    lines.append(f"TRIVIAL threshold (D1 det>=0.95 & median|g_hat-g|<=0.05, het off, pair+square): {trivial}")
    lines.append(f"LEAK threshold (drop >0.15 off->on): {any(v > 0.15 for v in drops.values())} drops={drops}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1667)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--engine", default="compile")
    ap.add_argument("--no-selfmodel", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / "A1_episodes.csv") if a.summary else run(a.n, a.batch, a.seed, a.engine,
                                                                       not a.no_selfmodel)
    sm = summarise(df)
    sm.to_csv(RES / "A1_summary.csv", index=False)
    print(sm[sm.g_bin == "all"].to_string(index=False))
    v = verdict(sm)
    print(v)
    (RES / "A1_verdict.txt").write_text(v + "\n(source: results/stageA/A1_summary.csv; exploratory)\n")


if __name__ == "__main__":
    main()
