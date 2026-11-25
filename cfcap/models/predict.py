"""Predictor for the stress role's `t2_eval --mode evaluate` (T2 test plant): `--predictor
cfcap.models.predict:predict --models DIR`, DIR holding `weights.pt` (a train run's) and `calib.json` (conformal
constants fitted on T1 cal rows by scripts/models_export.py --calib-out; never on T2).

predict(models_dir, episode, v2_obs) -> {method: dict(lower, point, scale, route, posterior, train, alpha)} with
arrays (n_cand, 10) on rho_dec. Only the parent's T2 v2 observation exists, so merge candidates (which need the other
assembly's tokens) are refused (lower = -inf, point NaN); self and children are predicted. Methods:
  amortized_sim   direct head, simultaneous layer (q_sim)
  amortized_marg  direct head, marginal layer (q_marg)
  amortized_comp  posterior samples -> T1 rollouts of every candidate (GPU), simultaneous layer on them (q_comp)
Each method's dict also carries `seconds` (end-to-end wall for this episode).
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np

_CACHE = {}
L = 10


def _load(models_dir):
    import torch
    from cfcap.models import train as TR
    d = pathlib.Path(models_dir)
    if d not in _CACHE:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(d / "weights.pt", map_location=dev, weights_only=False)
        model = TR.build_model(ck["cfg"]).to(dev).eval()
        model.load_state_dict(ck["state"])
        cal = json.loads((d / "calib.json").read_text())
        _CACHE[d] = (model, ck["cfg"], cal, dev)
    return _CACHE[d]


def predict(models_dir, episode, v2_obs, S=16, compose=True):
    import torch
    from cfcap.gen import episode as E
    from cfcap.models import compose as CO
    from cfcap.models import posterior as M
    from cfcap.t1 import params as P
    from cfcap.t1 import reference
    model, cfg, cal, dev = _load(models_dir)
    t0 = time.perf_counter()
    ep = episode
    asm = ep.parent.asm
    n, m = asm.n, len(asm.modules)
    C = len(ep.cands)
    nan = np.full((C, L), np.nan)
    out = {k: dict(lower=np.full((C, L), -np.inf), point=nan.copy(), scale=nan.copy(), route=r, posterior="amortized",
                   train="T1", alpha=cal.get("alpha", 0.1))
           for k, r in (("amortized_sim", "simultaneous"), ("amortized_marg", "marginal"),
                        ("amortized_comp", "compositional"))}
    if v2_obs is None:
        return out
    duty, dcnt, imu, cmd = (np.asarray(v2_obs[k], np.float32) for k in ("duty", "dcnt", "imu", "cmd"))
    T = min(len(duty), len(cmd))
    duty, dcnt, imu, cmd = duty[:T], dcnt[:T], imu[:T], cmd[:T]
    ref = asm.ik(cmd.astype(np.float64), 0.0501) / 5.0
    speed = dcnt * (2 * math.pi / P.CPR / 0.02) / 5.0
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)
    wf = t(np.stack([ref, duty, speed], -1))[None]
    af = t(np.concatenate([cmd[:, :2] / 0.3, cmd[:, 2:], imu], -1))[None]
    geom = t(M.wheel_geometry(asm))[None]
    wm = t(np.asarray(asm.wheel_module), torch.long)[None]
    mask = torch.ones(1, n, dtype=torch.bool, device=dev)
    own = [ci for ci, c in enumerate(ep.cands) if all(r == 0 for r, _ in c.modules)]
    with torch.no_grad():
        zw, za = model.encode(wf, af, geom, mask)
        logits, ab, nu = model.latent_heads(zw, za, wm, mask, m)
        post = M.posterior_summary(dict(logits=logits, ab=ab, nuis=nu))
        if own:
            nmax = max(ep.cands[ci].asm.n for ci in own)
            pos = np.zeros((len(own), nmax), np.int64)
            cm = np.zeros((len(own), nmax), bool)
            cg = np.zeros((len(own), nmax, M.GEOM_DIM), np.float32)
            for j, ci in enumerate(own):
                c = ep.cands[ci]
                rows = np.concatenate([4 * k + np.arange(4) for _, k in c.modules])
                pos[j, :len(rows)], cm[j, :len(rows)] = rows, True
                cg[j, :len(rows)] = M.wheel_geometry(c.asm)
            zc = zw[0][t(pos, torch.long)]
            q = model.counterfactual(zc, t(cm, torch.bool)[:, None], t(cg)[:, None])[:, 0].float().cpu().numpy()
    sec_direct = time.perf_counter() - t0
    for j, ci in enumerate(own):
        p50, sc = q[j, :, 2], np.maximum(q[j, :, 2] - q[j, :, 0], 1e-3)
        for k, key in (("amortized_sim", "q_sim"), ("amortized_marg", "q_marg")):
            out[k]["point"][ci], out[k]["scale"][ci] = p50, sc
            out[k]["lower"][ci] = p50 - cal[key] * sc
    out["amortized_sim"]["seconds"] = out["amortized_marg"]["seconds"] = sec_direct
    if compose and own and dev == "cuda" and "q_comp" in cal:
        par, dr = E.params(ep.plate), E.domain_dr(ep.plate)
        rng = np.random.default_rng(int(ep.index))
        pp = dict(p=post["p"][0], ab=ab[0].float(), nu_mu=post["nu_mu"][0], nu_sd=post["nu_sd"][0])
        jobs = []
        for ci in own:
            c = ep.cands[ci]
            mods = [k for _, k in c.modules]
            widx = np.concatenate([4 * k + np.arange(4) for k in mods])
            lat = CO.sample_child_latents(c.asm, widx, pp, S, rng, dr, mods=mods)
            if c.payload is not None:
                f = lambda v: torch.full((S,), float(v), device="cuda")
                lat["payload"] = dict(mass=f(c.payload[0]), x=f(c.payload[1]), y=f(c.payload[2]))
            jobs.append((c.asm, lat))
        refs = reference.rho_ref_many([ep.cands[ci].asm for ci in own], ep.plate)
        res = CO.compositional_quantiles(jobs, par, ref=refs)
        for ci, (qq, _) in zip(own, res):
            p50, sc = qq[:, 2], np.maximum(qq[:, 2] - qq[:, 0], 1e-3)
            out["amortized_comp"]["point"][ci], out["amortized_comp"]["scale"][ci] = p50, sc
            out["amortized_comp"]["lower"][ci] = p50 - cal["q_comp"] * sc
        torch.cuda.synchronize()
    out["amortized_comp"]["seconds"] = time.perf_counter() - t0
    return out
