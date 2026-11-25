"""Compositional counterfactual route: posterior samples of module latents -> T1 closed-loop rollouts of the candidate
(`t1.capability`, relative=True) -> rho_rel quantiles. The comparator to the direct head in `posterior.py`.

Per-wheel modes are sampled mean-field from the mode posterior, weak gains from the Beta head, per-module mu / stall /
no-load from the nuisance head; everything the posterior does not model (trial DR, per-wheel motor spread around the
module mean, mounting yaw, load split) is drawn from the T1 prior.
"""
from __future__ import annotations

import numpy as np
import torch

from cfcap.models import posterior as M
from cfcap.t1 import capability as cap
from cfcap.t1 import plant


def sample_child_latents(child_asm, wheel_idx, post, S, rng, dr, het=plant.HET_FULL, mods=None, oracle=None):
    """post: per-episode posterior of the parent (p (n,4), ab (n,2), nu_mu/nu_sd (M,K)) as torch on cuda;
    wheel_idx: parent wheel indices of the child's wheels (child order); mods: parent module indices (child order).
    oracle: optional (mode (n,), g (n,)) true parent faults to use instead of posterior samples.
    Returns child latents for S samples."""
    lat = plant.sample_latents(child_asm, S, rng, het, dr=dr)
    dev = lat["mode"].device
    wi = torch.as_tensor(wheel_idx, device=dev)
    if oracle is None:
        p = post["p"][wi]                                                    # (n_c, 4)
        mode = torch.multinomial(p, S, replacement=True).T                   # (S, n_c)
        a, b = post["ab"][wi, 0], post["ab"][wi, 1]
        g = torch.distributions.Beta(a, b).sample((S,))
        g = torch.where(mode == plant.WEAK, g, torch.ones_like(g))
    else:
        mode = torch.as_tensor(oracle[0], device=dev)[wi].long().expand(S, -1).clone()
        g = torch.as_tensor(oracle[1], device=dev)[wi].float().expand(S, -1).clone()
    lat["mode"], lat["g"] = mode, g
    if oracle is None and mods is not None:
        mi = torch.as_tensor(mods, device=dev)
        nu = post["nu_mu"][mi] + post["nu_sd"][mi] * torch.randn(S, len(mods), len(M.NUIS), device=dev)
        d = lat["dom"]
        lat["mu_mod"] = nu[..., 0].exp() / d["mu"][:, None]
        wm = torch.as_tensor(np.asarray(child_asm.wheel_module), device=dev)
        oh = torch.nn.functional.one_hot(wm, len(mods)).float()
        for key, k, dk in (("stall_w", 1, "stall_scale"), ("noload_w", 2, "noload_scale")):
            mean_w = (lat[key] @ oh) / oh.sum(0)                             # (S, m_c)
            tgt = nu[..., k].exp() / d[dk][:, None]
            lat[key] = lat[key] * (tgt / mean_w)[:, wm]
    return lat


def compositional_quantiles(jobs, par, taus=M.QUANTILES, ref=None):
    """jobs: list of (child_asm, lat with S rows). -> list of (L, Q) quantile arrays, and the raw samples (S, L).
    ref None: rho_rel samples (paired fault-free rollout); ref (len(jobs), L): rho_dec = rho_abs / ref samples."""
    res = cap.capability(jobs, par, relative=ref is None)
    out = []
    for i, r in enumerate(res):
        s = r["rho_rel"].cpu().numpy() if ref is None else r["rho"].cpu().numpy() / ref[i]
        out.append((np.nanquantile(s, taus, axis=0).T, s))
    return out


def sample_from_episode(ep, cand, post, wheel_idx, mods, S, rng, dr, device="cuda"):
    """Candidate latents for S posterior samples, built on the generator's own plant latents of the candidate
    (`cfcap.gen.episode.plant_latents`) so every public per-module attribute the generator carries (module type,
    payload, the candidate's normal-load share) is kept, while everything the robot does not know is replaced:
    trial domain and mounting-yaw errors from the T1 prior, per-wheel motor spread from the prior around the
    posterior's module means, battery offsets from the prior, modes / g / mu from the posterior."""
    from capxfer import sim
    from cfcap.gen import episode as E
    lat = E.plant_latents(ep, cand.modules, cand.load_z, cand.payload, B=S)
    lat = {k: ({kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict)
               else (v.to(device) if torch.is_tensor(v) else v)) for k, v in lat.items()}
    prior = plant.sample_latents(cand.asm, S, rng, E.HET, dr=dr, device=device)
    for k in ("dom", "yaw_err", "vbatt_off", "stall_w", "noload_w"):
        lat[k] = prior[k]
    pc = sample_child_latents(cand.asm, wheel_idx, post, S, rng, dr, het=E.HET, mods=mods)
    lat["dom"] = pc["dom"]
    for k in ("mode", "g", "mu_mod", "stall_w", "noload_w"):
        lat[k] = pc[k]
    return lat
