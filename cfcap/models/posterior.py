"""Amortized morphology-conditioned posterior over module latents (PLAN §2 contribution 4).

A set transformer over wheel tokens. Each wheel token is a per-wheel temporal encoding (TCN) of (IK reference speed,
pre_duty, encoder speed) plus an embedding of the wheel's geometry: position in the assembly frame, rolling and roller
axes, its module's centre, and its offset from that centre. There are no wheel- or module-index embeddings, so the
model is equivariant to relabelling wheels and modules (the shortcut a fixed index would give is ruled out). One extra
token carries the assembly channels (command, IMU yaw rate and planar acceleration).

Heads:
  * per wheel: mode logits over (ok, weak, coast, brake) and a Beta(a, b) over the weak gain g;
  * per module (wheel tokens mean-pooled by module): Gaussian over log mu, log stall scale, log no-load scale;
  * optional direct counterfactual head (the "black-box" comparator): given a candidate's wheel tokens with their
    geometry in the candidate's frame, monotone quantiles of rho_rel on each decision leg. A merge is the same call on
    the concatenated token sets of the two parents (`counterfactual` takes any token set).

Feature conventions follow capxfer selfmodel: wheel channels (ref/5, pre_duty, speed/5), assembly channels
(vx/0.3, vy/0.3, wz, gz, ax, ay).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ("ok", "weak", "coast", "brake")
GEOM_DIM = 10
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
N_LEGS = 10
NUIS = ("log_mu", "log_stall", "log_noload")


def wheel_geometry(asm):
    """(n, GEOM_DIM) float32: wheel pos /0.5, w_hat, r_hat, module centre /0.5, offset from module centre /0.5."""
    wm = np.asarray(asm.wheel_module)
    mc = asm.module_pos[wm]
    g = np.concatenate([asm.wheel_pos / 0.5, asm.w_hat, asm.r_hat, mc / 0.5, (asm.wheel_pos - mc) / 0.5], 1)
    return g.astype(np.float32)


def wheel_features(asm, cmd, pre_duty, speed):
    """cmd (B,T,3), pre_duty/speed (B,T,n) -> (B,T,n,3) as capxfer build_features (firmware IK on nominal radius)."""
    ref = asm.ik(np.asarray(cmd, np.float64), 0.0501)
    return np.stack([ref / 5.0, np.asarray(pre_duty), np.asarray(speed) / 5.0], -1).astype(np.float32)


def asm_features(cmd, gz, ax, ay):
    cmd = np.asarray(cmd)
    return np.stack([cmd[..., 0] / 0.3, cmd[..., 1] / 0.3, cmd[..., 2], gz, ax, ay], -1).astype(np.float32)


class TCN(nn.Module):
    """Strided dilated temporal encoder: (N, C, T) -> (N, h) by mean + max pooling."""

    def __init__(self, c_in, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, h, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(h, h, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(h, h, 5, stride=2, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(h, h, 5, padding=8, dilation=4), nn.GELU())
        self.out = nn.Linear(2 * h, h)

    def forward(self, x):
        z = self.net(x)
        return self.out(torch.cat([z.mean(-1), z.amax(-1)], -1))


def mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


def _block(h, heads):
    return nn.TransformerEncoderLayer(h, heads, 4 * h, dropout=0.0, batch_first=True, norm_first=True,
                                      activation="gelu")


class MorphPosterior(nn.Module):
    def __init__(self, h=96, layers=3, heads=4, cf_layers=1, direct=True, n_legs=N_LEGS, quantiles=QUANTILES):
        super().__init__()
        self.h = h
        self.wheel_tcn = TCN(3, h)
        self.asm_tcn = TCN(6, h)
        self.geom = mlp(GEOM_DIM, h, h)
        self.type_emb = nn.Parameter(torch.randn(2, h) * 0.02)           # 0 wheel token, 1 assembly token
        self.layers = layers
        if layers > 0:
            self.enc = nn.TransformerEncoder(_block(h, heads), layers, enable_nested_tensor=False)
        else:                          # DeepSets ablation (capxfer-like pooled context, no attention)
            self.ctx = mlp(2 * h, h, h)
        self.norm = nn.LayerNorm(h)
        self.mode_head = mlp(h, h, len(CLASSES) + 2)
        self.nuis_head = mlp(2 * h, h, 2 * len(NUIS))
        self.direct = direct
        self.quantiles = tuple(quantiles)
        if direct:
            self.cf_geom = mlp(GEOM_DIM, h, h)
            self.cf_enc = nn.TransformerEncoder(_block(h, heads), cf_layers, enable_nested_tensor=False)
            self.leg_q = nn.Parameter(torch.randn(n_legs, h) * 0.02)
            self.cf_att = nn.MultiheadAttention(h, heads, batch_first=True)
            self.cf_out = mlp(2 * h, h, len(quantiles))

    # ------------------------------------------------------------------ encoder
    def encode(self, wf, af, geom, mask):
        """wf (B,T,n,3), af (B,T,6), geom (B,n,G), mask (B,n) bool -> wheel tokens (B,n,h), assembly token (B,h)."""
        B, T, n, _ = wf.shape
        x = wf.permute(0, 2, 3, 1).reshape(B * n, 3, T)
        tw = self.wheel_tcn(x).reshape(B, n, -1) + self.geom(geom) + self.type_emb[0]
        ta = self.asm_tcn(af.permute(0, 2, 1)) + self.type_emb[1]
        tok = torch.cat([ta[:, None], tw], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask], 1)
        if self.layers > 0:
            z = self.norm(self.enc(tok, src_key_padding_mask=pad))
        else:
            keep = (~pad).float()[..., None]
            mean = (tok * keep).sum(1, keepdim=True) / keep.sum(1, keepdim=True).clamp(min=1)
            z = self.norm(tok + self.ctx(torch.cat([tok, mean.expand_as(tok)], -1)))
        return z[:, 1:], z[:, 0]

    # ------------------------------------------------------------------ heads
    def latent_heads(self, zw, za, wm, mask, n_mod):
        """wm (B,n) module index per wheel. Returns mode logits (B,n,4), Beta (a,b) (B,n,2), nuisance (B,M,2*K)."""
        o = self.mode_head(zw)
        logits = o[..., :len(CLASSES)]
        ab = 1.0 + F.softplus(o[..., len(CLASSES):])
        oh = F.one_hot(wm.clamp(min=0), n_mod).float() * mask[..., None].float()          # (B,n,M)
        zm = torch.einsum("bnm,bnh->bmh", oh, zw) / oh.sum(1)[..., None].clamp(min=1)
        nu = self.nuis_head(torch.cat([zm, za[:, None].expand_as(zm)], -1))
        return logits, ab, nu

    def counterfactual(self, zw, member, cgeom, za=None):
        """Direct head. zw (B,n,h) wheel tokens (any token set, e.g. two parents concatenated for a merge),
        member (B,C,n) bool: wheels in candidate c, cgeom (B,C,n,G): wheel geometry in the candidate's frame.
        -> rho_rel quantiles (B,C,L,Q), non-decreasing in Q."""
        B, C, n = member.shape
        x = zw[:, None].expand(B, C, n, self.h) + self.cf_geom(cgeom)
        x = x.reshape(B * C, n, self.h)
        m = member.reshape(B * C, n)
        x = self.cf_enc(x, src_key_padding_mask=~m)
        q = self.leg_q[None].expand(B * C, -1, -1)
        a, _ = self.cf_att(q, x, x, key_padding_mask=~m)
        pooled = (x * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        o = self.cf_out(torch.cat([a, pooled[:, None].expand_as(a)], -1))               # (BC,L,Q)
        base = o[..., :1]
        inc = F.softplus(o[..., 1:])
        out = torch.cat([base, base + torch.cumsum(inc, -1)], -1)
        return out.reshape(B, C, -1, len(self.quantiles))

    def forward(self, wf, af, geom, wm, mask, n_mod, member=None, cgeom=None):
        zw, za = self.encode(wf, af, geom, mask)
        logits, ab, nu = self.latent_heads(zw, za, wm, mask, n_mod)
        out = dict(logits=logits, ab=ab, nuis=nu)
        if self.direct and member is not None:
            out["cf"] = self.counterfactual(zw, member & mask[:, None], cgeom)
        return out


# ---------------------------------------------------------------------- losses
MODE_W = (1.0, 2.0, 2.0, 2.0)


def beta_nll(ab, g):
    a, b = ab[..., 0], ab[..., 1]
    g = g.clamp(1e-3, 1 - 1e-3)
    return -(torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b) + (a - 1) * torch.log(g)
             + (b - 1) * torch.log1p(-g))


def pinball(q, y, taus):
    """q (...,Q), y (...) -> mean pinball loss."""
    t = torch.as_tensor(taus, device=q.device, dtype=q.dtype)
    d = y[..., None] - q
    return torch.maximum(t * d, (t - 1) * d).mean()


def loss_fn(out, batch, w_g=1.0, w_nu=0.3, w_cf=5.0, w_ce=1.0, taus=QUANTILES):
    mask, y, g = batch["mask"], batch["mode"], batch["g"]
    wts = torch.tensor(MODE_W, device=y.device)
    ce = F.cross_entropy(out["logits"][mask], y[mask], weight=wts)
    wk = mask & (y == 1)
    lg = beta_nll(out["ab"][wk], g[wk]).mean() if wk.any() else ce * 0
    K = len(NUIS)
    mu, ls = out["nuis"][..., :K], out["nuis"][..., K:].clamp(-6, 3)
    mm = batch["mod_mask"][..., None]
    nll = 0.5 * ((batch["nuis"] - mu) / ls.exp()) ** 2 + ls
    lnu = (nll * mm).sum() / (mm.sum() * K).clamp(min=1)
    parts = dict(ce=ce, beta=lg, nuis=lnu)
    loss = w_ce * ce + w_g * lg + w_nu * lnu
    if "cf" in out:
        lc = pinball(out["cf"], batch.get("target", batch.get("rho_rel")), taus)
        parts["cf"] = lc
        loss = loss + w_cf * lc
    return loss, parts


@torch.no_grad()
def posterior_summary(out):
    """-> p (B,n,4), g_mean (B,n) (Beta mean), nuisance mean/sd (B,M,K)."""
    p = torch.softmax(out["logits"], -1)
    a, b = out["ab"][..., 0], out["ab"][..., 1]
    K = len(NUIS)
    return dict(p=p, g_mean=a / (a + b), nu_mu=out["nuis"][..., :K], nu_sd=out["nuis"][..., K:].clamp(-6, 3).exp())
