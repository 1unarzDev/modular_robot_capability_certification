"""Train the amortized posterior (models pilot, exploratory). Wall-clock budgeted; cosine LR over the budget.

  scripts/heavy.sh --gpu $PY scripts/models_pilot_train.py --minutes 12 --tag n4
  scripts/heavy.sh --gpu $PY scripts/models_pilot_train.py --minutes 6 --shapes single,pair --tag n2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time

import numpy as np
import torch

from cfcap.models import data as D
from cfcap.models import posterior as M

OUT = pathlib.Path(os.environ.get("CFCAP_OUT", pathlib.Path.home() / ".cache/modular-runs/2026-11/cfcap")) / "models"


def load(split, shapes):
    return [D.ShapeSet(OUT / "pilot" / f"{split}_{s}.npz", s) for s in shapes]


@torch.no_grad()
def evaluate(model, sets, wheels=2048):
    model.eval()
    tot, n = {}, 0
    for ss in sets:
        bs = max(8, wheels // ss.n)
        for i in range(0, ss.N, bs):
            b = ss.batch(torch.arange(i, min(ss.N, i + bs), device="cuda"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b["wf"], b["af"], b["geom"], b["wm"], b["mask"], b["n_mod"], b["member"], b["cgeom"])
            out = {k: v.float() for k, v in out.items()}
            _, parts = M.loss_fn(out, b)
            for k, v in parts.items():
                tot[k] = tot.get(k, 0.0) + float(v) * len(b["mode"])
            n += len(b["mode"])
    model.train()
    return {k: v / n for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="single,pair,L3,square")
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--tag", default="n4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--h", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wheels", type=int, default=1536, help="wheel sequences per step")
    ap.add_argument("--no-direct", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    shapes = a.shapes.split(",")
    tr, va = load("train", shapes), load("val", shapes)
    model = M.MorphPosterior(h=a.h, layers=a.layers, direct=not a.no_direct).cuda()
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    w = np.array([s.N for s in tr], float)
    w /= w.sum()
    budget = a.minutes * 60
    log = OUT / f"train_{a.tag}_s{a.seed}.log"
    f = log.open("w")
    pr = lambda s: (print(s, flush=True), f.write(s + "\n"), f.flush())
    pr(f"params {nparam} train {[(s.shape, s.N) for s in tr]}")
    t0 = time.time()
    step, eps, warm = 0, 0, 200
    hist = []
    while True:
        el = time.time() - t0
        if el > budget:
            break
        frac = el / budget
        lr = a.lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        ss = tr[rng.choice(len(tr), p=w)]
        bs = max(8, a.wheels // ss.n)
        b = ss.batch(torch.as_tensor(rng.integers(ss.N, size=bs), device="cuda"))
        b["af"][..., 3:] += 0.01 * torch.randn(bs, 1, 3, device="cuda")            # capxfer augmentation
        b["wf"][..., 2] *= 1 + 0.01 * torch.randn(bs, 1, ss.n, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["wf"], b["af"], b["geom"], b["wm"], b["mask"], b["n_mod"], b["member"], b["cgeom"])
        out = {k: v.float() for k, v in out.items()}
        loss, parts = M.loss_fn(out, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        eps += bs
        if step % 500 == 0:
            v = evaluate(model, va)
            hist.append(dict(step=step, t=time.time() - t0, eps=eps, **{f"val_{k}": x for k, x in v.items()}))
            pr(f"step {step} {time.time() - t0:.0f}s eps {eps} lr {lr:.2e} loss {float(loss):.4f} "
               f"val {json.dumps({k: round(x, 4) for k, x in v.items()})}")
    v = evaluate(model, va)
    wall = time.time() - t0
    meta = dict(tag=a.tag, seed=a.seed, shapes=shapes, h=a.h, layers=a.layers, direct=not a.no_direct, params=nparam,
                steps=step, episodes_seen=eps, wall_s=wall, eps_per_s=eps / wall, val=v, hist=hist)
    torch.save(dict(state=model.state_dict(), meta=meta), OUT / f"model_{a.tag}_s{a.seed}.pt")
    pr(f"done {json.dumps({k: meta[k] for k in ('steps', 'episodes_seen', 'wall_s', 'eps_per_s')})} val {v}")


if __name__ == "__main__":
    main()
