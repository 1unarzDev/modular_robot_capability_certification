"""N7: size-aware two-simulator margin for single and pair children (targets: docs/board/stress.md "N7 targets").

    scripts/heavy.sh $PY scripts/stress_N7.py     # -> results/stress/N7_margin.csv, N7_verdict.json
"""
import csv
import json
import os
import pathlib
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stress_A5 as A5             # noqa: E402
import stress_A4_merge as A4       # noqa: E402

N = 200
SEED0 = 906000                     # stress_A5m.SEED0
TAUS = (0.7, 0.8)
USTAR = 0.01


def main():
    from cfcap.t1 import capability as cap1, params as P, plant
    from cfcap.t2 import build, capability as cap2
    par = P.base_params("plated")
    th = P.load_theta("plated")
    dr = P.theta_dr(th)
    R = np.load(A5.OUT / "A5_refs.npz")
    zm = np.load(A5.OUT / "A5m_rows.npz")
    rng = np.random.default_rng(SEED0)
    rows, verdict = [], {}
    for s in A5.SHAPES:                                   # replay A5m's draws in order; use single and pair
        asm = P.named_assembly(s, par)
        lat = plant.sample_latents(asm, N, rng, plant.HET_FULL, dr=dr, faults=A5.faults_k(N, asm.n, rng), device="cpu")
        if s not in ("single", "pair"):
            continue
        ref1 = R[f"{s}/T1"]
        r1 = cap1.capability([(asm, lat)], par, device="cpu")[0]["rho"].cpu().numpy() / ref1[None]
        assert np.allclose(r1, zm[f"{s}/T1/rho_dec"], atol=1e-4), "replayed latents differ from A5m"
        lp = A4.perturb(lat, np.random.default_rng([SEED0, 77, len(s)]))
        r1p = cap1.capability([(asm, lp)], par, device="cpu")[0]["rho"].cpu().numpy() / ref1[None]
        q = float(np.percentile(np.maximum(r1 - r1p, 0), 99))
        eff = build.ABLATIONS["all+motor"]
        tr = zm[f"{s}/all+motor/rho_dec"] * cap2.rho_ref(asm, par, "plated", eff)[None] / ref1[None]   # fixed ref
        faulted = (lat["mode"].numpy() > 0).any(1)
        for meth, marg in (("t1_only", 0.0), ("size_aware", q)):
            low = r1 - marg
            go = np.stack([low >= t for t in TAUS])
            cap_ = np.stack([tr >= t for t in TAUS])
            unsafe = float((go & ~cap_).mean())
            cf = float((go & cap_).sum() / max(1, cap_.sum()))
            rows.append(dict(shape=s, method=meth, q=marg, decisions=int(go.size), unsafe=unsafe, cert_frac=cf,
                             unsafe_faulted=float((go & ~cap_)[:, faulted].mean()),
                             cf_faulted=float((go & cap_)[:, faulted].sum() / max(1, cap_[:, faulted].sum()))))
        a, b = rows[-2], rows[-1]
        verdict[s] = dict(q=q, unsafe_t1_only=a["unsafe"], unsafe_size_aware=b["unsafe"], cf_t1_only=a["cert_frac"],
                          cf_size_aware=b["cert_frac"], cf_loss_pts=100 * (a["cert_frac"] - b["cert_frac"]),
                          bar_met=bool(b["unsafe"] <= 2 * USTAR and 100 * (a["cert_frac"] - b["cert_frac"]) <= 5))
        print(s, verdict[s], flush=True)
    A5.RES.mkdir(parents=True, exist_ok=True)
    with open(A5.RES / "N7_margin.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (A5.RES / "N7_verdict.json").write_text(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
