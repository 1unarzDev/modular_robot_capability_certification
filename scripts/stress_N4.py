"""N4: merge certificate with inferred latents (targets: docs/board/stress.md "N4 targets").

    scripts/heavy.sh --gpu $PY scripts/stress_N4.py run --arm P0 --shard 0     # 250 pairs -> $CFCAP_OUT/stress/N4_*.npz
    $PY scripts/stress_N4.py analyze           # -> results/stress/N4_merge.csv, N4_verdict.json, paper/dec_merge_n4.csv

Pairs are the E2 pairs (stress_A4_merge.run export replay). Each G_j is probed alone with the v2 routine (no jitter,
as cfcap.baselines.episode.observe_parent: T1 on the GPU, plant.observe, duty noise 0.005), no payload. Methods from
cfcap/baselines/compose.py: enum_known (composed posterior), copy_parent, detector_gated. Truth: E2's T1 and T2 rho_abs
over the FIXED T1 reference (baselines.episode.ref_table).
"""
import argparse
import csv
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stress_A4_merge as A4    # noqa: E402

TAUS = (0.6, 0.7, 0.8)
ALPHA = 0.1


def run(arm, shard, pairs=250, limit=None, device="cuda", start=0, stop=None):
    import torch
    from cfcap.baselines import compose as CO
    from cfcap.baselines.episode import DUTY_NOISE, lat_to, ref_table
    from cfcap.t1 import params as P, plant
    par = P.base_params("plated")
    recs = A4.run(arm, pairs, shard, export=[])
    recs = recs[start:stop]
    if limit:
        recs = recs[:limit]
    z = np.load(A4.OUT / f"A4_{arm}_s{shard}.npz")
    key = {(int(p), int(m)): i for i, (p, m) in enumerate(zip(z["meta_pair"], z["meta_merge"]))}
    out = {k: [] for k in ("pair", "merge", "n_mod", "n_fault", "truth_T1", "truth_T2", "enum_point", "enum_lower",
                           "enum_scale", "copy_parent", "detector_gated", "enum_abstain")}
    tab, T = None, None
    t0 = time.time()
    for ri, rec in enumerate(recs):
        probed = []
        for j, (g, l) in enumerate(((rec["g1"], rec["l1"]), (rec["g2"], rec["l2"]))):
            tab, T = plant.v2_probe(g, par, 1, jitter=False)
            ld = lat_to(dict(l, payload=None), device)
            R = plant.simulate_bucket([dict(asm=g, lat=ld, cmd=tab[0])], par, T, device=device)[0]
            obs = plant.observe(R, ld, seed=rec["pair"] * 7 + j)
            gen = torch.Generator(device=obs["pre_duty"].device).manual_seed(rec["pair"] * 7 + j + 1)
            obs["pre_duty"] = obs["pre_duty"] + DUTY_NOISE * torch.randn(obs["pre_duty"].shape, generator=gen,
                                                                         device=obs["pre_duty"].device)
            probed.append(CO.Probed(g, tab[0], T, {k: v[0].cpu() for k, v in obs.items()}, dict(l, payload=None)))
        mods = [(0, k) for k in range(rec["k1"])] + [(1, k) for k in range(rec["k2"])]
        cands = [CO.MergeCand(asm, list(mods), None, pay) for _, asm, pay in rec["merges"]]
        pe = CO.merge_enum_known(probed, cands, device=device)
        pc = CO.merge_copy_parent(probed, cands)
        pd = CO.merge_detector_gated(probed, cands)
        ref = ref_table(cands, "plated", device)
        for ci, (mi, asm, pay) in enumerate(rec["merges"]):
            i = key[(rec["pair"], mi)]
            out["pair"].append(rec["pair"]); out["merge"].append(mi)
            out["n_mod"].append(rec["k1"] + rec["k2"]); out["n_fault"].append(int(z["meta_n_fault"][i]))
            out["truth_T1"].append(z["T1_truth"][i] * z["ref1"][i] / ref[ci])
            out["truth_T2"].append(z["T2_truth"][i] * z["ref2"][i] / ref[ci])
            out["enum_point"].append(pe.point[ci]); out["enum_lower"].append(pe.lower[ci])
            out["enum_scale"].append(pe.scale[ci]); out["copy_parent"].append(pc.lower[ci])
            out["detector_gated"].append(pd.lower[ci])
            out["enum_abstain"].append(int(not np.isfinite(pe.lower[ci]).all()))
        if ri % 25 == 0:
            print(arm, shard, ri, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(A4.OUT / f"N4_{arm}_s{shard}_{start:03d}.npz", **{k: np.asarray(v) for k, v in out.items()})


def _conf_q(short, pairs):
    sc = np.sort([short[p] for p in pairs])
    if not len(sc):
        return np.inf
    return float(sc[min(int(np.ceil((len(sc) + 1) * (1 - ALPHA))), len(sc)) - 1])


def _metrics(lower, truth, faulted, pair):
    go = np.stack([lower >= t for t in TAUS], -1)
    cap = np.stack([np.broadcast_to(truth[..., None] >= np.array(TAUS), go.shape)], 0)[0]
    strat = np.stack([(faulted[:, None] | (np.abs(truth - t) < 0.1)) for t in TAUS], -1)
    up = np.unique(pair)
    mis = np.mean([(truth[pair == p] < lower[pair == p]).any() for p in up])
    return dict(cf_all=float((go & cap).sum() / max(1, cap.sum())),
                cf_stratum=float((go & cap & strat).sum() / max(1, (cap & strat).sum())),
                unsafe=float((go & ~cap).mean()), unsafe_stratum=float((go & ~cap & strat).sum() / max(1, strat.sum())),
                miscov_pair=float(mis))


def analyze():
    rows, verdict, dec = [], {}, []
    margins = {}
    qp = {}
    for r in csv.DictReader(open(A4.RES / "A4_merge.csv")):
        if r["group"] == "all" and r["plant"] == "T1" and r.get("q_pert_by_nmod"):
            qp[r["arm"]] = {int(k): v for k, v in json.loads(r["q_pert_by_nmod"]).items()}
    for arm in ("P0", "P5"):
        fs = sorted(A4.OUT.glob(f"N4_{arm}_s*.npz"))
        if not fs:
            continue
        zs = [np.load(f) for f in fs]
        z = {k: np.concatenate([q[k] for q in zs]) for k in zs[0].files}
        pair, faulted = z["pair"], z["n_fault"] > 0
        lo = np.where(np.isfinite(z["enum_lower"]), z["enum_lower"], -np.inf)
        up = np.unique(pair)
        short = {p: float(np.maximum(lo[pair == p] - z["truth_T1"][pair == p], 0).max()) for p in up}
        ev, od = up[up % 2 == 0], up[up % 2 == 1]
        q_od, q_ev = _conf_q(short, ev), _conf_q(short, od)          # q for odd pairs from even, and vice versa
        qcf = np.where(pair % 2 == 1, q_od, q_ev)[:, None]
        margins[arm] = dict(q_cf_all=_conf_q(short, up), q_cf_crossfit=[q_od, q_ev],
                            q_pert_by_nmod={str(k): v for k, v in qp.get(arm, {}).items()}, alpha=ALPHA,
                            source="exploratory E2 pairs (stress_A4_merge seeds 907000), N4 enum_known lower")
        lcf = lo - qcf
        lsz = lcf - np.array([qp.get(arm, {}).get(int(n), 0.0) for n in z["n_mod"]])[:, None]
        meths = dict(comp_enum=lcf, comp_enum_size=lsz, comp_enum_raw=lo, copy_parent=z["copy_parent"],
                     detector_gated=z["detector_gated"])
        for plant in ("T1", "T2"):
            tr = z[f"truth_{plant}"]
            for m, low in meths.items():
                r = dict(arm=arm, plant=plant, method=m, merges=len(tr), pairs=len(up), q_cf=json.dumps([q_od, q_ev]),
                         abstain=float(z["enum_abstain"].mean()), **_metrics(low, tr, faulted, pair))
                rows.append(r)
                for i in range(len(tr)):
                    for l in range(tr.shape[1]):
                        dec.append(dict(method=m, route="compositional" if m.startswith("comp") else "raw",
                                        posterior="enum" if m.startswith("comp") else "none", train="T1",
                                        test_plant=plant, split="merge_" + arm, eid=int(pair[i]), shape="", family="",
                                        cand=int(z["merge"][i]), kind="merge", cand_size=int(z["n_mod"][i]),
                                        has_fault=int(faulted[i]), leg=l, truth=float(tr[i, l]),
                                        point=float(z["enum_point"][i, l]) if m.startswith("comp") else float(low[i, l]),
                                        scale=float(z["enum_scale"][i, l]) if m.startswith("comp") else np.nan,
                                        lower=float(low[i, l]), alpha=ALPHA, seconds=np.nan))
        g = {(r["plant"], r["method"]): r for r in rows if r["arm"] == arm}
        ce1, cs2, cp1 = g[("T1", "comp_enum")], g[("T2", "comp_enum_size")], g[("T1", "copy_parent")]
        verdict[arm] = dict(cf_stratum_T1=ce1["cf_stratum"], unsafe_T1=ce1["unsafe"], miscov_T1=ce1["miscov_pair"],
                            cf_stratum_T2_size=cs2["cf_stratum"], unsafe_T2_size=cs2["unsafe"],
                            miscov_T2_size=cs2["miscov_pair"], cf_copy_parent_T1=cp1["cf_stratum"],
                            bar_cf=bool(ce1["cf_stratum"] >= 0.85 and ce1["unsafe"] <= 0.01),
                            bar_miscov=bool(ce1["miscov_pair"] <= 0.1 and cs2["miscov_pair"] <= 0.15),
                            bar_vs_copy_parent=bool(ce1["cf_stratum"] - cp1["cf_stratum"] >= 0.10))
    A4.RES.mkdir(parents=True, exist_ok=True)
    (A4.RES / "paper").mkdir(exist_ok=True)
    for name, rr in (("N4_merge.csv", rows), ("paper/dec_merge_n4.csv", dec)):
        with open(A4.RES / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    (A4.RES / "N4_verdict.json").write_text(json.dumps(verdict, indent=1))
    (A4.RES / "N4_margins.json").write_text(json.dumps(margins, indent=1))     # frozen for C-MERGE
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "analyze"))
    ap.add_argument("--arm", default="P0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    a = ap.parse_args()
    run(a.arm, a.shard, limit=a.limit, start=a.start, stop=a.stop) if a.cmd == "run" else analyze()
