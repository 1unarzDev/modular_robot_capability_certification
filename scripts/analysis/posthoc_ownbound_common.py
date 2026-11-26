"""Post-hoc: own-bound CF / E_dec / E_unsafe of ours vs CapXfer (and ghat-physics) on the C-MAIN T1 episodes both
streams share. Reproduces the run-1 cells first (asserts equality), then restricts to the eid intersection.
Writes results/confirmatory/posthoc_p1p2/ownbound_common.csv. Run 1 verdicts unchanged."""
import pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import confirmatory as CF
import common as C

R = pathlib.Path(C.RES)
dm = pd.read_csv(R / "models/confirmatory/dec_conf_models.csv.gz")
db = pd.read_csv(R / "baselines/confirmatory/dec_conf_baselines.csv.gz")
d = pd.concat([dm, db], ignore_index=True)
d = d[(d.claim == "C-MAIN") & (d.split == "conf_test") & (d.test_plant == "T1") &
      d.eid.between(CF.CMAIN_IDX[0], CF.CMAIN_IDX[1] - 1)]
if "shard" in d:
    d = d[d.shard != 0]
cells = pd.read_csv(R / "confirmatory/cells.csv")
meths = {"amortized_comp": "ours", "capxfer": "capxfer", "ghat_physics": "ghat"}
common = None
for m in meths:
    e = set(d[d.method == m].eid.unique())
    common = e if common is None else common & e
rows = []
for m, lab in meths.items():
    ev = d[d.method == m].reset_index(drop=True)
    for blk in ("stratum", "lone-faulted", "merge"):
        full = CF.cluster_stats(ev, ev["lower"], CF._sel(blk))
        ref = cells[(cells.claim == "C-MAIN") & (cells.plant == "T1") & (cells.method == m) & (cells.block == blk)].iloc[0]
        cf_full = C._ratio(full.cert.sum(), full.cap.sum())
        assert abs(cf_full - ref.CF_own) < 1e-9, (m, blk, cf_full, ref.CF_own)
        sub = full.loc[full.index.isin(common)]
        boot = CF.Boot(np.sort(list(common)))
        e_cf, r_cf = boot.ratio(sub, "cert", "cap")
        e_ed, _ = boot.ratio(sub, "uns", "inc")
        rows.append(dict(method=lab, block=blk, n_common=len(common), CF_own=e_cf,
                         CF_lo=np.nanpercentile(r_cf, 2.5), CF_hi=np.nanpercentile(r_cf, 97.5),
                         E_dec_own=e_ed, n_incapable=int(sub.inc.sum()), E_unsafe=float(sub.ep_unsafe.mean())))
out = pd.DataFrame(rows)
p = R / "confirmatory/posthoc_p1p2/ownbound_common.csv"
out.to_csv(p, index=False)
print(out.to_string())
