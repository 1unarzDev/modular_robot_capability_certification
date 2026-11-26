#!/usr/bin/env python3
"""Exploratory independent baseline re-analysis; never consumes confirmatory inputs."""
from pathlib import Path
import glob
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/codex/re_baselines.csv'
TAUS = (.6, .7, .8)

def ratio(a, b):
    return a / b if b else np.nan

def score(d, lower, stratum):
    truth = d.truth.to_numpy(float)
    fault = d.has_fault.to_numpy(float) > .5
    lower = np.asarray(lower, float)
    valid = np.isfinite(truth)
    covered = ~np.isfinite(lower) | (lower <= truth)
    row = {}
    for tau in TAUS:
        keep = valid & ((fault | (np.abs(truth - tau) < .1)) if stratum == 'decision' else True)
        capable = keep & (truth >= tau)
        incapable = keep & (truth < tau)
        go = keep & np.isfinite(lower) & (lower >= tau)
        row[f'coverage_{tau}'] = ratio(int((covered & keep).sum()), int(keep.sum()))
        row.update({f'n_cap_{tau}': int(capable.sum()), f'n_inc_{tau}': int(incapable.sum()),
                    f'cert_frac_{tau}': ratio(int((go & capable).sum()), int(capable.sum())),
                    f'unsafe_rate_{tau}': ratio(int((go & incapable).sum()), int(incapable.sum())),
                    f'ep_any_unsafe_{tau}': float(pd.Series(go & incapable).groupby(d.eid.to_numpy()).any().mean())})
    keep = valid & ((fault | np.logical_or.reduce([abs(truth-t) < .1 for t in TAUS])) if stratum == 'decision' else True)
    row['n_entries'] = int(keep.sum())
    row['entry_coverage'] = ratio(int((covered & keep).sum()), int(keep.sum()))
    row['episode_coverage'] = float(pd.Series(covered[keep]).groupby(d.eid.to_numpy()[keep]).all().mean()) if keep.any() else np.nan
    return row

def margin(cal, stratum):
    p = cal.point.to_numpy(float); s = cal.scale.to_numpy(float); lo = cal.lower.to_numpy(float)
    missing = ~np.isfinite(p); p = np.where(missing, lo, p)
    s = np.where(missing | ~np.isfinite(s) | (s <= 0), 1., s)
    truth = cal.truth.to_numpy(float); fault = cal.has_fault.to_numpy(float) > .5
    zs, allz = [], []
    for t in TAUS:
        mask = np.isfinite(truth) & ((fault | (abs(truth-t) < .1)) if stratum == 'decision' else True)
        z = (p-t)/s; allz.extend(z[mask & np.isfinite(z)]); zs.extend(z[mask & (truth < t) & np.isfinite(z)])
    if not zs:
        return min(allz) if allz else np.inf
    zs.sort(reverse=True); z = zs[math.floor(.01*len(zs))]
    return z + 1e-9*max(1.,abs(z))

def main():
    paths = [ROOT/'results/baselines/paper/dec_baselines.csv.gz', ROOT/'results/baselines/paper/dec_trivial.csv']
    frames = [pd.read_csv(p).assign(source=str(p.relative_to(ROOT))) for p in paths if p.exists()]
    # These exploratory run names are exact; duplicate units from retries are de-duplicated by decision key.
    runpaths = sorted(glob.glob(str(Path.home()/'.cache/vast-out/*/crane/cfcap-baselines-b2r.b2-e3-pilotbank.s*/units/*.npz')))
    for path in runpaths:
        with np.load(path, allow_pickle=False) as a:
            frames.append(pd.DataFrame({k:a[k] for k in a.files}).assign(source=path))
    rows = []
    if frames:
        d = pd.concat(frames, ignore_index=True)
        key = ['method','split','eid','shape','cand','leg','alpha']
        # Raw pilot units are used only when no derived table for that method/split exists.
        d = d.drop_duplicates(key, keep='first')
        for (method, shape, alpha, split), g in d.groupby(['method','shape','alpha','split'], dropna=False):
            if split == 'cal': continue
            cal = d[(d.method == method) & (d['split'] == 'cal') & (d.alpha == alpha)] if split == 'test' else d.iloc[0:0]
            for st in ('all','decision'):
                base = dict(status='exploratory', method=method, shape=shape, alpha=alpha, split=split, stratum=st,
                            source=';'.join(sorted(g.source.unique())), n_episodes=g.eid.nunique())
                rows.append(dict(base, analysis='raw_lower', note='raw run lower; not a calibrated simultaneous bound', **score(g,g.lower,st)))
                if cal.empty:
                    rows.append(dict(base, analysis='matched', note='unavailable: no cal rows; no test fitting'))
                else:
                    k = margin(cal,st)
                    p=g.point.to_numpy(float); s=g.scale.to_numpy(float); miss=~np.isfinite(p)
                    p=np.where(miss,g.lower.to_numpy(float),p); s=np.where(miss | ~np.isfinite(s) | (s<=0),1.,s)
                    rows.append(dict(base, analysis='matched', margin=k, note='cal only; pooled shapes and thresholds', **score(g,p-k*s,st)))
    rows.append(dict(status='exploratory', analysis='E3', note='unavailable: learned method, model_free, and calibrated comparison rows absent'))
    sweep = []
    pattern = str(Path.home()/'.cache/vast-out/*/crane/cfcap-baselines-smc-small.smc-sweep-p*/units/*.npz')
    for path in sorted(glob.glob(pattern)):
        with np.load(path, allow_pickle=False) as a:
            sweep.append(pd.DataFrame({k: a[k] for k in a.files}).assign(source=path))
    if sweep:
        x = pd.concat(sweep, ignore_index=True).drop_duplicates(['method','split','eid','cand','leg'])
        x['particles'] = x.method.str.extract(r'p(\d+)$').astype(int)
        # The sweep plan picks 5 pair and 5 square IDs; completeness is checked by count and equality.
        sets = {n: set(g.eid) for n,g in x.groupby('particles')}
        common = set.intersection(*sets.values()) if len(sets) == 3 else set()
        complete = set(sets) == {256,1024,2048} and all(len(v) == 10 for v in sets.values()) and len(common) == 10
        for n,g in x.groupby('particles'):
            for subset, y in [('available',g), ('common',g[g.eid.isin(common)])]:
                if y.empty: continue
                ep = y.drop_duplicates(['split','eid'])
                finite = np.isfinite(y.truth) & np.isfinite(y.lower)
                relevant = (y.has_fault > .5) | ((y.truth-.8).abs() < .1)
                rows.append(dict(status='exploratory', analysis='SMC_sweep', method=f'nuisance_smc_p{n}', split='test',
                    stratum=subset, source=pattern, n_episodes=len(ep), truth_retained=ep.truth_retained.mean(),
                    q05_coverage=(y.loc[finite,'lower'] <= y.loc[finite,'truth']).mean(),
                    q05_coverage_relevant=(y.loc[finite & relevant,'lower'] <= y.loc[finite & relevant,'truth']).mean(),
                    seconds_mean=ep.seconds.mean(), note='partial sweep; selection withheld' if not complete else 'complete sweep'))
        if complete:
            z = [r for r in rows if r.get('analysis') == 'SMC_sweep' and r['stratum'] == 'common']
            ref = next(r for r in z if r['method'].endswith('2048'))
            eligible = [r for r in z if all(np.isfinite(r[k]) and r[k] >= ref[k]-.01 for k in ('truth_retained','q05_coverage','q05_coverage_relevant'))]
            chosen = min(eligible, key=lambda r: r['seconds_mean'])['method'] if eligible else 'unavailable'
        else: chosen = 'unavailable: incomplete planned sweep'
        rows.append(dict(status='exploratory', analysis='SMC_selection', source=pattern, note=chosen))
    else:
        rows.append(dict(status='exploratory', analysis='SMC_selection', note='unavailable: no sweep units', source=pattern))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT,index=False)
    print('wrote',OUT,'rows',len(rows))

if __name__ == '__main__': main()
