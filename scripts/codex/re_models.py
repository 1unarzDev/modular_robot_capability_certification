#!/usr/bin/env python3
"""Exploratory independent analysis; reads only named exploratory splits.

Unsafe rate uses incapable decisions as denominator; certified fraction uses capable
ones. Thresholds are pooled across 0.6, 0.7 and 0.8. Confidence intervals
resample whole episodes, separately within each reported group.
"""
import json, pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cfcap.models.train import grid

CACHE = pathlib.Path.home() / '.cache/vast-out'
SPLITS = ('cal', 'test', 'H3', 'H4', 'H4b', 'het_cal', 'het_test', 'H1', 'H2')
TAUS = np.array([.6, .7, .8])
B = 1000


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return pd.DataFrame({k: z[k] for k in z.files})


def conformal(cal, prefix):
    p = cal[prefix+'q50'].to_numpy(dtype=float)
    s = np.maximum(p-cal[prefix+'q05'].to_numpy(dtype=float), .001)
    y = cal.truth.to_numpy(dtype=float)
    ok = np.isfinite(p) & np.isfinite(s) & np.isfinite(y)
    score = pd.DataFrame({'eid':cal.eid[ok], 'score':(p[ok]-y[ok])/s[ok]}).groupby('eid').score.max().to_numpy()
    if not len(score): return None
    k = int(np.ceil((len(score)+1)*.9))
    return np.inf if k > len(score) else np.partition(score, k-1)[k-1]


def analyze(df, prefix, q, tag, split, source):
    p = df[prefix+'q50'].to_numpy(dtype=float)
    ok = np.isfinite(p)
    if not ok.any() or q is None: return []
    df = df.loc[ok].copy()
    p = p[ok]; y = df.truth.to_numpy(dtype=float)
    lower = p-q*np.maximum(p-df[prefix+'q05'].to_numpy(dtype=float), .001)
    go = lower[:,None] >= TAUS
    cap = y[:,None] >= TAUS
    df['_cov90'] = ((y >= df[prefix+'q05']) & (y <= df[prefix+'q95'])).astype(int)
    df['_entry_cov'] = (y >= lower).astype(int)
    df['_cert_num'] = (go & cap).sum(axis=1)
    df['_cert_den'] = cap.sum(axis=1)
    df['_unsafe_num'] = (go & ~cap).sum(axis=1)
    df['_unsafe_den'] = (~cap).sum(axis=1)
    df['_one'] = 1
    rel = (df.has_fault.to_numpy() > 0) | (np.min(np.abs(y[:,None]-TAUS),axis=1) < .1)
    result = []
    for n in ['all'] + sorted(df.parent_n.unique().tolist()):
        for st in ('all','relevant'):
            mask = np.ones(len(df),dtype=bool) if n == 'all' else df.parent_n.to_numpy() == n
            if st == 'relevant': mask &= rel
            d = df.loc[mask]
            if d.empty: continue
            sums = d.groupby('eid')[['_cov90','_entry_cov','_cert_num','_cert_den','_unsafe_num','_unsafe_den','_one']].sum().to_numpy(dtype=float)
            totals = sums.sum(axis=0)
            rng = np.random.default_rng(20261125)
            boot = np.zeros((B,7))
            # Batch limits the peak allocation for large episode sets.
            for start in range(0,B,100):
                ix = rng.integers(0,len(sums),size=(min(100,B-start),len(sums)))
                boot[start:start+len(ix)] = sums[ix].sum(axis=1)
            for metric,num,den in [('cov90',0,6),('entry_cov',1,6),('cert_frac',2,3),('unsafe_commit',4,5)]:
                val = totals[num]/totals[den] if totals[den] else np.nan
                vals = np.divide(boot[:,num],boot[:,den],out=np.full(B,np.nan),where=boot[:,den]>0)
                finite = vals[np.isfinite(vals)]
                ci = np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
                result.append(dict(model=tag,route='direct_sim' if not prefix else 'compositional',split=split,parent_n=n,stratum=st,metric=metric,value=val,lo=ci[0],hi=ci[1],n_ep=len(sums),n_rows=len(d),denominator=totals[den],cal_q=q,source=source,status='exploratory'))
    return result


def main():
    runs = {}
    # Match only train / exploratory evaluation run names: never enter conf runs.
    for m in list(CACHE.glob('*/crane/cfcap-models*/manifest.json')) + list(CACHE.glob('coordinator-artifacts/cfcap-models*/artifacts/out/manifest.json')):
        if 'conf' in str(m).lower(): continue
        man = json.loads(m.read_text())
        if man.get('task') != 'train': continue
        argv=man.get('task_argv') or []
        name=argv[argv.index('--grid')+1] if '--grid' in argv else 'b1'
        try: cfg=grid(name)[int(man['shard'])]
        except (IndexError,KeyError,ValueError): continue
        tag=f"{cfg['arch']}-{cfg['div']}-{cfg['seed']}"
        if not (m.parent/'units/eval-cal.npz').exists(): continue
        key=(man.get('status')=='complete',man.get('updated',''))
        if tag not in runs or key > runs[tag][0]: runs[tag]=(key,m.parent)
    out=[]
    for tag,(_,run) in sorted(runs.items()):
        cal=read_npz(run/'units/eval-cal.npz')
        for pre in ('','c'):
            q=conformal(cal,pre)
            for split in SPLITS[:5]:
                path=run/'units'/f'eval-{split}.npz'
                if path.exists(): out += analyze(read_npz(path),pre,q,tag,split,str(path))
    # Stage-two heterogeneous units are partitioned by split and model. Exact
    # duplicate copies are suppressed by unit name and content hash.
    parts={}; seen=set()
    for m in CACHE.glob('*/crane/cfcap-models*.models-eval2-het.*/manifest.json'):
        man=json.loads(m.read_text())
        if man.get('task') != 'models_eval2': continue
        for rec in man.get('files',[]):
            unit=rec.get('unit',''); split=unit.rsplit('-',2)[0]
            if split not in SPLITS or 'conf' in unit.lower(): continue
            f=m.parent/'units'/f'{unit}.npz'
            if not f.exists(): continue
            key=(unit,rec.get('sha256'))
            if key in seen: continue
            seen.add(key)
            meta=json.loads(f.with_suffix('.json').read_text()).get('meta',{})
            tags=str(meta.get('models','')).split(',')
            df=read_npz(f)
            for mi,g in df.groupby('model'):
                if int(mi)<len(tags): parts.setdefault((tags[int(mi)],split),[]).append(g)
    for tag in sorted({k[0] for k in parts}):
        data={s:pd.concat(v,ignore_index=True).drop_duplicates(['eid','cand','leg']) for (t,s),v in parts.items() if t==tag}
        if 'het_cal' not in data: continue
        for pre in ('','c'):
            q=conformal(data['het_cal'],pre)
            for split,df in data.items(): out += analyze(df,pre,q,tag,split,'~/.cache/vast-out/*/crane/cfcap-models*.models-eval2-het.*/units/'+split+'-*.npz')
    result=pd.DataFrame(out)
    dest=ROOT/'results/codex/re_models.csv'
    result.to_csv(dest,index=False)
    print('wrote',len(result),'rows',dest)
    print('runs',runs)
    print('stage2',[(k,sum(len(x) for x in v)) for k,v in sorted(parts.items())])

if __name__=='__main__': main()
