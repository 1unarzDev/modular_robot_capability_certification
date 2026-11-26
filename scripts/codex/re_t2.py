"""Exploratory independent fixed-reference re-score; run via scripts/heavy.sh.
A5m T2 absolute values are reconstructed from saved own-reference labels using
its original eight-draw T2 reference; T1 reference is cfcap.gen.rho_ref.
A5y supplies fixed-reference labels directly. Bootstrap resamples episodes.
"""
import csv, os, pathlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('NUMBA_NUM_THREADS','4')
import numpy as np
from cfcap.t1 import params as P, capability as C
from cfcap.t2 import capability as C2, build
from cfcap.gen import rho_ref as RR

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(os.environ['CFCAP_OUT']) / 'stress'
SHAPES = ('single','pair','L3','square','2x3')
LT = np.array(['yaw' if x.startswith('yaw') else 'diag' if x.startswith('diag') else 'axis' for x in C.LEGS])
TAUS = np.array([.7,.8])
rows=[]
def add(**kw): rows.append(kw)
def ci(x, seed=73):
    rng=np.random.default_rng(seed); n=len(x)
    # x holds per-episode numerators and denominators
    idx=rng.integers(0,n,(2000,n)); a=x[idx].sum(axis=1)
    return np.quantile(a[:,0]/a[:,1],[.025,.975])
def summarize(src, cond, shape, a, b):
    dom=np.array(['dominated' if t=='axis' else 'undominated' if t=='diag' or shape not in ('single','pair') else 'other' for t in LT])
    flip=(a[:,:,None]>=TAUS)!=(b[:,:,None]>=TAUS)
    shift=np.abs(b-a)>.05
    for leg in ('axis','diag','yaw','any'):
        for cl in ('dominated','undominated','other','any'):
            mask=((LT==leg) if leg!='any' else np.ones(10,bool)) & ((dom==cl) if cl!='any' else np.ones(10,bool))
            if not mask.any(): continue
            f=flip[:,mask,:].sum((1,2)); d=shift[:,mask].sum(1)
            nf=int(mask.sum()*2); nd=int(mask.sum())
            flo,fhi=ci(np.column_stack((f,np.full(len(f),nf))))
            dlo,dhi=ci(np.column_stack((d,np.full(len(d),nd))))
            add(source=src,condition=cond,shape=shape,leg_type=leg,dominance=cl,metric='flip',episodes=len(a),numerator=int(f.sum()),denominator=len(a)*nf,estimate=f.mean()/nf,ci_low=flo,ci_high=fhi)
            add(source=src,condition=cond,shape=shape,leg_type=leg,dominance=cl,metric='shift05',episodes=len(a),numerator=int(d.sum()),denominator=len(a)*nd,estimate=d.mean()/nd,ci_low=dlo,ci_high=dhi)
def auc(x,p,n):
    a=x[p]; b=x[n]
    return (np.greater(a[:,None],b).sum()+.5*np.equal(a[:,None],b).sum())/(len(a)*len(b)) if len(a) and len(b) else np.nan
def aucs(src,cond,records, target, negative, shape='pooled'):
    p=np.array([r[target] for r in records],bool); n=~p
    if negative=='faulted': n &= np.array([r['fault']>0 for r in records])
    for probe in ('v2','clamp','yawc'):
        x=np.array([r.get(probe,np.nan) for r in records]); valid=np.isfinite(x); pp=p&valid; nn=n&valid
        if not pp.any() or not nn.any(): continue
        add(source=src,condition=cond,shape=shape,leg_type='yaw' if target.startswith('yaw') else 'any',dominance='any',metric='auc_'+probe+'_'+target+'_'+negative,episodes=int(valid.sum()),numerator=int(pp.sum()),denominator=int(nn.sum()),estimate=auc(x,pp,nn),ci_low='',ci_high='')

def main():
    par=P.base_params('plated'); m=np.load(OUT/'A5m_rows.npz'); y=np.load(OUT/'A5y.npz'); old=np.load(OUT/'A5_refs.npz')
    with open(ROOT/'results/stress/A5m_rows.csv',newline='') as fh: raw=list(csv.DictReader(fh))
    for shape in SHAPES:
        asm=P.named_assembly(shape,par)
        lat=RR.nominal_latents(asm,'plated',1)
        ref=C.capability([(asm,lat)],par,device='cpu')[0]['rho'].cpu().numpy()[0]
        RR.put('plated',RR.key(asm),ref); ref=RR.get('plated',RR.key(asm))
        if not np.allclose(ref,old[f'{shape}/T1'],rtol=1e-5,atol=1e-6): raise ValueError('T1 reference mismatch '+shape)
        a=m[f'{shape}/T1/rho_dec']
        for cond in ('motor','all+motor'):
            ref2=C2.rho_ref(asm,par,'plated',build.ABLATIONS[cond])
            b=m[f'{shape}/{cond}/rho_dec']*ref2/ref
            summarize('A5m',cond,shape,a,b)
            rec=[]
            for ep in range(len(a)):
                r=next(r for r in raw if r['shape']==shape and int(r['ep'])==ep)
                fl=np.any((a[ep,:,None]>=TAUS)!=(b[ep,:,None]>=TAUS)); shift=np.any(abs(a[ep]-b[ep])>.05)
                rec.append(dict(fault=int(r['n_fault']),any_flip=fl,any_shift=shift,v2=float(r[cond+'.D_imu_v2']),clamp=float(r[cond+'.D_imu_clamp'])))
            aucs('A5m',cond,rec,'any_flip','faulted',shape)
        if shape in ('L3','square','2x3'):
            summarize('A5y','all+motor',shape,y[f'{shape}/T1'],y[f'{shape}/T2_fixed'])
    rec=[]
    for shape in ('L3','square','2x3'):
        a=y[f'{shape}/T1']; b=y[f'{shape}/T2_fixed']; fl=np.any((a[:,:,None]>=TAUS)!=(b[:,:,None]>=TAUS),axis=2); sh=abs(a-b)>.05
        for i in range(len(a)):
            rec.append(dict(fault=int(y[f'{shape}/n_fault'][i]),yaw_flip=fl[i,LT=='yaw'].any(),yaw_target=(fl[i,LT=='yaw']|sh[i,LT=='yaw']).any(),any_flip=fl[i].any(),any_target=(fl[i]|sh[i]).any(),**{k:float(y[f'{shape}/D_imu_{k}'][i]) for k in ('v2','clamp','yawc')}))
    for target,neg in [('yaw_target','all'),('yaw_flip','faulted'),('any_target','all'),('any_flip','faulted')]: aucs('A5y','all+motor',rec,target,neg)
    path=ROOT/'results/codex/re_t2.csv'; fields=['source','condition','shape','leg_type','dominance','metric','episodes','numerator','denominator','estimate','ci_low','ci_high']
    with path.open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('wrote',path,len(rows))
if __name__=='__main__': main()
