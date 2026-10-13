import pickle, glob, numpy as np, collections
def load(run, d):
    G=[]
    for f in sorted(glob.glob(f'{d}/{run}__*.pkl')): G+=pickle.load(open(f,'rb'))['gt']
    return G
FSD='/home/juyoung/meridian_ws/analysis/fail_explore/recs'
D={'FS':load('fs01',FSD),'v1':load('v101','recs'),'v9':load('v901','recs')}
for k,G in D.items():
    L=[g for g in G if g['status']=='low_iou']
    biou=np.array([g['best_iou'] for g in L]); cov=np.array([g.get('c_cov',np.nan) for g in L]); ins=np.array([g.get('c_inside',np.nan) for g in L])
    print(f"{k}: low_iou {len(L)} ({len(L)/len(G)*100:.1f}%) best_iou med {np.median(biou):.2f}; best-pred covers GT med {np.nanmedian(cov):.2f}; pred inside GT med {np.nanmedian(ins):.2f}")
    # type: pred too big (inside<0.5, cov>0.5) / too small (cov<0.5, inside>0.5) / offset
    tb=np.mean((cov>=0.5)&(ins<0.5)); ts=np.mean((cov<0.5)&(ins>=0.5)); print(f'    too-big(spills) {tb:.2f}  too-small(partial) {ts:.2f}  other {1-tb-ts:.2f}')
    c=collections.Counter((g['seq'][:4],g['cls']) for g in L); print('   ', c.most_common(8))
    # tp IoU quality
    T=[g['best_iou'] for g in G if g['status']=='tp']; print('    TP IoU median %.3f mean %.3f'%(np.median(T),np.mean(T)))
