import pickle, glob, numpy as np, collections
def load(run, d):
    P=[];G=[];nk=0
    for f in sorted(glob.glob(f'{d}/{run}__*.pkl')):
        x=pickle.load(open(f,'rb')); P+=x['pred']; G+=x['gt']; nk+=x['n_kf']
    return P,G,nk
for nm,(r,d) in {'FS01':('fs01','/home/juyoung/meridian_ws/analysis/fail_explore/recs'),'v1_01':('v101','recs'),'v9_01':('v901','recs')}.items():
    P,G,nk=load(r,d)
    c=collections.Counter(p['status'] for p in P)
    conf=np.array([p['conf'] for p in P]); st=np.array([p['status'] for p in P]); area=np.array([p['area'] for p in P]); sf=np.array([p['struct_frac'] for p in P])
    print(nm, 'kf',nk,'preds/kf %.1f'%(len(P)/nk), dict(c))
    tpc=np.array([g.get('m_conf',np.nan) for g in G if g['status']=='tp'])
    print('   TP conf median %.2f  p25 %.2f  share TP conf<0.2 %.2f'%(np.nanmedian(tpc),np.nanpercentile(tpc,25),np.nanmean(tpc<0.2)))
    for lo,hi in ((0.1,0.2),(0.2,0.4),(0.4,1.01)):
        m=(conf>=lo)&(conf<hi); cc=collections.Counter(st[m]); print(f'   conf {lo}-{hi}: n/kf {m.sum()/nk:.1f} ', dict(cc.most_common(5)))
    fp=st=='fp'
    print('   fp: struct_frac>0.5 share %.2f, median area %d'%((sf[fp]>0.5).mean(), np.median(area[fp])))
