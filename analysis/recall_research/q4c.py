import pickle, glob, numpy as np, collections
def load(run, d):
    G=[]
    for f in sorted(glob.glob(f'{d}/{run}__*.pkl')):
        G+=pickle.load(open(f,'rb'))['gt']
    return G
FS=load('fs01','/home/juyoung/meridian_ws/analysis/fail_explore/recs'); V=load('v901','recs')
for nm,G in (('FS',FS),('v9',V)):
    sel=[g for g in G if np.sqrt(g['area']*6.25)*1.6>=192]
    bad=[g for g in sel if g['status'] in('missed','low_iou','merged')]
    print(nm,'n>=192px',len(sel),'not tp',len(bad))
    c=collections.Counter((g['seq'][:6], g['cls'], g['status']) for g in bad)
    for k,v in c.most_common(14): print('   ',k,v)
    print('   trunc share among not-tp %.2f vs all %.2f'%(np.mean([g['trunc'] for g in bad]), np.mean([g['trunc'] for g in sel])))
    print('   ring_ceil>0.3 share among not-tp %.2f'%np.mean([g['ring_ceil']>0.3 for g in bad]))
# per-class tp by size for v9 vs FS in 64-128 vs 192+
def rate(G, f):
    s=[g for g in G if f(g)]; return (np.mean([g['status']=='tp' for g in s])*100 if s else float('nan'), len(s))
for cls in ['appliance','unknown','wall','lamp','bed','screens','furniture','books','chair','plant','table','door','window']:
    out=[]
    for lo,hi in ((64,128),(128,192),(192,5000)):
        f=lambda g: g['cls']==cls and lo<=np.sqrt(g['area']*6.25)*1.6<hi
        a=rate(FS,f); b=rate(V,f); out.append(f'{lo}-{hi}: FS {a[0]:.0f}({a[1]}) v9 {b[0]:.0f}({b[1]})')
    print(cls, ' | '.join(out))
