import pickle, glob, numpy as np, collections
def load(run, d):
    G=[]
    for f in sorted(glob.glob(f'{d}/{run}__*.pkl')): G+=pickle.load(open(f,'rb'))['gt']
    return G
FSD='/home/juyoung/meridian_ws/analysis/fail_explore/recs'
D={'FS':load('fs01',FSD),'v1(100ep)':load('v101','recs'),'v9(5ep)':load('v901','recs'),'v5(10ep,human)':load('v501',FSD)}
keys=collections.Counter((g['seq'][:3],g['cls']) for g in D['FS'])
print('seq,cls | n(FS) | ' + ' | '.join(D))
rows=[]
for k,n in keys.most_common(22):
    r=[]
    for nm,G in D.items():
        s=[g for g in G if (g['seq'][:3],g['cls'])==k]; st=collections.Counter(g['status'] for g in s)
        r.append(f"{st['tp']/len(s)*100:4.1f} (m{st['merged']/len(s)*100:3.0f} x{st['missed']/len(s)*100:3.0f} l{st['low_iou']/len(s)*100:3.0f})")
    # contribution to v9 vs FS R gap (pts of total)
    print(f'{k} | {n} | '+' | '.join(r))
