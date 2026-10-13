import pickle, glob, numpy as np, collections
def load(run, d):
    G=[]
    for f in sorted(glob.glob(f'{d}/{run}__*.pkl')):
        x=pickle.load(open(f,'rb'))
        for g in x['gt']: G.append(g)
    return G
runs={'FS01':('fs01','/home/juyoung/meridian_ws/analysis/fail_explore/recs'),'v1_01':('v101','recs'),'v9_01':('v901','recs'),'v5_01':('v501','/home/juyoung/meridian_ws/analysis/fail_explore/recs')}
edges=[64,96,128,192,256,384,5000]
print('input-px sqrt-area bin | ' + ' | '.join(f'{k} tp%/merged%/missed%' for k in runs))
D={k:load(*v) for k,v in runs.items()}
for i in range(len(edges)-1):
    row=[]
    for k,G in D.items():
        sz=np.array([np.sqrt(g['area']*6.25)*1.6 for g in G]); st=np.array([g['status'] for g in G])
        m=(sz>=edges[i])&(sz<edges[i+1]); c=collections.Counter(st[m]); n=m.sum()
        row.append(f"{c['tp']/n*100:4.1f}/{c['merged']/n*100:4.1f}/{c['missed']/n*100:4.1f} (n{n})")
    print(f'{edges[i]}-{edges[i+1]}: '+' | '.join(row))
for k,G in D.items():
    print(k, collections.Counter(g['status'] for g in G).most_common())
