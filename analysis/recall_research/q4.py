import pickle, glob, numpy as np
R = pickle.load(open('label_stats.pkl','rb')); rng=np.random.default_rng(0)
T=np.vstack([r['targets'] for r in R]).astype(float)
# training: input px = native * s, s~U(.5,1.5) (mosaic+RandomPerspective). COCO W~640 so native==640-scale.
W=np.array([r['W'] for r in R]); print('COCO width median', np.median(W))
s=rng.uniform(0.5,1.5,len(T)); tr=np.sqrt(T[:,0])*s
# uH2 obj GT (v9 run keyframes), area in cells (6.25 native px) -> input px at 1024: *6.25*1.6^2
G=[]; Gs=[]
for f in glob.glob('recs/v901__*.pkl'):
    d=pickle.load(open(f,'rb'))
    for g in d['gt']:
        G.append(np.sqrt(g['area']*6.25)*1.6); Gs.append(g['status'])
G=np.array(G); Gs=np.array(Gs)
qs=[5,10,25,50,75,90,95]
print('sqrt-area in MODEL INPUT px (percentiles %s)'%qs)
print(' train targets v1 (x s):', np.percentile(tr,qs).round(0))
print(' train targets >= 64px only:', np.percentile(tr[tr>=64],qs).round(0), ' share of targets >=64px: %.1f%%'%((tr>=64).mean()*100))
print(' uH2 obj GT @1024:', np.percentile(G,qs).round(0))
edges=[0,32,64,96,128,192,256,384,2000]
h1=np.histogram(tr,edges)[0]/len(tr); h2=np.histogram(G,edges)[0]/len(G)
for i in range(len(edges)-1):
    m=(G>=edges[i])&(G<edges[i+1])
    print(f'  {edges[i]:4d}-{edges[i+1]:4d}px: train {h1[i]*100:5.1f}%  uH2 {h2[i]*100:5.1f}%  v9 tp-rate {np.mean(Gs[m]=="tp")*100 if m.any() else 0:5.1f}  n={m.sum()}')
# what if trained at 1024 (imgsz 1024 => native*1.6*s) or scale range changes
for name,ss in [('imgsz640 s U(.5,1.5)',rng.uniform(.5,1.5,len(T))),('imgsz640 s U(.8,2.0)',rng.uniform(.8,2.0,len(T))),('imgsz1024 (=x1.6) s U(.5,1.5)',1.6*rng.uniform(.5,1.5,len(T)))]:
    x=np.sqrt(T[:,0])*ss; x=x[x>=64]
    print(f' {name}: targets>=64px per img-equivalent {len(x)/len(R):.1f}, median {np.median(x):.0f}, p90 {np.percentile(x,90):.0f}')
