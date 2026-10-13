import pickle, numpy as np, collections
R = pickle.load(open('label_stats.pkl','rb'))
n=len(R)
n05=np.array([r['n05'] for r in R]); nl=np.array([r['n0205'] for r in R]); nz=np.array([r['n_lt02'] for r in R]); na=np.array([r['n_all'] for r in R])
print(f'per image: all {na.mean():.1f}  >=0.5 {n05.mean():.1f}  0.2-0.5 {nl.mean():.1f}  <0.2 {nz.mean():.1f}')
L=[(r,x) for r in R for x in r['low']]
print('low masks with pixels', len(L), 'per image %.2f'%(len(L)/n))
fr=np.array([x['frac'] for _,x in L]); ar=np.array([x['area'] for _,x in L])
print('size (img frac) pct 10/50/90:', np.percentile(fr,[10,50,90]))
for lo,hi in [(0,0.001),(0.001,0.005),(0.005,0.02),(0.02,0.05),(0.05,1)]:
    s=(fr>=lo)&(fr<hi); print(f'  {lo*100:.1f}-{hi*100:.1f}%: {s.sum()/n:.2f}/img')
# where do their pixels go in v1 labels
ig=np.array([x['in_ignore'] for _,x in L]); st=np.array([x['on_struct'] for _,x in L]); ob=np.array([x['on_obj'] for _,x in L])
w=ar/ar.sum()
print('pixel share: in v1 ignore %.1f%%, on COCO structure %.1f%%, on COCO obj/stuff(non-structure) %.1f%%, void %.1f%%'%((ig*w).sum()*100,(st*w).sum()*100,(ob*w).sum()*100,((1-st-ob)*w).sum()*100))
print('-> taught as background: %.1f%% of their pixels'%((1-ig)*w).sum()*100 if False else '')
print('taught as background pixel share %.1f%%'%(((1-ig)*w).sum()*100))
# categories
def cls(x):
    if x['on_struct']>=0.5: return 'struct(COCO 구조물 위)'
    if x['nb_status']=='kept' and x['kept_ring']>=0.7 and x['nb_area']>=3*x['area']: return 'enclosed-in-kept(부품/구멍 후보)'
    if x['kept_ring']>=0.7: return 'between-kept(물체 사이 틈)'
    return 'outside(놓친 물체 후보)'
C=collections.Counter(); P=collections.Counter(); B=collections.Counter()
for r,x in L:
    c=cls(x); C[c]+=1; P[c]+=x['area']; B[c]+=x['area']*(1-x['in_ignore'])
tot=sum(P.values())
for c in C: print(f'{c:34s} {C[c]/n:.2f}/img  px {P[c]/tot*100:.1f}%  bg-taught px share {B[c]/max(P[c],1)*100:.0f}%')
# small, outside, not ignored, not struct, frac>=0.001 : missed-object candidates taught as bg
cand=[(r,x) for r,x in L if cls(x).startswith('outside') and x['frac']>=0.001]
print('outside & >=0.1%%img: %.2f/img; with in_ignore<0.5: %.2f/img'%(len(cand)/n, sum(x['in_ignore']<0.5 for _,x in cand)/n))
enc=[(r,x) for r,x in L if cls(x).startswith('enclosed') and x['frac']>=0.001]
print('enclosed & >=0.1%%img: %.2f/img; in_ignore<0.5 %.2f/img'%(len(enc)/n, sum(x['in_ignore']<0.5 for _,x in enc)/n))
sc=np.array([x['score'] for _,x in L]); print('score hist', np.histogram(sc,[0.2,0.3,0.4,0.5])[0]/n)
pickle.dump(dict(cand=[(r['id'],x) for r,x in cand], enc=[(r['id'],x) for r,x in enc], struct=[(r['id'],x) for r,x in L if cls(x).startswith('struct') and x['frac']>=0.001]), open('q2_sel.pkl','wb'))
