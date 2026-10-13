"""천장(ceiling-merged)이 가장 많이 겹치는 선생 마스크를 사진마다 넓이 순으로: 1 등 = 천장 판, 나머지 = 천장에 붙은 것 후보.
버려진(drop=경우2, siglip=경우4) 2 등 이하 마스크를 세고 그림으로."""
import pickle, random, sys, collections, numpy as np
sys.path.insert(0,'.'); from grid import render
R = pickle.load(open('label_stats.pkl','rb')); imgs=[r for r in R if r['has_ceil']]
rows=[]
for r in imgs:
    C=sorted([x for x in r['removed']+r['kept_struct'] if x['kind']=='ceil'], key=lambda x:-x['area'])
    tot=sum(x['area'] for x in C)
    for i,x in enumerate(C):
        x0,y0,x1,y1=x['box']; w,h=x1-x0,y1-y0
        rows.append(dict(id=r['id'],rank=i,share=x['area']/max(tot,1),elong=max(w,h)/max(min(w,h),1),**x))
print('ceiling images',len(imgs))
c=collections.Counter((min(x['rank'],2),x['status']) for x in rows)
for rk in (0,1,2):
    print(f" rank {'1' if rk==0 else '2' if rk==1 else '3+'}: drop {c[(rk,'drop')]} siglip {c[(rk,'siglip')]} kept {c[(rk,'kept')]}")
sec=[x for x in rows if x['rank']>0 and x['status']!='kept']
print('secondary removed: %d (%.2f/ceiling-img); drop %d siglip %d; frac>=0.5%% %d; elongated(>=3) %d'%(len(sec),len(sec)/len(imgs),sum(x['status']=='drop' for x in sec),sum(x['status']=='siglip' for x in sec),sum(x['frac']>=0.005 for x in sec),sum(x['elong']>=3 for x in sec)))
big=[x for x in sec if x['status']=='drop']
print('secondary case-2 drops: cover-of-ceiling-seg median %.2f, frac median %.3f'%(np.median([x['cover'] for x in big]),np.median([x['frac'] for x in big])))
random.seed(5); random.shuffle(big)
render([(x['id'],x['k'],f"r{x['rank']+1} {x['frac']*100:.1f}% cov{x['cover']:.2f} e{x['elong']:.0f}") for x in big[:48]],'q1b_ceiling_secondary_drop.jpg')
top=[x for x in rows if x['rank']==0 and x['status']=='drop']; random.shuffle(top)
render([(x['id'],x['k'],f"r1 {x['frac']*100:.1f}% cov{x['cover']:.2f}") for x in top[:24]],'q1b_ceiling_rank1_drop.jpg')
pickle.dump(rows,open('q1b_rows.pkl','wb'))
