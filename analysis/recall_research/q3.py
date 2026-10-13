import pickle, numpy as np
R = pickle.load(open('label_stats.pkl','rb')); rng=np.random.default_rng(0)
t1=np.array([len(r['targets']) for r in R]); t10=np.array([len(r['targets'])+len(r['targets10']) for r in R])
print('targets/img v1: mean %.1f median %d p90 %d p99 %d max %d'%(t1.mean(),np.median(t1),np.percentile(t1,90),np.percentile(t1,99),t1.max()))
print('targets/img v10: mean %.1f median %d p90 %d p99 %d max %d'%(t10.mean(),np.median(t10),np.percentile(t10,90),np.percentile(t10,99),t10.max()))
print('images over max_inst_per_src 256: v1 %d v10 %d'%((t1>256).sum(),(t10>256).sum()))
print('n_kept %.1f n_ex %.2f n_ex10-added %.2f  drop %.2f siglip %.2f'%(np.mean([r['n_kept'] for r in R]),np.mean([r['n_ex'] for r in R]),np.mean([r['n_ex10']-r['n_ex'] for r in R]),np.mean([r['n_drop'] for r in R]),np.mean([r['n_sig'] for r in R])))
# Mosaic simulation: 4 tiles, crop & scale; estimate fraction of targets lost by (a) w2/h2<=2px (b) visible-area ratio <=0.1 (c) G=256 cap
S=640
def sim(key, trials=20000):
    lost_small=lost_vis=lost_cap=tot=0; ngt=[]
    for _ in range(trials):
        idx=rng.integers(0,len(R),4); xc,yc=rng.uniform(S/2,1.5*S,2); s=rng.uniform(0.5,1.5)
        tx=rng.uniform(0.4,0.6)*S-s*S; ty=rng.uniform(0.4,0.6)*S-s*S
        kept=0
        for t,i in enumerate(idx):
            r=R[i]; W,H=r['W'],r['H']
            T=r['targets'] if key=='v1' else np.vstack([r['targets'],r['targets10']])
            if len(T)==0: continue
            # tile placement (same as dataset.cpp)
            if t==0: x1a,y1a,x2a,y2a=max(xc-W,0),max(yc-H,0),xc,yc; x1b,y1b=W-(x2a-x1a),H-(y2a-y1a)
            elif t==1: x1a,y1a,x2a,y2a=xc,max(yc-H,0),min(xc+W,2*S),yc; x1b,y1b=0,H-(y2a-y1a)
            elif t==2: x1a,y1a,x2a,y2a=max(xc-W,0),yc,xc,min(2*S,yc+H); x1b,y1b=W-(x2a-x1a),0
            else: x1a,y1a,x2a,y2a=xc,yc,min(xc+W,2*S),min(2*S,yc+H); x1b,y1b=0,0
            # random box positions: we only stored w,h -> place uniformly in image
            w=T[:,1].astype(float); h=T[:,2].astype(float)
            bx=rng.uniform(0,np.maximum(W-w,1)); by=rng.uniform(0,np.maximum(H-h,1))
            dx,dy=x1a-x1b,y1a-y1b
            X1=np.maximum(bx+dx,x1a); Y1=np.maximum(by+dy,y1a); X2=np.minimum(bx+w+dx,x2a); Y2=np.minimum(by+h+dy,y2a)
            ok=(X2>X1)&(Y2>Y1)
            w1=(X2-X1)*s; h1=(Y2-Y1)*s
            O1=np.clip(s*X1+tx,0,S); O2=np.clip(s*X2+tx,0,S); P1=np.clip(s*Y1+ty,0,S); P2=np.clip(s*Y2+ty,0,S)
            w2=O2-O1; h2=P2-P1
            vis=ok&(w2>0)&(h2>0)
            small=vis&((w2<=2)|(h2<=2))
            lowvis=vis&~small&(w2*h2/(w1*h1+1e-16)<=0.1)
            good=vis&~small&~lowvis
            # count only targets whose box reaches the output (vis); 'small' are those visible but <=2px
            tot+=vis.sum(); lost_small+=small.sum(); lost_vis+=lowvis.sum(); kept+=good.sum()
        ngt.append(kept); lost_cap+=max(0,kept-256)
    ngt=np.array(ngt)
    print(f'{key}: per mosaic sample GT kept mean {ngt.mean():.1f} p99 {np.percentile(ngt,99):.0f} max {ngt.max()}; of visible targets: <=2px dropped {lost_small/tot*100:.1f}%, crop<10% dropped {lost_vis/tot*100:.1f}%, G=256 cap {lost_cap/tot*100:.2f}%')
sim('v1'); sim('v10')
# tiny target share
T=np.vstack([r['targets'] for r in R]); mn=np.minimum(T[:,1],T[:,2])
print('v1 targets with min side <4px(native) %.1f%%, <8px %.1f%%, area<64px %.1f%%, area<1600px(=uH2 scoring min at native) %.1f%%'%((mn<4).mean()*100,(mn<8).mean()*100,(T[:,0]<64).mean()*100,(T[:,0]<1600).mean()*100))
