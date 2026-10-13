"""모자이크+RandomPerspective 를 dataset.cpp 규칙대로 흉내 내 '평가에 쓰이는 크기 띠(64–384 px)' 정답 수와 가장자리 잘림 비율을 잰다."""
import pickle, numpy as np
R = pickle.load(open('label_stats.pkl','rb')); rng=np.random.default_rng(0); S=640
def sim(lo, hi, mosaic=1.0, trials=20000):
    band=0; trunc=0; big=0; small=0
    for _ in range(trials):
        nt = 4 if rng.random() < mosaic else 1
        idx=rng.integers(0,len(R),nt); xc,yc=rng.uniform(S/2,1.5*S,2); s=rng.uniform(lo,hi)
        tx=rng.uniform(0.4,0.6)*S-s*S; ty=rng.uniform(0.4,0.6)*S-s*S
        for t,i in enumerate(idx):
            r=R[i]; W,H=r['W'],r['H']; T=r['targets']
            if len(T)==0: continue
            if nt==1: x1a,y1a=(2*S-W)//2,(2*S-H)//2; x2a,y2a=x1a+W,y1a+H; x1b=y1b=0
            elif t==0: x1a,y1a,x2a,y2a=max(xc-W,0),max(yc-H,0),xc,yc; x1b,y1b=W-(x2a-x1a),H-(y2a-y1a)
            elif t==1: x1a,y1a,x2a,y2a=xc,max(yc-H,0),min(xc+W,2*S),yc; x1b,y1b=0,H-(y2a-y1a)
            elif t==2: x1a,y1a,x2a,y2a=max(xc-W,0),yc,xc,min(2*S,yc+H); x1b,y1b=W-(x2a-x1a),0
            else: x1a,y1a,x2a,y2a=xc,yc,min(xc+W,2*S),min(2*S,yc+H); x1b,y1b=0,0
            w=T[:,1].astype(float); h=T[:,2].astype(float); fill=T[:,0]/(w*h)
            bx=rng.uniform(0,np.maximum(W-w,1)); by=rng.uniform(0,np.maximum(H-h,1)); dx,dy=x1a-x1b,y1a-y1b
            X1=np.maximum(bx+dx,x1a); Y1=np.maximum(by+dy,y1a); X2=np.minimum(bx+w+dx,x2a); Y2=np.minimum(by+h+dy,y2a)
            O1=np.clip(s*X1+tx,0,S); O2=np.clip(s*X2+tx,0,S); P1=np.clip(s*Y1+ty,0,S); P2=np.clip(s*Y2+ty,0,S)
            w2=np.maximum(O2-O1,0); h2=np.maximum(P2-P1,0); w1=np.maximum(X2-X1,0)*s; h1=np.maximum(Y2-Y1,0)*s
            ok=(w2>2)&(h2>2)&(w2*h2/(w1*h1+1e-16)>0.1)
            vis=np.sqrt(w2*h2*fill)*1.0  # 보이는 넓이(상자 채움비 가정)
            # 평가 엔진은 1024 입력: 학습 640 의 화소 x1.6 이 아니라 '학습 입력 화소' 그대로 비교(모델은 같은 격자 stride 로 본다)
            m=ok&(vis>=64)&(vis<384)
            band+=m.sum(); small+=(ok&(vis<64)).sum(); big+=(ok&(vis>=384)).sum()
            full_w=w*s; full_h=h*s
            trunc+=(m&((w2<full_w*0.98)|(h2<full_h*0.98))).sum()
    return band/trials, trunc/max(band,1), small/trials, big/trials
for lo,hi,mo in [(0.5,1.5,1),(0.8,2.0,1),(1.0,2.2,1),(0.5,1.5,0),(1.0,2.2,0)]:
    b,t,sm,bg=sim(lo,hi,mo)
    print(f'scale U({lo},{hi}) mosaic={mo}: per 640 sample: targets in 64-384px band {b:.1f} (truncated {t*100:.0f}%), <64px {sm:.1f}, >=384px {bg:.2f}')
