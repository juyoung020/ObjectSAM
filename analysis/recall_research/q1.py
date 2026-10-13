import pickle, numpy as np, collections
R = pickle.load(open('label_stats.pkl','rb'))
n = len(R)
print('images', n)
def bins(fr):
    return '<0.1%' if fr<0.001 else '0.1-0.5%' if fr<0.005 else '0.5-2%' if fr<0.02 else '2-5%' if fr<0.05 else '>=5%'
for kind, flag in (('ceil','has_ceil'),('wall','has_wall')):
    imgs = [r for r in R if r[flag]]
    print(f'\n== {kind}: images with that COCO structure: {len(imgs)} ({len(imgs)/n*100:.1f}%)')
    rem = [x for r in imgs for x in r['removed'] if x['kind']==kind]
    kept = [x for r in imgs for x in r['kept_struct'] if x['kind']==kind]
    c = collections.Counter((bins(x['frac']), x['status']) for x in rem)
    ck = collections.Counter(bins(x['frac']) for x in kept)
    print('size | drop(case2/3) | siglip>=0.8 | kept(case4, siglip<0.8)   [per image with structure]')
    for b in ['<0.1%','0.1-0.5%','0.5-2%','2-5%','>=5%']:
        print(f"{b:9s} | {c[(b,'drop')]:6d} ({c[(b,'drop')]/len(imgs):.2f}) | {c[(b,'siglip')]:6d} ({c[(b,'siglip')]/len(imgs):.2f}) | {ck[b]:6d} ({ck[b]/len(imgs):.2f})")
    small = [x for x in rem if x['frac']<0.02]
    print('removed small(<2% img) per image', len(small)/len(imgs), ' of which drop', sum(x['status']=='drop' for x in small), 'siglip', sum(x['status']=='siglip' for x in small))
    per = np.array([sum(1 for x in r['removed'] if x['kind']==kind and x['frac']<0.02) for r in imgs])
    print('per-image dist small removed: mean %.2f median %d p90 %d max %d, frac images >=1: %.2f' % (per.mean(), np.median(per), np.percentile(per,90), per.max(), (per>0).mean()))
    # siglip prob histogram for small kept/removed cand
    sp = np.array([x['sig'] for x in small if x['status']=='siglip'])
    print('drop small: cover stats', np.percentile([x['cover'] for x in small if x['status']=='drop'] or [0],[10,50,90]))
    # removed small pixel share of the image
    pix = sum(x['area'] for x in small)/sum(r['H']*r['W'] for r in imgs)
    print(f'removed small pixels = {pix*100:.2f}% of pixels in these images')
