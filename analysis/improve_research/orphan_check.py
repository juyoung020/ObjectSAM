"""학습 라벨(v5 판정 파일)에서 '배경으로 학습되는 물체 같은 화소' 세기(CPU 만).
화소 나누기: 양성(남은 선생 마스크 ∪ 사람 정답·부품) / 구조물(WCF) / 무시(crowd) / 나머지 = 고아(음성으로 학습됨).
고아 화소가 어느 선생 마스크에 속하는지(점수 구간, 판정에서 버려졌는지)로 나누고,
'고아 비율 ≥ 70 %, ≥ 100 px 인 선생 마스크' = 배경으로 학습되는 물체 후보를 센다.
  python -I orphan_check.py --plan data/teacher/train2017_human_os_parts.plan --every 200 --out DIR
"""
import argparse, io, json, os, struct, sys, zipfile, collections
import numpy as np, cv2
sys.path.insert(0, '/home/juyoung/ObjectSAM/pytools')
import ftl
from label_plan import WCF

def leb_runs(b):
    vals, p, v, s = [], 0, 0, 0
    out = []
    while p < len(b):
        c = b[p]; p += 1
        v |= (c & 0x7F) << s; s += 7
        if not c & 0x80:
            out.append(v); v = s = 0
    return out

def rle_mask(b, n):
    m = np.zeros(n, bool); pos = 0; one = False
    for r in leb_runs(b):
        if one and r: m[pos:pos + r] = True
        pos += r; one = not one
    return m

def read_plan(path):
    data = open(path, 'rb').read(); p = 0
    while p < len(data):
        assert data[p:p+4] == b'FTP1'
        iid, nd, nc, ne, _ = struct.unpack_from('<IHHHH', data, p + 4); q = p + 16
        drop = np.frombuffer(data, '<u2', nd, q); q += 2 * nd + 2 * nc
        ex = []
        for e in range(ne + 1):
            (ln,) = struct.unpack_from('<I', data, q); q += 4
            ex.append(data[q:q+ln]); q += ln
        yield iid, drop.copy(), ex[:-1], ex[-1]
        p = q

ap = argparse.ArgumentParser()
ap.add_argument('--plan', default='/home/juyoung/ObjectSAM/data/teacher/train2017_human_os_parts.plan')
ap.add_argument('--every', type=int, default=200)
ap.add_argument('--out', default='/home/juyoung/ObjectSAM/analysis/improve_research/orphan')
ap.add_argument('--draw', type=int, default=12)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
pan = json.load(open('/home/juyoung/ObjectSAM/data/coco/panoptic_train2017.json'))
cats = {c['id']: c['name'] for c in pan['categories']}
anns = {x['image_id']: x for x in pan['annotations']}
del pan
z = zipfile.ZipFile('/home/juyoung/ObjectSAM/data/coco/panoptic_train2017.zip')
B = [0, .1, .2, .3, .4, .5, 1.01]
tot = collections.Counter(); objs = collections.Counter(); objs_in = collections.Counter(); nimg = nin = 0
area_hist = collections.Counter(); tot2 = collections.Counter(); reason = collections.Counter(); drawn = 0; examples = []
for i, (r, (pid, drop, ex, ig)) in enumerate(zip(ftl.read('/home/juyoung/ObjectSAM/data/teacher/train2017.ftl'), read_plan(a.plan))):
    if i % a.every: continue
    assert r['id'] == pid
    H, W = r['H'], r['W']; n = H * W
    x = anns[pid]
    png = cv2.imdecode(np.frombuffer(z.read('panoptic_train2017/' + x['file_name']), np.uint8), cv2.IMREAD_COLOR)
    rgb = png[..., ::-1].astype(np.int64); ids = (rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]).reshape(-1)
    names = {s['id']: cats[s['category_id']] for s in x['segments_info']}
    wcf = np.isin(ids, [k for k, v in names.items() if v in WCF])
    indoor = any(v in ('ceiling-merged', 'wall-other-merged', 'wall-wood', 'wall-brick', 'wall-stone', 'wall-tile') for v in names.values())
    tm = ftl.idmap(r).reshape(-1); sc = r['scores']
    keep = sc >= 0.5; keep[drop] = False
    pos = np.zeros(n, bool)
    if keep.any(): pos |= keep[np.maximum(tm, 0)] & (tm >= 0)
    for e in ex: pos |= rle_mask(e, n)
    ign = rle_mask(ig, n) if ig else np.zeros(n, bool)
    orph = ~pos & ~wcf & ~ign
    nimg += 1; nin += indoor
    tot['px'] += n; tot['pos'] += pos.sum(); tot['wcf'] += (wcf & ~pos).sum(); tot['ign'] += (ign & ~pos).sum(); tot['orph'] += orph.sum()
    s_px = np.where(tm >= 0, sc[np.maximum(tm, 0)], -1)
    dropped = np.zeros(len(sc), bool); dropped[drop] = True
    d_px = np.where(tm >= 0, dropped[np.maximum(tm, 0)], False)
    tot['orph_dropped_hi'] += (orph & d_px).sum()
    for lo, hi in zip(B[:-1], B[1:]):
        tot[f'orph_s{lo:.1f}'] += (orph & ~d_px & (s_px >= lo) & (s_px < hi)).sum()
    tot['orph_nomask'] += (orph & (tm < 0)).sum()
    # 고아 화소의 panoptic 범주·버린 까닭(실내만)
    if indoor:
        cid = {sid: names[sid] for sid in names}
        u, c = np.unique(ids[orph], return_counts=True)
        for uu, cc in zip(u, c): tot2[cid.get(int(uu), 'VOID')] += cc
        wfrac = np.bincount(tm[tm >= 0], weights=wcf[tm >= 0], minlength=len(sc)) / np.maximum(np.bincount(tm[tm >= 0], minlength=len(sc)), 1)
        for k in drop:
            mk_or = (orph & (tm == k)).sum()
            if mk_or: reason[('struct' if wfrac[k] >= 0.5 else 'frag/blob')] += mk_or
        tot2['_orph_in'] += orph.sum()
    # 마스크 단위
    cnt_all = np.bincount(tm[tm >= 0], minlength=len(sc)); cnt_or = np.bincount(tm[orph & (tm >= 0)], minlength=len(sc))
    for k in np.where((cnt_all >= 100) & (cnt_or >= 0.7 * cnt_all))[0]:
        key = 'dropped>=.5' if dropped[k] else f's{max(b for b in B if b <= sc[k]):.1f}'
        objs[key] += 1
        if indoor: objs_in[key] += 1
        ar = cnt_all[k] / n
        area_hist[(key, 'small<1%' if ar < .01 else ('mid<5%' if ar < .05 else 'big'))] += 1
        if 0.2 <= sc[k] < 0.5 and indoor: examples.append((pid, int(k), float(sc[k]), int(cnt_all[k])))
    if drawn < a.draw and indoor and orph.mean() > 0.05:
        img = cv2.imread(f'/home/juyoung/ObjectSAM/data/coco/train2017/{pid:012d}.jpg')
        ov = img.copy(); o2 = orph.reshape(H, W)
        ov[o2] = (0.4 * ov[o2] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        p2 = pos.reshape(H, W); ov[p2] = (0.7 * ov[p2] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
        # 저점수 선생 마스크 테두리(노랑)
        for k in np.where((sc >= 0.2) & (sc < 0.5))[0]:
            mk = (tm == k).reshape(H, W).astype(np.uint8)
            if mk.sum() >= 100 and (orph.reshape(H, W) & (mk > 0)).sum() >= 0.7 * mk.sum():
                cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ov, cs, -1, (0, 255, 255), 2)
        cv2.imwrite(f'{a.out}/{pid:012d}.jpg', np.concatenate([img, ov], 1))
        drawn += 1
P = tot['px']
print(f'images {nimg} (indoor {nin}), every {a.every}')
print('pixel split %: pos {:.1f} struct {:.1f} ignore {:.2f} orphan(negative, non-struct) {:.1f}'.format(*(100 * tot[k] / P for k in ('pos', 'wcf', 'ign', 'orph'))))
O = tot['orph']
print('orphan px by teacher mask: dropped(score>=.5) {:.1f}% | '.format(100 * tot['orph_dropped_hi'] / O) +
      ' | '.join(f'{k[6:]} {100 * tot[k] / O:.1f}%' for k in tot if k.startswith('orph_s')) + f' | no mask {100*tot["orph_nomask"]/O:.1f}%')
print('teacher masks >=100px that are >=70% orphan (taught as background), per image:')
for k in sorted(objs): print(f'  {k}: all {objs[k]/nimg:.2f}/img, indoor {objs_in[k]/max(nin,1):.2f}/img')
print('  by area:', {f'{k[0]}|{k[1]}': round(v / nimg, 3) for k, v in sorted(area_hist.items())})
json.dump(examples[:400], open(f'{a.out}/lowscore_examples.json', 'w'))

O2 = tot2.pop('_orph_in')
print('INDOOR orphan px by panoptic class (top 25, % of indoor orphan px):')
print('  ' + ', '.join(f'{k} {100*v/O2:.1f}' for k, v in tot2.most_common(25)))
print('INDOOR orphan px from dropped masks by reason:', {k: f'{100*v/O2:.1f}%' for k, v in reason.items()})
