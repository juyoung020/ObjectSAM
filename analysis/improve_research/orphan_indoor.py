"""천장(ceiling-merged)이 있는 COCO 사진(진짜 실내)만: 경계 띠를 뺀(2 px 침식) 고아 덩어리(≥ 200 px) 수와 출처.
출처: blob/frag 로 버린 선생 마스크의 나머지 / 구조물로 버린 마스크의 비구조물 부분 / 점수 0.2–0.5 선생 마스크 / 그 밖."""
import sys, json, zipfile, collections, os
import numpy as np, cv2
sys.path.insert(0, '/home/juyoung/ObjectSAM/analysis/improve_research'); sys.path.insert(0, '/home/juyoung/ObjectSAM/pytools')
from orphan_check_lib import read_plan, rle_mask
import ftl
from label_plan import WCF
EVERY = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = '/home/juyoung/ObjectSAM/analysis/improve_research/orphan_indoor'; os.makedirs(OUT, exist_ok=True)
pan = json.load(open('/home/juyoung/ObjectSAM/data/coco/panoptic_train2017.json'))
cats = {c['id']: c['name'] for c in pan['categories']}; anns = {x['image_id']: x for x in pan['annotations']}; del pan
z = zipfile.ZipFile('/home/juyoung/ObjectSAM/data/coco/panoptic_train2017.zip')
src = collections.Counter(); nimg = 0; comp_area = []; drawn = 0; under_class = collections.Counter()
k5 = np.ones((5, 5), np.uint8)
for i, (r, (pid, drop, ex, ig)) in enumerate(zip(ftl.read('/home/juyoung/ObjectSAM/data/teacher/train2017.ftl'), read_plan('/home/juyoung/ObjectSAM/data/teacher/train2017_human_os_parts.plan'))):
    if i % EVERY: continue
    x = anns[pid]; names = {s['id']: cats[s['category_id']] for s in x['segments_info']}
    if 'ceiling-merged' not in names.values(): continue
    H, W = r['H'], r['W']; n = H * W
    png = cv2.imdecode(np.frombuffer(z.read('panoptic_train2017/' + x['file_name']), np.uint8), cv2.IMREAD_COLOR)
    rgb = png[..., ::-1].astype(np.int64); ids = (rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]).reshape(-1)
    wcf = np.isin(ids, [k for k, v in names.items() if v in WCF])
    tm = ftl.idmap(r).reshape(-1); sc = r['scores']
    keep = sc >= 0.5; keep[drop] = False
    pos = keep[np.maximum(tm, 0)] & (tm >= 0)
    for e in ex: pos |= rle_mask(e, n)
    ign = rle_mask(ig, n) if ig else np.zeros(n, bool)
    orph = (~pos & ~wcf & ~ign).reshape(H, W).astype(np.uint8)
    orph = cv2.morphologyEx(orph, cv2.MORPH_OPEN, k5)   # 가는 경계 띠 제거
    nimg += 1
    dropped = np.zeros(len(sc), bool); dropped[drop] = True
    wfrac = np.bincount(tm[tm >= 0], weights=wcf[tm >= 0], minlength=len(sc)) / np.maximum(np.bincount(tm[tm >= 0], minlength=len(sc)), 1)
    nc, lab, stats, _ = cv2.connectedComponentsWithStats(orph, connectivity=8)
    for c in range(1, nc):
        if stats[c, cv2.CC_STAT_AREA] < 200: continue
        m = (lab == c).reshape(-1)
        k = np.bincount(tm[m & (tm >= 0)], minlength=len(sc)).argmax()
        if dropped[k]: s = 'dropped_struct' if wfrac[k] >= 0.5 else 'dropped_frag_blob'
        elif sc[k] >= 0.2: s = 'lowscore_0.2-0.5'
        else: s = 'lowscore_<0.2'
        src[s] += 1; comp_area.append(stats[c, cv2.CC_STAT_AREA] / n)
        u, cc = np.unique(ids[m], return_counts=True); under_class[names.get(int(u[cc.argmax()]), 'VOID')] += 1
    if drawn < 16 and nc > 2:
        img = cv2.imread(f'/home/juyoung/ObjectSAM/data/coco/train2017/{pid:012d}.jpg'); ov = img.copy()
        o2 = orph > 0; ov[o2] = (0.4 * ov[o2] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        p2 = pos.reshape(H, W); ov[p2] = (0.75 * ov[p2] + 0.25 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.imwrite(f'{OUT}/{pid:012d}.jpg', np.concatenate([img, ov], 1)); drawn += 1
ca = np.array(comp_area)
print(f'ceiling images {nimg} (every {EVERY})  orphan blobs >=200px after 2px opening: {len(ca)/nimg:.2f}/img')
print('  source:', {k: round(v / nimg, 2) for k, v in src.items()})
print('  area (% image) quartiles:', np.round(np.percentile(ca * 100, [25, 50, 75, 90]), 2) if len(ca) else None)
print('  majority panoptic class under blob (top 15):', under_class.most_common(15))
