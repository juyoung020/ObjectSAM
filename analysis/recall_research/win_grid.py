"""COCO 창·문·커튼 사진에서 v1 정답(남은 선생 마스크) 분할을 색으로 — 창틀이 따로 나오나 보기."""
import json, pickle, random, sys, cv2, numpy as np
sys.path.insert(0, '/home/juyoung/ObjectSAM/pytools'); import ftl
R = {r['id']: r for r in pickle.load(open('label_stats.pkl', 'rb'))}
pan = json.load(open('/home/juyoung/ObjectSAM/data/coco/panoptic_train2017.json'))
cats = {c['id']: c['name'] for c in pan['categories']}
want = [a['image_id'] for a in pan['annotations'] if a['image_id'] in R and
        any(cats[s['category_id']] in ('window-other', 'curtain') and s['area'] > 15000 for s in a['segments_info'])
        and any(cats[s['category_id']].startswith(('wall', 'ceiling')) for s in a['segments_info'])]
random.seed(3); random.shuffle(want); want = set(want[:16])
sig = {}
sys.path.insert(0, '.'); from label_stats import plans, sig_read
S = sig_read('/home/juyoung/ObjectSAM/data/teacher/train2017.siglip')
tiles = []
pl = {p[0]: p for p in plans('/home/juyoung/ObjectSAM/data/teacher/train2017.plan') if p[0] in want}
rng = np.random.default_rng(0); pal = rng.integers(40, 255, (400, 3))
for r in ftl.read('/home/juyoung/ObjectSAM/data/teacher/train2017.ftl'):
    if r['id'] not in want: continue
    m = ftl.idmap(r); im = cv2.imread(f"/home/juyoung/ObjectSAM/data/coco/train2017/{r['id']:012d}.jpg")
    drop = set(pl[r['id']][1]); sg = S.get(r['id'], {})
    col = np.zeros_like(im)
    for k in range(len(r['scores'])):
        if r['scores'][k] < 0.5 or k in drop or sg.get(k, 0) >= 0.8: continue
        col[m == k] = pal[k % 400]
    v = cv2.addWeighted(im, 0.45, col, 0.55, 0)
    v = cv2.resize(v, (320, int(320 * im.shape[0] / im.shape[1])))
    t = np.full((260, 320, 3), 255, np.uint8); t[:min(260, v.shape[0])] = v[:260]
    tiles.append(t)
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4) if len(tiles[i:i + 4]) == 4]
cv2.imwrite('q5_window_partition.jpg', np.vstack(rows)); print(len(tiles))
