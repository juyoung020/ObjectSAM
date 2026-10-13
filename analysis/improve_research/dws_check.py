"""문·창·계단(DWS) 정답 점검(CPU): (1) panoptic 조각 하나가 떨어진 덩어리 여럿(≥200 px)인 비율 = 창 여러 개를 한 정답으로 가르침,
(2) v5 규칙(--keep-parts)에서 그 정답 안에 살아남는 선생 부품 마스크 수, (3) 정답 안 선생 마스크(≥100 px, 80 % 이상 안) 분포."""
import sys, json, zipfile, collections
import numpy as np, cv2
sys.path.insert(0, '/home/juyoung/ObjectSAM/pytools')
import ftl
EVERY = int(sys.argv[1]) if len(sys.argv) > 1 else 10
pan = json.load(open('/home/juyoung/ObjectSAM/data/coco/panoptic_train2017.json'))
cats = {c['id']: c['name'] for c in pan['categories']}
z = zipfile.ZipFile('/home/juyoung/ObjectSAM/data/coco/panoptic_train2017.zip')
want = {'window-other', 'door-stuff', 'curtain', 'window-blind', 'shelf', 'cabinet-merged', 'light'}
ann_by = {}
for x in pan['annotations']:
    names = [cats[s['category_id']] for s in x['segments_info']]
    if want & set(names): ann_by[x['image_id']] = x
del pan
st = collections.defaultdict(collections.Counter); nimg = 0
for i, r in enumerate(ftl.read('/home/juyoung/ObjectSAM/data/teacher/train2017.ftl')):
    if r['id'] not in ann_by or i % EVERY: continue
    x = ann_by[r['id']]; nimg += 1
    png = cv2.imdecode(np.frombuffer(z.read('panoptic_train2017/' + x['file_name']), np.uint8), cv2.IMREAD_COLOR)
    rgb = png[..., ::-1].astype(np.int64); ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    tm = ftl.idmap(r); sc = r['scores']
    for s in x['segments_info']:
        nm = cats[s['category_id']]
        if nm not in want: continue
        g = ids == s['id']; ga = g.sum()
        if ga < 200: continue
        n, lab, sts, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), connectivity=8)
        ncc = int((sts[1:, cv2.CC_STAT_AREA] >= 200).sum())
        c = st[nm]; c['segs'] += 1; c['multi_cc'] += ncc >= 2; c['cc_total'] += ncc
        # 정답 안 선생 마스크
        tk = tm[g]; tk = tk[tk >= 0]
        if len(tk) == 0: continue
        inside = np.bincount(tk, minlength=len(sc)); full = np.bincount(tm[tm >= 0].ravel(), minlength=len(sc))
        ks = np.where((full >= 100) & (inside >= 0.8 * full) & (sc >= 0.5))[0]
        fr = full[ks] / ga
        parts = ks[(fr >= 0.01) & (fr <= 0.7)]
        c['part_masks'] += len(parts); c['segs_with_part'] += len(parts) > 0
        c['whole_like'] += int(((fr > 0.7)).sum())
print(f'images sampled {nimg} (every {EVERY} among images containing these stuff classes)')
print('class | segs | multi-CC(≥2 pieces ≥200px) | mean pieces | teacher part masks/seg (v5 keep-parts 조건) | segs with ≥1 part')
for nm, c in sorted(st.items()):
    S = max(c['segs'], 1)
    print(f"{nm} | {c['segs']} | {100*c['multi_cc']/S:.0f}% | {c['cc_total']/S:.2f} | {c['part_masks']/S:.2f} | {100*c['segs_with_part']/S:.0f}%")
