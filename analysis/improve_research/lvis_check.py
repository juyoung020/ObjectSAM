"""LVIS 쪽 점검(CPU): (1) lvis_v1_val 에 든 COCO train2017 사진(우리 파이프라인이 안 읽음) 수,
(2) 실패 유형 관련 LVIS 범주의 인스턴스 수, (3) not_exhaustive 비율(빠진 라벨 → 배경으로 학습될 위험)."""
import json, os, re, collections
D = os.path.expanduser('~/robot-agent/training/data/embed/data')
tr = json.load(open(f'{D}/lvis_v1_train.json'))
va = json.load(open(f'{D}/lvis_v1_val.json'))
coco_tr = {int(f[:-4]) for f in os.listdir('/home/juyoung/ObjectSAM/data/coco/train2017') if f.endswith('.jpg')}
coco_va = {int(f[:-4]) for f in os.listdir('/home/juyoung/ObjectSAM/data/coco/val2017') if f.endswith('.jpg')}
tr_ids = {im['id'] for im in tr['images']}; va_ids = {im['id'] for im in va['images']}
print('LVIS train imgs', len(tr_ids), 'in coco train2017', len(tr_ids & coco_tr), 'in val2017', len(tr_ids & coco_va))
print('LVIS val imgs', len(va_ids), 'in coco train2017', len(va_ids & coco_tr), 'in val2017', len(va_ids & coco_va))
va_tr = va_ids & coco_tr
nann = sum(1 for a in va['annotations'] if a['image_id'] in va_tr)
print('LVIS val annotations on COCO train2017 images (not used now):', nann)
print('COCO train2017 images with no LVIS at all:', len(coco_tr - tr_ids - va_ids))
cats = {c['id']: c for c in tr['categories']}
pat = re.compile(r'(lamp|light|bulb|chandelier|fan|vent|pipe|duct|window|curtain|blind|shade|door|knob|handle|drawer|book|shelf|frame|cushion|pillow|blanket|cabinet|chair|sofa|bed|towel|switch|socket|outlet|mirror|picture|poster|radiator|sprinkler|smoke|fire_alarm|ceiling)')
cnt = collections.Counter(a['category_id'] for a in tr['annotations'])
imgc = collections.defaultdict(set)
for a in tr['annotations']: imgc[a['category_id']].add(a['image_id'])
ne = collections.Counter(); 
for im in tr['images']:
    for c in im.get('not_exhaustive_category_ids', []): ne[c] += 1
rows = []
for cid, c in cats.items():
    if pat.search(c['name']):
        rows.append((cnt[cid], c['name'], len(imgc[cid]), ne[cid], c.get('frequency')))
rows.sort(reverse=True)
print('\nname | inst | imgs | not_exhaustive imgs | freq')
for n, name, ni, nn, f in rows:
    if n >= 50: print(f'{name} | {n} | {ni} | {nn} ({nn/max(ni,1)*100:.0f}%) | {f}')
tot_ne = sum(1 for im in tr['images'] if im.get('not_exhaustive_category_ids'))
print('\nimages with any not_exhaustive cat:', tot_ne, '/', len(tr['images']))
