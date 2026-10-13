"""학습 라벨에서 '부품'이 얼마나 있고 얼마나 버려지나(COCO 표본, v2 판정 규칙 기준).

  .venv/bin/python pytools/part_analysis.py [--n 1000]
- 정답(panoptic things·문·창·계단·LVIS) 안에 50 % 이상 들어간 CropFormer 마스크(점수 ≥ 0.5) = 정답 조각 → v2 규칙은 모두 버린다.
  그 조각이 정답 넓이의 몇 % 인지로 나눈다: <5 %(부스러기) · 5-30 % · 30-70 %(부품 같은 것) · ≥70 %(정답과 거의 같음).
- 정답끼리 포함 관계(LVIS 베개가 panoptic 침대 안 등) = 이미 있는 부품 정답 수.
"""
import argparse
import collections
import json
import os
import random
import sys
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402
from label_human import DWS, LVIS, lvis_mask  # noqa: E402
from label_plan import WCF  # noqa: E402

COCO = 'data/coco'
BINS = [0, 0.05, 0.3, 0.7, 1.01]
BN = ['<5%', '5-30%', '30-70%', '>=70%']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    a = ap.parse_args()
    random.seed(1)
    pan = json.load(open(f'{COCO}/panoptic_train2017.json'))
    cats = {c['id']: c for c in pan['categories']}
    ann = {x['image_id']: x for x in pan['annotations']}
    del pan
    pick = []
    for i, r in enumerate(ftl.read('data/teacher/train2017.ftl')):   # 저수지 표본
        if len(pick) < a.n:
            pick.append(r)
        elif random.random() < a.n / (i + 1):
            pick[random.randrange(a.n)] = r
    ids = {r['id'] for r in pick}
    L = json.load(open(LVIS))
    lvc = {c['id']: c['name'] for c in L['categories']}
    lv = collections.defaultdict(list)
    for x in L['annotations']:
        if x['image_id'] in ids:
            lv[x['image_id']].append((x['segmentation'], lvc[x['category_id']]))
    del L
    z = zipfile.ZipFile(f'{COCO}/panoptic_train2017.zip')
    S = collections.Counter()
    part_cls = collections.Counter()
    nest = collections.Counter()
    for r in pick:
        x = ann[r['id']]
        png = cv2.imdecode(np.frombuffer(z.read('panoptic_train2017/' + x['file_name']), np.uint8), 1)[..., ::-1].astype(np.int64)
        pid = png[..., 0] + 256 * png[..., 1] + 65536 * png[..., 2]
        H, W = pid.shape
        gts, gn, wcf = [], [], np.zeros((H, W), bool)
        for s in x['segments_info']:
            nm = cats[s['category_id']]['name']
            m = pid == s['id']
            if nm in WCF:
                wcf |= m
            elif s.get('iscrowd'):
                continue
            elif cats[s['category_id']]['isthing'] or nm in DWS:
                gts.append(m); gn.append(nm)
        npan = len(gts)
        for seg, nm in lv[r['id']]:
            m = lvis_mask(seg, H, W)
            if m.any() and not any((m & g).sum() / max((m | g).sum(), 1) > 0.7 for g in gts[:npan]):
                gts.append(m); gn.append(nm)
        if not gts:
            S['imgs'] += 1
            continue
        G = np.stack(gts).reshape(len(gts), -1)
        ga = G.sum(1)
        # 정답끼리 포함(작은 것이 큰 것 안에 80 % 이상, 넓이 70 % 미만)
        inter_gg = G.astype(np.float32) @ G.T.astype(np.float32)
        for i in range(len(gts)):
            for j in range(len(gts)):
                if i != j and ga[i] < 0.7 * ga[j] and inter_gg[i, j] >= 0.8 * ga[i]:
                    S['nested_gt'] += 1
                    nest[f'{gn[i]} in {gn[j]}'] += 1
        tm = ftl.idmap(r).reshape(-1)
        for k in np.where(r['scores'] >= 0.5)[0]:
            mk = tm == k
            n = mk.sum()
            if n == 0 or wcf.reshape(-1)[mk].mean() >= 0.5:
                continue
            inter = G[:, mk].sum(1)
            j = int(np.argmax(inter))
            if inter[j] / n >= 0.5:
                f = inter[j] / ga[j]
                b = next(i for i in range(4) if f < BINS[i + 1])
                S[f'frag_{b}'] += 1
                if b in (1, 2):
                    part_cls[gn[j]] += 1
        S['imgs'] += 1
    n = S['imgs']
    print(f'표본 {n} 장 (v2 판정 규칙 기준)')
    print('정답 안 CropFormer 조각(모두 버려짐), 정답 넓이 대비 크기별 장당: ' + ' · '.join(f"{BN[b]} {S[f'frag_{b}'] / n:.2f}" for b in range(4)))
    print(f"  부품 같은 조각(5-70 %) 장당 {(S['frag_1'] + S['frag_2']) / n:.2f} — 어느 정답 안에 많나:", part_cls.most_common(15))
    print(f"이미 있는 부품 정답(정답 안의 정답) 장당 {S['nested_gt'] / n:.2f}:", nest.most_common(15))


if __name__ == '__main__':
    main()
