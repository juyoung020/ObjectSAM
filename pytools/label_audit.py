"""선생 라벨 전체 점검: 점수 ≥ score 마스크를 COCO panoptic 사람 라벨과 맞춰 문제 유형별로 센다(SPEC 4 절).

  .venv/bin/python pytools/label_audit.py [--workers 24]   → 표 출력 + data/teacher/audit.json (유형별 사례)

유형(마스크마다): A1 큰 구조물, A2 구조물+물체 합쳐짐, A3 작은 구조물 후보(SigLIP2 판정 대상),
B1 물체 마스크가 구조물로 샘(배경 20~50 %), B2 COCO 물체 둘 이상을 합침, B3 COCO 물체 하나가 여러 선생 마스크로 갈림,
C 물체 화소 빈틈.
"""
import argparse
import collections
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402
from coco_build import BG  # noqa: E402

COCO = 'data/coco'
_G = {}


def _init(a):
    cv2.setNumThreads(1)
    pan = json.load(open(f'{COCO}/panoptic_train2017.json'))
    _G['cats'] = {c['id']: c for c in pan['categories']}
    _G['anns'] = {x['image_id']: x for x in pan['annotations']}
    _G['zip'] = zipfile.ZipFile(f'{COCO}/panoptic_train2017.zip')
    _G['a'] = a


def one(r):
    a = _G['a']
    x = _G['anns'][r['id']]
    png = cv2.imdecode(np.frombuffer(_G['zip'].read('panoptic_train2017/' + x['file_name']), np.uint8), cv2.IMREAD_COLOR)
    rgb = png[..., ::-1].astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    seg = {}  # id → (종류 0 배경 1 thing 2 stuff 물체, 넓이)
    for s in x['segments_info']:
        c = _G['cats'][s['category_id']]
        k = 0 if c['name'] in BG else (1 if c['isthing'] and not s['iscrowd'] else 2)
        seg[s['id']] = (k, s['area'], c['name'])
    kind = np.full(ids.shape, 3, np.int8)
    for sid, (k, _, _) in seg.items():
        kind[ids == sid] = k
    m = ftl.idmap(r)
    keep = np.where(r['scores'] >= a.score)[0]
    cnt = collections.Counter()
    ex = []  # (유형, 크기 지표, image_id, 마스크)
    best = collections.defaultdict(list)  # thing id → [선생 마스크가 덮는 비율]
    covered = np.zeros(m.shape, bool)
    for k in keep:
        mk = m == k
        area = int(mk.sum())
        if area == 0:
            continue
        covered |= mk
        kk = kind[mk]
        bgf = float((kk == 0).mean())
        objf = float(((kk == 1) | (kk == 2)).mean())
        u, c = np.unique(ids[mk], return_counts=True)
        if bgf >= 0.5:
            bsel = [(int(cc), int(uu)) for uu, cc in zip(u, c) if int(uu) in seg and seg[int(uu)][0] == 0]
            cmax, sid = max(bsel)
            if cmax / seg[sid][1] >= a.cover:
                t = 'A2' if objf >= 0.2 else 'A1'
            else:
                t = 'A3'
            cnt[t] += 1
            if t == 'A2':
                ex.append((t, objf, r['id'], int(k)))
        else:
            cnt['obj'] += 1
            if bgf >= 0.2:
                cnt['B1'] += 1
                ex.append(('B1', bgf, r['id'], int(k)))
            things = [(int(uu), int(cc)) for uu, cc in zip(u, c) if int(uu) in seg and seg[int(uu)][0] == 1]
            for uu, cc in things:
                best[uu].append(cc / seg[uu][1])
            big = [uu for uu, cc in things if cc / seg[uu][1] >= 0.5 and cc / area >= 0.15]
            if len(big) >= 2:
                cnt['B2'] += 1
                ex.append(('B2', len(big), r['id'], int(k)))
    for uu, fr in best.items():
        if max(fr) < 0.5 and sum(f >= 0.2 for f in fr) >= 2 and seg[uu][1] >= 1024:
            cnt['B3'] += 1
            ex.append(('B3', -max(fr), r['id'], uu))
    objpx = (kind == 1) | (kind == 2)
    hole = float((objpx & ~covered).sum() / max(1, objpx.sum()))
    cnt['things'] = sum(1 for v in seg.values() if v[0] == 1)
    return r['id'], cnt, ex, hole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score', type=float, default=0.5)
    ap.add_argument('--cover', type=float, default=0.3)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    tot = collections.Counter()
    imgs = collections.Counter()
    ex = collections.defaultdict(list)
    holes = []
    n = 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a,)) as pool:
        for iid, cnt, e, hole in pool.map(one, ftl.read('data/teacher/train2017.ftl'), chunksize=64):
            n += 1
            tot.update(cnt)
            for t in cnt:
                if cnt[t]:
                    imgs[t] += 1
            for t, v, i, k in e:
                ex[t].append((v, i, k))
            holes.append((hole, iid))
    masks = tot['A1'] + tot['A2'] + tot['A3'] + tot['obj']
    print(f'images {n}, masks (score ≥ {a.score}) {masks}, per image {masks / n:.1f}, COCO things {tot["things"]}')
    rows = [('A1', '큰 구조물(뺌)'), ('A2', '구조물+물체 합쳐짐(나눔)'), ('A3', '작은 구조물 후보(SigLIP2)'),
            ('B1', '물체가 구조물로 샘(배경 20~50%)'), ('B2', 'COCO 물체 둘 이상 합침'), ('B3', 'COCO 물체 하나가 여러 마스크로')]
    for t, d in rows:
        print(f'{t} {d:28s} {tot[t]:9d} masks  {imgs[t]:7d} images ({imgs[t] / n * 100:5.1f}%)')
    h = np.array([x for x, _ in holes])
    print(f'C  물체 화소 빈틈: 사진당 중앙값 {np.median(h) * 100:.2f}%, >20% {int((h > 0.2).sum())} 장')
    for t in ex:
        ex[t].sort(reverse=True)
        ex[t] = ex[t][:200]
    ex['C'] = [(x, i, -1) for x, i in sorted(holes, reverse=True)[:200]]
    json.dump(ex, open('data/teacher/audit.json', 'w'))


if __name__ == '__main__':
    main()
