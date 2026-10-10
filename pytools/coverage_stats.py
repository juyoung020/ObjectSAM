"""선생 라벨이 사진을 얼마나 덮나: 점수 ≥ score 마스크로 덮이지 않은 화소를 COCO panoptic 종류별로 센다(전체 사진).

  .venv/bin/python pytools/coverage_stats.py [--score 0.5 --workers 24]
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
KINDS = ['thing', 'stuff-obj', 'bg', 'unlabeled']
_G = {}


def _init(score):
    cv2.setNumThreads(1)
    pan = json.load(open(f'{COCO}/panoptic_train2017.json'))
    _G['cats'] = {c['id']: c for c in pan['categories']}
    _G['anns'] = {x['image_id']: x for x in pan['annotations']}
    _G['zip'] = zipfile.ZipFile(f'{COCO}/panoptic_train2017.zip')
    _G['score'] = score


def one(r):
    x = _G['anns'][r['id']]
    png = cv2.imdecode(np.frombuffer(_G['zip'].read('panoptic_train2017/' + x['file_name']), np.uint8), cv2.IMREAD_COLOR)
    rgb = png[..., ::-1].astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    kind = np.full(ids.shape, 3, np.int8)  # 3 = 미라벨
    for s in x['segments_info']:
        c = _G['cats'][s['category_id']]
        kind[ids == s['id']] = 2 if c['name'] in BG else (0 if c['isthing'] else 1)
    m = ftl.idmap(r)
    valid = m >= 0
    keep = r['scores'] >= _G['score']
    covered = np.zeros(m.shape, bool)
    covered[valid] = keep[m[valid]]
    out = np.zeros((4, 3), np.int64)  # [종류][전체, 마스크 없음, 낮은 점수만]
    for k in range(4):
        sel = kind == k
        out[k] = sel.sum(), (sel & ~valid).sum(), (sel & valid & ~covered).sum()
    hole = ~covered & (kind <= 1)
    frac = hole.sum() / max(1, (kind <= 1).sum())
    return r['id'], out, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score', type=float, default=0.5)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    tot = np.zeros((4, 3), np.int64)
    fracs = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.score,)) as pool:
        for iid, out, frac in pool.map(one, ftl.read('data/teacher/train2017.ftl'), chunksize=64):
            tot += out
            fracs.append((frac, iid))
    print(f'images {len(fracs)}')
    for k, nm in enumerate(KINDS):
        t, nomask, low = tot[k]
        print(f'{nm:10s} {t / 1e9:6.2f} G px  no mask {nomask / t * 100:5.2f}%  only low-score {low / t * 100:5.2f}%  '
              f'total {(nomask + low) / t * 100:5.2f}%')
    f = np.array([x for x, _ in fracs])
    print('per image, share of object pixels uncovered: median %.2f%%, 90th %.2f%%, 99th %.2f%%, >20%%: %d images' %
          (np.median(f) * 100, np.percentile(f, 90) * 100, np.percentile(f, 99) * 100, (f > 0.2).sum()))
    worst = sorted(fracs, reverse=True)[:10]
    print('worst:', [(i, f'{x * 100:.0f}%') for x, i in worst])


if __name__ == '__main__':
    main()
