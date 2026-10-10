"""label_audit.py 의 유형별 사례를 그림으로(data/teacher/audit.json).

  .venv/bin/python pytools/audit_view.py B1 [--n 4] → ~/Desktop/유형_B1.jpg
B1: 선생 물체 마스크(초록) 중 COCO 배경 부분(빨강). B2: 선생 마스크(흰 테두리) 안의 COCO 물체들(색마다 다른 물체).
B3: COCO 물체 하나(흰 테두리)를 나눠 가진 선생 마스크들(색마다). C: 어느 마스크에도 안 덮인 COCO 물체 화소(자홍).
"""
import argparse
import json
import os
import sys
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402
from coco_build import BG  # noqa: E402

COCO = 'data/coco'


def contour(vis, m, col=(255, 255, 255), t=2):
    cv2.drawContours(vis, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, col, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['A2', 'B1', 'B2', 'B3', 'C'])
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    ex = json.load(open('data/teacher/audit.json'))[a.kind]
    rng = np.random.default_rng(a.seed)
    seen, pick = set(), []
    for i in rng.permutation(len(ex)):
        v, iid, k = ex[i]
        if iid not in seen:
            seen.add(iid)
            pick.append((v, iid, k))
        if len(pick) == a.n:
            break
    want = {p[1] for p in pick}
    recs = {r['id']: r for r in ftl.read('data/teacher/train2017.ftl') if r['id'] in want}
    pan = json.load(open(f'{COCO}/panoptic_train2017.json'))
    cats = {c['id']: c for c in pan['categories']}
    anns = {x['image_id']: x for x in pan['annotations']}
    z = zipfile.ZipFile(f'{COCO}/panoptic_train2017.zip')
    col = np.random.default_rng(5)
    tiles = []
    for v, iid, k in pick:
        x = anns[iid]
        png = cv2.imdecode(np.frombuffer(z.read('panoptic_train2017/' + x['file_name']), np.uint8), cv2.IMREAD_COLOR)
        rgb = png[..., ::-1].astype(np.int64)
        ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
        info = {s['id']: (cats[s['category_id']]['name'], cats[s['category_id']]['isthing']) for s in x['segments_info']}
        bg = np.isin(ids, [i for i, (nm, _) in info.items() if nm in BG])
        obj = np.isin(ids, [i for i, (nm, _) in info.items() if nm not in BG])
        r = recs[iid]
        m = ftl.idmap(r)
        img = cv2.imread(f'{COCO}/train2017/{iid:012d}.jpg')
        vis = img.astype(np.float32)
        title = ''
        if a.kind in ('A2', 'B1'):
            mk = m == k
            vis[mk & ~bg] = 0.45 * vis[mk & ~bg] + 0.55 * np.array([60, 220, 60.])
            vis[mk & bg] = 0.4 * vis[mk & bg] + 0.6 * np.array([0, 0, 255.])
            vis = vis.astype(np.uint8)
            contour(vis, mk)
            title = f'structure share {float(bg[mk].mean()) * 100:.0f}%'
        elif a.kind == 'B2':
            mk = m == k
            names = []
            for sid, (nm, th) in info.items():
                p = (ids == sid) & mk
                if th and p.sum() > 0.5 * (ids == sid).sum():
                    c = col.integers(60, 255, 3)
                    vis[p] = 0.4 * vis[p] + 0.6 * c
                    names.append(nm)
            vis = vis.astype(np.uint8)
            contour(vis, mk)
            title = 'one teacher mask over: ' + ', '.join(names[:4])
        elif a.kind == 'B3':
            thing = ids == k
            ks = [kk for kk in np.unique(m[thing]) if kk >= 0 and r['scores'][kk] >= 0.5]
            for kk in ks:
                p = (m == kk) & thing
                vis[p] = 0.4 * vis[p] + 0.6 * col.integers(60, 255, 3)
            vis = vis.astype(np.uint8)
            contour(vis, thing)
            title = f'COCO {info[k][0]} split into {len(ks)} teacher masks'
        else:
            keep = r['scores'] >= 0.5
            cov = np.zeros(m.shape, bool)
            cov[m >= 0] = keep[m[m >= 0]]
            hole = obj & ~cov
            vis[hole] = 0.3 * vis[hole] + 0.7 * np.array([255, 0, 255.])
            vis = vis.astype(np.uint8)
            title = f'uncovered object pixels {v * 100:.0f}%'
        cv2.putText(vis, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        t = np.concatenate([img, vis], 1)
        tiles.append(cv2.resize(t, (960, int(960 * t.shape[0] / t.shape[1]))))
    out = os.path.expanduser(f'~/Desktop/유형_{a.kind}.jpg')
    cv2.imwrite(out, np.concatenate([cv2.copyMakeBorder(t, 4, 4, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                                     for t in tiles], 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == '__main__':
    main()
