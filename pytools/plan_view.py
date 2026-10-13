"""판정 파일 그림: 원본 | 선생 마스크 전부 | 학습 정답(물체 = 색, 뺀 마스크 = 어둡게).

  .venv/bin/python pytools/plan_view.py [--plan data/teacher/train2017_wcf.plan --n 8 --seed 0]  → ~/Desktop/학습정답.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402
from final_view import read_plan_full  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default='data/teacher/train2017_wcf.plan')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.expanduser('~/Desktop/학습정답.jpg'))
    a = ap.parse_args()
    ids = [int(l.split()[0]) for l in open('data/teacher/train2017.list') if l.strip()]
    plans = read_plan_full(a.plan, set(ids))
    pick = set(np.random.default_rng(a.seed).choice(sorted(plans), a.n, replace=False).tolist())
    recs = {r['id']: r for r in ftl.read('data/teacher/train2017.ftl') if r['id'] in pick}
    tiles = []
    for iid in sorted(pick):
        r = recs[iid]
        drop, _, extras, ign = plans[iid]
        img = cv2.imread(f'data/coco/train2017/{iid:012d}.jpg')
        if img is None:
            continue
        m = ftl.idmap(r)
        col = np.random.default_rng(1).integers(60, 255, (len(r['scores']), 3))
        raw = img.astype(np.float32)
        fin = img.astype(np.float32) * 0.3
        nk = nd = 0
        for k in range(len(r['scores'])):
            mk = m == k
            if r['scores'][k] < 0.5 or not mk.any():
                continue
            raw[mk] = 0.45 * raw[mk] + 0.55 * col[k]
            if k in drop:
                nd += 1
                continue
            nk += 1
            fin[mk] = 0.45 * img[mk] + 0.55 * col[k]
        from final_view import rle_mask
        H, W = img.shape[:2]
        ex = [rle_mask(e, H, W) for e in extras]
        for j, p in enumerate(ex):
            fin[p] = 0.45 * img[p] + 0.55 * np.array([255, 200, 90.]) * (0.6 + 0.4 * ((j * 37) % 7) / 6)
        if ign:
            p = rle_mask(ign, H, W)
            fin[p] = 0.3 * img[p] + 0.7 * np.array([255, 0, 255.])
        raw, fin = raw.astype(np.uint8), fin.astype(np.uint8)
        for p in ex:
            cs = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(fin, cs, -1, (0, 255, 255), 2)
        for k in range(len(r['scores'])):
            if r['scores'][k] >= 0.5 and k not in drop:
                cs = cv2.findContours((m == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(fin, cs, -1, (255, 255, 255), 1)
        for t, txt in ((raw, 'teacher: all masks'), (fin, f'GT: human {len(ex)} (yellow edge) + teacher {nk}, removed {nd}')):
            cv2.putText(t, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(t, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        t = np.concatenate([img, raw, fin], 1)
        tiles.append(cv2.resize(t, (1500, int(1500 * t.shape[0] / t.shape[1]))))
    cv2.imwrite(a.out, np.concatenate([cv2.copyMakeBorder(t, 4, 4, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                                       for t in tiles], 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(a.out)


if __name__ == '__main__':
    main()
