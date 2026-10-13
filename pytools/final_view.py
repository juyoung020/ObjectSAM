"""최종 라벨 그림: 원본 | 선생 마스크 전부 | 최종(판정 파일 + SigLIP2 적용) — docs/LABELS.md 3 절.

  .venv/bin/python pytools/final_view.py [--n 6 --seed 0 --need-siglip]  → ~/Desktop/최종라벨.jpg

최종 칸: 물체 = 색, 큰 구조물로 뺀 곳 = 어둡게, SigLIP2 로 뺀 조각 = 빨강, 나눠 살린 COCO 조각 = 흰 테두리, 무시 영역 = 자홍.
"""
import argparse
import os
import struct
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402


def leb_runs(b):
    vals, p = [], 0
    while p < len(b):
        v = s = 0
        while True:
            c = b[p]
            p += 1
            v |= (c & 0x7F) << s
            s += 7
            if not c & 0x80:
                break
        vals.append(v)
    return vals


def rle_mask(b, H, W):
    m = np.zeros(H * W, bool)
    pos, one = 0, False
    for v in leb_runs(b):
        if one:
            m[pos:pos + v] = True
        pos += v
        one = not one
    return m.reshape(H, W)


def read_plan_full(path, want):
    d = open(path, 'rb').read()
    p, out = 0, {}
    while p < len(d):
        iid, nd, nc, ne, _ = struct.unpack_from('<IHHHH', d, p + 4)
        q = p + 16
        drop = struct.unpack_from(f'<{nd}H', d, q)
        q += 2 * nd
        cand = struct.unpack_from(f'<{nc}H', d, q)
        q += 2 * nc
        extras = []
        for _ in range(ne):
            (ln,) = struct.unpack_from('<I', d, q)
            extras.append(d[q + 4:q + 4 + ln])
            q += 4 + ln
        (ln,) = struct.unpack_from('<I', d, q)
        ign = d[q + 4:q + 4 + ln]
        p = q + 4 + ln
        if iid in want:
            out[iid] = (set(drop), set(cand), extras, ign)
    return out


def read_siglip(path):
    d = open(path, 'rb').read()
    p, out = 0, {}
    while p + 12 <= len(d):
        iid, n = struct.unpack_from('<IH', d, p + 4)
        out[iid] = {struct.unpack_from('<H', d, p + 12 + 6 * j)[0]: struct.unpack_from('<f', d, p + 14 + 6 * j)[0]
                    for j in range(n)}
        p += 12 + 6 * n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--thr', type=float, default=0.8)
    ap.add_argument('--out', default=os.path.expanduser('~/Desktop/최종라벨.jpg'))
    a = ap.parse_args()
    sg = read_siglip('data/teacher/train2017.siglip')
    hit = sorted(i for i, v in sg.items() if any(p >= a.thr for p in v.values()))
    rng = np.random.default_rng(a.seed)
    pick = set(rng.choice(hit, a.n, replace=False).tolist())
    plans = read_plan_full('data/teacher/train2017.plan', pick)
    recs = {r['id']: r for r in ftl.read('data/teacher/train2017.ftl') if r['id'] in pick}
    tiles = []
    for iid in sorted(pick):
        r = recs[iid]
        drop, cand, extras, ign = plans[iid]
        img = cv2.imread(f'data/coco/train2017/{iid:012d}.jpg')
        H, W = img.shape[:2]
        m = ftl.idmap(r)
        keepscore = r['scores'] >= 0.5
        col = np.random.default_rng(1).integers(60, 255, (len(r['scores']), 3))
        raw = img.astype(np.float32)
        fin = img.astype(np.float32) * 0.35
        nsig = 0
        for k in range(len(r['scores'])):
            if not keepscore[k]:
                continue
            mk = m == k
            if not mk.any():
                continue
            raw[mk] = 0.45 * raw[mk] + 0.55 * col[k]
            if k in drop:
                continue
            if k in cand and sg.get(iid, {}).get(k, 0) >= a.thr:
                fin[mk] = 0.3 * img[mk] + 0.7 * np.array([0, 0, 255.])
                nsig += 1
                continue
            fin[mk] = 0.45 * img[mk] + 0.55 * col[k]
        for e in extras:
            p = rle_mask(e, H, W)
            fin[p] = 0.45 * img[p] + 0.55 * np.array([255, 220, 120.])
        if ign:
            p = rle_mask(ign, H, W)
            fin[p] = 0.3 * img[p] + 0.7 * np.array([255, 0, 255.])
        raw, fin = raw.astype(np.uint8), fin.astype(np.uint8)
        for e in extras:
            p = rle_mask(e, H, W).astype(np.uint8)
            cv2.drawContours(fin, cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 255, 255), 1)
        for t, txt in ((raw, 'teacher: all masks'), (fin, f'final: {nsig} removed by SigLIP2 (red)')):
            cv2.putText(t, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(t, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        t = np.concatenate([img, raw, fin], 1)
        tiles.append(cv2.resize(t, (1440, int(1440 * t.shape[0] / t.shape[1]))))
    cv2.imwrite(a.out, np.concatenate([cv2.copyMakeBorder(t, 4, 4, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                                       for t in tiles], 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(a.out)


if __name__ == '__main__':
    main()
