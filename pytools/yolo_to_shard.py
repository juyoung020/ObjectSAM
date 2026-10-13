"""옛 ObjectSAM 학습 데이터(YOLO 다각형 라벨: 사람 정답 통째 + FastSAM-s 채우기, 벽·천장·바닥 없음) → 학습 샤드(FSR3).

  .venv/bin/python pytools/yolo_to_shard.py [--list ~/robot-agent/.../train_n26_uw.txt] [--out data/old_shards]
- COCO 장(coco_train__*)은 뺀다(COCO 11.3 만 장은 판정 파일 쪽으로 따로 들어간다). ADE20K 실내 + BEHAVIOR 시뮬만.
- 목록에 같은 장이 여러 번 있으면(시뮬 ×2·×4) 그 횟수만큼 레코드를 쓴다(옛 학습과 같은 비율).
- 라벨 없는 장은 정답 0 개로 넣는다(옛 학습과 같음).
레코드 형식은 coco_build.py 머리말(FSR3).
"""
import argparse
import os
import struct
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from coco_build import leb128, rle_runs

Y = os.path.expanduser('~/robot-agent/training/data/fastsam_obj/yolo')


def one(name):
    jpg = open(f'{Y}/images/train/{name}', 'rb').read()
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    lab = f'{Y}/labels/train/{os.path.splitext(name)[0]}.txt'
    gts = []
    if os.path.exists(lab):
        for line in open(lab):
            v = line.split()
            if len(v) < 7:
                continue
            pts = (np.array(v[1:], np.float32).reshape(-1, 2) * [W, H]).round().astype(np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [pts], 1)
            if m.any():
                gts.append(m.astype(bool))
    gts.sort(key=lambda m: -int(m.sum()))
    body = [b'FSR3' + struct.pack('<HHHHIHH', W, H, W, H, len(jpg), len(gts), 0), jpg]
    for m in gts:
        ys, xs = np.nonzero(m)
        r = leb128(rle_runs(m))
        body.append(struct.pack('<4fI', float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1), len(r)))
        body.append(r)
    return b''.join(body), len(gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', default=f'{Y}/train_n26_uw.txt')
    ap.add_argument('--out', default='data/old_shards')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--no-sim', action='store_true')   # 2026-10-12 사용자 지시: BEHAVIOR 시뮬은 학습에서 뺀다 → ADE20K 만
    a = ap.parse_args()
    reps = Counter(os.path.basename(l.strip()) for l in open(a.list) if l.strip())
    names = sorted(n for n in reps if not n.startswith('coco_train__') and (not a.no_sim or n.startswith('ade_')))
    os.makedirs(a.out, exist_ok=True)
    base = f"{a.out}/{'ade' if a.no_sim else 'old'}_00"
    offs, n_img, n_gt, src = [0], 0, 0, Counter()
    with open(base + '.bin.tmp', 'wb') as f, ProcessPoolExecutor(a.workers) as pool:
        for name, (rec, ng) in zip(names, pool.map(one, names, chunksize=16)):
            for _ in range(reps[name]):
                f.write(rec)
                offs.append(offs[-1] + len(rec))
                n_img += 1; n_gt += ng
                src['ade' if name.startswith('ade_') else 'sim'] += 1
    np.asarray(offs, np.uint64).tofile(base + '.idx')
    os.replace(base + '.bin.tmp', base + '.bin')
    print(f'{base}.bin: {n_img} records ({len(names)} images) {dict(src)}, {n_gt / n_img:.1f} gts/record')


if __name__ == '__main__':
    main()
