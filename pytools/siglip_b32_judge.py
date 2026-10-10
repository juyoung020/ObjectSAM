"""SigLIP2 B/32 판정(가벼운 판, robot-agent 의 TensorRT 엔진): 판정 파일의 후보 마스크마다 구조물 확률 합.

  TRT_PY=~/ovdet_venv/bin/python .venv-teacher/bin/python pytools/siglip_b32_judge.py [--limit N]
    → data/teacher/train2017.siglip (FTS1, siglip_judge.py 와 같은 형식)

글자 임베딩은 미리 data/teacher/siglip_b32_text.npz (clip_venv 의 open_clip, giant 와 같은 문장).
영상: 마스크 상자를 10 % 넓힌 정사각형을 256 으로(주변 그대로), 마스크는 32 화소 칸마다 비율(wpatch)로 엔진의 MAP
풀링에 들어간다(robot-agent/src/scene_graph/clip/tools/export_siglip2.py).
"""
import argparse
import os
import struct
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402
from siglip_judge import read_plan  # noqa: E402

ENGINE = os.path.expanduser('~/robot-agent/models/ovdet/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan')
S, G = 256, 8


def crops_for(job):
    rec, cand = job
    cv2.setNumThreads(1)
    img = cv2.imread(f"data/coco/train2017/{rec['id']:012d}.jpg")
    if img is None:
        img = cv2.imread(f"data/teacher/nonjpeg/{rec['id']:012d}.ppm")
    rgb = img[..., ::-1]
    H, W = rgb.shape[:2]
    m = ftl.idmap(rec)
    ims, wps = [], []
    for k in cand:
        mk = m == k
        ys, xs = np.nonzero(mk)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        s = int(max(y1 - y0, x1 - x0) * 1.1) + 2
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        ya, xa = cy - s // 2, cx - s // 2
        crop = np.zeros((s, s, 3), np.uint8)
        cm = np.zeros((s, s), np.float32)
        sy, sx = slice(max(ya, 0), min(ya + s, H)), slice(max(xa, 0), min(xa + s, W))
        dy, dx = slice(sy.start - ya, sy.stop - ya), slice(sx.start - xa, sx.stop - xa)
        crop[dy, dx] = rgb[sy, sx]
        cm[dy, dx] = mk[sy, sx]
        crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA if s > S else cv2.INTER_LINEAR)
        cm = cv2.resize(cm, (S, S), interpolation=cv2.INTER_AREA if s > S else cv2.INTER_LINEAR)
        ims.append(crop)
        wps.append(cm.reshape(G, S // G, G, S // G).mean((1, 3)).reshape(-1))
    return rec['id'], cand, np.stack(ims), np.stack(wps).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=-1)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--out', default='data/teacher/train2017.siglip')
    a = ap.parse_args()
    sys.path.insert(0, os.path.expanduser('~/robot-agent/src/scene_graph/clip/tools'))
    from trt_run import Engine
    eng = Engine(ENGINE, max_batch=8)
    tx = np.load('data/teacher/siglip_b32_text.npz')
    temb, ns, scale, bias = tx['emb'], int(tx['nstruct']), float(tx['scale']), float(tx['bias'])
    plan = {iid: c for iid, c in read_plan('data/teacher/train2017.plan') if c}
    jobs = ((r, plan[r['id']]) for r in ftl.read('data/teacher/train2017.ftl') if r['id'] in plan)
    if a.limit > 0:
        jobs = (j for _, j in zip(range(a.limit), jobs))
    f = open(a.out + '.tmp', 'wb')
    t0, n, nc = time.time(), 0, 0

    def bounded(pool, it, window):  # 처리 중인 작업 수 제한(메모리)
        q = deque()
        for j in it:
            q.append(pool.submit(crops_for, j))
            if len(q) >= window:
                yield q.popleft().result()
        while q:
            yield q.popleft().result()

    with ProcessPoolExecutor(a.workers) as pool:
        for iid, cand, ims, wps in bounded(pool, jobs, 4 * a.workers):
            x = (ims.astype(np.float32).transpose(0, 3, 1, 2) / 255.0 - 0.5) / 0.5
            emb = eng.run(x, wps)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            lg = emb @ temb.T * scale + bias
            lg -= lg.max(1, keepdims=True)
            p = np.exp(lg)
            p = p[:, :ns].sum(1) / p.sum(1)
            f.write(b'FTS1' + struct.pack('<IHH', iid, len(cand), 0) +
                    b''.join(struct.pack('<Hf', k, float(v)) for k, v in zip(cand, p)))
            n += 1
            nc += len(cand)
            if n % 5000 == 0:
                el = time.time() - t0
                print(f'{n} images, {nc} crops, {nc / el:.0f} crops/s, eta {(558576 - nc) / (nc / el) / 60:.1f} min', flush=True)
    f.close()
    os.replace(a.out + '.tmp', a.out)
    print(f'done: {n} images, {nc} crops in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
