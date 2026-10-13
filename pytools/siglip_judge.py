"""SigLIP2 판정(docs/LABELS.md 3 절 경우 4): 판정 파일의 후보 마스크마다 구조물 확률 합(벽+천장+바닥+하늘)을 구한다.

  .venv-teacher/bin/python pytools/siglip_judge.py [--limit N]  → data/teacher/train2017.siglip

레코드(판정 파일과 같은 순서, 후보가 있는 사진만): 'FTS1', u32 image_id, u16 n, u16 0, n × [u16 선생 마스크 번호, f32 확률].
"""
import argparse
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402
from structure_review import OBJECT, SIGLIP, STRUCT, crop_of  # noqa: E402

SIZE = 384


def read_plan(path):
    data = open(path, 'rb').read()
    p = 0
    while p < len(data):
        assert data[p:p + 4] == b'FTP1'
        iid, nd, nc, ne, _ = struct.unpack_from('<IHHHH', data, p + 4)
        q = p + 16 + 2 * nd
        cand = list(struct.unpack_from(f'<{nc}H', data, q))
        q += 2 * nc
        for _ in range(ne):
            (ln,) = struct.unpack_from('<I', data, q)
            q += 4 + ln
        (ln,) = struct.unpack_from('<I', data, q)
        p = q + 4 + ln
        yield iid, cand


def crops_for(job):
    """사진 하나의 후보 조각들(384×384 RGB u8, SigLIP 정규화 전)."""
    rec, cand = job
    cv2.setNumThreads(1)
    img = cv2.imread(f"data/coco/train2017/{rec['id']:012d}.jpg")
    if img is None:  # PNG 1 장
        from PIL import Image
        img = np.asarray(Image.open(f"data/coco/train2017/{rec['id']:012d}.jpg").convert('RGB'))[..., ::-1]
    rgb = np.ascontiguousarray(img[..., ::-1])
    m = ftl.idmap(rec)
    out = []
    for k in cand:
        c = crop_of(rgb, m == k)
        out.append(cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR if c.shape[0] < SIZE else cv2.INTER_AREA))
    return rec['id'], cand, np.stack(out) if out else np.zeros((0, SIZE, SIZE, 3), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=-1)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', default='data/teacher/train2017.siglip')
    a = ap.parse_args()
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(SIGLIP)
    model = AutoModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).cuda().eval()
    texts = [t for v in STRUCT.values() for t in v] + OBJECT
    ns = sum(len(v) for v in STRUCT.values())
    with torch.no_grad():
        ti = proc(text=[f'This is a photo of {t}.' for t in texts], padding='max_length', max_length=64,
                  return_tensors='pt').to('cuda')
        tf = torch.nn.functional.normalize(model.get_text_features(**ti).float(), dim=-1)
    scale, bias = model.logit_scale.exp().float(), model.logit_bias.float()
    ip = proc.image_processor
    mean = torch.tensor(ip.image_mean, device='cuda').view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device='cuda').view(1, 3, 1, 1)

    plan = {iid: c for iid, c in read_plan('data/teacher/train2017.plan') if c}
    jobs = ((r, plan[r['id']]) for r in ftl.read('data/teacher/train2017.ftl') if r['id'] in plan)
    if a.limit > 0:
        jobs = (j for i, j in zip(range(a.limit), jobs))
    f = open(a.out, 'wb')
    pending = []  # (id, cand, 시작, 끝)
    buf, done, ncrop = [], 0, 0
    t0 = time.time()

    @torch.no_grad()
    def flush(final=False):
        nonlocal buf, pending, done
        while buf and (len(buf) >= a.batch or final):
            take = buf[:a.batch]
            x = torch.from_numpy(np.stack(take)).cuda().permute(0, 3, 1, 2).half().div(255)
            x = (x - mean) / std
            feat = torch.nn.functional.normalize(model.get_image_features(pixel_values=x).float(), dim=-1)
            p = ((feat @ tf.T) * scale + bias).softmax(-1)[:, :ns].sum(-1).cpu().numpy()
            probs.extend(p.tolist())
            buf = buf[a.batch:]
        # 확률이 다 나온 사진부터 기록
        while pending and pending[0][3] <= len(probs) + base[0]:
            iid, cand, s, e = pending.pop(0)
            pr = probs[s - base[0]:e - base[0]]
            f.write(b'FTS1' + struct.pack('<IHH', iid, len(cand), 0) +
                    b''.join(struct.pack('<Hf', k, v) for k, v in zip(cand, pr)))
            done += 1
        if pending:
            drop = pending[0][2] - base[0]
        else:
            drop = len(probs)
        del probs[:drop]
        base[0] += drop

    probs, base = [], [0]
    def bounded(pool, it, window):
        """pool.map 은 작업을 한꺼번에 넣어 결과가 GPU 보다 빨리 쌓인다(메모리 부족) → 처리 중인 작업 수를 제한."""
        from collections import deque
        q = deque()
        for j in it:
            q.append(pool.submit(crops_for, j))
            if len(q) >= window:
                yield q.popleft().result()
        while q:
            yield q.popleft().result()

    with ProcessPoolExecutor(a.workers) as pool:
        for iid, cand, crops in bounded(pool, jobs, 4 * a.workers):
            s = ncrop
            ncrop += len(cand)
            pending.append((iid, cand, s, ncrop))
            buf.extend(list(crops))
            before = done
            flush()
            if done // 2000 != before // 2000:
                f.flush()
                el = time.time() - t0
                print(f'{done} images, {ncrop} crops, {ncrop / el:.1f} crops/s, eta {(558576 - ncrop) / (ncrop / el) / 3600:.1f} h',
                      flush=True)
    flush(final=True)
    f.close()
    el = time.time() - t0
    print(f'done: {done} images, {ncrop} crops in {el:.0f} s ({ncrop / el:.1f} crops/s)')


if __name__ == '__main__':
    main()
