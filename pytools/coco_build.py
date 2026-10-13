"""COCO 2017 panoptic → 학습 샤드(FSR3). 규칙은 docs/SPEC.md 4 절.

  .venv/bin/python pytools/coco_build.py [--out data/coco_shards --workers 24]

- 나누기: train2017 에서 5,000 장(고정 시드)을 검증으로 떼고 나머지는 학습. val2017 은 시험이라 여기서 다루지 않는다.
- 배경(정답 없음) stuff 19 종: 벽 5·천장·바닥 2·땅 10·하늘. 그 밖 stuff 는 물체로, 떨어진 조각(연결 성분)마다 정답 하나.
- 물체(thing) 80 종: 개체마다 정답 하나. iscrowd 는 지금은 정답에서 뺀다(무시 영역 처리는 학습기에 넣을 때).
- 영상은 원본 JPEG 바이트를 그대로 넣는다(COCO 는 긴 변 ≤ 640, 다시 부호화하지 않음). 마스크는 영상 해상도.

레코드(리틀 엔디언): 'FSR3', u16 영상 w,h, u16 마스크 w,h, u32 JPEG 길이, u16 정답 수, u16 0, JPEG,
  정답마다(넓이 큰 순) [f32 상자 x1,y1,x2,y2 (영상 화소), u32 RLE 바이트 수, RLE].
RLE: 행 우선, 0 부터 시작하는 교대 길이들을 LEB128 가변 길이 정수로.
"""
import argparse
import json
import os
import struct
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

COCO = 'data/coco'
BG = {  # 배경 stuff 22 종(SPEC 4 절)
    'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'wall-other-merged',
    'ceiling-merged',
    'floor-wood', 'floor-other-merged', 'road', 'pavement-merged', 'dirt-merged', 'sand', 'gravel', 'snow',
    'grass-merged', 'playingfield', 'railroad', 'platform',
    'sky-other-merged',
    'building-other-merged', 'house', 'roof',  # 건물도 구조물(2026-10-11)
}
N_VAL = 5000
MIN_PIECE = 16          # stuff 조각 최소 넓이(화소) — 1 칸도 안 되는 부스러기만
SHARD_IMAGES = 14000


def rle_runs(m):
    f = m.reshape(-1)
    idx = np.flatnonzero(f[1:] != f[:-1]) + 1
    runs = np.diff(np.concatenate(([0], idx, [f.size])))
    if f[0]:
        runs = np.concatenate(([0], runs))
    return runs.astype(np.uint32)


def leb128(v):
    nb = 1 + (v >= 1 << 7) + (v >= 1 << 14) + (v >= 1 << 21)
    out = np.zeros(int(nb.sum()), np.uint8)
    pos = np.concatenate(([0], np.cumsum(nb)[:-1]))
    for k in range(4):
        sel = nb > k
        byte = (v[sel] >> (7 * k)) & 0x7F
        out[pos[sel] + k] = (byte | ((nb[sel] > k + 1).astype(np.uint32) << 7)).astype(np.uint8)
    return out.tobytes()


_ZIP = None
_CATS = None


def _init(cats):
    global _CATS
    import cv2
    cv2.setNumThreads(1)
    _CATS = cats


def build_one(args):
    """영상 하나 → 레코드 바이트, 통계."""
    global _ZIP
    import cv2
    ann, split_dir = args
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(f'{COCO}/panoptic_train2017.zip')
    jpg = open(f"{COCO}/{split_dir}/{ann['file_name'].replace('.png', '.jpg')}", 'rb').read()
    png = cv2.imdecode(np.frombuffer(_ZIP.read(f"panoptic_train2017/{ann['file_name']}"), np.uint8), cv2.IMREAD_COLOR)
    rgb = png[..., ::-1].astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    H, W = ids.shape
    st = dict(thing=0, crowd=0, bg=0, stuff_pieces=0)
    gts = []
    for s in ann['segments_info']:
        c = _CATS[s['category_id']]
        m = ids == s['id']
        if c['isthing']:
            if s['iscrowd']:
                st['crowd'] += 1
                continue
            gts.append(m)
            st['thing'] += 1
        elif c['name'] in BG:
            st['bg'] += 1
        else:
            n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
            for k in range(1, n):
                piece = lab == k
                if piece.sum() >= MIN_PIECE:
                    gts.append(piece)
                    st['stuff_pieces'] += 1
    gts.sort(key=lambda m: -int(m.sum()))
    body = [b'FSR3' + struct.pack('<HHHHIHH', W, H, W, H, len(jpg), len(gts), 0), jpg]
    for m in gts:
        ys, xs = np.nonzero(m)
        r = leb128(rle_runs(m))
        body.append(struct.pack('<4fI', float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1), len(r)))
        body.append(r)
    return b''.join(body), st


def write_shards(name, anns, split_dir, out, workers, cats):
    t0 = time.time()
    tot = dict(images=0, gts=0, thing=0, crowd=0, bg=0, stuff_pieces=0)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(cats,)) as pool:
        for si in range(0, len(anns), SHARD_IMAGES):
            part = anns[si:si + SHARD_IMAGES]
            base = f'{out}/{name}_{si // SHARD_IMAGES:02d}'
            offs = [0]
            with open(base + '.bin.tmp', 'wb') as f:
                for rec, st in pool.map(build_one, [(a, split_dir) for a in part], chunksize=16):
                    f.write(rec)
                    offs.append(offs[-1] + len(rec))
                    tot['images'] += 1
                    for k, v in st.items():
                        tot[k] += v
            np.asarray(offs, np.uint64).tofile(base + '.idx')
            os.replace(base + '.bin.tmp', base + '.bin')
            print(f'{base}.bin: {len(part)} images ({time.time() - t0:.0f} s)', flush=True)
    tot['gts'] = tot['thing'] + tot['stuff_pieces']
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/coco_shards')
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = json.load(open(f'{COCO}/panoptic_train2017.json'))
    cats = {c['id']: c for c in d['categories']}
    assert BG <= {c['name'] for c in d['categories']}, BG - {c['name'] for c in d['categories']}
    anns = sorted(d['annotations'], key=lambda x: x['image_id'])
    rng = np.random.default_rng(0)
    val_idx = set(rng.choice(len(anns), N_VAL, replace=False).tolist())
    val = [x for i, x in enumerate(anns) if i in val_idx]
    train = [x for i, x in enumerate(anns) if i not in val_idx]
    json.dump(sorted(x['image_id'] for x in val), open(f'{a.out}/val_image_ids.json', 'w'))
    stats = {'val': write_shards('val', val, 'train2017', a.out, a.workers, cats),
             'train': write_shards('train', train, 'train2017', a.out, a.workers, cats),
             'rules': dict(bg=sorted(BG), min_piece=MIN_PIECE, n_val=N_VAL, crowd='dropped', seed=0)}
    json.dump(stats, open(f'{a.out}/stats.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != 'rules'}))


if __name__ == '__main__':
    sys.exit(main())
