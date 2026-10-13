"""판정 파일: 선생 라벨(FTL1)을 COCO panoptic 과 맞춰 사진마다 무엇을 빼고 나누고 무시할지 한 번 계산한다(docs/LABELS.md 3 절).

  .venv/bin/python pytools/label_plan.py [--workers 24]  → data/teacher/train2017.plan

레코드(리틀 엔디언, 선생 라벨과 같은 순서): 'FTP1', u32 image_id, u16 n_drop, u16 n_cand, u16 n_extra, u16 0,
  u16 drop[n_drop]  (경우 2·3: 정답에서 뺄 선생 마스크 번호),
  u16 cand[n_cand]  (경우 4: SigLIP2 판정 대상, 판정 전에는 물체),
  n_extra × [u32 RLE 바이트 수, RLE]  (경우 3 에서 나눠 살린 COCO 물체 조각, 사진 해상도),
  u32 RLE 바이트 수, RLE  (무시 영역; 0 이면 없음).
선생 마스크 번호는 FTL1 의 순위 번호, 점수 ≥ score 인 것만 정답 후보. RLE 는 FSR3 와 같다(행 우선, 0 부터 교대 길이, LEB128).
"""
import argparse
import json
import os
import struct
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402
from coco_build import BG, leb128, rle_runs  # noqa: E402

WCF = {n for n in BG if n.startswith(('wall', 'ceiling', 'floor'))} | {  # 벽·천장·바닥, 바깥 땅도 바닥
    'road', 'pavement-merged', 'dirt-merged', 'sand', 'gravel', 'snow', 'grass-merged', 'playingfield', 'railroad', 'platform'}

COCO = 'data/coco'
_G = {}


def _init(a):
    cv2.setNumThreads(1)
    pan = json.load(open(f'{COCO}/panoptic_train2017.json'))
    _G['cats'] = {c['id']: c for c in pan['categories']}
    _G['anns'] = {x['image_id']: x for x in pan['annotations']}
    _G['zip'] = zipfile.ZipFile(f'{COCO}/panoptic_train2017.zip')
    _G['a'] = a


def rle(m):
    return leb128(rle_runs(m)) if m.any() else b''


def one(r):
    a = _G['a']
    x = _G['anns'][r['id']]
    png = cv2.imdecode(np.frombuffer(_G['zip'].read('panoptic_train2017/' + x['file_name']), np.uint8), cv2.IMREAD_COLOR)
    rgb = png[..., ::-1].astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    H, W = ids.shape
    isbg = {s['id']: _G['cats'][s['category_id']]['name'] in BG for s in x['segments_info']}
    area = {s['id']: s['area'] for s in x['segments_info']}
    if a.simple:  # 벽·천장·바닥만, 다른 조건 없이(2026-10-11): 그 화소가 절반 이상이면 뺀다
        isbg = {s['id']: _G['cats'][s['category_id']]['name'] in WCF for s in x['segments_info']}
    bg = np.isin(ids, [i for i, b in isbg.items() if b])
    obj = np.isin(ids, [i for i, b in isbg.items() if not b])
    m = ftl.idmap(r)
    drop, cand, extra = [], [], []
    covered = np.zeros((H, W), bool)
    ignore = np.zeros((H, W), bool)
    min_piece = a.min_piece * H * W
    st = dict(obj=0, big=0, split=0, cand=0, pieces=0, tiny=0)
    for k in np.where(r['scores'] >= a.score)[0]:
        mk = m == k
        if not mk.any():
            continue
        bgf = bg[mk].mean()
        if a.simple:
            if bgf >= a.bg_frac:
                drop.append(int(k))
                st['big'] += 1
            else:
                st['obj'] += 1
            continue
        if bgf < a.bg_frac:
            covered |= mk
            st['obj'] += 1
            continue
        u, c = np.unique(ids[mk & bg], return_counts=True)
        j = int(np.argmax(c))
        if c[j] / area[int(u[j])] >= a.cover:
            drop.append(int(k))
            if obj[mk].mean() >= a.mix:
                st['split'] += 1
                # 섞인 COCO 물체 조각을 떨어진 조각마다 살린다
                for sid in np.unique(ids[mk & obj]):
                    part = (mk & (ids == sid)).astype(np.uint8)
                    n, lab = cv2.connectedComponents(part, connectivity=8)
                    for q in range(1, n):
                        p = lab == q
                        if p.sum() >= min_piece:
                            extra.append(rle(p))
                            covered |= p
                            st['pieces'] += 1
                        else:
                            ignore |= p
                            st['tiny'] += 1
            else:
                st['big'] += 1
        else:
            cand.append(int(k))
            covered |= mk
            st['cand'] += 1
    if not a.simple:
        ignore |= obj & ~covered  # 빈틈은 무시 영역
    rec = [b'FTP1', struct.pack('<IHHHH', r['id'], len(drop), len(cand), len(extra), 0),
           struct.pack(f'<{len(drop)}H', *drop), struct.pack(f'<{len(cand)}H', *cand)]
    for e in extra:
        rec += [struct.pack('<I', len(e)), e]
    ig = rle(ignore)
    rec += [struct.pack('<I', len(ig)), ig]
    st['ignore_px'] = int(ignore.sum())
    st['px'] = H * W
    return b''.join(rec), st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score', type=float, default=0.5)
    ap.add_argument('--bg_frac', type=float, default=0.5)
    ap.add_argument('--cover', type=float, default=0.3)
    ap.add_argument('--mix', type=float, default=0.2)
    ap.add_argument('--min_piece', type=float, default=0.001)
    ap.add_argument('--simple', action='store_true', help='벽·천장·바닥(땅 포함)만 과반이면 뺀다. 나누기·SigLIP 후보·무시 영역 없음')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--out', default='data/teacher/train2017.plan')
    a = ap.parse_args()
    tot = {}
    n = 0
    with open(a.out + '.tmp', 'wb') as f, ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a,)) as pool:
        for rec, st in pool.map(one, ftl.read('data/teacher/train2017.ftl'), chunksize=64):
            f.write(rec)
            n += 1
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
    os.replace(a.out + '.tmp', a.out)
    json.dump({'args': vars(a), 'images': n, **tot}, open(a.out + '.json', 'w'), indent=1)
    print(f'{a.out}: {n} images', {k: v for k, v in tot.items() if k not in ('px',)},
          f"ignore {tot['ignore_px'] / tot['px'] * 100:.2f}% of pixels")


if __name__ == '__main__':
    main()
