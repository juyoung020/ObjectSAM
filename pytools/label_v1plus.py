"""v1 판정 파일(CropFormer 거의 전부 + SigLIP) 위에 사람 정답을 '보태기만' 한다(2026-10-12, v10).

  .venv/bin/python pytools/label_v1plus.py [--workers 16] [--dup 0.8] → data/teacher/train2017_v1plus.plan
- v1 의 drop·cand(SigLIP 판정 대상)·extra·무시 영역은 그대로 둔다(CropFormer 마스크는 하나도 더 버리지 않는다).
- 사람 정답(COCO panoptic things(crowd 제외) + 문·창·계단 + LVIS, label_human.py 와 같은 규칙) 중
  남는 선생 마스크나 v1 extra 와 IoU ≥ --dup 인 것이 없는 것만 extra 로 덧붙인다(중복 정답 막기).
- 덧붙인 정답 화소는 무시 영역에서 뺀다(v1 은 'COCO 물체인데 선생이 안 덮은 곳'을 무시 영역으로 뒀다).
"""
import argparse
import json
import os
import struct
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402
from label_human import DWS, LVIS, lvis_mask, rle  # noqa: E402

COCO = 'data/coco'
_G = {}


def unrle(b, n):
    vals, v, s = [], 0, 0
    for c in b:
        v |= (c & 0x7F) << s; s += 7
        if not c & 0x80:
            vals.append(v); v = s = 0
    m = np.zeros(n, bool); p = 0
    for j, ln in enumerate(vals):
        if j % 2:
            m[p:p + ln] = True
        p += ln
    return m


def plans(path):
    d = open(path, 'rb').read(); q = 0
    while q < len(d):
        iid, nd, nc, ne, _ = struct.unpack_from('<IHHHH', d, q + 4); q += 16
        drop = list(struct.unpack_from(f'<{nd}H', d, q)); q += 2 * nd
        cand = list(struct.unpack_from(f'<{nc}H', d, q)); q += 2 * nc
        ex = []
        for _ in range(ne):
            (l,) = struct.unpack_from('<I', d, q); ex.append(d[q + 4:q + 4 + l]); q += 4 + l
        (l,) = struct.unpack_from('<I', d, q); ig = d[q + 4:q + 4 + l]; q += 4 + l
        yield iid, drop, cand, ex, ig


def _init(a):
    cv2.setNumThreads(1)
    _G['zip'] = zipfile.ZipFile(f'{COCO}/panoptic_train2017.zip')
    _G['a'] = a


def one(job):
    r, (iid, drop, cand, ex, ig), x, lv = job
    a = _G['a']
    png = cv2.imdecode(np.frombuffer(_G['zip'].read('panoptic_train2017/' + x['file_name']), np.uint8), cv2.IMREAD_COLOR)
    rgb = png[..., ::-1].astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    H, W = ids.shape; n = H * W
    gts = []
    for s in x['segments_info']:
        if s.get('iscrowd'):
            continue
        if s['isthing'] or s['name'] in DWS:
            gts.append((ids == s['id']).reshape(-1))
    npan = len(gts)
    for seg in lv:
        m = lvis_mask(seg, H, W).reshape(-1)
        if m.any() and not any((m & g).sum() / max((m | g).sum(), 1) > 0.7 for g in gts[:npan]):
            gts.append(m)
    tm = ftl.idmap(r).reshape(-1)
    dset = set(drop)
    have = [tm == k for k in np.where(r['scores'] >= a.score)[0] if int(k) not in dset]
    have += [unrle(e, n) for e in ex]
    ignore = unrle(ig, n) if ig else np.zeros(n, bool)
    add = []
    st = dict(gt=len(gts), added=0, dup=0, not_nested=0)
    for g in gts:
        ga = g.sum()
        if ga == 0:
            continue
        if any((g & h).sum() / max((g | h).sum(), 1) >= a.dup for h in have if (g & h).any()):
            st['dup'] += 1
            continue
        if a.nested_only:   # 다른(선생·v1 extra) 마스크 안에 80 % 이상 들어가고 그 넓이의 70 % 이하 = 부품·내용물만
            if not any((g & h).sum() >= 0.8 * ga and ga <= 0.7 * h.sum() for h in have if (g & h).any()):
                st['not_nested'] = st.get('not_nested', 0) + 1
                continue
        add.append(g); st['added'] += 1
        ignore &= ~g
    extra = ex + [rle(g.reshape(H, W)) for g in add]
    rec = [b'FTP1', struct.pack('<IHHHH', iid, len(drop), len(cand), len(extra), 0),
           struct.pack(f'<{len(drop)}H', *drop), struct.pack(f'<{len(cand)}H', *cand)]
    for e in extra:
        rec += [struct.pack('<I', len(e)), e]
    igr = rle(ignore.reshape(H, W))
    rec += [struct.pack('<I', len(igr)), igr]
    return b''.join(rec), st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score', type=float, default=0.5)
    ap.add_argument('--dup', type=float, default=0.8)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=-1)
    ap.add_argument('--nested-only', action='store_true')   # 2026-10-12 v18: 사람 정답 중 부품·내용물(다른 마스크 안)만 보탠다
    ap.add_argument('--base', default='data/teacher/train2017.plan')
    ap.add_argument('--out', default='data/teacher/train2017_v1plus.plan')
    a = ap.parse_args()
    lv = json.load(open(LVIS)); by = defaultdict(list)
    for x in lv['annotations']:
        by[x['image_id']].append(x['segmentation'])
    del lv
    pan = json.load(open(f'{COCO}/panoptic_train2017.json'))
    cats = {c['id']: c for c in pan['categories']}
    anns = {x['image_id']: {'file_name': x['file_name'], 'segments_info': [
        {'id': s['id'], 'iscrowd': s.get('iscrowd', 0), 'name': cats[s['category_id']]['name'],
         'isthing': cats[s['category_id']]['isthing']} for s in x['segments_info']]} for x in pan['annotations']}
    del pan
    jobs = ((r, p, anns[r['id']], by.get(r['id'], [])) for r, p in zip(ftl.read('data/teacher/train2017.ftl'), plans(a.base)))
    if a.limit > 0:
        jobs = (j for _, j in zip(range(a.limit), jobs))
    tot, n = defaultdict(int), 0
    with open(a.out + '.tmp', 'wb') as f, ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a,)) as pool:
        for rec, st in pool.map(one, jobs, chunksize=16):
            f.write(rec); n += 1
            for k, v in st.items():
                tot[k] += v
    os.replace(a.out + '.tmp', a.out)
    json.dump({'args': vars(a), 'images': n, **tot}, open(a.out + '.json', 'w'), indent=1)
    print(f'{a.out}: {n} images', {k: round(v / n, 2) for k, v in tot.items()}, '(per image)')


if __name__ == '__main__':
    main()
