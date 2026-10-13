"""v1 판정 파일(train2017.plan + siglip) 표본 분석 — Q1(천장·벽 위 버려진 작은 마스크), Q2(점수 0.2–0.5 마스크),
Q3(장당 정답 수·상한), Q4(정답 크기 분포). 읽기만 한다.

  cd ~/ObjectSAM && .venv/bin/python analysis/recall_research/label_stats.py [--every 20] [--workers 12]
  → analysis/recall_research/label_stats.pkl
"""
import argparse
import json
import os
import pickle
import struct
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

ROOT = '/home/juyoung/ObjectSAM'
sys.path.insert(0, f'{ROOT}/pytools')
import ftl  # noqa: E402
from coco_build import BG  # noqa: E402

COCO = f'{ROOT}/data/coco'
_G = {}


def unrle(b, n):
    vals, v, s = [], 0, 0
    for c in b:
        v |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            vals.append(v)
            v = s = 0
    m = np.zeros(n, bool)
    p = 0
    for j, ln in enumerate(vals):
        if j % 2:
            m[p:p + ln] = True
        p += ln
    return m


def plans(path):
    d = open(path, 'rb').read()
    q = 0
    while q < len(d):
        iid, nd, nc, ne, _ = struct.unpack_from('<IHHHH', d, q + 4)
        q += 16
        drop = list(struct.unpack_from(f'<{nd}H', d, q)); q += 2 * nd
        cand = list(struct.unpack_from(f'<{nc}H', d, q)); q += 2 * nc
        ex = []
        for _ in range(ne):
            (ln,) = struct.unpack_from('<I', d, q); ex.append(d[q + 4:q + 4 + ln]); q += 4 + ln
        (ln,) = struct.unpack_from('<I', d, q); ig = d[q + 4:q + 4 + ln]; q += 4 + ln
        yield iid, drop, cand, ex, ig


def sig_read(path):
    s = open(path, 'rb').read()
    out = {}
    q = 0
    while q + 12 <= len(s):
        iid, n = struct.unpack_from('<IH', s, q + 4)
        v = {}
        for k in range(n):
            m, pr = struct.unpack_from('<Hf', s, q + 12 + k * 6)
            v[m] = pr
        out[iid] = v
        q += 12 + n * 6
    return out


def _init():
    cv2.setNumThreads(1)
    pan = json.load(open(f'{COCO}/panoptic_train2017.json'))
    _G['cats'] = {c['id']: c['name'] for c in pan['categories']}
    _G['anns'] = {x['image_id']: x for x in pan['annotations']}
    _G['zip'] = zipfile.ZipFile(f'{COCO}/panoptic_train2017.zip')


def bbox(mk):
    ys, xs = np.nonzero(mk)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def one(job):
    r, (pid, drop, cand, ex, ig), ex10, sig = job
    assert r['id'] == pid
    x = _G['anns'][r['id']]
    png = cv2.imdecode(np.frombuffer(_G['zip'].read('panoptic_train2017/' + x['file_name']), np.uint8), cv2.IMREAD_COLOR)
    rgb = png[..., ::-1].astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    H, W = ids.shape
    N = H * W
    cat = {s['id']: _G['cats'][s['category_id']] for s in x['segments_info']}
    segarea = {s['id']: s['area'] for s in x['segments_info']}
    bgids = [i for i, c in cat.items() if c in BG]
    bg = np.isin(ids, bgids)
    obj = np.isin(ids, [i for i, c in cat.items() if c not in BG])
    ceil = np.isin(ids, [i for i, c in cat.items() if c == 'ceiling-merged'])
    wall = np.isin(ids, [i for i, c in cat.items() if c.startswith('wall')])
    m = ftl.idmap(r)
    sc = r['scores']
    K = len(sc)
    ignore = unrle(ig, N).reshape(H, W) if ig else np.zeros((H, W), bool)
    dset = set(drop)
    status = {}
    for k in range(K):
        if sc[k] < 0.5:
            continue
        if k in dset:
            status[k] = 'drop'
        elif sig.get(k, 0.0) >= 0.8:
            status[k] = 'siglip'
        else:
            status[k] = 'kept'
    keptmap = np.zeros((H, W), bool)
    for k, s in status.items():
        if s == 'kept':
            keptmap |= m == k
    out = dict(id=r['id'], H=H, W=W, n_all=K, n05=int((sc >= 0.5).sum()), n0205=int(((sc >= 0.2) & (sc < 0.5)).sum()),
               n_lt02=int((sc < 0.2).sum()), n_drop=len(drop), n_sig=sum(1 for s in status.values() if s == 'siglip'),
               n_kept=sum(1 for s in status.values() if s == 'kept'), n_ex=len(ex), n_ex10=len(ex10),
               has_ceil=bool(ceil.any()), ceil_frac=float(ceil.mean()), has_wall=bool(wall.any()),
               ignore_frac=float(ignore.mean()))
    # 정답 크기(원본 화소): 남은 선생 마스크 + v1 extra (+ v10 extra 따로)
    tg = []
    for k, s in status.items():
        if s != 'kept':
            continue
        mk = m == k
        a = int(mk.sum())
        if a == 0:
            continue
        x0, y0, x1, y1 = bbox(mk)
        tg.append((a, x1 - x0, y1 - y0))
    for e in ex:
        mk = unrle(e, N).reshape(H, W)
        if mk.any():
            x0, y0, x1, y1 = bbox(mk)
            tg.append((int(mk.sum()), x1 - x0, y1 - y0))
    out['targets'] = np.array(tg, np.int32).reshape(-1, 3)
    tg10 = []
    for e in ex10[len(ex):]:   # v1plus: v1 extra 뒤에 덧붙인 사람 정답
        mk = unrle(e, N).reshape(H, W)
        if mk.any():
            x0, y0, x1, y1 = bbox(mk)
            tg10.append((int(mk.sum()), x1 - x0, y1 - y0))
    out['targets10'] = np.array(tg10, np.int32).reshape(-1, 3)
    # Q1: 버려진(drop·siglip) 마스크 중 구조물 위 — 어느 구조물이 가장 많이 겹치나
    rem = []
    kept_struct = []
    for k, s in status.items():
        mk = m == k
        a = int(mk.sum())
        if a == 0:
            continue
        bgf = float(bg[mk].mean())
        if bgf < 0.5:
            continue
        u, c = np.unique(ids[mk & bg], return_counts=True)
        j = int(np.argmax(c))
        dom = cat[int(u[j])]
        cover = float(c[j] / segarea[int(u[j])])
        kind = 'ceil' if dom == 'ceiling-merged' else ('wall' if dom.startswith('wall') else 'other')
        x0, y0, x1, y1 = bbox(mk)
        row = dict(k=int(k), status=s, area=a, frac=a / N, cover=cover, dom=dom, kind=kind, sig=float(sig.get(k, -1)),
                   score=float(sc[k]), box=(x0, y0, x1, y1))
        if s == 'kept':
            kept_struct.append(row)
        else:
            rem.append(row)
    out['removed'] = rem
    out['kept_struct'] = kept_struct
    # Q2: 점수 0.2–0.5 마스크
    low = []
    ker = np.ones((5, 5), np.uint8)
    for k in np.where((sc >= 0.2) & (sc < 0.5))[0]:
        mk = m == k
        a = int(mk.sum())
        if a == 0:
            continue
        ring = cv2.dilate(mk.astype(np.uint8), ker).astype(bool) & ~mk
        rl = m[ring]
        nr = max(len(rl), 1)
        # 둘레에서 가장 많이 맞닿은 이웃 마스크
        lab, cnt = np.unique(rl, return_counts=True)
        j = int(np.argmax(cnt))
        nb = int(lab[j])
        nb_frac = float(cnt[j] / nr)
        nb_status = status.get(nb, 'low' if nb >= 0 else 'none')
        kept_ring = float(keptmap[ring].mean()) if ring.any() else 0.0
        nb_area = int((m == nb).sum()) if nb >= 0 else 0
        x0, y0, x1, y1 = bbox(mk)
        low.append(dict(k=int(k), score=float(sc[k]), area=a, frac=a / N, on_struct=float(bg[mk].mean()),
                        on_obj=float(obj[mk].mean()), on_ceil=float(ceil[mk].mean()), on_wall=float(wall[mk].mean()),
                        in_ignore=float(ignore[mk].mean()), nb=nb, nb_frac=nb_frac, nb_status=nb_status,
                        nb_area=nb_area, kept_ring=kept_ring, box=(x0, y0, x1, y1)))
    out['low'] = low
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every', type=int, default=20)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', default=f'{ROOT}/analysis/recall_research/label_stats.pkl')
    a = ap.parse_args()
    sig = sig_read(f'{ROOT}/data/teacher/train2017.siglip')
    val = set(json.load(open(f'{ROOT}/data/coco_shards/val_image_ids.json')))

    def jobs():
        P10 = plans(f'{ROOT}/data/teacher/train2017_v1plus.plan')
        for i, (r, p) in enumerate(zip(ftl.read(f'{ROOT}/data/teacher/train2017.ftl'), plans(f'{ROOT}/data/teacher/train2017.plan'))):
            p10 = next(P10)
            if i % a.every or r['id'] in val:
                continue
            assert p10[0] == r['id']
            yield r, p, p10[3], sig.get(r['id'], {})

    res = []
    with ProcessPoolExecutor(a.workers, initializer=_init) as pool:
        for o in pool.map(one, jobs(), chunksize=8):
            res.append(o)
            if len(res) % 500 == 0:
                print(len(res), flush=True)
    pickle.dump(res, open(a.out, 'wb'))
    print('done', len(res))


if __name__ == '__main__':
    main()
