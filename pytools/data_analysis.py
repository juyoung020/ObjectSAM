"""학습 데이터 분석(COCO 표본): 학습 정답의 크기 분포, 선생(CropFormer) 마스크가 버려지는 이유와 크기,
사람 라벨이 없는 곳을 CropFormer 와 FastSAM-s 가 얼마나 덮는지.

  .venv/bin/python pytools/data_analysis.py [--n 800] [--plan data/teacher/train2017_human.plan]
판정 규칙은 label_human.py 와 같게 다시 계산한다(버림 이유를 알려고).
"""
import argparse
import collections
import json
import os
import random
import sys
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402
from label_human import DWS, LVIS, OBJ_STUFF, lvis_mask  # noqa: E402
from label_plan import WCF  # noqa: E402

COCO = 'data/coco'
SIZE = [0, 0.001, 0.005, 0.02, 0.1, 1.01]   # 화면 넓이 비율 구간
SN = ['<0.1%', '0.1-0.5%', '0.5-2%', '2-10%', '>10%']


def sbin(f):
    return next(i for i in range(5) if f < SIZE[i + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--fastsam', default=os.path.expanduser('~/meridian_ws/models/fastsam/FastSAM-s.pt'))
    a = ap.parse_args()
    random.seed(0)
    pan = json.load(open(f'{COCO}/panoptic_train2017.json'))
    cats = {c['id']: c for c in pan['categories']}
    ann = {x['image_id']: x for x in pan['annotations']}
    del pan
    recs = list(ftl.read('data/teacher/train2017.ftl'))
    pick = random.sample(recs, a.n)
    del recs
    ids = {r['id'] for r in pick}
    lv = collections.defaultdict(list)
    lvc = {}
    L = json.load(open(LVIS))
    lvc = {c['id']: c['name'] for c in L['categories']}
    for x in L['annotations']:
        if x['image_id'] in ids:
            lv[x['image_id']].append((x['segmentation'], lvc[x['category_id']]))
    del L
    z = zipfile.ZipFile(f'{COCO}/panoptic_train2017.zip')
    from ultralytics import YOLO
    fs = YOLO(a.fastsam)

    S = collections.Counter()
    gt_size = collections.Counter(); t_size = collections.Counter(); drop_size = collections.defaultdict(collections.Counter)
    for r in pick:
        x = ann[r['id']]
        png = cv2.imdecode(np.frombuffer(z.read('panoptic_train2017/' + x['file_name']), np.uint8), 1)[..., ::-1].astype(np.int64)
        pid = png[..., 0] + 256 * png[..., 1] + 65536 * png[..., 2]
        H, W = pid.shape
        px = H * W
        gts, gname, crowd, wcf = [], [], np.zeros((H, W), bool), np.zeros((H, W), bool)
        stuffobj = np.zeros((H, W), bool)
        for s in x['segments_info']:
            nm = cats[s['category_id']]['name']
            m = pid == s['id']
            if nm in WCF:
                wcf |= m
            elif s.get('iscrowd'):
                crowd |= m
            elif cats[s['category_id']]['isthing'] or nm in DWS:
                gts.append(m); gname.append(nm)
            elif nm in OBJ_STUFF:
                stuffobj |= m
        npan = len(gts)
        for seg, nm in lv[r['id']]:
            m = lvis_mask(seg, H, W)
            if m.any() and not any((m & g).sum() / max((m | g).sum(), 1) > 0.7 for g in gts[:npan]):
                gts.append(m); gname.append('lvis:' + nm)
        G = np.stack(gts).reshape(len(gts), -1) if gts else np.zeros((0, px), bool)
        garea = G.sum(1); gunion = G.any(0).reshape(H, W)
        for g in garea:
            gt_size[sbin(g / px)] += 1
        tm = ftl.idmap(r)
        kept = np.zeros((H, W), bool)
        for k in np.where(r['scores'] >= 0.5)[0]:
            mk = tm == k
            n = mk.sum()
            if n == 0:
                continue
            b = sbin(n / px)
            mf = mk.reshape(-1)
            if wcf[mk].mean() >= 0.5:
                why = 'struct'
            else:
                inter = G[:, mf].sum(1) if len(G) else np.zeros(0)
                if len(G) and (inter.max() / n >= 0.5 or gunion[mk].mean() >= 0.7):
                    why = 'frag'
                elif len(G) and (inter / np.maximum(garea, 1)).max() >= 0.2:
                    why = 'blob'
                    # 같은 범주 정답 여럿을 덮는 덩어리(책 줄 등)인가
                    cov = [gname[j] for j in np.flatnonzero(inter / np.maximum(garea, 1) >= 0.2)]
                    base = {c.split(':')[-1] for c in cov}
                    if len(cov) >= 2 and len(base) == 1:
                        S['blob_samecls'] += 1
                        S['blob_samecls_' + base.pop()] += 1
                else:
                    why = 'kept'
                    kept |= mk
            drop_size[why][b] += 1
            t_size[b] += 1
        # 사람 라벨 없는 곳(정답·구조물·crowd 밖)을 누가 덮나
        free = ~(gunion | wcf | crowd)
        S['free_px'] += int(free.sum()); S['px'] += px
        S['free_cf'] += int((free & kept).sum())
        S['free_stuffobj'] += int((free & stuffobj).sum())
        img = cv2.imread(f'{COCO}/train2017/{r["id"]:012d}.jpg')
        if img is None:
            continue
        res = fs.predict(img, imgsz=640, conf=0.4, iou=0.7, retina_masks=True, verbose=False)[0]
        fsu = np.zeros((H, W), bool)
        if res.masks is not None:
            for m in res.masks.data.cpu().numpy() > 0.5:
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                n = m.sum()
                if n and (m & free).sum() / n >= 0.5 and wcf[m].mean() < 0.5:   # 빈 곳에 주로 있는 물체 같은 마스크
                    fsu |= m
                    S['fs_free_masks'] += 1
                    S[f'fs_free_size{sbin(n / px)}'] += 1
        S['free_fs'] += int((free & fsu).sum())
        S['free_both'] += int((free & fsu & kept).sum())
        S['imgs'] += 1
    n = S['imgs']
    print(f'표본 {n} 장')
    print('== 학습 정답(사람 라벨) 크기 분포 (장당 개수)')
    print('  ' + ' '.join(f'{SN[b]} {gt_size[b] / n:.2f}' for b in range(5)))
    print('== CropFormer 마스크(점수 ≥ 0.5) 처리 — 크기별 장당 개수 [남김 / 구조물 / 정답 조각 / 덩어리]')
    for b in range(5):
        print(f'  {SN[b]:9s} ' + ' / '.join(f"{drop_size[w][b] / n:.2f}" for w in ('kept', 'struct', 'frag', 'blob')))
    print(f"  덩어리 중 같은 범주 정답 여럿을 덮은 것(책 줄 같은): 장당 {S['blob_samecls'] / n:.3f}, 범주 상위:",
          sorted(((k[13:], v) for k, v in S.items() if k.startswith('blob_samecls_')), key=lambda t: -t[1])[:8])
    print('== 사람 라벨 없는 곳(정답·구조물·crowd 밖) 덮기')
    f = S['free_px']
    print(f"  빈 곳 = 화면의 {f / S['px'] * 100:.1f}% · CropFormer 남김이 덮음 {S['free_cf'] / f * 100:.1f}% · FastSAM-s 가 덮음 {S['free_fs'] / f * 100:.1f}% "
          f"· 둘 다 {S['free_both'] / f * 100:.1f}% · 물체 같은 stuff(--obj-stuff 대상) {S['free_stuffobj'] / f * 100:.1f}%")
    print(f"  FastSAM-s 빈 곳 마스크 장당 {S['fs_free_masks'] / n:.2f} (크기별 " + ' '.join(f"{SN[b]} {S[f'fs_free_size{b}'] / n:.2f}" for b in range(5)) + ')')
    print(f"  CropFormer 남김 장당 {sum(drop_size['kept'].values()) / n:.2f}")


if __name__ == '__main__':
    main()
