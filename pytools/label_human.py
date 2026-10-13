"""판정 파일(사람 라벨 + CropFormer 채우기, 2026-10-11): 이전 ObjectSAM 방식에서 FastSAM-s 자리에 CropFormer 선생.

  .venv/bin/python pytools/label_human.py [--workers 24]  → data/teacher/train2017_human.plan (FTP1, 학습기가 그대로 읽음)

사진마다
1. 사람 정답 통째 마스크(extra RLE, 정답끼리 합치지 않음):
   COCO panoptic things(crowd 제외) + 문·창·계단 stuff(door-stuff, window-other, stairs)
   + LVIS v1 인스턴스(panoptic 정답과 IoU > 0.7 이면 LVIS 쪽 버림).
2. CropFormer 마스크(점수 ≥ 0.5) 중 다음은 뺀다(drop):
   - 화소 50 % 이상이 벽·천장·바닥(바깥 땅 포함, label_plan.WCF 18 종)
   - 50 % 이상이 정답 하나 위, 또는 70 % 이상이 정답 합집합 위(정답 조각; 정답 통째가 대신)
   - 정답 하나를 그 넓이의 20 % 이상 덮음(정답에 벽·이웃 물체가 붙은 덩어리)
   남은 것은 물체(사람 라벨이 없는 물체).
3. 무시 영역: COCO crowd 조각.
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
from pycocotools import mask as mu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402
from coco_build import leb128, rle_runs  # noqa: E402
from label_plan import WCF  # noqa: E402

COCO = 'data/coco'
LVIS = os.path.expanduser('~/robot-agent/training/data/embed/data/lvis_v1_train.json')
DWS = {'door-stuff', 'window-other', 'stairs'}
# --obj-stuff: 물체 같은 stuff 도 정답(연결 성분마다). 2026-10-12 — 조명 조각의 69 % 가 CropFormer 마스크(IoU ≥ 0.5)가 없어
# 학습에서 배경으로 남았다(1,500 장 표본). 벽·천장·바닥·땅·하늘·건물·나무·물 같은 덩어리 stuff 는 넣지 않는다.
OBJ_STUFF = {'light', 'mirror-stuff', 'pillow', 'towel', 'blanket', 'curtain', 'shelf', 'cabinet-merged', 'counter',
             'table-merged', 'window-blind', 'banner', 'cardboard'}
MIN_PIECE = 200
# --keep-parts 에서 빼는 정답(사람·동물): 로봇이 부품 단위로 조작할 대상이 아니고, 옷·팔다리 조각이 사람을 쪼개도록 가르친다(표본 그림 2026-10-12)
LIVING = {'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}
_G = {}


def _init(a):  # 작업자는 zip 만 연다. 주석은 사진마다 job 으로 받는다(메모리)
    cv2.setNumThreads(1)
    _G['zip'] = zipfile.ZipFile(f'{COCO}/panoptic_train2017.zip')
    _G['a'] = a


def rle(m):
    return leb128(rle_runs(m)) if m.any() else b''


def lvis_mask(seg, H, W):
    if isinstance(seg, list):
        r = mu.merge(mu.frPyObjects(seg, H, W))
    else:
        r = mu.frPyObjects(seg, H, W) if isinstance(seg['counts'], list) else seg
    return mu.decode(r).astype(bool)


def one(job):
    r, x, lv = job
    a = _G['a']
    png = cv2.imdecode(np.frombuffer(_G['zip'].read('panoptic_train2017/' + x['file_name']), np.uint8), cv2.IMREAD_COLOR)
    rgb = png[..., ::-1].astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    H, W = ids.shape
    gts, crowd, wcf_ids = [], np.zeros((H, W), bool), []
    living = []   # 정답이 사람·동물인가(--keep-parts 는 무생물 정답의 부품만)
    ceil_ids = []
    for s in x['segments_info']:
        if s['name'] in WCF:
            wcf_ids.append(s['id'])
            if s['name'] == 'ceiling-merged':
                ceil_ids.append(s['id'])
        elif s.get('iscrowd'):
            crowd |= ids == s['id']
        elif a.split_dws and s['name'] in DWS:
            # --split-dws: 문·창·계단 정답을 떨어진 덩어리마다 하나씩(작은 조각은 무시 영역)
            n_, lab = cv2.connectedComponents((ids == s['id']).astype(np.uint8), connectivity=8)
            for k_ in range(1, n_):
                pc = lab == k_
                if pc.sum() >= MIN_PIECE:
                    gts.append(pc); living.append(False)
                else:
                    crowd |= pc
        elif s['isthing'] or s['name'] in DWS:
            gts.append(ids == s['id'])
            living.append(s['name'] in LIVING)
        elif a.obj_stuff and s['name'] in OBJ_STUFF:
            n, lab = cv2.connectedComponents((ids == s['id']).astype(np.uint8), connectivity=8)
            for k in range(1, n):
                if (lab == k).sum() >= MIN_PIECE:
                    gts.append(lab == k)
                    living.append(False)
    npan = len(gts)
    st = dict(gt_pan=npan, gt_lvis=0, lvis_dup=0, t_obj=0, t_struct=0, t_frag=0, t_blob=0, t_part=0, t_clip=0, t_ceil=0)
    for seg in lv:
        m = lvis_mask(seg, H, W)
        if not m.any():
            continue
        if any((m & g).sum() / max((m | g).sum(), 1) > 0.7 for g in gts[:npan]):
            st['lvis_dup'] += 1
            continue
        gts.append(m)
        living.append(False)
        st['gt_lvis'] += 1
    wcf = np.isin(ids, wcf_ids)
    G = np.stack(gts).reshape(len(gts), -1) if gts else np.zeros((0, H * W), bool)
    garea = G.sum(1)
    gunion = G.any(0)
    tm = ftl.idmap(r).reshape(-1)
    drop = []
    orphan_from = []   # --ignore-orphans: frag/blob 로 버린 마스크(정답에 안 덮인 나머지를 무시 영역으로)
    clipped = []   # --clip-struct: 구조물에 걸친 마스크에서 구조물 화소를 잘라낸 것(정답 목록 뒤에 붙인다)
    for k in np.where(r['scores'] >= a.score)[0]:
        mk = tm == k
        n = mk.sum()
        if n == 0:
            continue
        if wcf.reshape(-1)[mk].mean() >= 0.5:
            if a.ceil_small and ceil_ids:
                cl = np.isin(ids, ceil_ids).reshape(-1)
                if cl[mk].mean() >= 0.5 and n < a.ceil_small * max(cl.sum(), 1):
                    st['t_ceil'] = st.get('t_ceil', 0) + 1   # --ceil-small: 천장에 붙은 작은 것(조명·환기구·표지판) → 물체로 남긴다
                    continue
            drop.append(int(k)); st['t_struct'] += 1
            continue
        inter = G[:, mk].sum(1) if len(G) else np.zeros(0)
        j = int(np.argmax(inter)) if len(G) else -1
        if a.keep_parts and j >= 0 and not living[j] and inter[j] / n >= 0.8 and n >= 100 and a.part_lo <= inter[j] / garea[j] <= 0.7:
            st['t_part'] = st.get('t_part', 0) + 1   # --keep-parts: 정답 하나 안의 부품(정답 넓이의 part_lo~70 %) → 남긴다
            continue
        if len(G) and (inter.max() / n >= 0.5 or gunion[mk].mean() >= 0.7):
            drop.append(int(k)); st['t_frag'] += 1
            orphan_from.append(mk)
            continue
        if len(G) and (inter / np.maximum(garea, 1)).max() >= 0.2:
            drop.append(int(k)); st['t_blob'] += 1
            orphan_from.append(mk)
            continue
        wf = wcf.reshape(-1)[mk].mean()
        if a.clip_struct and wf >= a.clip_struct:
            m2 = (mk & ~wcf.reshape(-1)).reshape(H, W)
            drop.append(int(k)); st['t_clip'] = st.get('t_clip', 0) + 1
            if m2.sum() >= 100:
                clipped.append(m2)
            continue
        st['t_obj'] += 1
    extra = [rle(g) for g in gts] + [rle(m) for m in clipped]
    rec = [b'FTP1', struct.pack('<IHHHH', r['id'], len(drop), 0, len(extra), 0), struct.pack(f'<{len(drop)}H', *drop)]
    for e in extra:
        rec += [struct.pack('<I', len(e)), e]
    if a.ignore_orphans:
        free = ~(gunion | wcf.reshape(-1))
        orph = np.zeros(H * W, bool)
        for mk in orphan_from:
            rest = mk & free
            if 0 < rest.sum() < 0.05 * H * W:
                orph |= rest
        for k in np.where((r['scores'] >= 0.2) & (r['scores'] < a.score))[0]:   # 점수 0.2~0.5 선생 마스크의 안 덮인 부분
            mk = tm == k
            rest = mk & free
            if 0 < rest.sum() < 0.05 * H * W:
                orph |= rest
        # 남은 선생 마스크(정답으로 쓰는 것)와 겹치는 곳은 무시하지 않는다
        kept = np.isin(tm, [k for k in np.where(r['scores'] >= a.score)[0] if int(k) not in set(drop)])
        orph &= ~kept
        st['orphan_px'] = int(orph.sum())
        crowd = crowd | orph.reshape(H, W)
    ig = rle(crowd)
    rec += [struct.pack('<I', len(ig)), ig]
    st['ignore_px'] = int(crowd.sum())
    st['px'] = H * W
    return b''.join(rec), st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score', type=float, default=0.5)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=-1)
    ap.add_argument('--obj-stuff', action='store_true')
    ap.add_argument('--keep-parts', action='store_true')   # 2026-10-12: GT 가 부품 단위(로봇 조작) — 정답 안 부품 마스크를 통째 정답과 함께 가르친다
    ap.add_argument('--part-lo', type=float, default=0.01)
    # 2026-10-12: 남는 CropFormer 마스크 중 장당 0.79 개가 구조물에 20~50 % 걸친 '물체+벽' 덩어리(표본 694 장).
    # uHumans2 에서 놓친 물체의 대부분이 창틀+벽·천장, 커튼+벽 덩어리에 삼켜졌다 → 그 몫(구조물 화소)만 잘라낸다.
    ap.add_argument('--clip-struct', type=float, default=0.0)
    # 2026-10-12: 천장 있는 COCO 사진에서 장당 1.52 개의 작은 CropFormer 마스크(천장 조각 넓이의 20 % 미만, 천장 위 50 % 이상)가
    # '구조물'로 버려졌다 — 그림으로 보니 조명·형광등 판·표지판·환기구(표본 541 장). → 천장 조각 넓이의 이 비율 미만이면 물체로 둔다.
    ap.add_argument('--ceil-small', type=float, default=0.0)
    # 2026-10-12 개선안 조사(analysis/improve_research/REPORT.md):
    ap.add_argument('--ignore-orphans', action='store_true')   # 1 위: 버린 선생 마스크의 남은 부분(사진 5 % 미만)·점수 0.2~0.5 마스크 → 무시 영역
    ap.add_argument('--split-dws', action='store_true')        # 3 위: 문·창·계단 정답을 덩어리별로(창 정답 47 %·문 31 % 가 여러 덩어리)
    ap.add_argument('--lvis-val', action='store_true')         # 4 위: lvis_v1_val.json 의 train2017 사진 정답도 읽는다(약 1.4 만 장, 19 만 개)
    ap.add_argument('--out', default='data/teacher/train2017_human.plan')
    a = ap.parse_args()
    lv = json.load(open(LVIS))
    by = defaultdict(list)
    for x in lv['annotations']:
        by[x['image_id']].append(x['segmentation'])
    del lv
    if a.lvis_val:
        lv = json.load(open(LVIS.replace('lvis_v1_train', 'lvis_v1_val')))
        for x in lv['annotations']:
            by[x['image_id']].append(x['segmentation'])   # val2017 사진은 학습 목록에 없어 자연히 빠진다
        del lv
    pan = json.load(open(f'{COCO}/panoptic_train2017.json'))
    cats = {c['id']: c for c in pan['categories']}
    anns = {}
    for x in pan['annotations']:
        anns[x['image_id']] = {'file_name': x['file_name'], 'segments_info': [
            {'id': s['id'], 'iscrowd': s.get('iscrowd', 0), 'name': cats[s['category_id']]['name'],
             'isthing': cats[s['category_id']]['isthing']} for s in x['segments_info']]}
    del pan
    import gc
    gc.collect()
    jobs = ((r, anns[r['id']], by.get(r['id'], [])) for r in ftl.read('data/teacher/train2017.ftl'))
    if a.limit > 0:
        jobs = (j for _, j in zip(range(a.limit), jobs))
    tot, n = {}, 0
    with open(a.out + '.tmp', 'wb') as f, ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a,)) as pool:
        for rec, st in pool.map(one, jobs, chunksize=32):
            f.write(rec)
            n += 1
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
    os.replace(a.out + '.tmp', a.out)
    json.dump({'args': vars(a), 'images': n, **tot}, open(a.out + '.json', 'w'), indent=1)
    print(f'{a.out}: {n} images', {k: round(v / n, 2) for k, v in tot.items() if k not in ('px', 'ignore_px')},
          f"(per image) ignore {tot['ignore_px'] / tot['px'] * 100:.2f}% of pixels")


if __name__ == '__main__':
    main()
