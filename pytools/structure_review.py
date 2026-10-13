"""구조물 빼기 훑어보기(SPEC 4 절): 선생 마스크 중 COCO 배경이 대부분인 것을 SigLIP2 로 구조물/물체 판정해 그림으로.

  .venv-teacher/bin/python pytools/structure_review.py --n 8 --out ~/Desktop/구조물_판정.jpg

빨강 = 구조물로 뺌: COCO 배경 ≥ bg_frac 이고, 그 배경 조각을 cover 이상 덮거나(글자 C) SigLIP2 구조물 확률 합 ≥ thr.
노랑 = COCO 배경 대부분이지만 물체로 남김,
나머지 색 = 물체. 선생 점수 ≥ score 인 마스크만.
"""
import argparse
import json
import os
import sys
import zipfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftl  # noqa: E402
from coco_build import BG  # noqa: E402

COCO = 'data/coco'
SIGLIP = 'weights/siglip2-giant-opt-patch16-384'
STRUCT = {
    'wall': ['a wall', 'a painted wall', 'a brick wall', 'a tiled wall', 'a wooden wall'],
    'ceiling': ['a ceiling', 'a ceiling with lights'],
    'floor': ['a floor', 'a wooden floor', 'a tiled floor', 'a carpeted floor', 'the ground', 'a road', 'grass', 'sand',
              'snow', 'a sidewalk'],
    'sky': ['the sky', 'a cloudy sky'],
}
OBJECT = ['an object', 'a picture frame', 'a painting', 'a poster', 'a light switch', 'a power outlet', 'a window', 'a door',
          'a bag', 'a rug', 'a mirror', 'a shelf', 'a sign', 'a clock', 'a lamp', 'a piece of furniture', 'a person',
          'a plant', 'a box', 'a towel', 'a curtain', 'a cabinet', 'a building', 'a tree', 'a car', 'a table']


class Judge:
    def __init__(self):
        from transformers import AutoModel, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(SIGLIP)
        self.model = AutoModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).cuda().eval()
        self.texts = [t for v in STRUCT.values() for t in v] + OBJECT
        self.n_struct = sum(len(v) for v in STRUCT.values())
        ti = self.proc(text=[f'This is a photo of {t}.' for t in self.texts], padding='max_length', max_length=64,
                       return_tensors='pt').to('cuda')
        with torch.no_grad():
            self.tf = torch.nn.functional.normalize(self.model.get_text_features(**ti).float(), dim=-1)

    @torch.no_grad()
    def struct_prob(self, crops):
        """crops: RGB uint8 목록 → 구조물(벽·천장·바닥·하늘) 확률의 합(문장 전체 softmax)."""
        out = []
        for i in range(0, len(crops), 32):
            pi = self.proc(images=crops[i:i + 32], return_tensors='pt').to('cuda')
            f = torch.nn.functional.normalize(self.model.get_image_features(pixel_values=pi['pixel_values'].half()).float(),
                                              dim=-1)
            logits = f @ self.tf.T * self.model.logit_scale.exp().float() + self.model.logit_bias.float()
            p = logits.softmax(-1)
            out += p[:, :self.n_struct].sum(-1).tolist()
        return out


def crop_of(img_rgb, m):
    """마스크 상자를 정사각형으로 넓혀 자르고 마스크 밖은 회색."""
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    H, W = m.shape
    out = np.full((s, s, 3), 128, np.uint8)
    ya, xa = cy - s // 2, cx - s // 2
    for_y = slice(max(ya, 0), min(ya + s, H))
    for_x = slice(max(xa, 0), min(xa + s, W))
    sub = img_rgb[for_y, for_x].copy()
    sub[~m[for_y, for_x]] = 128
    out[for_y.start - ya:for_y.stop - ya, for_x.start - xa:for_x.stop - xa] = sub
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--score', type=float, default=0.5)
    ap.add_argument('--bg_frac', type=float, default=0.5)
    ap.add_argument('--thr', type=float, default=0.8)
    ap.add_argument('--cover', type=float, default=0.3)
    ap.add_argument('--scene', default='indoor', choices=['indoor', 'outdoor'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.expanduser('~/Desktop/구조물_판정.jpg'))
    a = ap.parse_args()
    pan = json.load(open(f'{COCO}/panoptic_train2017.json'))
    cats = {c['id']: c['name'] for c in pan['categories']}
    anns = {x['image_id']: x for x in pan['annotations']}
    z = zipfile.ZipFile(f'{COCO}/panoptic_train2017.zip')
    # 벽·바닥·천장이 함께 있는 실내 사진에서 고른다
    want = []
    for iid, x in anns.items():
        names = {cats[s['category_id']] for s in x['segments_info']}
        if a.scene == 'indoor':
            ok = any(n.startswith('wall') for n in names) and any(n.startswith('floor') for n in names) and 'ceiling-merged' in names
        else:
            ok = 'sky-other-merged' in names and bool(names & {'road', 'pavement-merged', 'grass-merged', 'dirt-merged', 'sand',
                                                                   'snow', 'playingfield', 'gravel'})
        if ok:
            want.append(iid)
    rng = np.random.default_rng(a.seed)
    pick = set(rng.choice(sorted(want), a.n, replace=False).tolist())
    recs = [r for r in ftl.read('data/teacher/train2017.ftl') if r['id'] in pick]
    judge = Judge()
    tiles, stats = [], dict(masks=0, cand=0, removed=0)
    for r in recs:
        x = anns[r['id']]
        png = cv2.imdecode(np.frombuffer(z.read('panoptic_train2017/' + x['file_name']), np.uint8), cv2.IMREAD_COLOR)
        rgb = png[..., ::-1].astype(np.int64)
        ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
        bgmask = np.zeros(ids.shape, bool)
        seg_area = {}
        for s in x['segments_info']:
            if cats[s['category_id']] in BG:
                bgmask |= ids == s['id']
                seg_area[s['id']] = s['area']
        img = cv2.imread(f"{COCO}/train2017/{r['id']:012d}.jpg")
        m = ftl.idmap(r)
        keep = [k for k in range(len(r['scores'])) if r['scores'][k] >= a.score]
        cand, crops, big = [], [], set()
        for k in keep:
            mk = m == k
            if bgmask[mk].mean() >= a.bg_frac:
                # 가장 많이 겹치는 COCO 배경 조각을 얼마나 덮나
                u, c = np.unique(ids[mk & bgmask], return_counts=True)
                j = int(np.argmax(c))
                if c[j] / seg_area[int(u[j])] >= a.cover:
                    big.add(k)
                else:
                    cand.append(k)
                    crops.append(crop_of(img[..., ::-1], mk))
        probs = judge.struct_prob(crops) if crops else []
        verdict = {k: p for k, p in zip(cand, probs)}
        verdict.update({k: 1.0 for k in big})
        vis = img.astype(np.float32)
        col = np.random.default_rng(1)
        labels = []
        for k in keep:
            mk = m == k
            if k in verdict:
                c = np.array([0, 0, 255.]) if verdict[k] >= a.thr else np.array([0, 230, 255.])
            else:
                c = col.integers(60, 200, 3) * np.array([0.6, 1.0, 0.6])
            vis[mk] = 0.45 * vis[mk] + 0.55 * c
            cv2.drawContours(vis, cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                             (255, 255, 255), 1)
            if k in verdict:
                ys, xs = np.nonzero(mk)
                labels.append(('C' if k in big else f'{verdict[k]:.2f}', (int(xs.mean()) - 15, int(ys.mean()))))
        stats['masks'] += len(keep)
        stats['cand'] += len(cand) + len(big)
        stats['removed'] += sum(p >= a.thr for p in probs) + len(big)
        vis = vis.astype(np.uint8)
        for txt, pos in labels:
            cv2.putText(vis, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(vis, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        t = np.concatenate([img, vis], 1)
        t = cv2.resize(t, (960, int(960 * t.shape[0] / t.shape[1])))
        tiles.append(t)
    sheet = np.concatenate([cv2.copyMakeBorder(t, 4, 4, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles], 0)
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(a.out, stats)


if __name__ == '__main__':
    main()
