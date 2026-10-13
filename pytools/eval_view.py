"""평가 그림: 원본 | 정답(초록 = 새 모델이 찾음, 빨강 = 놓침, 회색 = 구조물) | FastSAM-s 검출 | 새 모델 검출.

  .venv/bin/python pytools/eval_view.py --set sim_eval --n 6 --conf 0.05  → ~/Desktop/평가_<set>.jpg
놓친 비율이 큰 프레임을 고른다(FastSAM-s 는 찾고 새 모델은 놓친 정답 수 순).
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ext  # noqa: E402  (RA 경로를 sys.path 에 넣는다)
import eval_det as ev  # noqa: E402


def paint(rgb, masks, seed=1):
    out = rgb.astype(np.float32) * 0.45
    col = np.random.default_rng(seed).integers(60, 255, (max(len(masks), 1), 3))
    for k, m in enumerate(masks):
        out[m] = 0.35 * rgb[m] + 0.65 * col[k]
    out = out.astype(np.uint8)
    for m in masks:
        cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(out, cs, -1, (255, 255, 255), 1)
    return out


def label(t, s):
    cv2.putText(t, s, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(t, s, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='sim_eval')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--new', default='runs/main/final.pt@640')
    ap.add_argument('--out')
    a = ap.parse_args()
    out = a.out or os.path.expanduser(f'~/Desktop/평가_{a.set}.jpg')
    fs = eval_ext.make('weights/FastSAM-s.pt@640', 0.25)
    nw = eval_ext.make(a.new, a.conf)
    cand = []
    for s in ev.samples(a.set, None, a.every):
        if not s['gts']:
            continue
        d0, d1 = fs.detect(s['rgb']), nw.detect(s['rgb'])
        r0, r1 = ev.frame_eval(s, d0), ev.frame_eval(s, d1)
        miss = sum(g0['found'] and not g1['found'] for g0, g1 in zip(r0['gts'], r1['gts']))
        cand.append((miss, s, d0, d1, r0, r1))
        cand.sort(key=lambda c: -c[0])
        del cand[a.n:]
    tiles = []
    for miss, s, d0, d1, r0, r1 in cand:
        rgb = s['rgb']
        bgr = lambda x: np.ascontiguousarray(x[..., ::-1])
        gt = rgb.astype(np.float32) * 0.45
        gt[s['struct']] = 0.5 * rgb[s['struct']] + 0.5 * np.array([90, 90, 90.])
        for (m, *_), g in zip(s['gts'], r1['gts']):
            c = np.array([40, 220, 40.]) if g['found'] else np.array([230, 40, 40.])
            gt[m] = 0.35 * rgb[m] + 0.65 * c
        gt = gt.astype(np.uint8)
        nf0 = sum(g['found'] for g in r0['gts'])
        nf1 = sum(g['found'] for g in r1['gts'])
        ng = len(s['gts'])
        p = [bgr(rgb), bgr(gt), bgr(paint(rgb, d0['masks'])), bgr(paint(rgb, d1['masks']))]
        label(p[0], s['id'])
        label(p[1], f'GT {ng}: green found / red missed (new)')
        label(p[2], f'FastSAM-s: {r0["n"]} dets, found {nf0}/{ng}, wall {r0["struct_fp"]}')
        label(p[3], f'new@{a.conf}: {r1["n"]} dets, found {nf1}/{ng}, wall {r1["struct_fp"]}')
        t = np.concatenate([np.concatenate(p[:2], 1), np.concatenate(p[2:], 1)], 0)
        tiles.append(cv2.resize(t, (1400, int(1400 * t.shape[0] / t.shape[1]))))
    cv2.imwrite(out, np.concatenate([cv2.copyMakeBorder(t, 6, 6, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                                     for t in tiles], 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == '__main__':
    main()
