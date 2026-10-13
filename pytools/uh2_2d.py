"""uHumans2 2D 마스크 평가(물체 GT 만): 키프레임 GT 인스턴스(runs/frontend_eval/gt_labels_2d)와 예측 마스크를 IoU ≥ 0.5 로 1:1 매칭.

  .venv/bin/python pytools/uh2_2d.py <이름>=<모델.pt> ... [--conf 0.1 0.4] [--step 2]
GT 범주 = ~/meridian_ws/tools/kimera/gt_class.json (obj_recall.py 가 만든 것).
- 물체 GT = 벽·천장·바닥·unknown 이 아닌 인스턴스.
- 예측이 화소 50 % 이상 벽·천장·바닥(struct), unknown, 또는 라벨 없음 위면 FP 에서 뺀다(무시). struct 위 예측 수는 따로 센다.
- P = TP / (TP + FP), R = TP / 물체 GT, F1, mIoU = 매칭된 쌍 평균 IoU(SQ), bestIoU = 물체 GT 마다 가장 잘 맞는 예측의 IoU 평균.
추론은 frontend 와 같게 imgsz 1024, NMS IoU 0.7, 면적 ≥ 100 화소(640 기준; frontend 1024 기준 256).
"""
import argparse
import collections
import glob
import json
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

WS = os.path.expanduser('~/meridian_ws')
GT = f'{WS}/runs/frontend_eval/gt_labels_2d'
STRUCT = {'wall', 'ceiling', 'floor'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--conf', type=float, nargs='+', default=[0.1, 0.4])
    ap.add_argument('--step', type=int, default=2)
    a = ap.parse_args()
    cls = json.load(open(f'{WS}/tools/kimera/gt_class.json'))
    seqs = sorted(os.path.basename(d)[len('uHumans2_'):] for d in glob.glob(f'{GT}/uHumans2_*'))
    lo = min(a.conf)
    for spec in a.models:
        name, path = spec.split('=')
        m = YOLO(path)
        st = {c: collections.Counter() for c in a.conf}
        ious = {c: [] for c in a.conf}
        best = {c: [] for c in a.conf}
        for seq in seqs:
            cl = cls[seq]
            imgs = sorted(glob.glob(f'{WS}/datasets/unpacked/uHumans2_{seq}/left_cam_rgb/*'))
            for p in sorted(glob.glob(f'{GT}/uHumans2_{seq}/*.png'))[::a.step]:
                L = cv2.imread(p, -1)
                img = np.ascontiguousarray(cv2.imread(imgs[int(os.path.basename(p)[:-4])])[:, 40:680])  # frontend 처럼 가운데 640 폭
                r = m.predict(img, imgsz=1024, conf=lo, iou=0.7, retina_masks=True, max_det=300, verbose=False)[0]
                if r.masks is None:
                    P, cf = np.zeros((0,) + L.shape, bool), np.zeros(0)
                else:
                    P = r.masks.data.cpu().numpy() > 0.5
                    cf = r.boxes.conf.cpu().numpy()
                    keep = P.reshape(len(P), -1).sum(1) >= 100
                    P, cf = P[keep], cf[keep]
                ids = [e for e in np.unique(L) if e]
                kind = {e: cl.get(str(e - 1), 'unknown') for e in ids}
                obj = [e for e in ids if kind[e] not in STRUCT and kind[e] != 'unknown']
                ign = np.isin(L, [0] + [e for e in ids if kind[e] == 'unknown'])
                sm = np.isin(L, [e for e in ids if kind[e] in STRUCT])
                G = np.stack([L == e for e in obj]).reshape(len(obj), -1) if obj else np.zeros((0, L.size), bool)
                Pf = P.reshape(len(P), -1)
                inter = Pf.astype(np.float32) @ G.T.astype(np.float32) if len(P) and len(G) else np.zeros((len(P), len(G)))
                iou = inter / np.maximum(Pf.sum(1)[:, None] + G.sum(1)[None, :] - inter, 1)
                on_s = (Pf & sm.reshape(-1)).sum(1) / np.maximum(Pf.sum(1), 1)
                on_i = (Pf & ign.reshape(-1)).sum(1) / np.maximum(Pf.sum(1), 1)
                for c in a.conf:
                    sel = np.where(cf >= c)[0]
                    io = iou[sel]
                    used, matched = set(), {}
                    for k in np.argsort(-io, axis=None):  # 큰 IoU 부터 탐욕 1:1
                        i, j = np.unravel_index(k, io.shape)
                        if io[i, j] < 0.5:
                            break
                        if i in used or j in matched:
                            continue
                        used.add(i); matched[j] = io[i, j]
                    t = st[c]
                    t['gt'] += len(obj); t['tp'] += len(matched); ious[c] += list(matched.values())
                    best[c] += list(io.max(0)) if len(sel) else [0.0] * len(obj)
                    for n, i in enumerate(sel):
                        if n in used:
                            continue
                        if on_s[i] >= 0.5:
                            t['struct'] += 1
                        elif on_i[i] >= 0.5:
                            t['ign'] += 1
                        else:
                            t['fp'] += 1
                    t['pred'] += len(sel); t['frames'] += 1
                    t[seq + '_gt'] += len(obj); t[seq + '_tp'] += len(matched)
        for c in a.conf:
            t = st[c]
            P_ = t['tp'] / max(t['tp'] + t['fp'], 1); R_ = t['tp'] / max(t['gt'], 1)
            per = ' '.join(f"{s.replace('_s1_', '')} {t[s + '_tp'] / max(t[s + '_gt'], 1) * 100:.1f}" for s in seqs)
            print(f"{name:10s} @{c:.1f} | P {P_ * 100:.1f} R {R_ * 100:.1f} F1 {2 * P_ * R_ / max(P_ + R_, 1e-9) * 100:.1f} "
                  f"| mIoU {np.mean(ious[c]) * 100:.1f} bestIoU {np.mean(best[c]) * 100:.1f} "
                  f"| pred/fr {t['pred'] / t['frames']:.1f} struct/fr {t['struct'] / t['frames']:.1f} | R: {per}", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
