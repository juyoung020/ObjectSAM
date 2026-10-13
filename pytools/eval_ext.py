"""외부 평가셋(정답 있음)에서 모델 비교 — robot-agent ObjectSAM 평가 도구(eval_det.py)를 그대로 쓰고 검출기만 GPU 판.

  .venv/bin/python pytools/eval_ext.py --set coco_val --every 2 --out data/eval/coco_val.json \
      --models fs1024=weights/FastSAM-s.pt@1024 objsam=$RA/.../ObjectSAM-n-416.onnx ours=runs/r1/ours.pt@1024

세트: sim_eval(BEHAVIOR 학습에 안 쓴 집 10 곳), coco_val(1,500), ade_val(실내 1,021).
지표(eval_det.frame_eval): 재현율(검출 마스크 픽셀 50 % 이상이 정답 위), 통째 IoU ≥ 0.5, 벽·천장·바닥 헛검출/프레임,
  찾은 물체당 검출 수(더 나눔), 덜 나눔/프레임. 앞 모델 대비 재현율 차이의 프레임 짝 부트스트랩 95 % 구간.
후처리는 ref_detect.OrtDetector 와 같다(letterbox, conf 0.25, NMS 0.7, 상자 안 proto>0, 최소 넓이, 마스크 중복 0.7).
최소 넓이는 416 의 24 칸(원본 화소로 같은 크기)을 입력 크기에 맞춰 바꾼다.
"""
import argparse
import os
import sys

import numpy as np

RA = os.environ.get('RA_FASTSAM', os.path.expanduser('~/robot-agent/training/fastsam'))
sys.path.insert(0, RA)


class TorchDetector:
    """Ultralytics .pt 를 GPU(fp16)로 돌려 ONNX 와 같은 원시 출력을 만든다. 후처리는 OrtDetector 의 것."""

    def __init__(self, pt, imgsz, conf_th=0.25):
        import torch
        from ultralytics import YOLO
        from ref_detect import OrtDetector
        self.torch = torch
        self.model = YOLO(pt).model.cuda().half().eval()
        self.model.fuse()
        det = OrtDetector.__new__(OrtDetector)
        det.S = imgsz
        det.conf_th, det.nms_iou, det.mask_iou, det.max_det = conf_th, 0.7, 0.7, 100
        det.area_min = int(round(24 * (imgsz / 416) ** 2))
        det.ms, det.mem_mb = [], 0.0
        det.s = self
        self.det = det
        self.ms = det.ms
        self.mem_mb = 0.0

    # OrtDetector 가 부르는 ONNX 세션 흉내
    def get_inputs(self):
        class I:
            name = 'images'
        return [I()]

    def run(self, _, feed):
        torch = self.torch
        x = torch.from_numpy(next(iter(feed.values()))).cuda().half()
        with torch.no_grad():
            y = self.model(x)
        o0, o1 = y[0][0] if isinstance(y[0], (list, tuple)) else y[0], y[0][1] if isinstance(y[0], (list, tuple)) else y[1]
        return [o0.float().cpu().numpy(), o1.float().cpu().numpy()]

    def detect(self, rgb, full=True):
        return self.det.detect(rgb, full)


def make(path, conf):
    imgsz = None
    if '@' in path:
        path, imgsz = path.rsplit('@', 1)
        imgsz = int(imgsz)
    if path.endswith('.pt'):
        return TorchDetector(path, imgsz or 1024, conf)
    import common as c
    return c.make_detector(path, conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True)
    ap.add_argument('--models', nargs='+', required=True, help='이름=경로.pt@입력크기 | 이름=경로.onnx')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--conf', type=float, default=0.25)
    a = ap.parse_args()
    import eval_det as ev
    dets = {}
    for p in a.models:
        n, path = p.split('=', 1)
        dets[n] = make(path, a.conf)
    import json
    import time
    frames = []
    t0 = time.time()
    for s in ev.samples(a.set, a.limit, a.every):
        fr = {'id': s['id'], 'tags': [sorted(g[1]) for g in s['gts']], 'res': {}}
        for n, d in dets.items():
            fr['res'][n] = ev.frame_eval(s, d.detect(s['rgb']))
        frames.append(fr)
        if len(frames) % 200 == 0:
            print(f'[eval] {a.set} {len(frames)} frames {time.time() - t0:.0f}s', flush=True)
    names = list(dets)
    res = {'set': a.set, 'models': dict(p.split('=', 1) for p in a.models),
           'summary': {n: ev.summarize(frames, n) for n in names},
           'latency_ms': {n: float(np.median(d.ms)) for n, d in dets.items()}, 'engine_mb': {n: 0.0 for n in names}}
    if len(names) > 1:
        res['delta_recall_ci'] = {n: {b: ev.paired_boot(frames, names[0], n, b) for b in ev.BUCKETS} for n in names[1:]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    with open(a.out.replace('.json', '.frames.json'), 'w') as f:
        json.dump(frames, f)
    res['plans'] = res['models']
    ev.print_table(res)


if __name__ == '__main__':
    main()
