"""YOLOv8-seg 계열 .pt → FSOW 가중치(학습기 시작점). BN running 통계와 DFL 고정 가중치도 넣는다.

  .venv/bin/python pytools/dump_weights.py weights/FastSAM-s.pt weights/FastSAM-s.fsow
  .venv/bin/python pytools/dump_weights.py weights/yolov8s-seg.pt weights/yolov8s-seg-nc1.fsow --nc1

--nc1: 클래스 출력층(head cv3.*.2, 80 종)을 1 종으로 새로 만든다. 가중치는 nn.Conv2d 기본 초기화(시드 0),
bias 는 Ultralytics Detect.bias_init 과 같은 log(5 / nc / (640 / stride)²).
"""
import math
import sys

import numpy as np
import torch

import fsow


def main(src, dst, nc1=False):
    from ultralytics import YOLO
    m = YOLO(src).model.float()
    t = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items() if 'num_batches' not in k}
    if nc1:
        head = m.model[-1]
        torch.manual_seed(0)
        for i, (seq, s) in enumerate(zip(head.cv3, head.stride.tolist())):
            old = seq[2]
            new = torch.nn.Conv2d(old.in_channels, 1, 1)
            new.bias.data[:] = math.log(5 / 1 / (640 / s) ** 2)
            pre = f'model.{len(m.model) - 1}.cv3.{i}.2.'
            assert t[pre + 'weight'].shape[0] == old.out_channels, pre
            t[pre + 'weight'] = new.weight.detach().numpy().astype(np.float32)
            t[pre + 'bias'] = new.bias.detach().numpy().astype(np.float32)
            print(f'{pre}: {old.out_channels} → 1 classes, bias {new.bias.item():.3f}')
    fsow.save(dst, t)
    print(f'{dst}: {len(t)} tensors, {sum(a.size for a in t.values()):,} values')


if __name__ == '__main__':
    with torch.no_grad():
        main(sys.argv[1], sys.argv[2], '--nc1' in sys.argv[3:])
