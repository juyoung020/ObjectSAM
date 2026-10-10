"""FastSAM-s.pt → FSOW 가중치(학습기 시작점). BN running 통계와 DFL 고정 가중치도 넣는다.

  .venv/bin/python pytools/dump_weights.py weights/FastSAM-s.pt weights/FastSAM-s.fsow
"""
import sys

import numpy as np
import torch

import fsow


def main(src, dst):
    from ultralytics import YOLO
    m = YOLO(src).model.float()
    t = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items() if 'num_batches' not in k}
    fsow.save(dst, t)
    print(f'{dst}: {len(t)} tensors, {sum(a.size for a in t.values()):,} values')


if __name__ == '__main__':
    with torch.no_grad():
        main(*sys.argv[1:3])
