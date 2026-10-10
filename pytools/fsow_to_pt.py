"""학습기 EMA 가중치(FSOW, PyTorch 이름) → Ultralytics .pt (추론·ONNX/TensorRT 내보내기용).

  .venv/bin/python pytools/fsow_to_pt.py runs/r1/ema_last.fsow runs/r1/FastSAM-s-obj.pt [--base weights/FastSAM-s.pt]
"""
import argparse

import torch

import fsow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--base', default=None, help='기본: reg_max 16 → FastSAM-s.pt, 26 → FastSAM-s-r26.pt')
    a = ap.parse_args()
    from ultralytics import YOLO
    t = fsow.load(a.src)
    if a.base is None:
        a.base = 'weights/FastSAM-s-r26.pt' if t['model.22.cv2.0.2.weight'].shape[0] == 104 else 'weights/FastSAM-s.pt'
    y = YOLO(a.base)
    sd = y.model.state_dict()
    n = 0
    for k, v in t.items():
        assert k in sd, f'unknown tensor {k}'
        assert tuple(sd[k].shape) == v.shape, f'{k}: {tuple(sd[k].shape)} vs {v.shape}'
        sd[k] = torch.from_numpy(v).to(sd[k].dtype)
        n += 1
    missing = [k for k in sd if k not in t and 'num_batches' not in k and 'dfl' not in k]
    assert not missing, f'missing in {a.src}: {missing[:5]}'
    y.model.load_state_dict(sd)
    y.save(a.dst)
    print(f'{a.dst}: {n} tensors from {a.src}')


if __name__ == '__main__':
    main()
