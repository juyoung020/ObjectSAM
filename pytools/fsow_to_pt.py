"""학습기 EMA 가중치(FSOW, PyTorch 이름) → Ultralytics .pt (추론·ONNX/TensorRT 내보내기용).

  .venv/bin/python pytools/fsow_to_pt.py runs/r1/ema_last.fsow runs/r1/model.pt

모델 정의는 FastSAM-s 와 같은 YOLOv8s-seg(클래스 1)이고, reg_max 는 가중치의 상자 가지 출력(4·reg_max)에서 읽는다.
"""
import argparse

import torch

import fsow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--yaml-from', default='weights/FastSAM-s.pt', help='모델 정의(yaml)를 가져올 Ultralytics 파일')
    a = ap.parse_args()
    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel
    t = fsow.load(a.src)
    y = YOLO(a.yaml_from)
    cfg = dict(y.model.yaml)
    cfg['reg_max'] = int(t['model.22.cv2.0.2.weight'].shape[0] // 4)
    m = SegmentationModel(cfg, ch=3, nc=1, verbose=False)
    sd = m.state_dict()
    for k, v in t.items():
        assert k in sd, f'unknown tensor {k}'
        assert tuple(sd[k].shape) == v.shape, f'{k}: {tuple(sd[k].shape)} vs {v.shape}'
        sd[k] = torch.from_numpy(v).to(sd[k].dtype)
    missing = [k for k in sd if k not in t and 'num_batches' not in k and 'dfl' not in k]
    assert not missing, f'missing in {a.src}: {missing[:5]}'
    m.load_state_dict(sd)
    y.model = m
    y.save(a.dst)
    print(f'{a.dst}: {len(t)} tensors from {a.src}, reg_max {cfg["reg_max"]}')


if __name__ == '__main__':
    main()
