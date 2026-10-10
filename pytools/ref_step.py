"""정합성 기준: Ultralytics 로 고정 배치 하나의 정방향·손실·역방향을 fp32 로 돌려 FSOW 로 남긴다.

C++ 학습기(tests/parity)가 같은 가중치·같은 배치로 계산한 값과 층마다 비교한다.
  .venv/bin/python pytools/ref_step.py --out data/parity/ref.fsow [--imgsz 256 --batch 2]

배치: 실제 사진 대신 재현 가능한 합성 영상(부드러운 무늬 + 도형)과, 그 도형의 정답 마스크·상자.
정답 마스크 해상도 = imgsz/4 (학습 때와 같이 proto 와 같은 크기).
"""
import argparse
import os

import numpy as np
import torch

import fsow


def make_batch(b, s, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) / s
    imgs, bidx, boxes, masks = [], [], [], []
    ms = s // 4
    for i in range(b):
        im = np.stack([0.5 + 0.25 * np.sin(6.28 * (xx * rng.uniform(1, 4) + yy * rng.uniform(1, 4)) + rng.uniform(0, 6))
                       for _ in range(3)], 0)
        n = int(rng.integers(3, 7))
        for _ in range(n):
            cx, cy = rng.uniform(0.15, 0.85, 2)
            rw, rh = rng.uniform(0.05, 0.25, 2)
            kind = rng.integers(0, 2)
            if kind == 0:   # 타원
                m = ((xx - cx) / rw) ** 2 + ((yy - cy) / rh) ** 2 <= 1
            else:           # 사각
                m = (abs(xx - cx) <= rw) & (abs(yy - cy) <= rh)
            col = rng.uniform(0, 1, 3).astype(np.float32)
            im[:, m] = col[:, None] * 0.8 + 0.1
            mm = m.reshape(ms, 4, ms, 4).mean((1, 3)) >= 0.5        # imgsz/4 정답
            ys, xs = np.nonzero(m)
            x1, y1, x2, y2 = xs.min() / s, ys.min() / s, (xs.max() + 1) / s, (ys.max() + 1) / s
            bidx.append(i)
            boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
            masks.append(mm.astype(np.uint8))
        imgs.append(np.clip(im, 0, 1))
    return (np.stack(imgs).astype(np.float32), np.array(bidx, np.float32), np.array(boxes, np.float32),
            np.stack(masks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/FastSAM-s.pt')
    ap.add_argument('--out', default='data/parity/ref.fsow')
    ap.add_argument('--imgsz', type=int, default=256)
    ap.add_argument('--batch', type=int, default=2)
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8SegmentationLoss
    model = YOLO(a.weights).model.float().cuda()
    model.args = get_cfg()
    model.args.overlap_mask = False
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    w0 = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items() if 'num_batches' not in k}

    img, bidx, boxes, masks = make_batch(a.batch, a.imgsz)
    x = torch.from_numpy(img).cuda()
    acts = {}
    hooks = [m.register_forward_hook(lambda mod, inp, out, i=i: acts.__setitem__(i, out))
             for i, m in enumerate(model.model[:-1])]
    preds = model(x)
    for h in hooks:
        h.remove()

    crit = v8SegmentationLoss(model)
    keep = {}
    orig = crit.assigner.forward

    def spy(*args, **kw):
        r = orig(*args, **kw)
        keep['assign'] = [t.detach() for t in r]
        return r
    crit.assigner.forward = spy
    batch = {'batch_idx': torch.from_numpy(bidx).cuda(), 'cls': torch.zeros(len(bidx), 1).cuda(),
             'bboxes': torch.from_numpy(boxes).cuda(), 'masks': torch.from_numpy(masks).cuda()}
    for k in ('boxes', 'scores', 'mask_coefficient', 'proto'):
        preds[k].retain_grad()
    loss, items = crit(preds, batch)
    total = loss.sum()
    total.backward()

    t = {f'w.{k}': v for k, v in w0.items()}
    t['in.img'] = img
    t['in.batch_idx'] = bidx.astype(np.int32)
    t['in.bboxes'] = boxes
    t['in.masks'] = masks
    for i, v in acts.items():
        if isinstance(v, torch.Tensor):
            t[f'act.{i}'] = v.detach().cpu().numpy()
    for k in ('boxes', 'scores', 'mask_coefficient', 'proto'):
        t[f'out.{k}'] = preds[k].detach().cpu().numpy()
        t[f'dout.{k}'] = preds[k].grad.detach().cpu().numpy()
    lab, tb, ts, fg, tgi = keep['assign']
    t['asg.target_bboxes'] = tb.cpu().numpy()
    t['asg.target_scores'] = ts.float().cpu().numpy()
    t['asg.fg_mask'] = fg.cpu().numpy().astype(np.uint8)
    t['asg.target_gt_idx'] = tgi.cpu().numpy().astype(np.int32)
    t['loss.items'] = torch.stack([items[k] for k in crit.loss_names]).cpu().numpy()   # box seg cls dfl sem (배치 곱 전)
    t['loss.total'] = np.array([total.item()], np.float32)
    for k, p in model.named_parameters():
        if p.grad is not None:
            t[f'grad.{k}'] = p.grad.detach().cpu().numpy()
    for k, v in model.state_dict().items():
        if 'running' in k:
            t[f'post.{k}'] = v.detach().cpu().numpy()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fsow.save(a.out, t)
    names = crit.loss_names
    print(f'{a.out}: {len(t)} tensors; loss total {total.item():.6f}; ' +
          ' '.join(f'{n} {items[n].item():.5f}' for n in names) + f'; fg {int(fg.sum())}, gt {len(bidx)}')


if __name__ == '__main__':
    main()
