"""CropFormer Hornet-L(선생) 가중치와 C++ 포팅 정합성 기준값을 FSOW 로 덤프한다. 선생 환경(.venv-teacher)에서 돈다.

  .venv-teacher/bin/python pytools/teacher_dump.py weights      # → weights/teacher/cropformer_hornet_l.fsow
  .venv-teacher/bin/python pytools/teacher_dump.py ref IMG...   # → data/parity/teacher_ref_<이름>.fsow (fp32, TF32 끔)

기준값: 5 화면 입력(정규화·패딩 뒤), 백본 단계 출력, 픽셀 디코더 출력, 2d/3d 디코더 출력, 최종 마스크·점수.
"""
import os
import sys

CF = os.path.expanduser('~/ObjectSAM/third_party/detectron2/projects/CropFormer')
sys.path.insert(0, CF)
sys.path.insert(0, CF + '/demo_cropformer')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from predictor import CropFormerPredictor

import fsow

ROOT = os.path.expanduser('~/ObjectSAM')
CKPT = f'{ROOT}/weights/teacher/CropFormer_hornet_3x_03823a.pth'


def build():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(CF + '/configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml')
    cfg.MODEL.WEIGHTS = CKPT
    cfg.freeze()
    return cfg, CropFormerPredictor(cfg)


def np32(t):
    return t.detach().float().cpu().numpy()


def dump_weights():
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)['model']
    out = {k: v.float().numpy() for k, v in sd.items() if v.is_floating_point()}
    skipped = [k for k, v in sd.items() if not v.is_floating_point()]
    path = f'{ROOT}/weights/teacher/cropformer_hornet_l.fsow'
    fsow.save(path, out)
    print(f'{path}: {len(out)} tensors, {sum(v.size for v in out.values()) / 1e6:.1f} M, skipped {skipped}')


def dump_ref(paths):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, pred = build()
    m = pred.model
    rec = {}

    def keep(name, first_only=True):
        def f(mod, inp, out):
            if first_only and name in rec:
                return
            rec[name] = out
        return f

    m.backbone.register_forward_pre_hook(lambda mod, inp: rec.__setitem__('input', inp[0]))
    bb = m.backbone
    for i in range(4):
        bb.downsample_layers[i].register_forward_hook(keep(f'bb.down{i}'))
        bb.stages[i][0].register_forward_hook(keep(f'bb.s{i}.b0'))
        bb.stages[i].register_forward_hook(keep(f'bb.s{i}'))
    bb.register_forward_hook(keep('bb.out'))
    pd = m.sem_seg_head.pixel_decoder
    pd.transformer.encoder.layers[0].register_forward_hook(keep('pd.enc0'))
    pd.transformer.register_forward_hook(keep('pd.memory'))
    orig_ff = pd.forward_features

    def ff(features):
        r = orig_ff(features)
        rec['pd.out'] = r
        return r
    pd.forward_features = ff
    dec = m.sem_seg_head.predictor
    dec.transformer_cross_attention_layers[0].register_forward_hook(keep('dec.ca0'))
    dec.transformer_self_attention_layers[0].register_forward_hook(keep('dec.sa0'))
    dec.transformer_ffn_layers[0].register_forward_hook(keep('dec.ffn0'))
    dec.make_3d.register_forward_hook(keep('dec.make3d'))
    dec.register_forward_hook(keep('dec.out'))
    orig_iw = m.inference_whole_views

    def iw(*a, **k):
        rec['crop_regions'] = a[6]
        rec['image_sizes'] = a[5]
        rec['views_hw'] = [(b['height'], b['width']) for b in a[4]]
        return orig_iw(*a, **k)
    m.inference_whole_views = iw

    os.makedirs(f'{ROOT}/data/parity', exist_ok=True)
    for p in paths:
        rec.clear()
        img = cv2.imread(p)
        with torch.no_grad():
            ins = pred(img)['instances']
        o = {'image_bgr': img.astype(np.uint8), 'input': np32(rec['input'])}
        for i in range(4):
            for k in (f'bb.down{i}', f'bb.s{i}.b0', f'bb.s{i}'):
                o[k] = np32(rec[k])
        for k, v in rec['bb.out'].items():
            o[f'bb.{k}'] = np32(v)
        o['pd.enc0'] = np32(rec['pd.enc0'])
        o['pd.memory'] = np32(rec['pd.memory'][0])
        mf, _, ms = rec['pd.out']
        o['pd.mask_features'] = np32(mf)
        for i, f in enumerate(ms):
            o[f'pd.ms{i}'] = np32(f)
        for k in ('dec.ca0', 'dec.sa0', 'dec.ffn0', 'dec.make3d'):
            o[k] = np32(rec[k])
        o2, o3 = rec['dec.out']
        o['dec.logits2d'] = np32(o2['pred_logits'])
        o['dec.logits3d'] = np32(o3['pred_logits'])
        o['dec.masks3d'] = np32(o3['pred_masks'])
        o['crop_regions'] = np.asarray(rec['crop_regions'], np.int32)
        o['image_sizes'] = np.asarray(rec['image_sizes'], np.int32)
        o['views_hw'] = np.asarray(rec['views_hw'], np.int32)
        o['out.masks'] = ins.pred_masks.cpu().numpy().astype(np.uint8)
        o['out.scores'] = np32(ins.scores)
        name = os.path.splitext(os.path.basename(p))[0]
        path = f'{ROOT}/data/parity/teacher_ref_{name}.fsow'
        fsow.save(path, o)
        print(path, {k: v.shape for k, v in o.items() if k.startswith(('input', 'bb.res', 'pd.mask', 'dec.masks'))})


if __name__ == '__main__':
    if sys.argv[1] == 'weights':
        dump_weights()
    else:
        dump_ref(sys.argv[2:])
