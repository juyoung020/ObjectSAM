#!/bin/bash
# 기준선: 원래 FastSAM-s(1024, 416), ObjectSAM — 세 평가셋, 2 장마다 하나(ObjectSAM 결과표와 같은 조건)
cd "$(dirname "$0")/.."
OBJ=$HOME/robot-agent/training/data/fastsam_obj/release/ObjectSAM-n-416.onnx
for S in sim_eval coco_val ade_val; do
  .venv/bin/python pytools/eval_ext.py --set $S --every 2 --out data/eval/base_$S.json \
    --models fs1024=weights/FastSAM-s.pt@1024 fs416=weights/FastSAM-s.pt@416 objsam=$OBJ
done
