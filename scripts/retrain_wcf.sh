#!/bin/bash
# 다시 학습(2026-10-11): 벽·천장·바닥(땅 포함)만 빼는 판정(train2017_wcf.plan), SigLIP2·나누기·무시 영역 없음.
# 학습 → .pt → 1024 ONNX → uHumans2 frontend 평가(문턱 0.4, 0.1)
cd "$(dirname "$0")/.."
log() { echo "$(date '+%m-%d %H:%M') $*" >> runs/wcf.log; }
log "학습 시작"
./build/train --teacher data/teacher/train2017.ftl --plan data/teacher/train2017_wcf.plan \
  --batch 32 --imgsz 640 --lr0 0.01 --warmup-epochs 1 --log-every 200 --epochs 100 --out runs/wcf > runs/wcf_train.log 2>&1 \
  || { log "학습 실패: $(tail -1 runs/wcf_train.log)"; exit 1; }
log "학습 끝: $(grep ' it ' runs/wcf_train.log | tail -1 | cut -c1-90)"
.venv/bin/python pytools/fsow_to_pt.py runs/wcf/ema_last.fsow runs/wcf/final.pt > /dev/null 2>&1 || { log "pt 변환 실패"; exit 1; }
M=$HOME/meridian_ws/models_wcf
mkdir -p $M/fastsam && ln -sfn $HOME/meridian_ws/models/clip $M/clip
cp runs/wcf/final.pt $M/fastsam/objectsam_wcf.pt
.venv/bin/python -c "
from ultralytics import YOLO
YOLO('$M/fastsam/objectsam_wcf.pt').export(format='onnx', imgsz=1024, opset=17, simplify=True, dynamic=False)" > runs/wcf_export.log 2>&1
mv $M/fastsam/objectsam_wcf.onnx $M/fastsam/FastSAM-s-1024.onnx || { log "ONNX 실패"; exit 1; }
T=$HOME/meridian_ws/tools/fe_eval.sh
L=$HOME/meridian_ws/logs
bash $T wcf040 engines > $L/wcf_engines.log 2>&1 || { log "엔진 실패"; exit 1; }
SEQS="apartment_00h apartment_01h apartment_02h office_00h office_12h"
[ -f $HOME/meridian_ws/datasets/unpacked/uHumans2_office_s1_06h/metadata.json ] && SEQS="$SEQS office_06h"
for c in wcf040 wcf010; do
  bash $T $c run $SEQS > $L/${c}_run.log 2>&1 && log "uHumans2 $c 끝" || log "uHumans2 $c 실패"
done
