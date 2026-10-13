#!/bin/bash
# 다시 학습(2026-10-11): 사람 라벨(COCO things + 문·창·계단 + LVIS) + CropFormer 채우기(train2017_human.plan)
# 판정 파일 다 되기 기다림 → 학습 → .pt → 1024 ONNX → uHumans2 frontend 평가(문턱 0.4, 0.1)
cd "$(dirname "$0")/.."
log() { echo "$(date '+%m-%d %H:%M') $*" >> runs/human.log; }
while pgrep -f "[l]abel_human.py" > /dev/null; do sleep 30; done
[ -f data/teacher/train2017_human.plan ] || { log "판정 파일 없음: $(tail -2 runs/label_human.log)"; exit 1; }
log "판정 파일: $(tail -1 runs/label_human.log)"
log "학습 시작"
./build/train --teacher data/teacher/train2017.ftl --plan data/teacher/train2017_human.plan \
  --batch 32 --imgsz 640 --lr0 0.01 --warmup-epochs 1 --log-every 200 --epochs 100 --out runs/human > runs/human_train.log 2>&1 \
  || { log "학습 실패: $(tail -1 runs/human_train.log)"; exit 1; }
log "학습 끝: $(grep ' it ' runs/human_train.log | tail -1 | cut -c1-90)"
.venv/bin/python pytools/fsow_to_pt.py runs/human/ema_last.fsow runs/human/final.pt > /dev/null 2>&1 || { log "pt 변환 실패"; exit 1; }
M=$HOME/meridian_ws/models_wcf
mkdir -p $M/fastsam && ln -sfn $HOME/meridian_ws/models/clip $M/clip
cp runs/human/final.pt $M/fastsam/objectsam_human.pt
.venv/bin/python -c "
from ultralytics import YOLO
YOLO('$M/fastsam/objectsam_human.pt').export(format='onnx', imgsz=1024, opset=17, simplify=True, dynamic=False)" > runs/human_export.log 2>&1
mv $M/fastsam/objectsam_human.onnx $M/fastsam/FastSAM-s-1024.onnx || { log "ONNX 실패"; exit 1; }
T=$HOME/meridian_ws/tools/fe_eval.sh
L=$HOME/meridian_ws/logs
bash $T wcf040 engines > $L/human_engines.log 2>&1 || { log "엔진 실패"; exit 1; }
SEQS="apartment_00h apartment_01h apartment_02h office_00h office_12h"
[ -f $HOME/meridian_ws/datasets/unpacked/uHumans2_office_s1_06h/metadata.json ] && SEQS="$SEQS office_06h"
for c in wcf040 wcf010; do
  bash $T $c run $SEQS > $L/human_${c}_run.log 2>&1 && log "uHumans2 $c 끝" || log "uHumans2 $c 실패"
done
