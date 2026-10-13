#!/bin/bash
# 밤샘: 시험 학습 t1(5 epoch) → 평가 3 종 → 본 학습 main(100 epoch, epoch 마다 coco_val 평가)
cd "$(dirname "$0")/.."
T="--teacher data/teacher/train2017.ftl --plan data/teacher/train2017.plan --siglip data/teacher/train2017.siglip --batch 32 --imgsz 640 --lr0 0.01 --warmup-epochs 1 --log-every 200"
OBJ=$HOME/robot-agent/training/data/fastsam_obj/release/ObjectSAM-n-416.onnx
log() { echo "$(date '+%m-%d %H:%M') $*" >> runs/overnight.log; }

rm -rf runs/t1 runs/t1.log
log "t1 시작"
./build/train $T --epochs 5 --out runs/t1 > runs/t1.log 2>&1 || { log "t1 실패: $(tail -1 runs/t1.log)"; exit 1; }
grep -qi nan runs/t1.log && { log "t1 손실 NaN"; exit 1; }
log "t1 끝: $(grep ' it ' runs/t1.log | tail -1 | cut -c1-90)"
.venv/bin/python pytools/fsow_to_pt.py runs/t1/ema_last.fsow runs/t1/t1.pt > /dev/null 2>&1
for S in sim_eval coco_val ade_val; do
  .venv/bin/python pytools/eval_ext.py --set $S --every 2 --out runs/t1/eval_$S.json \
    --models fs640=weights/FastSAM-s.pt@640 objsam=$OBJ t1=runs/t1/t1.pt@640 > runs/t1/eval_$S.log 2>&1
  log "t1 평가 $S 끝"
done

log "main 시작"
./build/train $T --epochs 100 --out runs/main > runs/main.log 2>&1 &
TP=$!
# epoch 마다 EMA 를 coco_val(2 장마다 하나)로 평가
while true; do
  for f in runs/main/ema_ep*.fsow; do
    [ -e "$f" ] || continue
    n=$(basename $f .fsow); [ -f runs/main/eval_$n.json ] && continue
    .venv/bin/python pytools/fsow_to_pt.py $f runs/main/$n.pt > /dev/null 2>&1 && \
    .venv/bin/python pytools/eval_ext.py --set coco_val --every 2 --out runs/main/eval_$n.json --models $n=runs/main/$n.pt@640 \
      > runs/main/eval_$n.log 2>&1 && log "main $n 평가 끝"
  done
  kill -0 $TP 2>/dev/null || break
  sleep 120
done
wait $TP; log "main 종료 (코드 $?): $(tail -1 runs/main.log)"
