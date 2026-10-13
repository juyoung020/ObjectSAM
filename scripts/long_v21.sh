#!/bin/bash
# v21 을 100 epoch 까지 이어서 학습(사용자 지시 2026-10-13). runs/v21/last.fsow(5 epoch 끝, 17,700 반복)에서 --epochs 100 으로 resume.
#   학습률은 100 epoch 직선 감쇠 일정의 5 epoch 지점(약 0.0095)으로 다시 올라가 이어진다. mosaic 은 99 epoch 까지 켜고 마지막 1 epoch 끔(v21 처방 그대로).
#   끝나면 SKIP_TRAIN=1 scripts/exp.sh v21L 로 변환·평가. 중간 epoch EMA(ema_epN.fsow) 는 남는다(학습 중엔 prune 이 건드리지 않음).
cd "$(dirname "$0")/.."
N=v21L; NB=3540; OFF=$((99*NB - 5*NB))   # --max-steps 는 resume 지점부터 센다: 17,700 → 350,460(99 epoch 끝)
log() { echo "$(date '+%m-%d %H:%M') [$N] $*" >> runs/exp.log; }
A="--plan data/teacher/train2017.plan --siglip data/teacher/train2017.siglip --lr0 0.01 --warmup-epochs 0.3 --oversample data/teacher/indoor_ids.json --nbs 32"
TR="./build/train --teacher data/teacher/train2017.ftl --batch 32 --imgsz 640 --log-every 200 --epochs 100 --out runs/$N $A"
mkdir -p runs/$N
[ -f runs/$N/last.fsow ] || cp runs/v21/last.fsow runs/$N/last.fsow
log "학습 시작(v21 5 epoch 에서 이어서 100 epoch 까지): $A"
if ! grep -q "mosaic-off-phase" runs/${N}_train.log 2>/dev/null; then
  $TR --resume runs/$N/last.fsow --max-steps $OFF >> runs/${N}_train.log 2>&1 || { log "학습 실패: $(tail -1 runs/${N}_train.log)"; exit 1; }
fi
echo "mosaic-off-phase" >> runs/${N}_train.log
$TR --resume runs/$N/last.fsow --mosaic 0 >> runs/${N}_train.log 2>&1 || { log "학습 실패(mosaic 끈 구간): $(tail -1 runs/${N}_train.log)"; exit 1; }
log "학습 끝: $(grep ' it ' runs/${N}_train.log | tail -1 | cut -c1-90)"
SKIP_TRAIN=1 scripts/exp.sh $N
