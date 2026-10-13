#!/bin/bash
# 아이디어 실험 하나: 학습(5 epoch, 2026-10-12 사용자 지시 — EPOCHS 로 바꿈) → 1024 ONNX → seg 엔진 → uHumans2 5 시퀀스 전체 평가(문턱 0.1·0.4)
#   scripts/exp.sh <이름> <학습 인자...>     예: scripts/exp.sh mix --plan data/teacher/train2017_human.plan --extra-shards 'data/old_shards/*.bin'
# 결과: runs/<이름>/ · ~/meridian_ws/runs/full_exp_<이름>_{01,04} · 요약 ~/meridian_ws/logs/ck_full_summary.log
# 같은 GPU 를 쓰는 평가가 돌고 있으면 끝날 때까지 기다린다.
cd "$(dirname "$0")/.."
N=$1; shift
WS=$HOME/meridian_ws
log() { echo "$(date '+%m-%d %H:%M') [$N] $*" >> runs/exp.log; }
while pgrep -f "[c]k_full.sh|[e]val.sh run" > /dev/null; do sleep 30; done
log "학습 시작: $*"
TR="./build/train --teacher data/teacher/train2017.ftl --batch 32 --imgsz 640 --lr0 0.01 --warmup-epochs 1 --log-every 200 --epochs ${EPOCHS:-5} --out runs/$N"
if [ -n "$SKIP_TRAIN" ]; then   # 학습은 이미 따로 끝냄(2026-10-13, 긴 학습 scripts/long_v21.sh) — 변환·평가만
  log "학습 건너뜀(SKIP_TRAIN): runs/$N/ema_last.fsow 로 평가"
elif [ -n "$MOSAIC_OFF_AT" ]; then   # 마지막 구간은 mosaic 끄기(2026-10-12): MOSAIC_OFF_AT 반복까지 학습 → 이어서 --mosaic 0 으로 끝까지
  if grep -q "^done: $MOSAIC_OFF_AT iterations" runs/${N}_train.log 2>/dev/null && [ -f runs/$N/last.fsow ]; then
    log "앞 구간($MOSAIC_OFF_AT 반복)은 이미 끝남 — mosaic 끈 구간만 이어서"
  else
    $TR "$@" --max-steps $MOSAIC_OFF_AT > runs/${N}_train.log 2>&1 || { log "학습 실패: $(tail -1 runs/${N}_train.log)"; exit 1; }
  fi
  $TR "$@" --resume runs/$N/last.fsow --mosaic 0 >> runs/${N}_train.log 2>&1 || { log "학습 실패(mosaic 끈 구간): $(tail -1 runs/${N}_train.log)"; exit 1; }
else
  $TR "$@" > runs/${N}_train.log 2>&1 || { log "학습 실패: $(tail -1 runs/${N}_train.log)"; exit 1; }
fi
log "학습 끝: $(grep ' it ' runs/${N}_train.log | tail -1 | cut -c1-90)"
M=$WS/models_ck/exp_$N
mkdir -p $M/fastsam $M/frontend_rtx3060 && ln -sfn $WS/models/clip $M/clip
cp $WS/models/frontend_rtx3060/clip_image.plan $WS/models/frontend_rtx3060/manifest.json $M/frontend_rtx3060/
.venv/bin/python pytools/fsow_to_pt.py runs/$N/ema_last.fsow runs/$N/final.pt > /dev/null 2>&1 || { log "pt 변환 실패"; exit 1; }
cp runs/$N/final.pt $M/fastsam/m.pt
.venv/bin/python -c "
from ultralytics import YOLO
YOLO('$M/fastsam/m.pt').export(format='onnx', imgsz=1024, opset=17, simplify=True, dynamic=False)" > runs/${N}_export.log 2>&1
mv $M/fastsam/m.onnx $M/fastsam/FastSAM-s-1024.onnx && rm -f $M/fastsam/m.pt || { log "ONNX 실패"; exit 1; }
( source $WS/tools/env.sh > /dev/null 2>&1
  FB_MODELS=$M MERIDIAN_WS=$WS python3 $WS/src/meridian/benchmarks/meridian_benchmark/frontend_benchmark/build_engines.py --only seg ) \
  > runs/${N}_engine.log 2>&1 || { log "엔진 실패"; exit 1; }
# 평가: 빠른 경로(tools/fast_eval, C++/CUDA 커널, 기존과 S3 숫자 차이 0 검증 — 2026-10-12). 원래 지표 summary 가 필요하면 ck_full.sh.
bash $WS/tools/fast_eval/run.sh exp_$N 0.1 0.4 > $WS/logs/fast_exp_$N.log 2>&1 || { log "평가 실패(fast_eval): $(tail -1 $WS/logs/fast_exp_$N.log)"; exit 1; }
for th in 01 04; do tail -1 $WS/runs/fast_exp_${N}_$th/s3.txt; done >> $WS/logs/obj_eval_summary.log
log "물체 기준 평가: $(grep "exp_$N@" $WS/logs/obj_eval_summary.log | tail -2 | tr '\n' ' ')"
scripts/prune.sh 5 > /dev/null 2>&1   # 가중치 용량 관리: 성능 낮은 순으로 가중치 지우기(사용자 지시 2026-10-12)
