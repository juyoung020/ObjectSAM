#!/bin/bash
# 학습 중 epoch 마다 EMA 를 COCO val 로 평가(2 장마다 하나): runs/<run>/eval_ep<N>.json
cd "$(dirname "$0")/.."
R=$1
seen=0; last=0
while true; do
  for f in $R/ema_ep*.fsow; do
    [ -e "$f" ] || continue
    n=$(basename $f .fsow); [ -f $R/eval_$n.json ] && continue
    .venv/bin/python pytools/fsow_to_pt.py $f $R/$n.pt > /dev/null 2>&1 && \
    .venv/bin/python pytools/eval_ext.py --set coco_val --every 2 --out $R/eval_$n.json --models $n=$R/$n.pt@1024 \
      > $R/eval_$n.log 2>&1 && echo "$(date +%H:%M) $n done"
  done
  [ $last = 1 ] && break
  if pgrep -f "build/train --init" > /dev/null; then seen=1; elif [ $seen = 1 ]; then last=1; continue; fi  # 끝나면 한 번 더 훑고 종료
  [ -f $R/stop_eval ] && break
  sleep 120
done
