#!/bin/bash
# 본 학습이 끝나면 마지막 EMA 로 확신도 문턱 훑기(원래 평가의 문턱 보정): runs/main/sweep/<set>_c<t>.json
cd "$(dirname "$0")/.."
while pgrep -f "[o]vernight.sh" > /dev/null; do sleep 60; done
mkdir -p runs/main/sweep
.venv/bin/python pytools/fsow_to_pt.py runs/main/ema_last.fsow runs/main/final.pt > /dev/null 2>&1 || exit 1
for t in 0.25 0.15 0.1 0.07 0.05 0.03; do
  for S in sim_eval coco_val ade_val; do
    .venv/bin/python pytools/eval_ext.py --set $S --every 2 --conf $t --out runs/main/sweep/${S}_c$t.json \
      --models final=runs/main/final.pt@640 > runs/main/sweep/${S}_c$t.log 2>&1
  done
  echo "$(date '+%m-%d %H:%M') sweep t=$t 끝" >> runs/overnight.log
done
