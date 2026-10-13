#!/bin/bash
# 가중치 용량 관리(사용자 지시 2026-10-12): 성능 낮은 순으로 가중치를 지운다. 로그·평가 결과(s3.txt·csv·summary)는 남긴다.
#   scripts/prune.sh [남길 상위 수=5]
# 순위: 채점 S3 의 PQ@0.1 + PQ@0.4(logs/obj_eval_summary.log·docs 표). v1(runs/main) 의 최종 가중치와 지금 학습 중인 run 은 늘 남긴다.
# 지우는 것: (모든 run) 중간 epoch EMA(ema_epN.fsow) · (순위 밖 run) 남은 fsow·pt 전부, models_ck 의 그 모델 엔진·ONNX.
cd "$(dirname "$0")/.."
KEEP=${1:-5}; WS=$HOME/meridian_ws; LOG=runs/prune.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
# run 폴더 → models_ck 이름, 점수(PQ@0.1+PQ@0.4)
declare -A CK=( [main]=- [human]=human_ema [mix]=exp_mix [mix_os]=exp_mix_os [parts]=exp_parts [coco]=exp_coco [ceil]=exp_ceil
  [v8]=exp_v8 [v1r5]=exp_v1r5 [v9]=exp_v9 [v10]=exp_v10 [v11]=exp_v11 [v12]=exp_v12 [v13]=exp_v13 [v14]=exp_v14 [v15]=exp_v15 [wcf]=- [t1]=- )
declare -A SC=( [human]=0.973 [mix]=1.037 [mix_os]=1.027 [parts]=1.044 [coco]=1.007 [ceil]=0.999 [v8]=0.994 [v1r5]=1.026 [v9]=1.025
  [v10]=1.008 [v11]=1.045 [v12]=1.010 [v13]=1.044 [v14]=1.029 [v15]=1.051 [wcf]=0 [t1]=0 )
# 새 실험 점수는 obj_eval_summary 에서 읽는다(exp_<이름>@0.1 / @0.4 줄의 PQ = 5 번째 칸)
for d in runs/*/; do n=$(basename $d); [ -n "${SC[$n]}" ] && continue
  a=$(grep -E "^exp_$n@0\.1 " $WS/logs/obj_eval_summary.log | tail -1 | awk '{print $5}')
  b=$(grep -E "^exp_$n@0\.4 " $WS/logs/obj_eval_summary.log | tail -1 | awk '{print $5}')
  [ -n "$a" ] && [ -n "$b" ] && SC[$n]=$(python3 -c "print(round($a+$b,3))") && CK[$n]=exp_$n
done
running=$(ps -eo cmd | grep -oE "build/train .*--out runs/[A-Za-z0-9_]+" | grep -oE "runs/[A-Za-z0-9_]+$" | sed 's|runs/||')
ranked=$(for n in "${!SC[@]}"; do echo "${SC[$n]} $n"; done | sort -rn | awk '{print $2}')
top=$(echo "$ranked" | head -$KEEP)
log "순위(높은 순): $(echo $ranked | tr '\n' ' ') · 남김: main $top · 학습 중: ${running:-없음}"
before=$(df --output=avail -B1G / | tail -1)
for d in runs/*/; do n=$(basename $d)
  [[ " $running " == *" $n "* ]] && continue
  [ "$n" = v21L ] && continue   # 100 epoch 긴 학습: 중간 epoch 체크포인트·가중치 모두 남긴다(2026-10-13)
  rm -f runs/$n/ema_ep*.fsow runs/$n/ema_ep*.pt   # 중간 epoch EMA(fsow·pt) 는 어느 run 이든 필요 없음
  [ "$n" = main ] && continue
  if [ -n "${SC[$n]+x}" ] && ! echo "$top" | grep -qx "$n"; then
    rm -f runs/$n/*.fsow runs/$n/*.pt
    c=${CK[$n]}; [ -n "$c" ] && [ "$c" != - ] && rm -rf $WS/models_ck/${c}* && log "지움: runs/$n 가중치 + models_ck/${c}* (점수 ${SC[$n]})"
  fi
done
after=$(df --output=avail -B1G / | tail -1)
log "남은 공간 ${before}G → ${after}G"
tail -3 $LOG
