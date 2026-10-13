#!/bin/bash
# 5 epoch 관문(사용자 지시 2026-10-12): 기준은 **ObjectSAM v1 @0.1**(지금까지 최고, 채점 S3, 5 시퀀스) — 표시는 FastSAM-s 넘을 값 대비 이긴 칸 수도 함께.
#   scripts/ep5_gate.sh <이름> [안 씀]
# 5 epoch 평가(5 시퀀스, 문턱 0.1)를 v1 @0.1 의 12 칸에 댄다.
#   계속: v1 보다 FastSAM-s 를 이긴 칸 수가 많거나, FastSAM-s 에 지는 칸 중 하나라도 v1 보다 잡음 이상 FastSAM-s 에 가까울 때
#         (잡음: R·트랙 1.0 · mIoU 0.010 · 과소분할 5 % · 그 밖 P·F1 1.0, PQ·IDF1·HOTA 0.010, 과다분할·구조물 5 %).
#   취소: 그 밖. 프로세스는 PID 로만 죽인다(CLAUDE.md).
cd "$(dirname "$0")/.."
N=$1; P=$2; L=$HOME/meridian_ws/logs
log() { echo "$(date '+%m-%d %H:%M') [$N] $*" >> runs/exp.log; }
row() { grep -E "^[0-9-]+ [0-9:]+ +exp_$1_ema_ep5 .*\(1[0-9]{4}, 1[0-9]{3}\)" $L/ck_exp_$1_summary.log 2>/dev/null | tail -1; }   # 5 시퀀스 줄(2D GT 1 만 대)만
until [ -n "$(row $N)" ]; do
  pgrep -f "build/train .*--out runs/$N( |$)" > /dev/null || pgrep -f "ckpt_watch.sh .*runs/$N exp_$N" > /dev/null || exit 0
  sleep 30
done
A='v1 v1 v1 58.6 74.0 65.4 0.555 267 1349 91.3 0.467 0.527 0.682 27.7 43.1'; B=$(row $N)   # ObjectSAM v1 @0.1, S3 5 시퀀스(docs/BENCHMARK.md)
read -r verdict detail < <(python3 - "$A" "$B" <<'PY'
import sys
K = ['P', 'R', 'F1', 'PQ', 'split', 'merged', 'track', 'IDF1', 'HOTA', 'mIoU', 'struct_pred', 'struct_px']
UP = {'P', 'R', 'F1', 'PQ', 'track', 'IDF1', 'HOTA', 'mIoU'}
T = dict(P=48.7, R=78.8, F1=56.8, PQ=0.471, split=449, merged=613, track=93.8, IDF1=0.424, HOTA=0.485, mIoU=0.707, struct_pred=56.6, struct_px=69.2)  # FastSAM-s 넘을 값(S3, 5 시퀀스)
NOISE = dict(P=1.0, R=1.0, F1=1.0, PQ=0.010, track=1.0, IDF1=0.010, HOTA=0.010, mIoU=0.010)
def f(s):
    t = s.split()[3:]
    return {k: float(t[i]) for i, k in enumerate(K)}
def gap(v, k):   # FastSAM-s 까지 남은 차이(0 이하면 이김)
    return (T[k] - v[k]) if k in UP else (v[k] - T[k])
def tol(v, k):
    return NOISE[k] if k in NOISE else 0.05 * T[k]
b = f(sys.argv[2])
wb = sum(gap(b, k) < 0 for k in K)
if not sys.argv[1].strip():
    print('계속', f'앞 버전 5 시퀀스 기록 없음 — FastSAM-s 대비 {wb}/12 칸 이김'); sys.exit()
a = f(sys.argv[1]); wa = sum(gap(a, k) < 0 for k in K)
closer = [f"{k} FastSAM-s 까지 v1 {gap(a,k):+.3g} → 이번 {gap(b,k):+.3g}" for k in K if gap(a, k) > 0 and gap(a, k) - gap(b, k) >= tol(a, k)]
lose = [k for k in K if gap(b, k) >= 0]
ok = wb > wa or closer
msg = f"FastSAM-s 대비 이긴 칸 v1 {wa} → 이번 {wb}/12" + (f"; 좁혀짐: {', '.join(closer)}" if closer else '') + f"; 남은 진 칸: {','.join(lose)}"
print('계속' if ok else '취소', msg)
PY
)
if [ "$verdict" = "취소" ]; then
  for pid in $(pgrep -f "scripts/exp.sh $N( |$)") $(pgrep -f "build/train .*--out runs/$N( |$)") $(pgrep -f "ckpt_watch.sh .*runs/$N exp_$N"); do kill $pid 2>/dev/null; done
  log "5 epoch 관문: 취소 — $detail. 실험 실패(취소)"
else
  log "5 epoch 관문: 계속 — $detail"
fi
