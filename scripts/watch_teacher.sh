#!/usr/bin/env bash
# 선생 라벨 만들기 감시: 죽었으면 이어서 다시 돌리고(이어 하기 지원), 로그가 15 분 넘게 멈추면 끄고 다시 돌린다.
# 남은 사진이 0 이거나, 다시 돌려도 새로 쓴 사진이 0 장인 일이 두 번 이어지면(못 읽는 사진만 남음) 끝낸다.
#   nohup scripts/watch_teacher.sh > data/teacher/watch.log 2>&1 &
cd "$(dirname "$0")/.."
LIST=data/teacher/train2017.list OUT=data/teacher/train2017.ftl LOG=data/teacher/train2017.log
start() {
  echo "$(date '+%F %T') start teacher_label"
  nohup ./build/teacher_label --list $LIST --out $OUT --workers 8 >> $LOG 2>&1 &
  sleep 60
}
zero=0
while true; do
  pid=$(pgrep -x teacher_label | head -1)
  if [ -z "$pid" ]; then
    if tail -n 3 $LOG | grep -q " 0 to do"; then
      echo "$(date '+%F %T') all done"
      exit 0
    fi
    if tail -n 1 $LOG | grep -q "^wrote 0 records"; then
      zero=$((zero + 1))
      if [ $zero -ge 2 ]; then
        echo "$(date '+%F %T') nothing left that can be read; see 'skip:' lines in $LOG"
        exit 0
      fi
    else
      zero=0
    fi
    echo "$(date '+%F %T') not running; last log: $(tail -n 1 $LOG)"
    start
    continue
  fi
  age=$(( $(date +%s) - $(stat -c %Y $LOG) ))
  if [ $age -gt 900 ]; then
    echo "$(date '+%F %T') stalled ${age}s, killing $pid"
    kill $pid; sleep 20; kill -9 $pid 2>/dev/null
    start
    continue
  fi
  sleep 300
done
