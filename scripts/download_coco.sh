#!/bin/bash
# COCO 2017: train/val 사진 + 개체 정답(panoptic 정답은 robot-agent 에서 링크). 받고 풀고 zip 은 지운다.
cd "$(dirname "$0")/../data/coco"
for z in zips/train2017.zip zips/val2017.zip annotations/annotations_trainval2017.zip; do
  f=$(basename $z)
  d=${f%.zip}
  [ -f $d.done ] && continue
  aria2c -c -x 16 -s 16 -k 32M --file-allocation=none --max-tries=0 --retry-wait=5 --summary-interval=60 \
    --console-log-level=warn -o $f "http://images.cocodataset.org/$z" || exit 1
  unzip -q -o $f && rm -f $f && touch $d.done && echo "$(date +%H:%M) $f ok"
done
echo "$(date +%H:%M) coco done: $(ls train2017 | wc -l) train, $(ls val2017 | wc -l) val"
