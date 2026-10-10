# ObjectSAM 재학습 스펙

상태: 작성 중 (사용자와 하나씩 정함). 정해진 것은 **결정**, 아직인 것은 **미정**으로 둔다.
배경 조사: [FASTSAM_NETWORK.md](FASTSAM_NETWORK.md).

## 1. 목표

**한 줄 정의: 클래스 구분 없이 무엇이든 자르되, 구조물 배경(벽·바닥·천장·하늘)만 빼는 로봇용 SAM.**

영상 속 물체를 클래스 없이 모두 분할하되, 벽·천장·바닥·하늘에는 마스크를 내지 않고,
물체 하나에 마스크 하나(부분으로 쪼개지 않음)를 낸다. 로봇이 집을 물체를 찾는 데 쓴다 —
컵·리모컨·휴대폰처럼 화면에서 작게 보이는 물체를 놓치지 않는 것이 중요하다.

- **결정** 배경(마스크 없음): 벽, 천장, 바닥, 하늘 — 이 넷만. 바닥에는 실외의 걷는 면(도로·잔디·흙·모래 등)도 포함.
- **결정** 그 밖은 모두 물체: 문·창·계단, 물, 나무, 러그 등. 건물(지붕 포함)은 구조물(2026-10-11, 사용자).
- **결정** 물체는 통째로(부분 마스크 없음).

## 2. 신경망

- **결정** FastSAM 과 같은 신경망: YOLOv8-seg, 클래스 1("object"). FastSAM 고유의 구조 변경은 없다.
- **결정** 크기 s (YOLOv8s-seg, 약 1,179 만 파라미터) — FastSAM-s 와 같음.
- **결정** 입력 640, reg_max 16 (FastSAM-s 그대로). COCO 사진(최대 640)과 로봇 카메라(640×480)에 맞고,
  1024 는 정보 없이 시간만 약 2.5 배.

## 3. 데이터

- **결정** COCO 2017 (사람이 단 정답, 이름 있음). 나누기(92 : 4 : 4):
  - 학습: train2017 중 113,287 장
  - 검증: train2017 에서 뗀 5,000 장(고정 시드로 무작위) — 학습 중 곡선 보기·체크포인트 고르기에만
  - 시험: val2017 5,000 장 — 끝까지 손대지 않고 최종 점수에만(원래 FastSAM-s·예전 ObjectSAM 과 같은 조건 비교)
  - test2017 은 정답이 비공개라 우리 규칙(벽·바닥 제외)으로 채점할 수 없어 쓰지 않는다.
- **결정** 외부 시험(학습에 안 쓴 데이터, 로봇 실내 일반화 확인): BEHAVIOR 시뮬 집 10 곳, ADE20K 실내 val.

## 4. 라벨 규칙: 선생 모델 증류 + COCO 사람 라벨로 구조물 빼기

COCO 사람 정답만으로는 물체 80 종 밖이 비어 있고, stuff 53 종은 개체 구분 없는 덩어리다(선반 위 물건이 선반에 섞여 칠해짐,
떨어진 창문 셋이 마스크 하나 — 바탕화면의 예시 그림). 그래서 정답 마스크는 선생 모델로 만든다.

- **결정** 선생: CropFormer Hornet-L(개체 분할, 공개 모델 중 최고). 원본 그대로(화면 다섯 장) C++/CUDA fp16 로 돌린다
  — [TEACHER.md](TEACHER.md). 학습 사진 118,287 장(검증으로 뗀 5,000 장 포함) 모두.
- **결정** 선생 결과는 점수와 함께 모두 저장(data/teacher/train2017.ftl)하고, 아래 기준은 결과를 보며 정한다.
- **미정** 점수 기준(원본 데모는 0.5).
- **결정** 구조물(벽·천장·바닥·하늘) 빼기(2026-10-11, 사용자): 선생 마스크가 구조물이 되려면 둘 다 맞아야 한다.
  1. 같은 사진 COCO panoptic 사람 라벨상 그 마스크가 배경(아래 19 종)이 대부분이고,
  2. SigLIP2 가 마스크 조각을 구조물로 0.8 이상 확신: 벽·천장·바닥·하늘 확률의 합 ≥ 0.8(넷은 서로 비슷해 확률이 갈라지므로
     합으로 본다; 어느 구조물인지는 COCO 라벨이 정한다).
  둘 중 하나라도 아니면 물체로 남긴다(진짜 물체가 빠지는 쪽이 더 나쁘다). 0.8 이상으로 빠지는 마스크를 그림으로 훑어보고 확정한다.
  큰 구조물 조각(그 COCO 배경 조각을 30 % 이상 덮음)은 1 만으로 구조물. 선생이 구조물과 물체(문·창문·카운터 등)를 한 마스크로
  합친 경우는 통째로 빼지 않고 COCO 라벨로 나눈다: 배경 화소는 빼고, COCO 물체 화소는 그 COCO 조각(떨어진 조각마다)을 물체
  정답으로 남긴다(2026-10-11, 사용자). 카펫·물은 물체(그대로).
  **미정**: "대부분" 기준값(지금 0.5), 점수 기준(지금 0.5).
- 배경 19 종:
  - 벽: wall-brick, wall-stone, wall-tile, wall-wood, wall-other-merged
  - 천장: ceiling-merged
  - 바닥: floor-wood, floor-other-merged, road, pavement-merged, dirt-merged, sand, gravel, snow, grass-merged,
    playingfield, railroad, platform
  - 하늘: sky-other-merged
  - 건물: building-other-merged, house, roof (2026-10-11 추가)
- **결정** 최소 크기 없음: 로봇이 집는 물체는 작게 보이는 경우가 많다. 증강 뒤 상자가 2 화소 이하가 된 정답만 학습기가 자동으로
  뺀다(Ultralytics 와 같음).
- (이전 안, 쓰지 않음) COCO 정답만: 물체 80 종 + stuff 34 종을 떨어진 조각마다 나눠 정답으로. 위 품질 문제로 바꿨다.
  변환기 pytools/coco_build.py 는 시험(검증 지표, 구조물 화소)용으로 남긴다.

stuff 53 종: banner, blanket, bridge, cardboard, counter, curtain, door-stuff, floor-wood, flower, fruit, gravel, house,
light, mirror-stuff, net, pillow, platform, playingfield, railroad, river, road, roof, sand, sea, shelf, snow, stairs, tent,
towel, wall-brick, wall-stone, wall-tile, wall-wood, water-other, window-blind, window-other, tree-merged, fence-merged,
ceiling-merged, sky-other-merged, cabinet-merged, table-merged, floor-other-merged, pavement-merged, mountain-merged,
grass-merged, dirt-merged, paper-merged, food-other-merged, building-other-merged, rock-merged, wall-other-merged, rug-merged.

## 5. 학습

- **결정** 출발점은 YOLOv8s-seg(COCO) 가중치(2026-10-10, 사용자). FastSAM-s 는 SA-1B 2 %(약 22 만 장)를 SAM 이 자동으로
  만든 정답지(부분 마스크가 섞인 과분할)로 배워 그 습관을 안고 시작하게 된다. YOLOv8s-seg 는 COCO 의 사람이 그린 통째 마스크로
  배웠고 구조가 같다. 다른 점은 마지막 클래스 출력층(80 → 1)뿐이라 그 층만 새로 초기화(Ultralytics bias 초기값)한다.
- **결정** 학습기: 이 저장소의 C++/CUDA 학습기, fp16 + 동적 loss scaling, CUDA graph. Ultralytics 와 fp32 정합성 확인됨.
- **결정** 하이퍼파라미터: FastSAM 과 같은 Ultralytics 기본값 — SGD lr0 0.01, lrf 0.01, momentum 0.937,
  weight decay 5e-4, 워밍업 3 epoch, nbs 64, overlap_mask False, close_mosaic 0, mosaic·scale 0.5·translate 0.1·
  fliplr 0.5·HSV, EMA.
- **결정** epoch 상한 100(FastSAM 과 같음). lr 일정은 100 epoch 기준으로 고정.
- **결정** 매 epoch 검증(Ultralytics 기본 val=True 와 같음). 학습을 멈추지 않고, 직전 epoch EMA 가중치를 다른 프로세스가
  검증셋(5,000 장, 약 4 분)으로 잰다 — 학습은 약 5 % 만 느려진다(100 epoch 약 10 시간).
- **결정** 가장 좋은 epoch 의 가중치를 best 로 따로 남긴다. 최종 모델은 best.
- **결정** 조기 종료: 검증 지표가 20 epoch 연속 좋아지지 않으면 멈춘다.
- **결정** 배치 32(FastSAM 과 같음), nbs 64 에 맞춰 2 번 누적. 640 입력에서 GPU 8.3 GB, 스텝 90.5 ms(초당 354 장),
  epoch(113k 장)당 약 5.4 분.

## 6. 평가

- **결정** 검증 지표: 클래스 구분 없는 mask AP(COCO 방식, IoU 0.5~0.95, iscrowd 는 무시 영역) — best·조기 종료 기준.
  보조로 크기별 재현율, 벽·천장·바닥·하늘 위 헛 마스크 수, 물체당 마스크 수(과분할)를 함께 기록.
- **결정** 최종 시험: 원래 FastSAM-s, 예전 ObjectSAM 과 같은 평가 도구·같은 세트로 비교한다.
  - 예전 ObjectSAM 평가 세 곳(robot-agent `eval_det.py`, 2 장마다 하나): BEHAVIOR 시뮬 집 10 곳(689 프레임),
    COCO val2017 1,500 장, ADE20K 실내 val 1,021 장. 구조물 = 벽·천장·바닥(이 세트들의 정의).
  - COCO val2017 5,000 장(우리 규칙의 정답): 클래스 구분 없는 mask AP.
- **기준 수치**(GitHub juyoung020/ObjectSAM main 의 docs/TRAINING.md; 물체당 마스크 수의 COCO·ADE 는 같은 도구로 다시 잰
  `data/eval/base_*.json`. FastSAM-s 는 416 입력 기준):

  | | FastSAM-s | 예전 ObjectSAM |
  |---|---|---|
  | **BEHAVIOR** 재현율 전체 / 문·창·계단 / 처음 보는 범주 | 0.793 / 0.866 / 0.668 | 0.803 / 0.864 / 0.695 |
  | 구조물 헛 마스크 / 프레임 | 4.44 | 1.55 |
  | 물체당 마스크 수 | 2.77 | 2.51 |
  | 두 물체 합친 마스크 / 프레임 | 1.66 | 1.69 |
  | **COCO 1,500** 재현율 전체 / 작은 / 큰 / 문·창·계단 | 0.499 / 0.055 / 0.962 / 0.863 | 0.530 / 0.068 / 0.976 / 0.863 |
  | 구조물 헛 마스크 / 프레임 | 6.06 | 6.91 |
  | 물체당 마스크 수 | 3.14 | 3.97 |
  | 두 물체 합친 마스크 / 프레임 | 3.33 | 4.76 |
  | **ADE 실내** 재현율 전체 / 문·창·계단 | 0.581 / 0.785 | 0.628 / 0.816 |
  | 구조물 헛 마스크 / 프레임 | 10.05 | 10.81 |
  | 물체당 마스크 수 | 2.29 | 3.01 |
  | 두 물체 합친 마스크 / 프레임 | 3.75 | 6.31 |

- **결정** 합격 기준(세 세트 모두):
  1. 놓치지 않기: 재현율 전체·작은 물체가 FastSAM-s 보다 뚜렷이 낮지 않을 것(사진 짝 부트스트랩 95 % 구간 위 끝 ≥ 0,
     평균 차 ≥ −0.01).
  2. 구조물 헛 마스크 ≤ FastSAM-s 의 50 %: BEHAVIOR ≤ 2.22, COCO ≤ 3.03, ADE ≤ 5.03.
  3. 과분할 줄이기: 물체당 마스크 수 < 두 비교 대상의 작은 값: BEHAVIOR < 2.51, COCO < 3.14, ADE < 2.29.
  4. 종합: COCO val2017(5,000, 우리 규칙)의 mask AP 가 두 비교 대상보다 높을 것.

## 7. 산출물

- **결정** 산출물: ONNX(배포용 표준, ovdet 등이 받아 TensorRT 로 빌드) + .pt(Python·Ultralytics 용).
  TensorRT 엔진은 GPU·TensorRT 판마다 달라 배포하지 않고, 쓸 기기에서 ONNX 로 빌드하는 방법만 문서에 적는다.
  출력 모양은 FastSAM-s 와 같게(기존 후처리 그대로 쓰기).
- **미정** 공개 여부(GitHub).
