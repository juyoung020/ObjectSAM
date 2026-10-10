# FastSAM 신경망·학습 조사

작성 2026-10-10. 출처는 공개 가중치 파일(`FastSAM-s.pt`, `FastSAM-x.pt`)에 든 모델 정의와 학습 기록,
FastSAM 논문, SAM 논문이다. 가중치에서 직접 읽은 값과 논문에 적힌 값을 구분해 적는다.

## 요약

- FastSAM 의 신경망은 **표준 YOLOv8-seg 를 그대로** 쓰고 클래스 수만 1("object")로 바꾼 것이다.
  FastSAM 고유의 구조 변경은 없다. x 만 reg_max(상자 회귀 칸 수)를 16 → 26 으로 바꿨다.
- 두 모델 모두 **처음부터(무작위 초기화) 학습**했다. COCO 사전학습 가중치에서 출발하지 않았다.
- 학습 데이터는 SA-1B 의 2 %. SA-1B 마스크는 **SAM 이 자동으로 만든 것**이고 이름(클래스)이 없다.
  FastSAM 은 그 마스크를 전부 한 클래스로 놓고 배웠다 → SAM 이 자르는 대로(벽·하늘 포함, 전체·부분 모두) 자른다.
- FastSAM-s 는 **입력 640·reg_max 16**, FastSAM-x 는 **입력 1024·reg_max 26** 으로 학습했다.
  논문의 "1024, reg_max 26" 은 x 기준이다.

## 1. 모델 한눈에 (가중치에서 읽음)

| | FastSAM-s | FastSAM-x |
|---|---|---|
| 모델 정의 | `yolov8s-seg.yaml` | `yolov8x-seg.yaml` |
| 깊이 / 폭 / 최대 채널 | 0.33 / 0.50 / 1024 | 1.00 / 1.25 / 512 |
| 파라미터 | 11,790,483 | 72,234,149 |
| 층 수(모듈) | 23 | 23 |
| 클래스 수 `nc` | 1 | 1 |
| reg_max | 16 | 26 |
| 머리 가지 폭: 상자 c2 / 점수 c3 / 마스크 계수 c4 | 64 / 128 / 32 | 104 / 320 / 80 |
| proto 채널 / 마스크 계수 수 | 128 / 32 | 320 / 32 |
| 출력 단계(stride) | 8, 16, 32 | 8, 16, 32 |
| Ultralytics 판 | 8.0.107 (2023-06-21) | 8.0.107 (2023-06-17) |

머리 가지 폭은 Ultralytics `Detect` 의 규칙을 따른다: `c2 = max(16, ch0/4, 4·reg_max)`,
`c3 = max(ch0, min(nc, 100))`, `c4 = max(ch0/4, nm)` (ch0 = 가장 얕은 출력 단계 채널; s 128, x 320).

## 2. 학습 설정 (가중치의 `train_args`)

| 항목 | FastSAM-s | FastSAM-x |
|---|---|---|
| 시작 모델 `model` | `yolov8s-seg.yaml` | `yolov8x-seg.yaml` |
| `pretrained` | False | False |
| 데이터 `data` | `ultralytics/datasets/sa.yaml` (SA-1B) | 같음 |
| 입력 `imgsz` | **640** | **1024** |
| epoch / 배치 | 100 / 32 | 100 / 32 |
| 옵티마이저 / lr0 / lrf | SGD / 0.01 / 0.01 | 같음 |
| momentum / weight decay / warmup | 0.937 / 0.0005 / 3 epoch | 같음 |
| mosaic / close_mosaic | 1.0 / 0 (끝까지 mosaic) | 같음 |
| overlap_mask / mask_ratio | False / 4 | 같음 |
| 증강 | scale 0.5, translate 0.1, fliplr 0.5, hsv 0.015/0.7/0.4 | 같음 |
| nbs | 64 | 64 |

즉 reg_max(x)를 빼면 **Ultralytics 기본 하이퍼파라미터 그대로** 다.

## 3. 층 구성 (가중치에서 읽음)

YAML 의 반복 수는 깊이 배율이 곱해져 실제 반복이 된다(s: ×0.33 → 3→1, 6→2; x: ×1.0 그대로).
출력 채널은 폭 배율이 곱해진 실제 값.

### FastSAM-s (YOLOv8s-seg, nc=1, reg_max 16)

| # | 입력 | 모듈(실제 반복) | 출력 채널 | 크기(640 입력) | 파라미터 |
|---|---|---|---|---|---|
| 0 | 영상 | Conv 3×3 s2 | 32 | 320² | 928 |
| 1 | 0 | Conv 3×3 s2 | 64 | 160² | 18,560 |
| 2 | 1 | C2f ×1 (잔차) | 64 | 160² | 29,056 |
| 3 | 2 | Conv 3×3 s2 | 128 | 80² | 73,984 |
| 4 | 3 | C2f ×2 (잔차) | 128 | 80² | 197,632 |
| 5 | 4 | Conv 3×3 s2 | 256 | 40² | 295,424 |
| 6 | 5 | C2f ×2 (잔차) | 256 | 40² | 788,480 |
| 7 | 6 | Conv 3×3 s2 | 512 | 20² | 1,180,672 |
| 8 | 7 | C2f ×1 (잔차) | 512 | 20² | 1,838,080 |
| 9 | 8 | SPPF (maxpool 5 ×3) | 512 | 20² | 656,896 |
| 10 | 9 | Upsample ×2 (최근접) | 512 | 40² | 0 |
| 11 | 10, 6 | Concat | 768 | 40² | 0 |
| 12 | 11 | C2f ×1 | 256 | 40² | 591,360 |
| 13 | 12 | Upsample ×2 | 256 | 80² | 0 |
| 14 | 13, 4 | Concat | 384 | 80² | 0 |
| 15 | 14 | C2f ×1 → **P3** | 128 | 80² | 148,224 |
| 16 | 15 | Conv 3×3 s2 | 128 | 40² | 147,712 |
| 17 | 16, 12 | Concat | 384 | 40² | 0 |
| 18 | 17 | C2f ×1 → **P4** | 256 | 40² | 493,056 |
| 19 | 18 | Conv 3×3 s2 | 256 | 20² | 590,336 |
| 20 | 19, 9 | Concat | 768 | 20² | 0 |
| 21 | 20 | C2f ×1 → **P5** | 512 | 20² | 1,969,152 |
| 22 | 15, 18, 21 | Segment 머리 | — | — | 2,770,931 |

- **Conv** = Conv2d(bias 없음) + BatchNorm(eps 1e-3, momentum 0.03) + SiLU.
- **C2f** = 1×1 Conv 로 2c 채널 → 반으로 나눠, 뒤 절반에 Bottleneck(3×3 Conv 둘, 백본은 잔차 더하기) 을 n 번 잇고
  나온 것을 모두 이어(concat) 1×1 Conv.
- **SPPF** = 1×1 Conv(c/2) → 5×5 maxpool 을 세 번 이어 넷을 concat → 1×1 Conv.
- **Segment 머리**(단계 P3·P4·P5 마다): 상자 가지 Conv3×3(c2)→Conv3×3(c2)→Conv1×1(4·reg_max),
  점수 가지 Conv3×3(c3)→Conv3×3(c3)→Conv1×1(1), 마스크 계수 가지 Conv3×3(c4)→Conv3×3(c4)→Conv1×1(32).
  proto: P3 → Conv3×3(128) → ConvTranspose 2×2 s2 → Conv3×3(128) → Conv1×1(32), 즉 입력의 1/4 해상도 원형 32 장.
- **출력**: 앵커마다 상자(DFL 4·reg_max 칸의 기대값), 점수 1, 마스크 계수 32. 마스크 = 계수 · proto 를 상자로 자른 것.
- **손실**(Ultralytics v8SegmentationLoss): TAL 할당(top-10, α 0.5, β 6), 상자 CIoU ×7.5, 점수 BCE ×0.5,
  DFL ×1.5, 마스크 BCE(상자 안, 넓이로 나눔) ×7.5.

### FastSAM-x (YOLOv8x-seg, nc=1, reg_max 26)

같은 연결에 채널만 넓고(80/160/320/640) 반복이 깊다(C2f 3·6·6·3, 목 3).
층별 파라미터: 0 2,320 · 1 115,520 · 2 436,800 · 3 461,440 · 4 3,281,920 · 5 1,844,480 · 6 13,117,440 ·
7 3,687,680 · 8 6,969,600 · 9 1,025,920 · 12 7,379,200 · 15 1,948,800 · 16 922,240 · 18 7,174,400 ·
19 3,687,680 · 21 7,379,200 · 22(머리) 12,799,509. 머리 가지 폭 104 / 320 / 80, proto 320.

## 4. 논문에 적힌 것 (FastSAM, arXiv 2306.12156, 4장 Implementation Details)

- "FastSAM employs a YOLOv8-x model as the main part of its architecture, with the input size of 1024."
- "FastSAM's training was carried out on 2% of the SA-1B dataset."
- "We train the model for 100 epochs using the default hyper-parameter settings except that the reg max in the
  bounding box regression module is changed from 16 to 26 for predicting large instances."
- 3.2 절: 구조 설계는 YOLOv8-seg 를 그대로 쓰고, 더 정교한 설계는 범위 밖이라 했다.
- 한계(6 장): 작은 물체 마스크가 네모에 가깝고, 큰 물체는 상자 경계에 자국이 생긴다(YOLACT 방식의 약점).
  SA-1B 의 1/50 만 써서 데이터를 늘리면 더 좋아질 수 있다.

## 5. 학습 데이터 SA-1B 의 마스크 (SAM 논문, arXiv 2304.02643)

- SA-1B 는 SAM 과 함께 공개된 분할 학습용 데이터(1,100 만 장, 마스크 11 억 개). 마스크는 COCO RLE,
  **클래스 없음**("do not have classes"). 라이선스는 연구(비상업) 용.
- 공개 마스크는 data engine 의 **완전 자동 단계**에서 나온 것(전체의 99.1 %, "SA-1B only includes
  automatically generated masks"). 사람이 그린 마스크 세트는 따로 없다.
- 만든 모델(부록 B): 자동 생성용으로 따로 학습한 SAM — 수동·반자동 단계 데이터로만, 177,656 반복,
  점·마스크 프롬프트만, 마스크 디코더 3 층. 이 판의 이미지 인코더 크기는 부록에 명시돼 있지 않다
  (data engine 동안 인코더를 ViT-B → ViT-H 로 키웠다는 문장은 있다).
- 만든 방법: 32×32 격자 점 + 확대 영역 20 개(2×2·4×4 창)마다 마스크 후보 → 거르기:
  예측 IoU ≥ 88.0, 안정성(문턱 ±1 에서 IoU) ≥ 95.0, 영상 전체를 덮는 마스크 제거, NMS 0.7,
  100 화소 미만 조각 제거·작은 구멍 메우기.
- 품질: 자동 마스크 약 5 만 개를 전문가가 다듬은 판과 비교해 94 % 가 IoU 90 % 이상, 97 % 가 75 % 이상.
- 이 프로젝트에서 받아 본 tar 1,000 장(마스크 105,732 개): 장당 평균 106 개(2–562), predicted_iou 최소 0.880,
  stability_score 최소 0.950(위 거르기 기준과 같음), 마스크 넓이 중앙값은 영상의 0.1 %.
- 같은 물체가 전체·부분·하위 부분으로 겹쳐 들어 있다(점 하나에서 세 단계 후보를 내는 SAM 방식).
  FastSAM 이 벽·하늘을 자르고 물체를 조각내는(과분할) 것은 이 정답을 그대로 배웠기 때문이다.

## 6. 이 프로젝트와의 관계

- 학습기(`src/`)의 신경망은 FastSAM-s 와 같은 YOLOv8s-seg, nc 1. reg_max 는 가중치에서 읽어 16·26 모두 된다.
  Ultralytics 8.4.171 과 fp32 정합성 시험(`tests/parity.cpp`)을 통과했다.
- "FastSAM 과 똑같이" 를 COCO 2017 로 하면: YOLOv8s-seg, nc 1, 처음부터, 100 epoch, SGD lr0 0.01,
  overlap_mask False, close_mosaic 0. 입력은 FastSAM-s 그대로면 640·reg_max 16, x 방식이면 1024·reg_max 26.

## 출처

- FastSAM: Zhao et al., *Fast Segment Anything*, 2023 — https://arxiv.org/abs/2306.12156
- SAM / SA-1B: Kirillov et al., *Segment Anything*, 2023 — https://arxiv.org/abs/2304.02643
- SA-1B 다운로드·라이선스 — https://ai.meta.com/datasets/segment-anything-downloads/
- 공개 가중치: Ultralytics `FastSAM-s.pt`, `FastSAM-x.pt` (체크포인트의 `train_args`, 모델 정의)
