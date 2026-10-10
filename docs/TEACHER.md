# 선생 모델: CropFormer Hornet-L

학생(FastSAM 과 같은 YOLOv8s-seg)의 정답 마스크를 만드는 선생. 클래스 없이 화면의 모든 개체를 **물체 하나에 마스크 하나**로
자르는 개체 분할(entity segmentation) 모델이다.

## 왜 이 모델인가

- 과제가 우리와 같다: 이름 없이 모든 개체를 통째로 자른다. 부분 마스크를 내지 않는다.
- 코드와 가중치가 모두 공개된 모델 중 EntitySeg 개체 분할 점수가 가장 높다(2026-10 조사, 두 번 따로 확인).

  | 모델 | EntitySeg AP^e (저해상도 / 고해상도) |
  |---|---|
  | **CropFormer Hornet-L** | **49.1** / 51.5 |
  | Mask2Former Hornet-L | 47.1 / 49.2 |
  | E-SAM ViT-H (2025, 코드 미공개) | 48.9 / — |
  | CropFormer Swin-L | 45.8 / 48.2 |

  우리는 저해상도 설정(짧은 변 800)으로 돌린다. 고해상도 설정은 입력이 2700×4500 이라 16 GB 에 들어가지 않는다.
- 라이선스: 코드·가중치 CC BY-NC 4.0(비상업). 결과 학생 모델도 비상업 연구용이다(COCO 이미지 조건도 마찬가지).

출처: [CropFormer / EntitySeg](https://github.com/qqlu/Entity/blob/main/Entityv2/README.md),
[High-Quality Entity Segmentation](https://arxiv.org/abs/2211.05776), [E-SAM](https://arxiv.org/abs/2503.12094).

## 돌리는 방법(원본 데모 예측기와 같음)

- 화면 다섯 장: 원본을 짧은 변 800(긴 변 ≤ 1333)으로, 원본의 70 % 크기로 자른 네 장(보폭 60 %)도 같은 규칙으로.
  PIL 양선형 리사이즈, 정규화, 32 의 배수로 0 패딩.
- 백본 HorNet-L(2.2 억 파라미터 중 대부분) → MSDeformAttn 픽셀 디코더 6 층 → 디코더(질의 200 개): 화면마다 9 층,
  다섯 화면의 질의를 모은 3d 질의, 다시 9 층.
- 후처리: 다섯 화면의 마스크 확률을 원본 좌표에서 평균, 점수 높은 순으로 화소를 하나의 마스크에만 준다(겹치지 않음).
  점수 = 물체 확률 × 마스크 안 평균 확률.

## C++/CUDA 포팅(teacher/)

PyTorch 원본을 C++/CUDA(cuBLAS·cuBLASLt·cuDNN·cuFFT)로 옮겼다. 라벨 만들기에는 PyTorch 가 필요 없다.

| 파일 | 내용 |
|---|---|
| `teacher/tk.cu, tk.cuh` | 커널: LayerNorm, GroupNorm, 깊이별 conv, 전역 필터(FFT), MSDeformAttn, 마스크 어텐션, 후처리 |
| `teacher/tnet.cpp, tnet.h` | 네트워크. 정밀도는 부분마다 템플릿(`TeacherNetT<백본, 픽셀 디코더, 디코더>`) |
| `teacher/prep.cpp` | 전처리(Pillow Resample.c 를 그대로 옮긴 리사이즈) |
| `teacher/label.cpp` | 라벨 도구 `build/teacher_label`: 읽기 스레드 → GPU → 쓰기, 이어 하기 가능 |
| `tests/teacher_parity.cpp` | 단계별 정합성 시험 |
| `pytools/teacher_dump.py` | PyTorch 기준값 덤프(선생 환경, `scripts/setup_teacher.sh`) |
| `pytools/ftl.py` | 결과(FTL1) 읽기 |

### 정합성(fp32, PyTorch 와 비교, 사진 2 장)

| 단계 | 상대 오차(최대) |
|---|---|
| 전처리 | 0 (비트 단위로 같음) |
| 백본 res2..res5 | ≤ 2e-5 |
| 픽셀 디코더 | ≤ 8e-6 |
| 디코더(3d 마스크 로짓) | 1.2e-5 |
| 최종 마스크 | 다른 화소 0 개, 점수 1.7e-6 |

### 정밀도와 모델 자체의 흔들림

원본 모델(fp32)은 눈에 보이지 않는 입력 차이에도 결과가 바뀐다(복잡한 부엌 사진, 점수 ≥ 0.5 마스크 137 개 기준).

| 입력에 더한 잡음(정규화 단위; 픽셀값 1 단계 ≈ 0.017) | 평균 IoU | IoU < 0.5 |
|---|---|---|
| ±0.001 | 0.951 | 6 |
| ±0.005 | 0.907 | 11 |
| ±0.017 | 0.824 | 17 |
| 비교: 모두 fp16 으로 계산 | 0.95 ~ 0.98 | 2 ~ 6 |

fp16 의 차이는 모델이 원래 흔들리는 범위(픽셀값 1/17 단계 잡음) 안이라 라벨 품질에 영향이 없다. 그래서 모두 fp16 으로 돌린다.
(누적은 늘 fp32, 잔차 흐름도 fp32.)

### 속도(RTX 5070 Ti, 800×1088, 화면 다섯 장)

| | 한 장 |
|---|---|
| PyTorch fp32(원본) | 840 ms |
| C++ fp32 | 510 ms |
| C++ fp16 | 195 ms(GEMM 95 ms: fp32 누적 텐서코어 한계에 가깝다) |
| 라벨 도구(읽기·후처리·쓰기 포함, COCO 섞인 모양) | 약 240 ms |

## 결과 형식(FTL1)

`teacher/label.cpp` 머리말. 사진마다 겹치지 않는 마스크 지도(RLE)와 순위별 점수·넓이. 넓이 > 0 인 마스크는 점수와 관계없이
모두 남긴다(점수 기준은 학습 데이터를 만들 때 정한다). 사진당 약 15 KB.

## 다시 만들기

```bash
scripts/setup_teacher.sh                                   # PyTorch 선생 환경 + 가중치(정합성 기준값용)
.venv-teacher/bin/python pytools/teacher_dump.py ref data/coco/train2017/000000000164.jpg
cmake --build build -j --target teacher_parity teacher_label
build/teacher_parity weights/teacher/cropformer_hornet_l.fsow data/parity/teacher_ref_000000000164.fsow fp32
build/teacher_label --list data/teacher/train2017.list --out data/teacher/train2017.ftl
```
