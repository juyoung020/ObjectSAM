// GPU 증강: mosaic + affine(크기·이동) + HSV + 좌우 뒤집기 → 모델 입력, 정답 상자, 정답 마스크 비트.
//
// Ultralytics(Mosaic → RandomPerspective → RandomHSV → RandomFlip)와 같은 규칙. 다른 점 하나:
// mosaic 타일 밖으로 잘린 부분의 마스크·상자를 타일 안으로 자른다(Ultralytics 는 다각형을 캔버스로만 잘라
// 잘린 원본 영역이 이웃 타일 위로 번진 라벨이 생긴다).
#pragma once
#include "dataset.h"
#include "loss.cuh"

struct AugOut {
  act_t* img;          // [B, S, S, 3] NHWC, 0..1
  GtBatch gt;          // n_gt [B], box [B][G], mask [B][G][Hp*Wp/32]
  int* gt_inst;        // [B][G] 정답 → InstInfo 번호
  int G, Hp, Wp;
};

void augment(const BatchBuf& dev, int B, int S, const AugOut& out, cudaStream_t st);
