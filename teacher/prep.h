// 선생 입력 전처리: CropFormer 데모 예측기(demo_cropformer/predictor.py)와 비트 단위로 같게.
//   화면 0: 짧은 변 800(긴 변 ≤ 1333)으로 PIL 양선형 리사이즈.
//   화면 1..4: 원본에서 0.7 크기로 자른 네 장(EntityCrop, 보폭 0.6)을 같은 규칙으로 리사이즈.
//   (x − mean) / std (RGB), 다섯 장을 32 의 배수로 0 패딩.
#pragma once
#include <cstdint>
#include <vector>

#include "tk.cuh"

struct Prepared {
  int T = 5, Hp = 0, Wp = 0;
  ViewGeom g{};
  std::vector<float> x;  // [T, Hp, Wp, 3] 정규화 끝
};

// PIL Image.resize(BILINEAR) 와 같은 8 비트 RGB 리사이즈(Pillow Resample.c: 가로 다음 세로, 22 비트 고정소수점).
void pil_resize_rgb(const uint8_t* src, int sw, int sh, int sstride, uint8_t* dst, int dw, int dh);

// rgb: [h, w, 3] (행 간격 w·3).
void prepare(const uint8_t* rgb, int h, int w, Prepared& out);
