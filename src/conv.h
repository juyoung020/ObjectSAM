// cuDNN(legacy API) conv: NHWC packed, act_t 입출력, fp32 누적. 가장 빠른 알고리즘을 시작 때 한 번 재서 고른다.
// sm_120 에서는 graph API 의 융합·슬라이스 입출력이 지원되지 않거나 느려서(tests/probe_cudnn_strided.cu) 쓰지 않는다.
#pragma once
#include "core.h"

struct ConvGeom {
  int N, H, W, C;   // 입력
  int K, R, S;      // 출력 채널, 커널
  int stride, pad;
  int P() const { return (H + 2 * pad - R) / stride + 1; }
  int Q() const { return (W + 2 * pad - S) / stride + 1; }
};

class ConvOp {
 public:
  // transposed=true: ConvTranspose2d(입력 C→출력 K, 커널 R=S=stride, 패딩 0). 가중치는 PyTorch 모양 (C, K, R, S)
  // 그대로 = 같은 conv(K→C) 의 (K'=C, C'=K) 필터.
  // dt: 입출력 자료형(기본 act_t). 선생 추론은 fp16/fp32 를 섞어 쓴다.
  void init(cudnnHandle_t h, const ConvGeom& g, bool transposed = false, cudnnDataType_t dt = kActDnn);
  // 시작 때 한 번: 실제 버퍼로 알고리즘을 재고 필요한 작업 공간 크기를 돌려준다.
  size_t tune(act_t* x, act_t* w, act_t* y, act_t* dx, act_t* dw, act_t* dy, void* ws, size_t ws_limit);
  // 추론만: 정방향 알고리즘만 잰다(전치 아님). 포인터 형은 init 의 dt.
  size_t tune_fwd(void* x, void* w, void* y, void* ws, size_t ws_limit);
  void fwd_v(const void* x, const void* w, void* y, void* ws, size_t wsz) const;
  void fwd(const act_t* x, const act_t* w, act_t* y, void* ws, size_t wsz, float beta = 0.f) const;
  void dgrad(const act_t* dy, const act_t* w, act_t* dx, void* ws, size_t wsz, float beta = 0.f) const;
  void wgrad(const act_t* x, const act_t* dy, act_t* dw, void* ws, size_t wsz, float beta = 0.f) const;
  const ConvGeom& geom() const { return g_; }
  int64_t out_elems() const;
  float tuned_ms[3] = {0, 0, 0};

 private:
  cudnnHandle_t h_ = nullptr;
  ConvGeom g_{};
  bool tr_ = false;
  cudnnDataType_t dt_ = kActDnn;
  // tr_=false: x=입력(small or big), y=출력. tr_=true: 내부 conv 기준 big=출력 y, small=입력 x.
  cudnnTensorDescriptor_t xd_{}, yd_{};
  cudnnFilterDescriptor_t wd_{};
  cudnnConvolutionDescriptor_t cd_{};
  cudnnConvolutionFwdAlgo_t fa_{};
  cudnnConvolutionBwdDataAlgo_t da_{};
  cudnnConvolutionBwdFilterAlgo_t wa_{};
};
