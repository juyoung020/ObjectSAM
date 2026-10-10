// BatchNorm(학습 모드) + SiLU, 출력 여러 곳에 바로 쓰기(concat 슬라이스·2배 최근접 업샘플·잔차 더하기).
//
// 정방향: z(conv 출력, NHWC packed, [M=N*H*W, C]) → 채널 통계 → a = silu(γ·(z-μ)·istd + β) [+ res] → 목적지들.
// 역방향: dA = Σ 목적지 기울기(업샘플 목적지는 2×2 합) → g = dA·silu'(y) → dγ, dβ, dz.
// 통계는 블록별 fp32 부분합 → 채널별 double 합산(분산 상쇄 오차 방지).
#pragma once
#include "core.h"

constexpr int kMaxDest = 4;
struct Dest {
  act_t* p = nullptr;   // 목적지 버퍼(NHWC packed, 채널 ctot)
  act_t* g = nullptr;   // 같은 모양의 기울기 버퍼
  int ctot = 0;         // 목적지 채널 수
  int choff = 0;        // 목적지 안 시작 채널
  int c0 = 0, c1 = 0;   // 이 층 출력 중 보낼 채널 [c0, c1)
  int up = 1;           // 1 그대로, 2 최근접 2배 업샘플
};
struct DestList {
  Dest d[kMaxDest];
  int n = 0;
};

struct BnShape {
  int N, H, W, C;
  __host__ __device__ int64_t M() const { return int64_t(N) * H * W; }
};

// 시작 때 한 번(graph 캡처 전): 마지막 블록 판정용 장치 카운터를 잡는다.
void bn_init();

// 블록 수(부분합 버퍼 크기)를 정한다.
int bn_partial_blocks(const BnShape& s);

// 정방향 통계: z → mean, istd (fp32, [C]); running 통계 갱신(momentum, 불편 분산).
// scale = γ·istd, shift = β − μ·scale 도 만든다(적용 커널용).
void bn_forward_stats(const act_t* z, const BnShape& s, const float* gamma, const float* beta, float* run_mean,
                      float* run_var, float momentum, float eps, float2* partial, float* mean, float* istd,
                      float* scale, float* shift, cudaStream_t st);

// 평가 모드: running 통계로 scale/shift.
void bn_eval_scale(int C, const float* gamma, const float* beta, const float* run_mean, const float* run_var, float eps,
                   float* scale, float* shift, cudaStream_t st);

// 적용: a = silu(scale·z + shift) (+res) → 목적지들. act=false 면 silu 없이.
void bn_forward_apply(const act_t* z, const BnShape& s, const float* scale, const float* shift, const act_t* res,
                      const DestList& dl, bool act, cudaStream_t st);

// 역방향. dz 를 쓰고, dγ·dβ 는 gw_gamma/gw_beta 에 (accum ? 더하기 : 쓰기).
// res_grad != nullptr 이면 잔차 입력 기울기에 dA 를 (res_accum ? 더하기 : 쓰기).
void bn_backward(const act_t* z, const BnShape& s, const float* gamma, const float* mean, const float* istd,
                 const float* scale, const float* shift, const DestList& dl, bool act, float2* partial,
                 float* sums /*[2C] 작업 공간*/, act_t* dz, float* gw_gamma, float* gw_beta, bool accum,
                 act_t* res_grad, bool res_accum, cudaStream_t st);
