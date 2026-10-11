// 보조 커널: SPPF maxpool 사슬, 채널 합(bias 기울기), bias 더하기, 형 변환·누적.
#pragma once
#include "core.h"

// SPPF: cat[..., 0:c] 에 cv1 출력이 있을 때 5×5 stride1 pad2 maxpool 을 세 번 이어 cat 의 1·2·3 슬라이스에 쓴다.
void sppf_pool_fwd(act_t* cat, int N, int H, int W, int c, cudaStream_t st);
// 역방향: dcat 의 1·2·3 슬라이스 기울기를 maxpool 역으로 되돌려 dcat 의 0 슬라이스에 더한다.
// tmp: fp32 [N*H*W*c] 둘(작업 공간).
void sppf_pool_bwd(const act_t* cat, act_t* dcat, int N, int H, int W, int c, float* tmp_a, float* tmp_b,
                   cudaStream_t st);

// 채널 합: out[c] (=|+=) Σ_rows x[row, c]. partial: [블록 수 * C] 작업 공간(chan_sum_blocks).
int chan_sum_blocks(int64_t M, int C);
void chan_sum(const act_t* x, int64_t M, int C, float* partial, float* out, bool accum, cudaStream_t st);

// y[row, c] += bias[c]
void add_bias(act_t* y, int64_t M, int C, const float* bias, cudaStream_t st);

// 형 변환과 누적.
void f32_to_act(const float* src, act_t* dst, int64_t n, cudaStream_t st);
void act_accum_f32(const act_t* src, float* dst, int64_t n, bool accum, cudaStream_t st);
void fill_f32(float* p, float v, int64_t n, cudaStream_t st);
// acc[0..n) += src[0..n), acc[n] += 1 (학습 로그용 손실 합을 장치에서 모은다 — graph 안에서 호스트 왕복 없이)
void accum_items(const float* src, float* acc, int n, cudaStream_t st);

// 첫 층(입력 3 채널, 3×3 stride 2 pad 1, 출력 K=32) wgrad: dw(KRSC, act_t) = Σ dz ⊗ patch(x). 쓸 수 없는 모양·fp32 판은
// false(호출자가 cuDNN 으로). tmp: fp32 [K·27] 작업 공간.
bool wgrad_rgb_s2(const act_t* x, const act_t* dz, int N, int H, int W, int K, float* tmp, act_t* dw, cudaStream_t st);
