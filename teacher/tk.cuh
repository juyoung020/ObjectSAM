// 선생(CropFormer Hornet-L) 추론 커널. 배치는 행 우선 [토큰, 채널](NHWC).
// 활성값 형 A ∈ {float, __half} 는 템플릿: 부분마다 정밀도를 고른다(tnet.h). 행렬 곱은 cuBLAS(fp32 누적), 잔차 흐름은 fp32.
#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cufft.h>

#include <map>
#include <tuple>

#include <type_traits>

#include "core.h"

#define CUBLAS_CHECK(x)                                                                                  \
  do {                                                                                                   \
    cublasStatus_t e_ = (x);                                                                             \
    FSO_ASSERT(e_ == CUBLAS_STATUS_SUCCESS, "cuBLAS %s: %d", #x, int(e_));                               \
  } while (0)
#define CUFFT_CHECK(x)                                                                                   \
  do {                                                                                                   \
    cufftResult e_ = (x);                                                                                \
    FSO_ASSERT(e_ == CUFFT_SUCCESS, "cuFFT %s: %d", #x, int(e_));                                        \
  } while (0)

template <class A> constexpr cudnnDataType_t dnn_t() { return std::is_same<A, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF; }

// 행 우선 Y[M,N] = alpha·X[M,K]·W[N,K]ᵀ + beta·Y (nt) 또는 alpha·X[M,K]·W[K,N] + beta·Y (nn). X, W 는 A, Y 는 TY.
// batch>1 이면 X, W, Y 를 sx, sw, sy 원소씩 건너 batch 번. fp32 입력은 TF32 를 쓰지 않는다.
template <class A, class TY>
void gemm_nt(cublasHandle_t h, int64_t M, int64_t N, int64_t K, const A* X, int ldx, const A* W, int ldw, TY* Y, int ldy,
             float alpha = 1.f, float beta = 0.f, int batch = 1, int64_t sx = 0, int64_t sw = 0, int64_t sy = 0);
template <class A, class TY>
void gemm_nn(cublasHandle_t h, int64_t M, int64_t N, int64_t K, const A* X, int ldx, const A* W, int ldw, TY* Y, int ldy,
             float alpha = 1.f, float beta = 0.f, int batch = 1, int64_t sx = 0, int64_t sw = 0, int64_t sy = 0);

// cuBLASLt: 행렬 곱 + 에필로그(bias, ReLU+bias)를 한 번에. Y = act(X·Wᵀ + bias) + beta·Y.
// bias [N] 는 Y 와 같은 형(cuBLASLt 는 fp16 출력에 fp32 bias 를 받지 않는다). 같은 모양은 알고리즘을 한 번만 고른다.
struct LtCtx {
  cublasLtHandle_t h = nullptr;
  void* ws = nullptr;
  size_t wsz = size_t(32) << 20;
  struct Plan {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulAlgo_t algo;
  };
  std::map<std::tuple<int64_t, int64_t, int64_t, int, int, int, int, int, int, int>, Plan> plans;
  void init();
};
enum Epi { kEpiNone = 0, kEpiBias = 1, kEpiReluBias = 2 };
template <class A, class TY>
void lt_gemm(LtCtx& lt, int64_t M, int64_t N, int64_t K, const A* X, int ldx, const A* W, int ldw, TY* Y, int ldy,
             const TY* bias, Epi epi, float beta, cudaStream_t st);

// 행렬 보기: 행 간격 ld, 첫 채널 off.
template <class T>
struct Mat {
  T* p;
  int ld;
  __host__ __device__ T* row(int64_t r) const { return p + r * ld; }
};
template <class T> Mat<T> mat(T* p, int ld, int off = 0) { return Mat<T>{p + off, ld}; }

enum Act { kNone = 0, kRelu = 1, kGelu = 2 };

// y = LN(x + add)·g + b (행마다 C 채널), 그 뒤 ·scale. add: 채널마다 더할 값(없으면 nullptr; 미뤄 둔 잔차 bias).
template <class TI, class TO>
void layer_norm(Mat<const TI> x, Mat<TO> y, int64_t rows, int C, const float* g, const float* b, float eps, cudaStream_t st,
                float scale = 1.f, const float* add = nullptr);
// y = act(x + bias)·scale. bias 없으면 nullptr.
template <class TI, class TO>
void bias_act(Mat<const TI> x, Mat<TO> y, int64_t rows, int C, const float* bias, Act act, cudaStream_t st,
              float scale = 1.f);
// y = (x + bias) · m. bias 없으면 nullptr.
template <class TI, class A>
void bias_mul(Mat<const TI> x, const float* bias, Mat<const A> m, Mat<A> y, int64_t rows, int C, cudaStream_t st);
// r += gamma ⊙ (x + bias)
void res_add(Mat<float> r, Mat<const float> x, const float* bias, const float* gamma, int64_t rows, int C, cudaStream_t st);
// r = LN(r + x + bias)(트랜스포머 post-norm), ya 에 A 사본(없으면 p=nullptr). C ≤ 512.
template <class A>
void add_ln(Mat<float> r, Mat<const float> x, const float* bias, const float* g, const float* b, float eps, Mat<A> ya,
            int64_t rows, int C, cudaStream_t st);
// out = x + y[r mod rows_y] (위치 인코딩을 화면마다 같이 쓸 때). x 는 TI.
template <class TI, class A>
void add_rows(Mat<const TI> x, Mat<const float> y, int64_t rows_y, Mat<A> out, int64_t rows, int C, cudaStream_t st);
// 겹치지 않는 k×k 패치를 행으로 펼친다: x [B,H,W,C] → y [B·(H/k)·(W/k)][(ky·k + kx)·C + c].
template <class A> void patchify(const A* x, int B, int H, int W, int C, int k, A* y, cudaStream_t st);
// 깊이별 conv(K×K, 패딩 K/2), 입력 [B,H,W] 행. 출력 채널 c → y.row(r)[c·ostep]. w: [K·K][C] fp32.
template <class A>
void dwconv(Mat<const A> x, Mat<A> y, int ostep, int B, int H, int W, int C, int K, const float* w, const float* bias,
            float scale, cudaStream_t st);
// GroupNorm(G 그룹, 화면마다) + 선택 ReLU. x [B·HW, C] → yf(fp32, 없으면 nullptr), ya(A, 없으면 nullptr).
template <class TI, class A>
void group_norm(const TI* x, float* yf, A* ya, int B, int64_t HW, int C, int G, const float* g, const float* b, float eps,
                bool relu, float* stats, cudaStream_t st);
// PyTorch F.interpolate(bilinear, align_corners=False) NHWC: x [B,Hi,Wi,C] → y [B,Ho,Wo,C]. add 가 있으면 y = add + 보간.
template <class TI, class A>
void resize_bilinear(const TI* x, int B, int Hi, int Wi, int C, A* y, int Ho, int Wo, const A* add, cudaStream_t st);

// HorNet 전역 필터의 주파수 부분: x(채널 C2, 행 [B,H,W]) → rfft2 → ·weight → irfft2 → y 채널 c 를 y.row(r)[c·ostep].
// weight: 학습된 복소 가중치 [C2][wh][ww] 그대로; 곱할 때 (H, W/2+1) 로 양선형 보간(align_corners=True)하고 1/(HW) 를 곱한다
// (PyTorch: F.interpolate + ortho rfft2·irfft2). buf: fp32 [B·C2·H·W], spec: 복소 [B·C2·H·(W/2+1)]. FFT 는 fp32.
// 계획은 모양마다 만들지만 작업 메모리는 모두가 하나를 나눠 쓴다(모양이 수십 가지라 따로 잡으면 GPU 메모리가 바닥난다).
struct GlobalFilterPlan {
  int B = 0, H = 0, W = 0, C2 = 0;
  cufftHandle r2c = 0, c2r = 0;
  size_t work = 0;  // 필요한 작업 메모리
  void init(int B, int H, int W, int C2, cudaStream_t st);
};
template <class A>
void global_filter(const GlobalFilterPlan& p, Mat<const A> x, Mat<A> y, int ostep, const float2* weight, int wh, int ww,
                   float* buf, float2* spec, void* work, cudaStream_t st);

// MSDeformAttn 샘플링: value [B, L, 256] (8 머리 × 32), off [B, L, 8·3·4·2] + off_bias, aw [B, L, 8·3·4] + aw_bias
// (softmax 전). 기준점 = 질의 토큰 자신의 ((x+0.5)/W, (y+0.5)/H). out [B, L, 256].
struct DeformLevels {
  int n = 3;
  int h[3], w[3], start[3];
};
template <class A>
void ms_deform_sample(const A* value, const float* off, const float* off_bias, const float* aw, const float* aw_bias, A* out,
                      int B, int L, const DeformLevels& lv, cudaStream_t st);

// 점수 S [heads][rows][n](fp32) → P = softmax(S, 막힌 열 제외). 막힘: mask_logit [rows][n] < 0
// (Mask2Former 의 sigmoid < 0.5). 한 행이 모두 막히면 모두 연다. mask_logit 이 nullptr 이면 막힘 없음.
template <class A>
void masked_softmax(const float* S, const float* mask_logit, A* P, int heads, int64_t rows, int n, cudaStream_t st);

// 후처리(CropFormer inference_whole_views). logit [Q][T][Hm·Wm] fp32 (1/4 해상도, 패딩 포함 크기 Hm×Wm).
// 화면 v: 패딩 크기 (Hp, Wp) 로 키운 뒤 [:sh[v], :sw[v]] 를 (oh[v], ow[v]) 로 다시 키우고 sigmoid, 원래 사진 (OH, OW) 의
// (x0[v], y0[v]) 에 더한다(화면 0 은 전체). 평균한 확률 fused [Q][OH·OW].
struct ViewGeom {
  int T, Hm, Wm, Hp, Wp, OH, OW;
  int sh[8], sw[8], oh[8], ow[8], x0[8], y0[8];
};
void fuse_views(const float* logit, int Q, const ViewGeom& g, float* fused, float* cnt, cudaStream_t st);
// 점수 내림차순 순서 order[Q] 로 화소마다 fused > 0.5 인 첫 순위 → id [OH·OW] (없으면 -1). 겹치지 않는 마스크.
// 순위 k 의 마스크 점수 = Σ_{id==k} sigmoid(fused[order[k]]) / (넓이 + 1e-6) (CropFormer 그대로: 확률에 sigmoid 를 또 씌움)
// → mscore[k], area[k].
void assign_ids(const float* fused, const int* order, int Q, int64_t HW, int16_t* id, float* mscore, int* area,
                cudaStream_t st);
// 사인 위치 인코딩(Mask2Former PositionEmbeddingSine, normalize=True, 128+128) → out[(t·H·W + p)][256].
// add[l·256..] (수준 임베딩, 없으면 nullptr) 를 더하고, T > 0 이면 화면 t 의 z 인코딩(PositionEmbeddingSine3D2D)도 더한다.
void sine_pos(int H, int W, int T, const float* add, float* out, cudaStream_t st);
// 물체 확률 p[q] = softmax(cls[q])[0] 와 내림차순 순서 order[k](같으면 번호 작은 것 먼저 = 안정 정렬). Q ≤ 1024.
void rank_scores(const float* cls, int Q, float* p, int* order, cudaStream_t st);
// 최종 점수 score[k] = p[order[k]] · mscore[k]
void final_scores(const float* p, const int* order, const float* mscore, int Q, float* score, cudaStream_t st);
