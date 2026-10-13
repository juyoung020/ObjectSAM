// CropFormer Hornet-L(선생) 추론: 백본(HorNet-L) → 픽셀 디코더(MSDeformAttn 6 층 + FPN) → 디코더(2d 9 층, 3d 질의, 3d 9 층)
// → 후처리(5 화면 융합, 겹치지 않는 마스크). 원본: third_party/.../CropFormer, 기준값: pytools/teacher_dump.py.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "conv.h"
#include "fsow.h"
#include "tk.cuh"

// 장치 버퍼: 더 큰 크기가 필요할 때만 다시 잡는다.
inline size_t g_dbuf_held = 0;  // 지금 잡고 있는 DBuf 바이트 합(진단용, TEACHER_MEMLOG)
struct DBuf {
  void* p = nullptr;
  size_t cap = 0;
  template <class T> T* get(size_t n) {
    size_t b = n * sizeof(T);
    if (b > cap) {
      CUDA_CHECK(cudaDeviceSynchronize());  // 앞서 넣은 일이 옛 버퍼를 다 쓸 때까지
      if (p) CUDA_CHECK(cudaFree(p));
      g_dbuf_held += b - cap;
      if (getenv("TEACHER_MEMLOG") && b > (32 << 20)) {
        size_t fr, tt;
        cudaMemGetInfo(&fr, &tt);
        fprintf(stderr, "alloc %zu MB  held %zu MB  free %zu MB\n", b >> 20, g_dbuf_held >> 20, fr >> 20);
      }
      CUDA_CHECK(cudaMalloc(&p, b));
      cap = b;
    }
    return static_cast<T*>(p);
  }
  void release() {
    if (p) CUDA_CHECK(cudaFree(p));
    g_dbuf_held -= cap;
    p = nullptr, cap = 0;
  }
};

// 한 사진의 결과: 겹치지 않는 마스크(id 지도)와 순위별 점수.
struct TeacherResult {
  int OH = 0, OW = 0;
  std::vector<int16_t> id;     // [OH·OW] 순위 k(점수 내림차순) 또는 -1
  std::vector<float> score;    // [Q] 최종 점수(물체 확률 × 마스크 점수), 순위 순
  std::vector<int> area;       // [Q] 화소 수
};

// 정밀도: AB 백본, AP 픽셀 디코더, AD 디코더(어텐션 마스크가 작은 오차에 민감해 fp32 권장). 잔차·누적은 늘 fp32.
template <class AB, class AP, class AD>
class TeacherNetT {
 public:
  static constexpr int kQ = 200;

  void load(const std::string& fsow_path);
  // 모양 정하기(버퍼·계획 준비). T 화면, 패딩 크기 Hp×Wp (32 의 배수).
  void set_shape(int T, int Hp, int Wp);
  // x: 정규화·패딩된 [T, Hp, Wp, 3] (장치).
  void backbone(const AB* x);
  void pixel_decoder();
  void decoder();
  void postprocess(const ViewGeom& g, TeacherResult& out);
  // 호스트와 맞추지 않고 스트림에 넣기만 한다: 결과(id [OH·OW], 순위별 점수·넓이 [kQ])를 고정(pinned) 호스트 메모리로 복사.
  // 끝났는지는 호출한 쪽이 이 뒤에 기록한 이벤트로 안다.
  void postprocess_async(const ViewGeom& g, int16_t* h_id, float* h_score, int* h_area);
  // postprocess_async 바로 뒤에(같은 버퍼): 부품 후보(tk.cuh PartCfg)를 고정 호스트 메모리로. h_bits [OH·OW].
  // 메모리 아끼기(teacher_label --lowmem): 픽셀 디코더 뒤에 부르면 백본·픽셀 디코더 작업 버퍼를 놓아 디코더 최고 사용량을 줄인다.
  // decoder_too: 디코더 뒤에 부르면 디코더 작업 버퍼도. 호스트와 맞춘다(다음 사진에서 다시 잡는다).
  void release_scratch(bool decoder_too = false);
  // 백본을 화면 하나씩(같은 결과 모양, 작업 메모리 1/T; 백본에는 화면 사이 상호작용이 없다).
  void backbone_per_view(const AB* x);
  void parts_async(const ViewGeom& g, const PartCfg& cfg, PartInfo* h_info, int* h_n, unsigned long long* h_bits);

  cudaStream_t st = nullptr;
  cublasHandle_t blas = nullptr;
  LtCtx lt;
  cudnnHandle_t dnn = nullptr;

  // 단계 사이 결과(정합성 시험에서 읽고 덮어쓴다). 모두 행 우선 NHWC.
  int T = 0, Hp = 0, Wp = 0;
  int lh[4], lw[4];          // 백본 단계 s 의 크기(1/4, 1/8, 1/16, 1/32)
  AP* res[4] = {};           // 백본 출력 res2..res5 [T, lh, lw, 192·2^s]
  AD* ms[3] = {};            // 픽셀 디코더 다중 축척(0: 1/32, 1: 1/16, 2: 1/8) [T, h, w, 256]
  AD* mf = nullptr;          // mask features [T, lh0, lw0, 256]
  float* cls2d = nullptr;    // [T·Q, 2]
  float* q3d = nullptr;      // make_3d 출력 [Q, 256]
  float* cls3d = nullptr;    // [Q, 2]
  float* mask3d = nullptr;   // [Q, T, lh0·lw0]

 private:
  template <class A> const A* W(const std::string& n);        // Linear·1×1 conv 가중치 [out, in]
  const float* F(const std::string& n);                      // fp32 그대로
  template <class A> const A* Wpatch(const std::string& n);  // k×k conv → [out][(ky·k+kx)·in + c] (= cuDNN NHWC 필터)
  // 앞 rows 행(또는 원소)에 배율 s / 행마다 gamma 를 곱한 사본(Q 의 1/√d, layer scale 을 GEMM 에 접기).
  template <class A> const A* Wscaled(const std::string& n, int rows, float s);
  // bias 를 형 T 로(앞 count 개에 배율 s). cuBLASLt 에필로그 bias 는 출력과 같은 형이어야 한다.
  template <class T> const T* Bv(const std::string& n, int count = 0, float s = 1.f);
  template <class A> const A* Wgamma(const std::string& n, const std::string& gamma);
  // 미뤄 둔 잔차 bias: 단계 s 에서 블록 j 앞(k=0) 또는 블록 j 의 gnconv 뒤(k=1)까지 쌓인 Σ gamma⊙bias.
  // 잔차 GEMM 은 bias 없이 r 에 누적하고(cuBLASLt 의 beta=1 + bias 는 따로 도는 느린 epilogue), r 을 읽는 LN 이 더한다.
  const float* pend(int s, int j, int k);
  std::map<std::tuple<int, int, int>, float*> pend_;
  const float* Wdw(const std::string& n);               // 깊이별 [K·K][C]
  const float2* Wgf(int s, int j, int& wh, int& ww);    // 전역 필터 복소 가중치 [C2][wh][ww] (보간 전)

  void block(int s, int j, float* r, int H, int W);
  void attention(const std::string& mha, const std::string& norm, const AD* q_in, const AD* k_in, const AD* v_in, int nb,
                 int Lq, int Lk, const float* mask, float* tgt, AD* tgt_act);
  void ffn(const std::string& pre, float* tgt, AD* tgt_act, int rows);
  void head(const float* out, int rows, AD* me, float* cls);

  TensorMap host_;
  std::map<std::string, void*> dev_;
  std::map<std::tuple<int, int, int, int>, float2*> gfw_;
  std::map<std::tuple<int, int, int>, GlobalFilterPlan> gfp_;
  std::map<std::tuple<int, int, int>, ConvOp> fpn_conv_;
  ConvOp* conv3 = nullptr;
  size_t conv3_ws = 0;
  DBuf b_cnt;

  // 모양별 표(위치 인코딩)
  float* pd_pos = nullptr;      // [L, 256] 픽셀 디코더 (수준 임베딩 포함)
  float* dec_pos2[3] = {};      // [h·w, 256]
  float* dec_pos3[3] = {};      // [T·h·w, 256]
  DBuf b_pdpos, b_pos2[3], b_pos3[3];

  DBuf b_res[4], b_ms[3], b_mf, b_cls2d, b_q3d, b_cls3d, b_mask3d;
  DBuf b_r, b_r2, b_t, b_patch, b_f32a, b_f32b, b_fused, b_dwo, b_g, b_tmp2, b_xa, b_xb, b_h, b_gfbuf, b_gfspec, b_gfwork, b_stats;
  DBuf b_src, b_srca, b_q, b_val, b_off, b_aw, b_samp, b_ws;
  DBuf b_dsrc[3], b_kin2[3], b_kin3[3], b_mfl[3];
  DBuf b_out2, b_out2a, b_out3, b_out3a, b_qin, b_kin, b_me, b_ml, b_Qp, b_K, b_V, b_S, b_P, b_O, b_dn, b_post, b_order,
      b_id, b_ms_, b_area, b_p, b_score;
  DBuf b_pfull, b_pssum, b_pinter, b_pinfo, b_pn, b_pbits;
};

using TeacherFP32 = TeacherNetT<float, float, float>;     // 정합성 시험
using TeacherFast = TeacherNetT<__half, float, float>;    // 라벨 만들기
