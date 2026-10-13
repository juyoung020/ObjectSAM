#include "tnet.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>

namespace {
constexpr float kEpsHor = 1e-6f;  // HorNet LayerNorm
constexpr float kEpsLN = 1e-5f;   // nn.LayerNorm
constexpr float kEpsGN = 1e-5f;   // nn.GroupNorm
constexpr int kDepth[4] = {2, 3, 18, 2};
constexpr int kBase = 192;
const std::string kBB = "backbone.";
const std::string kPD = "sem_seg_head.pixel_decoder.";
const std::string kDEC = "sem_seg_head.predictor.";

int chans(int s) { return kBase << s; }
// gnconv 차수 s+2 의 dims(오름차순)
std::vector<int> gn_dims(int s) {
  int order = s + 2, C = chans(s);
  std::vector<int> d;
  for (int i = order - 1; i >= 0; i--) d.push_back(C >> i);
  return d;
}

// 계산 스트림(non-blocking)에 올리고 끝날 때까지 기다린다. 기본 스트림의 cudaMemcpy 는 계산 스트림과 순서가 없어서
// (pageable 복사는 DMA 가 끝나기 전에 돌아올 수 있다) 바로 뒤 커널이 덜 올라간 값을 읽을 수 있다.
template <class T> void put(T* p, const std::vector<T>& v, cudaStream_t st) {
  CUDA_CHECK(cudaMemcpyAsync(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
}
template <class T> T* upload(const std::vector<T>& v, DBuf& b, cudaStream_t st) {
  T* p = b.get<T>(v.size());
  put(p, v, st);
  return p;
}
template <class T> T* dev_copy(const std::vector<T>& v, cudaStream_t st) {
  T* p = dalloc<T>(v.size());
  put(p, v, st);
  return p;
}
template <class A> std::vector<A> cast(const float* p, size_t n) {
  std::vector<A> o(n);
  for (size_t i = 0; i < n; i++) o[i] = from_f<A>(p[i]);
  return o;
}
template <class A> const char* tag() { return std::is_same<A, float>::value ? "f:" : "h:"; }
}  // namespace

// ---------------------------------------------------------------- 가중치
template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::load(const std::string& path) {
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasCreate(&blas));
  CUBLAS_CHECK(cublasSetStream(blas, st));
  CUDNN_CHECK(cudnnCreate(&dnn));
  CUDNN_CHECK(cudnnSetStream(dnn, st));
  lt.init();
  host_ = fsow_load(path);
}
template <class AB, class AP, class AD>
template <class A>
const A* TeacherNetT<AB, AP, AD>::Wscaled(const std::string& n, int rows, float s) {
  auto k = std::string("S") + tag<A>() + n;
  if (!dev_.count(k)) {
    const HostTensor& t = host_.at(n);
    std::vector<float> o(t.f32(), t.f32() + t.numel());
    int64_t cols = t.numel() / t.shape[0];
    for (int64_t i = 0; i < int64_t(rows) * cols; i++) o[i] *= s;
    dev_[k] = dev_copy(cast<A>(o.data(), o.size()), st);
  }
  return static_cast<const A*>(dev_[k]);
}
template <class AB, class AP, class AD>
template <class T>
const T* TeacherNetT<AB, AP, AD>::Bv(const std::string& n, int count, float s) {
  auto k = std::string("B") + tag<T>() + std::to_string(count) + ":" + std::to_string(s) + ":" + n;
  if (!dev_.count(k)) {
    const HostTensor& t = host_.at(n);
    std::vector<float> o(t.f32(), t.f32() + t.numel());
    for (int i = 0; i < count; i++) o[i] *= s;
    dev_[k] = dev_copy(cast<T>(o.data(), o.size()), st);
  }
  return static_cast<const T*>(dev_[k]);
}
template <class AB, class AP, class AD>
template <class A>
const A* TeacherNetT<AB, AP, AD>::Wgamma(const std::string& n, const std::string& gamma) {
  auto k = std::string("G") + tag<A>() + n;
  if (!dev_.count(k)) {
    const HostTensor& t = host_.at(n);
    const float* g = host_.at(gamma).f32();
    std::vector<float> o(t.f32(), t.f32() + t.numel());
    int64_t cols = t.numel() / t.shape[0];
    for (int64_t r = 0; r < t.shape[0]; r++)
      for (int64_t c = 0; c < cols; c++) o[r * cols + c] *= g[r];
    dev_[k] = dev_copy(cast<A>(o.data(), o.size()), st);
  }
  return static_cast<const A*>(dev_[k]);
}
template <class AB, class AP, class AD>
const float* TeacherNetT<AB, AP, AD>::pend(int s, int j, int k) {
  auto key = std::make_tuple(s, j, k);
  auto it = pend_.find(key);
  if (it != pend_.end()) return it->second;
  int C = chans(s);
  std::vector<float> acc(C, 0.f);
  auto add = [&](int i, const char* b, const char* g) {
    std::string p = kBB + "stages." + std::to_string(s) + "." + std::to_string(i) + ".";
    const float* bb = host_.at(p + b).f32();
    const float* gg = host_.at(p + g).f32();
    for (int c = 0; c < C; c++) acc[c] += gg[c] * bb[c];
  };
  for (int i = 0; i < j; i++) add(i, "gnconv.proj_out.bias", "gamma1"), add(i, "pwconv2.bias", "gamma2");
  if (k == 1) add(j, "gnconv.proj_out.bias", "gamma1");
  float* p = dev_copy(acc, st);
  pend_[key] = p;
  return p;
}
template <class AB, class AP, class AD>
template <class A>
const A* TeacherNetT<AB, AP, AD>::W(const std::string& n) {
  auto k = std::string("W") + tag<A>() + n;
  if (!dev_.count(k)) {
    auto it = host_.find(n);
    FSO_ASSERT(it != host_.end(), "no weight %s", n.c_str());
    dev_[k] = dev_copy(cast<A>(it->second.f32(), it->second.numel()), st);
  }
  return static_cast<const A*>(dev_[k]);
}
template <class AB, class AP, class AD>
const float* TeacherNetT<AB, AP, AD>::F(const std::string& n) {
  auto k = "F:" + n;
  if (!dev_.count(k)) {
    auto it = host_.find(n);
    FSO_ASSERT(it != host_.end(), "no weight %s", n.c_str());
    const float* p = it->second.f32();
    dev_[k] = dev_copy(std::vector<float>(p, p + it->second.numel()), st);
  }
  return static_cast<const float*>(dev_[k]);
}
template <class AB, class AP, class AD>
template <class A>
const A* TeacherNetT<AB, AP, AD>::Wpatch(const std::string& n) {
  auto k = std::string("P") + tag<A>() + n;
  if (!dev_.count(k)) {
    const HostTensor& t = host_.at(n);  // (O, I, k, k)
    int O = t.shape[0], I = t.shape[1], K = t.shape[2];
    std::vector<float> o(t.numel());
    for (int a = 0; a < O; a++)
      for (int c = 0; c < I; c++)
        for (int ky = 0; ky < K; ky++)
          for (int kx = 0; kx < K; kx++)
            o[size_t(a) * K * K * I + (ky * K + kx) * I + c] = t.f32()[((size_t(a) * I + c) * K + ky) * K + kx];
    dev_[k] = dev_copy(cast<A>(o.data(), o.size()), st);
  }
  return static_cast<const A*>(dev_[k]);
}
template <class AB, class AP, class AD>
const float* TeacherNetT<AB, AP, AD>::Wdw(const std::string& n) {
  auto k = "D:" + n;
  if (!dev_.count(k)) {
    const HostTensor& t = host_.at(n);  // (C, 1, K, K)
    int C = t.shape[0], K = t.shape[2];
    std::vector<float> o(t.numel());
    for (int c = 0; c < C; c++)
      for (int q = 0; q < K * K; q++) o[size_t(q) * C + c] = t.f32()[size_t(c) * K * K + q];
    dev_[k] = dev_copy(o, st);
  }
  return static_cast<const float*>(dev_[k]);
}
// complex_weight (C2, h, w, 2) 그대로(보간은 곱하는 커널에서).
template <class AB, class AP, class AD>
const float2* TeacherNetT<AB, AP, AD>::Wgf(int s, int j, int& wh, int& ww) {
  const HostTensor& t = host_.at(kBB + "stages." + std::to_string(s) + "." + std::to_string(j) + ".gnconv.dwconv.complex_weight");
  wh = t.shape[1], ww = t.shape[2];
  auto key = std::make_tuple(s, j, 0, 0);
  auto it = gfw_.find(key);
  if (it != gfw_.end()) return it->second;
  std::vector<float2> o(t.numel() / 2);
  for (size_t i = 0; i < o.size(); i++) o[i] = make_float2(t.f32()[2 * i], t.f32()[2 * i + 1]);
  float2* p = dev_copy(o, st);
  gfw_[key] = p;
  return p;
}

// ---------------------------------------------------------------- 모양
template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::set_shape(int T_, int Hp_, int Wp_) {
  FSO_ASSERT(Hp_ % 32 == 0 && Wp_ % 32 == 0, "pad to 32: %d %d", Hp_, Wp_);
  bool same = T_ == T && Hp_ == Hp && Wp_ == Wp;
  T = T_, Hp = Hp_, Wp = Wp_;
  for (int s = 0; s < 4; s++) lh[s] = Hp >> (2 + s), lw[s] = Wp >> (2 + s);
  int64_t N0 = int64_t(T) * lh[0] * lw[0];
  for (int s = 0; s < 4; s++) res[s] = b_res[s].get<AP>(int64_t(T) * lh[s] * lw[s] * chans(s));
  for (int l = 0; l < 3; l++) ms[l] = b_ms[l].get<AD>(int64_t(T) * lh[3 - l] * lw[3 - l] * 256);
  mf = b_mf.get<AD>(N0 * 256);
  cls2d = b_cls2d.get<float>(int64_t(T) * kQ * 2);
  q3d = b_q3d.get<float>(kQ * 256);
  cls3d = b_cls3d.get<float>(kQ * 2);
  mask3d = b_mask3d.get<float>(int64_t(kQ) * N0);
  if (same) return;

  // 위치 인코딩 표(GPU 에서; 모양이 자주 바뀌어도 호스트와 맞추지 않는다)
  {
    const float* lvl = F(kPD + "transformer.level_embed");
    int64_t L = 0;
    for (int l = 0; l < 3; l++) L += int64_t(lh[3 - l]) * lw[3 - l];
    pd_pos = b_pdpos.get<float>(L * 256);
    int64_t off = 0;
    for (int l = 0; l < 3; l++) {
      int h = lh[3 - l], w = lw[3 - l];
      sine_pos(h, w, 0, lvl + l * 256, pd_pos + off * 256, st);
      off += int64_t(h) * w;
      dec_pos2[l] = b_pos2[l].get<float>(int64_t(h) * w * 256);
      dec_pos3[l] = b_pos3[l].get<float>(int64_t(T) * h * w * 256);
      sine_pos(h, w, 0, nullptr, dec_pos2[l], st);
      sine_pos(h, w, T, nullptr, dec_pos3[l], st);
    }
  }

  // FPN 3×3 conv (픽셀 디코더 정밀도)
  auto key = std::make_tuple(T, lh[0], lw[0]);
  if (!fpn_conv_.count(key)) {
    ConvOp& c = fpn_conv_[key];
    c.init(dnn, ConvGeom{T, lh[0], lw[0], 256, 256, 3, 3, 1, 1}, false, dnn_t<AP>());
    AP* x = b_xa.get<AP>(N0 * 256);
    AP* y = b_xb.get<AP>(N0 * 256);
    size_t lim = size_t(256) << 20;
    void* ws = b_ws.get<char>(lim);
    conv3_ws = std::max(conv3_ws, c.tune_fwd(x, const_cast<AP*>(Wpatch<AP>(kPD + "layer_1.weight")), y, ws, lim));
  }
  conv3 = &fpn_conv_[key];
}

// ---------------------------------------------------------------- 백본
template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::block(int s, int j, float* r, int H, int Wd) {
  using A = AB;
  const std::string p = kBB + "stages." + std::to_string(s) + "." + std::to_string(j) + ".";
  int C = chans(s);
  auto dims = gn_dims(s);
  int S = std::accumulate(dims.begin(), dims.end(), 0);
  int64_t N = int64_t(T) * H * Wd;
  A* t = b_t.get<A>(N * C);
  A* fused = b_fused.get<A>(N * 2 * C);
  A* dwo = b_dwo.get<A>(N * S);
  const float kScale = 1.f / 3.f;

  layer_norm<float, A>(mat<const float>(r, C), mat(t, C), N, C, F(p + "norm1.weight"), F(p + "norm1.bias"), kEpsHor, st, 1.f,
                       pend(s, j, 0));
  lt_gemm<A, A>(lt, N, 2 * C, C, t, C, W<A>(p + "gnconv.proj_in.weight"), C, fused, 2 * C, Bv<A>(p + "gnconv.proj_in.bias"), kEpiBias,
                0.f, st);
  Mat<const A> abc = mat<const A>(fused, 2 * C, dims[0]);
  if (s < 2) {
    dwconv<A>(abc, mat(dwo, S), 1, T, H, Wd, S, 7, Wdw(p + "gnconv.dwconv.weight"), F(p + "gnconv.dwconv.bias"), kScale, st);
  } else {
    A* g = b_g.get<A>(N * S);
    A* tmp2 = b_tmp2.get<A>(N * S);
    layer_norm<A, A>(abc, mat(g, S), N, S, F(p + "gnconv.dwconv.pre_norm.weight"), F(p + "gnconv.dwconv.pre_norm.bias"),
                     kEpsHor, st);
    // x1 = 앞 절반 → 3×3 깊이별 → 짝수 채널, x2 = 뒤 절반 → 주파수 필터 → 홀수 채널
    dwconv<A>(mat<const A>(g, S), mat(tmp2, S), 2, T, H, Wd, S / 2, 3, Wdw(p + "gnconv.dwconv.dw.weight"), nullptr, 1.f, st);
    auto pk = std::make_tuple(T, H, Wd * 10000 + S / 2);
    if (!gfp_.count(pk)) gfp_[pk].init(T, H, Wd, S / 2, st);
    int64_t nr = N * (S / 2);
    int64_t nc = int64_t(H) * (Wd / 2 + 1) * T * (S / 2);
    int wh, ww;
    const float2* gw = Wgf(s, j, wh, ww);
    global_filter<A>(gfp_[pk], mat<const A>(g, S, S / 2), mat(tmp2, S, 1), 2, gw, wh, ww, b_gfbuf.get<float>(nr),
                     b_gfspec.get<float2>(nc), b_gfwork.get<char>(std::max<size_t>(gfp_[pk].work, 1)), st);
    layer_norm<A, A>(mat<const A>(tmp2, S), mat(dwo, S), N, S, F(p + "gnconv.dwconv.post_norm.weight"),
                     F(p + "gnconv.dwconv.post_norm.bias"), kEpsHor, st, kScale);
  }
  // 재귀 게이트
  A* xa = b_xa.get<A>(N * C);
  A* xb = b_xb.get<A>(N * C);
  bias_mul<A, A>(mat<const A>(fused, 2 * C), nullptr, mat<const A>(dwo, S), mat(xa, dims[0]), N, dims[0], st);
  int off = dims[0];
  for (size_t i = 0; i + 1 < dims.size(); i++) {
    std::string pw = p + "gnconv.pws." + std::to_string(i) + ".";
    lt_gemm<A, A>(lt, N, dims[i + 1], dims[i], xa, dims[i], W<A>(pw + "weight"), dims[i], xb, dims[i + 1], Bv<A>(pw + "bias"),
                  kEpiBias, 0.f, st);
    bias_mul<A, A>(mat<const A>(xb, dims[i + 1]), nullptr, mat<const A>(dwo, S, off), mat(xb, dims[i + 1]), N, dims[i + 1], st);
    off += dims[i + 1];
    std::swap(xa, xb);
  }
  // r += gamma1 ⊙ proj_out(x): gamma 를 가중치에 접고 r 에 바로 누적(bias 는 pend 로 미룬다)
  lt_gemm<A, float>(lt, N, C, C, xa, C, Wgamma<A>(p + "gnconv.proj_out.weight", p + "gamma1"), C, r, C, nullptr, kEpiNone, 1.f,
                    st);
  // MLP
  A* h = b_h.get<A>(N * 4 * C);
  layer_norm<float, A>(mat<const float>(r, C), mat(t, C), N, C, F(p + "norm2.weight"), F(p + "norm2.bias"), kEpsHor, st, 1.f,
                       pend(s, j, 1));
  lt_gemm<A, A>(lt, N, 4 * C, C, t, C, W<A>(p + "pwconv1.weight"), C, h, 4 * C, Bv<A>(p + "pwconv1.bias"), kEpiBias, 0.f, st);
  bias_act<A, A>(mat<const A>(h, 4 * C), mat(h, 4 * C), N, 4 * C, nullptr, kGelu, st);  // GELU(erf) 는 따로(정확히)
  lt_gemm<A, float>(lt, N, C, 4 * C, h, 4 * C, Wgamma<A>(p + "pwconv2.weight", p + "gamma2"), 4 * C, r, C, nullptr, kEpiNone, 1.f,
                    st);
}

template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::backbone(const AB* x) {
  using A = AB;
  int64_t N = int64_t(T) * lh[0] * lw[0];
  float* r = b_r.get<float>(N * kBase);
  A* patch = b_patch.get<A>(N * 4 * kBase);
  patchify<A>(x, T, Hp, Wp, 3, 4, patch, st);
  lt_gemm<A, float>(lt, N, kBase, 48, patch, 48, Wpatch<A>(kBB + "downsample_layers.0.0.weight"), 48, r, kBase,
                    F(kBB + "downsample_layers.0.0.bias"), kEpiBias, 0.f, st);
  layer_norm<float, float>(mat<const float>(r, kBase), mat(r, kBase), N, kBase, F(kBB + "downsample_layers.0.1.weight"),
                           F(kBB + "downsample_layers.0.1.bias"), kEpsHor, st);
  for (int s = 0; s < 4; s++) {
    int C = chans(s);
    if (s > 0) {
      int Cp = chans(s - 1);
      int64_t Np = int64_t(T) * lh[s - 1] * lw[s - 1];
      std::string d = kBB + "downsample_layers." + std::to_string(s) + ".";
      A* t = b_t.get<A>(Np * Cp);
      layer_norm<float, A>(mat<const float>(r, Cp), mat(t, Cp), Np, Cp, F(d + "0.weight"), F(d + "0.bias"), kEpsHor, st, 1.f,
                           pend(s - 1, kDepth[s - 1], 0));
      patchify<A>(t, T, lh[s - 1], lw[s - 1], Cp, 2, patch, st);
      N = int64_t(T) * lh[s] * lw[s];
      float* r2 = b_r2.get<float>(N * C);
      lt_gemm<A, float>(lt, N, C, 4 * Cp, patch, 4 * Cp, Wpatch<A>(d + "1.weight"), 4 * Cp, r2, C, F(d + "1.bias"), kEpiBias, 0.f,
                        st);
      std::swap(b_r, b_r2);
      r = r2;
    }
    for (int j = 0; j < kDepth[s]; j++) block(s, j, r, lh[s], lw[s]);
    layer_norm<float, AP>(mat<const float>(r, C), mat(res[s], C), N, C, F(kBB + "norm" + std::to_string(s) + ".weight"),
                          F(kBB + "norm" + std::to_string(s) + ".bias"), kEpsHor, st, 1.f, pend(s, kDepth[s], 0));
  }
}

// ---------------------------------------------------------------- 픽셀 디코더
template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::pixel_decoder() {
  using A = AP;
  DeformLevels lv;
  int L = 0;
  for (int l = 0; l < 3; l++) {
    lv.h[l] = lh[3 - l], lv.w[l] = lw[3 - l], lv.start[l] = L;
    L += lv.h[l] * lv.w[l];
  }
  int64_t TL = int64_t(T) * L;
  float* src = b_src.get<float>(TL * 256);
  A* srca = b_srca.get<A>(TL * 256);
  float* stats = b_stats.get<float>(64 * T * 2);
  // 입력 투영 + GroupNorm → 토큰 [T][L][256]
  for (int l = 0; l < 3; l++) {
    int s = 3 - l, C = chans(s);
    int64_t hw = int64_t(lv.h[l]) * lv.w[l], n = T * hw;
    std::string p = kPD + "input_proj." + std::to_string(l) + ".";
    float* f = b_f32a.get<float>(n * 256);
    float* g = b_f32b.get<float>(n * 256);
    A* ga = b_t.get<A>(n * 256);
    lt_gemm<A, float>(lt, n, 256, C, res[s], C, W<A>(p + "0.weight"), C, f, 256, F(p + "0.bias"), kEpiBias, 0.f, st);
    group_norm<float, A>(f, g, ga, T, hw, 256, 32, F(p + "1.weight"), F(p + "1.bias"), kEpsGN, false, stats, st);
    size_t row = hw * 256;
    CUDA_CHECK(cudaMemcpy2DAsync(src + int64_t(lv.start[l]) * 256, L * 256 * 4, g, row * 4, row * 4, T,
                                 cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpy2DAsync(srca + int64_t(lv.start[l]) * 256, L * 256 * sizeof(A), ga, row * sizeof(A), row * sizeof(A),
                                 T, cudaMemcpyDeviceToDevice, st));
  }
  A* q = b_q.get<A>(TL * 256);
  A* val = b_val.get<A>(TL * 256);
  float* off = b_off.get<float>(TL * 192);
  float* aw = b_aw.get<float>(TL * 96);
  A* samp = b_samp.get<A>(TL * 256);
  float* f = b_f32a.get<float>(TL * 1024);
  A* h = b_h.get<A>(TL * 1024);
  for (int k = 0; k < 6; k++) {
    std::string p = kPD + "transformer.encoder.layers." + std::to_string(k) + ".";
    add_rows<float, A>(mat<const float>(src, 256), mat<const float>(pd_pos, 256), L, mat(q, 256), TL, 256, st);
    lt_gemm<A, A>(lt, TL, 256, 256, srca, 256, W<A>(p + "self_attn.value_proj.weight"), 256, val, 256,
                  Bv<A>(p + "self_attn.value_proj.bias"), kEpiBias, 0.f, st);
    gemm_nt<A, float>(blas, TL, 192, 256, q, 256, W<A>(p + "self_attn.sampling_offsets.weight"), 256, off, 192);
    gemm_nt<A, float>(blas, TL, 96, 256, q, 256, W<A>(p + "self_attn.attention_weights.weight"), 256, aw, 96);
    ms_deform_sample<A>(val, off, F(p + "self_attn.sampling_offsets.bias"), aw, F(p + "self_attn.attention_weights.bias"), samp,
                        T, L, lv, st);
    lt_gemm<A, float>(lt, TL, 256, 256, samp, 256, W<A>(p + "self_attn.output_proj.weight"), 256, f, 256,
                      F(p + "self_attn.output_proj.bias"), kEpiBias, 0.f, st);
    add_ln<A>(mat(src, 256), mat<const float>(f, 256), nullptr, F(p + "norm1.weight"),
              F(p + "norm1.bias"), kEpsLN, mat(srca, 256), TL, 256, st);
    lt_gemm<A, A>(lt, TL, 1024, 256, srca, 256, W<A>(p + "linear1.weight"), 256, h, 1024, Bv<A>(p + "linear1.bias"), kEpiReluBias,
                  0.f, st);
    lt_gemm<A, float>(lt, TL, 256, 1024, h, 1024, W<A>(p + "linear2.weight"), 1024, f, 256, F(p + "linear2.bias"), kEpiBias,
                      0.f, st);
    add_ln<A>(mat(src, 256), mat<const float>(f, 256), nullptr, F(p + "norm2.weight"), F(p + "norm2.bias"), kEpsLN,
              mat(srca, 256), TL, 256, st);
  }
  // 수준별로 되돌리기(fp32 잔차에서) → ms[l] [T, h, w, 256] (디코더 정밀도)
  for (int l = 0; l < 3; l++) {
    int64_t hw = int64_t(lv.h[l]) * lv.w[l];
    for (int t = 0; t < T; t++)
      bias_act<float, AD>(mat<const float>(src + (int64_t(t) * L + lv.start[l]) * 256, 256), mat(ms[l] + t * hw * 256, 256),
                          hw, 256, nullptr, kNone, st);
  }
  // FPN: res2 → 1×1(GN) + 1/8 수준 키운 것 → 3×3(GN, ReLU) → mask_features 1×1
  int64_t N0 = int64_t(T) * lh[0] * lw[0];
  float* fa = b_f32a.get<float>(N0 * 256);
  A* cur = b_t.get<A>(N0 * 256);
  A* y = b_xa.get<A>(N0 * 256);
  A* yc = b_xb.get<A>(N0 * 256);
  A* lvl2 = b_q.get<A>(int64_t(T) * lh[1] * lw[1] * 256);
  for (int t = 0; t < T; t++)
    bias_act<float, A>(mat<const float>(src + (int64_t(t) * L + lv.start[2]) * 256, 256),
                       mat(lvl2 + int64_t(t) * lh[1] * lw[1] * 256, 256), int64_t(lh[1]) * lw[1], 256, nullptr, kNone, st);
  gemm_nt<A, float>(blas, N0, 256, kBase, res[0], kBase, W<A>(kPD + "adapter_1.weight"), kBase, fa, 256);
  group_norm<float, A>(fa, nullptr, cur, T, int64_t(lh[0]) * lw[0], 256, 32, F(kPD + "adapter_1.norm.weight"),
                       F(kPD + "adapter_1.norm.bias"), kEpsGN, false, stats, st);
  resize_bilinear<A, A>(lvl2, T, lh[1], lw[1], 256, y, lh[0], lw[0], cur, st);
  conv3->fwd_v(y, Wpatch<A>(kPD + "layer_1.weight"), yc, b_ws.get<char>(conv3_ws), conv3_ws);
  group_norm<A, A>(yc, nullptr, cur, T, int64_t(lh[0]) * lw[0], 256, 32, F(kPD + "layer_1.norm.weight"),
                   F(kPD + "layer_1.norm.bias"), kEpsGN, true, stats, st);
  if constexpr (std::is_same<A, AD>::value || std::is_same<AD, float>::value)
    lt_gemm<A, AD>(lt, N0, 256, 256, cur, 256, W<A>(kPD + "mask_features.weight"), 256, mf, 256,
                   Bv<AD>(kPD + "mask_features.bias"), kEpiBias, 0.f, st);
  else {
    lt_gemm<A, float>(lt, N0, 256, 256, cur, 256, W<A>(kPD + "mask_features.weight"), 256, fa, 256,
                      F(kPD + "mask_features.bias"), kEpiBias, 0.f, st);
    bias_act<float, AD>(mat<const float>(fa, 256), mat(mf, 256), N0, 256, nullptr, kNone, st);
  }
}

// ---------------------------------------------------------------- 디코더
template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::attention(const std::string& mha, const std::string& norm, const AD* q_in, const AD* k_in,
                                        const AD* v_in, int nb, int Lq, int Lk, const float* mask, float* tgt, AD* tgt_act) {
  using A = AD;
  const A* Wi = W<A>(mha + "in_proj_weight");
  int64_t nq = int64_t(nb) * Lq, nk = int64_t(nb) * Lk;
  float* f = b_f32a.get<float>(std::max(nq, nk) * 256);
  A* Qp = b_Qp.get<A>(nq * 256);
  A* K = b_K.get<A>(nk * 256);
  A* V = b_V.get<A>(nk * 256);
  const float qs = 1.f / sqrtf(32.f);  // q 배율(PyTorch: bias 더한 뒤 곱함) → 가중치·bias 에 접는다
  lt_gemm<A, A>(lt, nq, 256, 256, q_in, 256, Wscaled<A>(mha + "in_proj_weight", 256, qs), 256, Qp, 256,
                Bv<A>(mha + "in_proj_bias", 256, qs), kEpiBias, 0.f, st);
  lt_gemm<A, A>(lt, nk, 256, 256, k_in, 256, Wi + 256 * 256, 256, K, 256, Bv<A>(mha + "in_proj_bias") + 256, kEpiBias, 0.f, st);
  lt_gemm<A, A>(lt, nk, 256, 256, v_in, 256, Wi + 512 * 256, 256, V, 256, Bv<A>(mha + "in_proj_bias") + 512, kEpiBias, 0.f, st);
  int64_t per_h = nq * Lk;
  float* S = b_S.get<float>(8 * per_h);
  A* P = b_P.get<A>(8 * per_h);
  A* O = b_O.get<A>(nq * 256);
  for (int h = 0; h < 8; h++)
    gemm_nt<A, float>(blas, Lq, Lk, 32, Qp + h * 32, 256, K + h * 32, 256, S + h * per_h, Lk, 1.f, 0.f, nb, int64_t(Lq) * 256,
                      int64_t(Lk) * 256, int64_t(Lq) * Lk);
  masked_softmax<A>(S, mask, P, 8, nq, Lk, st);
  for (int h = 0; h < 8; h++)
    gemm_nn<A, A>(blas, Lq, 32, Lk, P + h * per_h, Lk, V + h * 32, 256, O + h * 32, 256, 1.f, 0.f, nb, int64_t(Lq) * Lk,
                  int64_t(Lk) * 256, int64_t(Lq) * 256);
  lt_gemm<A, float>(lt, nq, 256, 256, O, 256, W<A>(mha + "out_proj.weight"), 256, f, 256, F(mha + "out_proj.bias"), kEpiBias,
                    0.f, st);
  add_ln<A>(mat(tgt, 256), mat<const float>(f, 256), nullptr, F(norm + "weight"), F(norm + "bias"), kEpsLN,
            mat(tgt_act, 256), nq, 256, st);
}

template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::ffn(const std::string& p, float* tgt, AD* tgt_act, int rows) {
  using A = AD;
  float* f = b_f32a.get<float>(int64_t(rows) * 2048);
  A* h = b_h.get<A>(int64_t(rows) * 2048);
  lt_gemm<A, A>(lt, rows, 2048, 256, tgt_act, 256, W<A>(p + "linear1.weight"), 256, h, 2048, Bv<A>(p + "linear1.bias"), kEpiReluBias,
                0.f, st);
  lt_gemm<A, float>(lt, rows, 256, 2048, h, 2048, W<A>(p + "linear2.weight"), 2048, f, 256, F(p + "linear2.bias"), kEpiBias, 0.f,
                    st);
  add_ln<A>(mat(tgt, 256), mat<const float>(f, 256), nullptr, F(p + "norm.weight"), F(p + "norm.bias"), kEpsLN,
            mat(tgt_act, 256), rows, 256, st);
}

template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::head(const float* out, int rows, AD* me, float* cls) {
  using A = AD;
  A* dn = b_dn.get<A>(int64_t(rows) * 256);
  A* h1 = b_xa.get<A>(int64_t(rows) * 256);
  A* h2 = b_xb.get<A>(int64_t(rows) * 256);
  float* f = b_f32b.get<float>(int64_t(rows) * 256);
  layer_norm<float, A>(mat<const float>(out, 256), mat(dn, 256), rows, 256, F(kDEC + "decoder_norm.weight"),
                       F(kDEC + "decoder_norm.bias"), kEpsLN, st);
  if (cls) {
    gemm_nt<A, float>(blas, rows, 2, 256, dn, 256, W<A>(kDEC + "class_embed.weight"), 256, cls, 2);
    bias_act<float, float>(mat<const float>(cls, 2), mat(cls, 2), rows, 2, F(kDEC + "class_embed.bias"), kNone, st);
  }
  const A* in = dn;
  A* outs[3] = {h1, h2, me};
  for (int k = 0; k < 3; k++) {
    std::string p = kDEC + "mask_embed.layers." + std::to_string(k) + ".";
    lt_gemm<A, A>(lt, rows, 256, 256, in, 256, W<A>(p + "weight"), 256, outs[k], 256, Bv<A>(p + "bias"),
                  k < 2 ? kEpiReluBias : kEpiBias, 0.f, st);
    in = outs[k];
  }
}

template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::decoder() {
  using A = AD;
  int hl[3], wl[3];
  int64_t hw[3];
  for (int l = 0; l < 3; l++) hl[l] = lh[3 - l], wl[l] = lw[3 - l], hw[l] = int64_t(hl[l]) * wl[l];
  A* src[3];
  A* kin2[3];
  A* kin3[3];
  A* mfl[3];
  const float* lvl = F(kDEC + "level_embed.weight");
  for (int l = 0; l < 3; l++) {
    int64_t n = T * hw[l];
    src[l] = b_dsrc[l].get<A>(n * 256);
    kin2[l] = b_kin2[l].get<A>(n * 256);
    kin3[l] = b_kin3[l].get<A>(n * 256);
    mfl[l] = b_mfl[l].get<A>(n * 256);
    float* sf = b_f32b.get<float>(n * 256);  // src(fp32) = ms + 수준 임베딩
    bias_act<A, float>(mat<const A>(ms[l], 256), mat(sf, 256), n, 256, lvl + l * 256, kNone, st);
    bias_act<float, A>(mat<const float>(sf, 256), mat(src[l], 256), n, 256, nullptr, kNone, st);
    add_rows<float, A>(mat<const float>(sf, 256), mat<const float>(dec_pos2[l], 256), hw[l], mat(kin2[l], 256), n, 256, st);
    add_rows<float, A>(mat<const float>(sf, 256), mat<const float>(dec_pos3[l], 256), n, mat(kin3[l], 256), n, 256, st);
    resize_bilinear<A, A>(mf, T, lh[0], lw[0], 256, mfl[l], hl[l], wl[l], nullptr, st);
  }
  const int Q = kQ;
  int64_t nq = int64_t(T) * Q;
  float* out2 = b_out2.get<float>(nq * 256);
  A* out2a = b_out2a.get<A>(nq * 256);
  A* qin = b_qin.get<A>(nq * 256);
  A* me = b_me.get<A>(nq * 256);
  float* ml = b_ml.get<float>(int64_t(Q) * T * hw[2]);
  const float* qfeat = F(kDEC + "query_feat.weight");
  const float* qpos = F(kDEC + "query_embed.weight");
  for (int t = 0; t < T; t++)
    CUDA_CHECK(cudaMemcpyAsync(out2 + int64_t(t) * Q * 256, qfeat, Q * 256 * 4, cudaMemcpyDeviceToDevice, st));
  // 어텐션 마스크 로짓: mask_features 를 그 수준 크기로 먼저 줄이고 곱한다(선형이라 곱한 뒤 줄이는 것과 같다).
  auto mask2d = [&](int l) {  // ml [T][Q][hw] = me_t · mfl_tᵀ
    gemm_nt<A, float>(blas, Q, hw[l], 256, me, 256, mfl[l], 256, ml, hw[l], 1.f, 0.f, T, int64_t(Q) * 256, hw[l] * 256,
                      int64_t(Q) * hw[l]);
  };
  auto mask3 = [&](int l) {  // ml [Q][T·hw]
    gemm_nt<A, float>(blas, Q, T * hw[l], 256, me, 256, mfl[l], 256, ml, T * hw[l]);
  };
  auto num = [](int i) { return std::to_string(i) + "."; };
  head(out2, nq, me, nullptr);
  mask2d(0);
  for (int i = 0; i < 9; i++) {
    int l = i % 3;
    std::string ca = kDEC + "transformer_cross_attention_layers." + num(i);
    std::string sa = kDEC + "transformer_self_attention_layers." + num(i);
    add_rows<float, A>(mat<const float>(out2, 256), mat<const float>(qpos, 256), Q, mat(qin, 256), nq, 256, st);
    attention(ca + "multihead_attn.", ca + "norm.", qin, kin2[l], src[l], T, Q, hw[l], ml, out2, out2a);
    add_rows<float, A>(mat<const float>(out2, 256), mat<const float>(qpos, 256), Q, mat(qin, 256), nq, 256, st);
    attention(sa + "self_attn.", sa + "norm.", qin, qin, out2a, T, Q, Q, nullptr, out2, out2a);
    ffn(kDEC + "transformer_ffn_layers." + num(i), out2, out2a, nq);
    head(out2, nq, me, i == 8 ? cls2d : nullptr);
    if (i < 8) mask2d((i + 1) % 3);
  }
  // 3d 질의: 화면 0 의 질의에서 출발해 모든 화면의 질의에 어텐션
  float* out3 = q3d;
  A* out3a = b_out3a.get<A>(int64_t(Q) * 256);
  A* kin = b_kin.get<A>(nq * 256);
  CUDA_CHECK(cudaMemcpyAsync(out3, out2, Q * 256 * 4, cudaMemcpyDeviceToDevice, st));
  std::string m3 = kDEC + "make_3d.";
  add_rows<float, A>(mat<const float>(out3, 256), mat<const float>(qpos, 256), Q, mat(qin, 256), Q, 256, st);
  add_rows<float, A>(mat<const float>(out2, 256), mat<const float>(qpos, 256), Q, mat(kin, 256), nq, 256, st);
  attention(m3 + "enc_crosattn_3d.0.multihead_attn.", m3 + "enc_crosattn_3d.0.norm.", qin, kin, out2a, 1, Q, T * Q, nullptr, out3,
            out3a);
  // make_3d 의 자기 어텐션은 위치 인코딩 없이(enc_selfattn_3d(output_3d))
  attention(m3 + "enc_selfattn_3d.0.self_attn.", m3 + "enc_selfattn_3d.0.norm.", out3a, out3a, out3a, 1, Q, Q, nullptr, out3,
            out3a);
  ffn(m3 + "enc_ffn_3d.0.", out3, out3a, Q);
  // q3d 는 make_3d 출력 그대로 두고 3d 디코더는 사본으로
  float* o3 = b_out3.get<float>(int64_t(Q) * 256);
  CUDA_CHECK(cudaMemcpyAsync(o3, out3, Q * 256 * 4, cudaMemcpyDeviceToDevice, st));
  head(o3, Q, me, nullptr);
  mask3(0);
  for (int i = 0; i < 9; i++) {
    int l = i % 3;
    std::string ca = kDEC + "transformer_cross_attention_layers." + num(i);
    std::string sa = kDEC + "transformer_self_attention_layers." + num(i);
    add_rows<float, A>(mat<const float>(o3, 256), mat<const float>(qpos, 256), Q, mat(qin, 256), Q, 256, st);
    attention(ca + "multihead_attn.", ca + "norm.", qin, kin3[l], src[l], 1, Q, T * hw[l], ml, o3, out3a);
    add_rows<float, A>(mat<const float>(o3, 256), mat<const float>(qpos, 256), Q, mat(qin, 256), Q, 256, st);
    attention(sa + "self_attn.", sa + "norm.", qin, qin, out3a, 1, Q, Q, nullptr, o3, out3a);
    ffn(kDEC + "transformer_ffn_layers." + num(i), o3, out3a, Q);
    head(o3, Q, me, i == 8 ? cls3d : nullptr);
    if (i < 8) mask3((i + 1) % 3);
  }
  int64_t full = int64_t(T) * lh[0] * lw[0];
  gemm_nt<A, float>(blas, Q, full, 256, me, 256, mf, 256, mask3d, full);
}

// ---------------------------------------------------------------- 후처리
template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::postprocess_async(const ViewGeom& g, int16_t* h_id, float* h_score, int* h_area) {
  const int Q = kQ;
  int64_t HW = int64_t(g.OH) * g.OW;
  float* p = b_p.get<float>(Q);
  int* od = b_order.get<int>(Q);
  float* fused = b_fused.get<float>(int64_t(Q) * HW);
  int16_t* id = b_id.get<int16_t>(HW);
  float* msc = b_ms_.get<float>(Q);
  int* area = b_area.get<int>(Q);
  float* score = b_score.get<float>(Q);
  rank_scores(cls3d, Q, p, od, st);
  fuse_views(mask3d, Q, g, fused, b_cnt.get<float>(HW), st);
  assign_ids(fused, od, Q, HW, id, msc, area, st);
  final_scores(p, od, msc, Q, score, st);
  CUDA_CHECK(cudaMemcpyAsync(h_id, id, HW * 2, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaMemcpyAsync(h_score, score, Q * 4, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaMemcpyAsync(h_area, area, Q * 4, cudaMemcpyDeviceToHost, st));
}

template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::backbone_per_view(const AB* x) {
  int T0 = T;
  AP* r0[4];
  for (int s = 0; s < 4; s++) r0[s] = res[s];
  for (int v = 0; v < T0; v++) {
    T = 1;
    for (int s = 0; s < 4; s++) res[s] = r0[s] + int64_t(v) * lh[s] * lw[s] * chans(s);
    backbone(x + int64_t(v) * Hp * Wp * 3);
  }
  T = T0;
  for (int s = 0; s < 4; s++) res[s] = r0[s];
}

template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::release_scratch(bool decoder_too) {
  CUDA_CHECK(cudaStreamSynchronize(st));
  for (DBuf* b : {&b_r, &b_r2, &b_patch, &b_t, &b_dwo, &b_g, &b_tmp2, &b_gfbuf, &b_gfspec, &b_gfwork, &b_fused, &b_src, &b_srca,
                  &b_q, &b_val, &b_off, &b_aw, &b_samp, &b_h, &b_xa, &b_xb, &b_f32a, &b_f32b, &b_ws, &b_res[0], &b_res[1], &b_res[2], &b_res[3]})
    b->release();
  if (!decoder_too) return;
  for (int l = 0; l < 3; l++) b_dsrc[l].release(), b_kin2[l].release(), b_kin3[l].release(), b_mfl[l].release();
  for (DBuf* b : {&b_out2, &b_out2a, &b_out3, &b_out3a, &b_qin, &b_kin, &b_me, &b_ml, &b_Qp, &b_K, &b_V, &b_S, &b_P, &b_O, &b_dn})
    b->release();
}

template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::parts_async(const ViewGeom& g, const PartCfg& cfg, PartInfo* h_info, int* h_n,
                                          unsigned long long* h_bits) {
  const int Q = kQ;
  int64_t HW = int64_t(g.OH) * g.OW;
  // postprocess_async 가 잡은 버퍼를 그대로 읽는다(get 이 더 크게 다시 잡지 않는 크기).
  auto* bits = b_pbits.get<unsigned long long>(HW);
  auto* info = b_pinfo.get<PartInfo>(kMaxParts);
  int* n = b_pn.get<int>(1);
  part_candidates(static_cast<float*>(b_fused.p), static_cast<int*>(b_order.p), static_cast<float*>(b_p.p),
                  static_cast<int16_t*>(b_id.p), static_cast<int*>(b_area.p), static_cast<float*>(b_score.p), Q, HW, cfg,
                  b_pfull.get<int>(Q), b_pssum.get<float>(Q), b_pinter.get<int>(int64_t(Q) * Q), info, n, bits, st);
  CUDA_CHECK(cudaMemcpyAsync(h_n, n, sizeof(int), cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaMemcpyAsync(h_info, info, kMaxParts * sizeof(PartInfo), cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaMemcpyAsync(h_bits, bits, HW * sizeof(unsigned long long), cudaMemcpyDeviceToHost, st));
}

template <class AB, class AP, class AD>
void TeacherNetT<AB, AP, AD>::postprocess(const ViewGeom& g, TeacherResult& out) {
  out.OH = g.OH, out.OW = g.OW;
  out.id.resize(int64_t(g.OH) * g.OW);
  out.score.resize(kQ);
  out.area.resize(kQ);
  postprocess_async(g, out.id.data(), out.score.data(), out.area.data());
  CUDA_CHECK(cudaStreamSynchronize(st));
}

template class TeacherNetT<float, float, float>;
template class TeacherNetT<__half, float, float>;
template class TeacherNetT<__half, __half, float>;
template class TeacherNetT<__half, __half, __half>;
