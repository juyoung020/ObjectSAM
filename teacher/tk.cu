#include "tk.cuh"

#include <algorithm>
#include <cfloat>
#include <cstdint>

namespace {

template <class T> constexpr cudaDataType_t blas_t();
template <> constexpr cudaDataType_t blas_t<float>() { return CUDA_R_32F; }
template <> constexpr cudaDataType_t blas_t<__half>() { return CUDA_R_16F; }

constexpr int kT = 256;
inline int nblk(int64_t n, int t = kT) { return int(std::min<int64_t>((n + t - 1) / t, 1 << 20)); }

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
// 블록 합(블록 크기 32 의 배수, ≤ 1024). sh: 32 float.
__device__ float block_sum(float v, float* sh) {
  v = warp_sum(v);
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  int nw = blockDim.x >> 5;
  v = l < nw ? sh[l] : 0.f;
  return warp_sum(v);
}
__device__ float block_max(float v, float* sh) {
  v = warp_max(v);
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  int nw = blockDim.x >> 5;
  v = l < nw ? sh[l] : -FLT_MAX;
  return warp_max(v);
}

template <class T> __device__ __forceinline__ float ld(const T* p) { return to_f(*p); }
template <class T> __device__ __forceinline__ void st_(T* p, float v) { *p = from_f<T>(v); }

// 8 개 묶음 읽기·쓰기(16 바이트 정렬). 호스트에서 vec_ok 로 확인한 경우만 쓴다.
__device__ __forceinline__ void ld8(const float* p, float* v) {
  float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
  v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}
__device__ __forceinline__ void ld8(const __half* p, float* v) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
  for (int k = 0; k < 4; k++) {
    float2 f = __half22float2(h[k]);
    v[2 * k] = f.x, v[2 * k + 1] = f.y;
  }
}
__device__ __forceinline__ void st8(float* p, const float* v) {
  reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}
__device__ __forceinline__ void st8(__half* p, const float* v) {
  uint4 u;
  __half2* h = reinterpret_cast<__half2*>(&u);
#pragma unroll
  for (int k = 0; k < 4; k++) h[k] = __floats2half2_rn(v[2 * k], v[2 * k + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}
template <class T> bool al16(const T* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }
template <class T> bool vec_ok(Mat<T> m, int C) { return al16(m.p) && m.ld % 8 == 0 && C % 8 == 0; }

template <class TI, class TO>
__global__ void k_bias_act_v(Mat<const TI> x, Mat<TO> y, int64_t rows, int C, const float* bias, int act, float scale) {
  int C8 = C / 8;
  int64_t n = rows * C8;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t r = i / C8;
    int c = int(i - r * C8) * 8;
    float v[8], b[8];
    ld8(x.row(r) + c, v);
    if (bias) ld8(bias + c, b);
#pragma unroll
    for (int k = 0; k < 8; k++) {
      float t = v[k] + (bias ? b[k] : 0.f);
      if (act == 1) t = fmaxf(t, 0.f);
      else if (act == 2) t = 0.5f * t * (1.f + erff(t * 0.70710678118654752f));
      v[k] = t * scale;
    }
    st8(y.row(r) + c, v);
  }
}
template <class TI, class A>
__global__ void k_bias_mul_v(Mat<const TI> x, const float* bias, Mat<const A> m, Mat<A> y, int64_t rows, int C) {
  int C8 = C / 8;
  int64_t n = rows * C8;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t r = i / C8;
    int c = int(i - r * C8) * 8;
    float v[8], b[8], w[8];
    ld8(x.row(r) + c, v);
    ld8(m.row(r) + c, w);
    if (bias) ld8(bias + c, b);
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = (v[k] + (bias ? b[k] : 0.f)) * w[k];
    st8(y.row(r) + c, v);
  }
}
__global__ void k_res_add_v(Mat<float> r_, Mat<const float> x, const float* bias, const float* gamma, int64_t rows, int C) {
  int C8 = C / 8;
  int64_t n = rows * C8;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t r = i / C8;
    int c = int(i - r * C8) * 8;
    float a[8], v[8], b[8], g[8];
    ld8(r_.row(r) + c, a);
    ld8(x.row(r) + c, v);
    ld8(bias + c, b);
    ld8(gamma + c, g);
#pragma unroll
    for (int k = 0; k < 8; k++) a[k] += g[k] * (v[k] + b[k]);
    st8(r_.row(r) + c, a);
  }
}
template <class TI, class A>
__global__ void k_add_rows_v(Mat<const TI> x, Mat<const float> y, int64_t rows_y, Mat<A> out, int64_t rows, int C) {
  int C8 = C / 8;
  int64_t n = rows * C8;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t r = i / C8;
    int c = int(i - r * C8) * 8;
    float v[8], w[8];
    ld8(x.row(r) + c, v);
    ld8(y.row(r % rows_y) + c, w);
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] += w[k];
    st8(out.row(r) + c, v);
  }
}
// LayerNorm, 워프당 한 행, 레인마다 NV 개 8-묶음을 레지스터에.
template <class TI, class TO, int NV>
__global__ void k_ln_v(Mat<const TI> x, Mat<TO> y, int64_t rows, int C, const float* g, const float* b, float eps,
                       float scale, const float* add) {
  int64_t r = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  int l = threadIdx.x & 31;
  if (r >= rows) return;
  int C8 = C / 8;
  float v[NV][8];
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < NV; k++) {
    int c8 = l + 32 * k;
    if (c8 < C8) {
      ld8(x.row(r) + c8 * 8, v[k]);
      if (add) {
        float a[8];
        ld8(add + c8 * 8, a);
#pragma unroll
        for (int e = 0; e < 8; e++) v[k][e] += a[e];
      }
#pragma unroll
      for (int e = 0; e < 8; e++) s += v[k][e];
    }
  }
  float mean = warp_sum(s) / C;
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < NV; k++)
    if (l + 32 * k < C8)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        float d = v[k][e] - mean;
        q += d * d;
      }
  float rstd = rsqrtf(warp_sum(q) / C + eps);
#pragma unroll
  for (int k = 0; k < NV; k++) {
    int c8 = l + 32 * k;
    if (c8 < C8) {
      float gg[8], bb[8], o[8];
      ld8(g + c8 * 8, gg);
      ld8(b + c8 * 8, bb);
#pragma unroll
      for (int e = 0; e < 8; e++) o[e] = ((v[k][e] - mean) * rstd * gg[e] + bb[e]) * scale;
      st8(y.row(r) + c8 * 8, o);
    }
  }
}

// ---------------------------------------------------------------- LayerNorm (워프당 한 행)
template <class TI, class TO>
__global__ void k_ln(Mat<const TI> x, Mat<TO> y, int64_t rows, int C, const float* g, const float* b, float eps,
                     float scale, const float* add) {
  int64_t r = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  int l = threadIdx.x & 31;
  if (r >= rows) return;
  const TI* xr = x.row(r);
  float s = 0.f;
  auto xv = [&](int c) { return ld(xr + c) + (add ? add[c] : 0.f); };
  for (int c = l; c < C; c += 32) s += xv(c);
  float mean = warp_sum(s) / C;
  float v = 0.f;
  for (int c = l; c < C; c += 32) {
    float d = xv(c) - mean;
    v += d * d;
  }
  float rstd = rsqrtf(warp_sum(v) / C + eps);
  TO* yr = y.row(r);
  for (int c = l; c < C; c += 32) st_(yr + c, ((xv(c) - mean) * rstd * g[c] + b[c]) * scale);
}

template <class TI, class TO>
__global__ void k_bias_act(Mat<const TI> x, Mat<TO> y, int64_t rows, int C, const float* bias, int act, float scale) {
  int64_t n = rows * C;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t r = i / C;
    int c = int(i - r * C);
    float v = ld(x.row(r) + c) + (bias ? bias[c] : 0.f);
    if (act == kRelu) v = fmaxf(v, 0.f);
    else if (act == kGelu) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    st_(y.row(r) + c, v * scale);
  }
}

template <class TI, class A>
__global__ void k_bias_mul(Mat<const TI> x, const float* bias, Mat<const A> m, Mat<A> y, int64_t rows, int C) {
  int64_t n = rows * C;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t r = i / C;
    int c = int(i - r * C);
    y.row(r)[c] = from_f<A>((ld(x.row(r) + c) + (bias ? bias[c] : 0.f)) * to_f(m.row(r)[c]));
  }
}

template <class A>
__global__ void k_patchify(const A* x, int B, int H, int W, int C, int k, A* y) {
  int Ho = H / k, Wo = W / k, KC = k * k * C;
  int64_t n = int64_t(B) * Ho * Wo * KC;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int col = int(i % KC);
    int64_t r = i / KC;
    int ox = int(r % Wo), oy = int((r / Wo) % Ho), b = int(r / (int64_t(Wo) * Ho));
    int c = col % C, kk = col / C, kx = kk % k, ky = kk / k;
    y[i] = x[((int64_t(b) * H + oy * k + ky) * W + ox * k + kx) * C + c];
  }
}

__global__ void k_res_add(Mat<float> r_, Mat<const float> x, const float* bias, const float* gamma, int64_t rows, int C) {
  int64_t n = rows * C;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t r = i / C;
    int c = int(i - r * C);
    r_.row(r)[c] += gamma[c] * (x.row(r)[c] + bias[c]);
  }
}

// r = LN(r + x + bias), 워프당 한 행, C ≤ 256·… (여기서는 256).
template <class A>
__global__ void k_add_ln(Mat<float> r_, Mat<const float> x, const float* bias, const float* g, const float* b, float eps,
                         Mat<A> ya, int64_t rows, int C) {
  int64_t r = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  int l = threadIdx.x & 31;
  if (r >= rows) return;
  constexpr int kMax = 16;  // C ≤ 512
  float v[kMax];
  float s = 0.f;
  float* rr = r_.row(r);
  const float* xr = x.row(r);
#pragma unroll
  for (int k = 0; k < kMax; k++) {
    int c = l + 32 * k;
    v[k] = c < C ? rr[c] + xr[c] + (bias ? bias[c] : 0.f) : 0.f;
    s += v[k];
  }
  float mean = warp_sum(s) / C;
  float q = 0.f;
#pragma unroll
  for (int k = 0; k < kMax; k++) {
    int c = l + 32 * k;
    float d = c < C ? v[k] - mean : 0.f;
    q += d * d;
  }
  float rstd = rsqrtf(warp_sum(q) / C + eps);
#pragma unroll
  for (int k = 0; k < kMax; k++) {
    int c = l + 32 * k;
    if (c < C) {
      float o = (v[k] - mean) * rstd * g[c] + b[c];
      rr[c] = o;
      if (ya.p) ya.row(r)[c] = from_f<A>(o);
    }
  }
}

template <class TI, class A>
__global__ void k_add_rows(Mat<const TI> x, Mat<const float> y, int64_t rows_y, Mat<A> out, int64_t rows, int C) {
  int64_t n = rows * C;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t r = i / C;
    int c = int(i - r * C);
    out.row(r)[c] = from_f<A>(ld(x.row(r) + c) + y.row(r % rows_y)[c]);
  }
}

// ---------------------------------------------------------------- 깊이별 conv. w: [K·K][C] fp32.
template <int K, class A>
__global__ void k_dw(Mat<const A> x, Mat<A> y, int ostep, int B, int H, int W, int C, const float* w,
                     const float* bias, float scale) {
  int64_t n = int64_t(B) * H * W * C;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t pix = i / C;
    int c = int(i - pix * C);
    int xx = int(pix % W);
    int yy = int((pix / W) % H);
    int64_t base = pix - int64_t(yy) * W - xx;  // 화면 첫 화소
    float acc = bias ? bias[c] : 0.f;
#pragma unroll
    for (int ky = 0; ky < K; ky++) {
      int sy = yy + ky - K / 2;
      if (sy < 0 || sy >= H) continue;
#pragma unroll
      for (int kx = 0; kx < K; kx++) {
        int sx = xx + kx - K / 2;
        if (sx < 0 || sx >= W) continue;
        acc += w[(ky * K + kx) * C + c] * to_f(x.row(base + int64_t(sy) * W + sx)[c]);
      }
    }
    y.row(pix)[int64_t(c) * ostep] = from_f<A>(acc * scale);
  }
}

template <int K, class A>
__global__ void k_dw_v(Mat<const A> x, Mat<A> y, int ostep, int B, int H, int W, int C, const float* w, const float* bias,
                       float scale) {
  int C8 = C / 8;
  int64_t n = int64_t(B) * H * W * C8;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t pix = i / C8;
    int c = int(i - pix * C8) * 8;
    int xx = int(pix % W);
    int yy = int((pix / W) % H);
    int64_t base = pix - int64_t(yy) * W - xx;
    float acc[8], v[8], wk[8];
    if (bias) ld8(bias + c, acc);
    else
#pragma unroll
      for (int e = 0; e < 8; e++) acc[e] = 0.f;
#pragma unroll
    for (int ky = 0; ky < K; ky++) {
      int sy = yy + ky - K / 2;
      if (sy < 0 || sy >= H) continue;
#pragma unroll
      for (int kx = 0; kx < K; kx++) {
        int sx = xx + kx - K / 2;
        if (sx < 0 || sx >= W) continue;
        ld8(x.row(base + int64_t(sy) * W + sx) + c, v);
        ld8(w + (ky * K + kx) * C + c, wk);
#pragma unroll
        for (int e = 0; e < 8; e++) acc[e] += wk[e] * v[e];
      }
    }
#pragma unroll
    for (int e = 0; e < 8; e++) acc[e] *= scale;
    if (ostep == 1) st8(y.row(pix) + c, acc);
    else
#pragma unroll
      for (int e = 0; e < 8; e++) y.row(pix)[int64_t(c + e) * ostep] = from_f<A>(acc[e]);
  }
}

// ---------------------------------------------------------------- GroupNorm
// stats[(b·G + g)·2] = (mean, rstd). 블록 (g, b).
template <class TI>
__global__ void k_gn_stats(const TI* x, int64_t HW, int C, int G, float eps, float* stats) {
  int g = blockIdx.x, b = blockIdx.y, cg = C / G;
  const TI* xb = x + int64_t(b) * HW * C + g * cg;
  double s = 0, q = 0;
  int64_t n = HW * cg;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    int64_t p = i / cg;
    float v = to_f(xb[p * C + (i - p * cg)]);
    s += v;
    q += double(v) * v;
  }
  __shared__ double sh[2][32];
  for (int o = 16; o; o >>= 1) {
    s += __shfl_xor_sync(0xffffffff, s, o);
    q += __shfl_xor_sync(0xffffffff, q, o);
  }
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) sh[0][w] = s, sh[1][w] = q;
  __syncthreads();
  if (threadIdx.x == 0) {
    double S = 0, Q = 0;
    for (int k = 0; k < int(blockDim.x >> 5); k++) S += sh[0][k], Q += sh[1][k];
    double mean = S / n, var = Q / n - mean * mean;
    stats[(b * G + g) * 2] = float(mean);
    stats[(b * G + g) * 2 + 1] = float(1.0 / sqrt(var + eps));
  }
}
template <class TI, class A>
__global__ void k_gn_apply(const TI* x, float* yf, A* ya, int B, int64_t HW, int C, int G, const float* g,
                           const float* b, bool relu, const float* stats) {
  int64_t n = int64_t(B) * HW * C;
  int cg = C / G;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int c = int(i % C);
    int bb = int(i / (HW * C));
    const float* s = stats + (bb * G + c / cg) * 2;
    float v = (to_f(x[i]) - s[0]) * s[1] * g[c] + b[c];
    if (relu) v = fmaxf(v, 0.f);
    if (yf) yf[i] = v;
    if (ya) ya[i] = from_f<A>(v);
  }
}

// ---------------------------------------------------------------- 양선형 보간(PyTorch align_corners=False)
struct Lerp {
  int i0, i1;
  float l0, l1;
};
__device__ __forceinline__ Lerp lerp_src(int dst, int in, int out) {
  float scale = float(in) / float(out);
  float src = scale * (dst + 0.5f) - 0.5f;
  if (src < 0.f) src = 0.f;
  int i0 = int(src);
  int p = i0 < in - 1 ? 1 : 0;
  float l1 = src - i0;
  return Lerp{i0, i0 + p, 1.f - l1, l1};
}
template <class TI, class A>
__global__ void k_resize(const TI* x, int B, int Hi, int Wi, int C, A* y, int Ho, int Wo, const A* add) {
  int64_t n = int64_t(B) * Ho * Wo * C;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int c = int(i % C);
    int64_t p = i / C;
    int xo = int(p % Wo), yo = int((p / Wo) % Ho), b = int(p / (int64_t(Wo) * Ho));
    Lerp ly = lerp_src(yo, Hi, Ho), lx = lerp_src(xo, Wi, Wo);
    const TI* xb = x + int64_t(b) * Hi * Wi * C + c;
    auto at = [&](int yy, int xx) { return to_f(xb[(int64_t(yy) * Wi + xx) * C]); };
    float v = ly.l0 * (lx.l0 * at(ly.i0, lx.i0) + lx.l1 * at(ly.i0, lx.i1)) +
              ly.l1 * (lx.l0 * at(ly.i1, lx.i0) + lx.l1 * at(ly.i1, lx.i1));
    if (add) v += to_f(add[i]);
    y[i] = from_f<A>(v);
  }
}

// ---------------------------------------------------------------- 전역 필터(FFT)
// x 행 r = (b, p) (p = yy·W + xx) 채널 c → buf[(b·C2 + c)·HW + p] (채널마다 연속; 32×32 타일 전치)
template <class A>
__global__ void k_gf_gather(Mat<const A> x, float* buf, int B, int64_t HW, int C2) {
  __shared__ float t[32][33];
  int b = blockIdx.z;
  int64_t p0 = int64_t(blockIdx.x) * 32;
  int c0 = blockIdx.y * 32;
  for (int k = threadIdx.y; k < 32; k += blockDim.y) {
    int64_t p = p0 + k;
    int c = c0 + threadIdx.x;
    t[k][threadIdx.x] = (p < HW && c < C2) ? to_f(x.row(int64_t(b) * HW + p)[c]) : 0.f;
  }
  __syncthreads();
  for (int k = threadIdx.y; k < 32; k += blockDim.y) {
    int c = c0 + k;
    int64_t p = p0 + threadIdx.x;
    if (p < HW && c < C2) buf[(int64_t(b) * C2 + c) * HW + p] = t[threadIdx.x][k];
  }
}
// spec [(b·C2 + c)][ky·Wr + kx] ·= 보간한 weight[c](ky, kx) / (H·W)
__global__ void k_gf_mul(float2* spec, const float2* w, int B, int H, int Wr, int C2, int wh, int ww, float sy, float sx,
                         float norm) {
  int64_t HWr = int64_t(H) * Wr;
  int64_t n = int64_t(B) * C2 * HWr;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t k = i % HWr;
    int c = int((i / HWr) % C2);
    int ky = int(k / Wr), kx = int(k % Wr);
    float fy = sy * ky, fx = sx * kx;
    int y0 = int(fy), yp = y0 < wh - 1 ? 1 : 0;
    int x0 = int(fx), xp = x0 < ww - 1 ? 1 : 0;
    float ly = fy - y0, hy = 1.f - ly, lx = fx - x0, hx = 1.f - lx;
    const float2* wc = w + int64_t(c) * wh * ww;
    float2 a00 = wc[y0 * ww + x0], a01 = wc[y0 * ww + x0 + xp], a10 = wc[(y0 + yp) * ww + x0], a11 = wc[(y0 + yp) * ww + x0 + xp];
    float qr = hy * (hx * a00.x + lx * a01.x) + ly * (hx * a10.x + lx * a11.x);
    float qi = hy * (hx * a00.y + lx * a01.y) + ly * (hx * a10.y + lx * a11.y);
    qr *= norm, qi *= norm;
    float2 a = spec[i];
    spec[i] = make_float2(a.x * qr - a.y * qi, a.x * qi + a.y * qr);
  }
}
template <class A>
__global__ void k_gf_scatter(const float* buf, Mat<A> y, int ostep, int B, int64_t HW, int C2) {
  __shared__ float t[32][33];
  int b = blockIdx.z;
  int64_t p0 = int64_t(blockIdx.x) * 32;
  int c0 = blockIdx.y * 32;
  for (int k = threadIdx.y; k < 32; k += blockDim.y) {
    int c = c0 + k;
    int64_t p = p0 + threadIdx.x;
    t[k][threadIdx.x] = (p < HW && c < C2) ? buf[(int64_t(b) * C2 + c) * HW + p] : 0.f;
  }
  __syncthreads();
  for (int k = threadIdx.y; k < 32; k += blockDim.y) {
    int64_t p = p0 + k;
    int c = c0 + threadIdx.x;
    if (p < HW && c < C2) y.row(int64_t(b) * HW + p)[int64_t(c) * ostep] = from_f<A>(t[threadIdx.x][k]);
  }
}

// ---------------------------------------------------------------- MSDeformAttn (워프 = (b, q, 머리), 레인 = 채널)
template <class A>
__global__ void k_deform(const A* value, const float* off, const float* off_b, const float* aw, const float* aw_b, A* out,
                         int B, int L, DeformLevels lv) {
  int64_t wid = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (wid >= int64_t(B) * L * 8) return;
  int h = int(wid % 8);
  int64_t bq = wid / 8;
  int q = int(bq % L), b = int(bq / L);
  // 기준점: 질의 토큰의 수준과 위치
  int lq = 0;
  while (lq + 1 < lv.n && q >= lv.start[lq + 1]) lq++;
  int qi = q - lv.start[lq];
  float ref_x = (float(qi % lv.w[lq]) + 0.5f) / float(lv.w[lq]);
  float ref_y = (float(qi / lv.w[lq]) + 0.5f) / float(lv.h[lq]);
  const float* o = off + bq * 192 + h * 24;
  const float* ob = off_b + h * 24;
  const float* a = aw + bq * 96 + h * 12;
  const float* ab = aw_b + h * 12;
  // softmax(12)
  float al = lane < 12 ? a[lane] + ab[lane] : -FLT_MAX;
  float mx = warp_max(al);
  float e = lane < 12 ? __expf(al - mx) : 0.f;
  e = e / warp_sum(e);
  float acc = 0.f;
  const A* vb = value + int64_t(b) * L * 256 + h * 32 + lane;
#pragma unroll
  for (int l = 0; l < 3; l++) {
    int Hl = lv.h[l], Wl = lv.w[l];
    const A* vl = vb + int64_t(lv.start[l]) * 256;
#pragma unroll
    for (int p = 0; p < 4; p++) {
      int k = l * 4 + p;
      float wgt = __shfl_sync(0xffffffff, e, k);
      float lx = ref_x + (o[k * 2] + ob[k * 2]) / float(Wl);
      float ly = ref_y + (o[k * 2 + 1] + ob[k * 2 + 1]) / float(Hl);
      float him = ly * Hl - 0.5f, wim = lx * Wl - 0.5f;
      if (him > -1.f && wim > -1.f && him < Hl && wim < Wl) {
        int h0 = floorf(him), w0 = floorf(wim), h1 = h0 + 1, w1 = w0 + 1;
        float lh = him - h0, lw = wim - w0, hh = 1.f - lh, hw = 1.f - lw;
        float v1 = (h0 >= 0 && w0 >= 0) ? to_f(vl[(int64_t(h0) * Wl + w0) * 256]) : 0.f;
        float v2 = (h0 >= 0 && w1 <= Wl - 1) ? to_f(vl[(int64_t(h0) * Wl + w1) * 256]) : 0.f;
        float v3 = (h1 <= Hl - 1 && w0 >= 0) ? to_f(vl[(int64_t(h1) * Wl + w0) * 256]) : 0.f;
        float v4 = (h1 <= Hl - 1 && w1 <= Wl - 1) ? to_f(vl[(int64_t(h1) * Wl + w1) * 256]) : 0.f;
        acc += wgt * (hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4);
      }
    }
  }
  out[bq * 256 + h * 32 + lane] = from_f<A>(acc);
}

// ---------------------------------------------------------------- 마스크 softmax (블록당 한 행)
template <class A>
__global__ void k_masked_softmax(const float* S, const float* ml, A* P, int64_t rows, int n) {
  __shared__ float sh[32];
  int64_t row = blockIdx.x;  // heads·rows
  const float* s = S + row * n;
  const float* m = ml ? ml + (row % rows) * n : nullptr;
  A* p = P + row * n;
  // 열린 칸 수와 최댓값
  float mx = -FLT_MAX, cnt = 0.f;
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    if (!m || !(m[j] < 0.f)) mx = fmaxf(mx, s[j]), cnt += 1.f;
  cnt = block_sum(cnt, sh);
  bool all_open = (m == nullptr) || cnt == 0.f;
  if (cnt == 0.f) {
    mx = -FLT_MAX;
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, s[j]);
  }
  mx = block_max(mx, sh);
  float sum = 0.f;
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    if (all_open || !(m[j] < 0.f)) sum += __expf(s[j] - mx);
  sum = block_sum(sum, sh);
  float inv = 1.f / sum;
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    p[j] = from_f<A>((all_open || !(m[j] < 0.f)) ? __expf(s[j] - mx) * inv : 0.f);
}

// ---------------------------------------------------------------- 후처리
// 화면 v 의 출력 (oh, ow) 화소마다 두 번 보간 → sigmoid → fused[q][(y0+Y)·OW + x0+X] += …
__global__ void k_fuse_view(const float* logit, int Q, int T, int Hm, int Wm, int Hp, int Wp, int v, int sh, int sw, int oh,
                            int ow, int x0, int y0, int OW, int64_t OHW, float* fused) {
  int64_t n = int64_t(Q) * oh * ow;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int X = int(i % ow);
    int64_t t = i / ow;
    int Y = int(t % oh);
    int q = int(t / oh);
    const float* lg = logit + (int64_t(q) * T + v) * Hm * Wm;
    // 패딩 크기 (Hp, Wp) 에서의 값
    auto up = [&](int yy, int xx) {
      Lerp a = lerp_src(yy, Hm, Hp), b = lerp_src(xx, Wm, Wp);
      auto at = [&](int r, int c) { return lg[int64_t(r) * Wm + c]; };
      return a.l0 * (b.l0 * at(a.i0, b.i0) + b.l1 * at(a.i0, b.i1)) + a.l1 * (b.l0 * at(a.i1, b.i0) + b.l1 * at(a.i1, b.i1));
    };
    Lerp a = lerp_src(Y, sh, oh), b = lerp_src(X, sw, ow);
    float val = a.l0 * (b.l0 * up(a.i0, b.i0) + b.l1 * up(a.i0, b.i1)) + a.l1 * (b.l0 * up(a.i1, b.i0) + b.l1 * up(a.i1, b.i1));
    fused[int64_t(q) * OHW + int64_t(y0 + Y) * OW + x0 + X] += 1.f / (1.f + expf(-val));
  }
}
__global__ void k_fuse_div(float* fused, const float* cnt, int Q, int64_t OHW) {
  int64_t n = int64_t(Q) * OHW;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    fused[i] = fused[i] / cnt[i % OHW];
}
__global__ void k_count(float* cnt, int OW, int64_t OHW, ViewGeom g) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < OHW; i += int64_t(gridDim.x) * blockDim.x) {
    int x = int(i % OW), y = int(i / OW);
    float c = 1e-16f;
    for (int v = 1; v < g.T; v++)
      if (x >= g.x0[v] && x < g.x0[v] + g.ow[v] && y >= g.y0[v] && y < g.y0[v] + g.oh[v]) c += 1.f;
    cnt[i] = c + 1.f;
  }
}
__global__ void k_assign(const float* fused, const int* order, int Q, int64_t HW, int16_t* id) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < HW; i += int64_t(gridDim.x) * blockDim.x) {
    int r = -1;
    for (int k = 0; k < Q; k++)
      if (fused[int64_t(order[k]) * HW + i] > 0.5f) {
        r = k;
        break;
      }
    id[i] = int16_t(r);
  }
}
__global__ void k_mscore(const float* fused, const int* order, const int16_t* id, int64_t HW, float* mscore, int* area) {
  __shared__ float sh[32];
  int k = blockIdx.x;
  const float* f = fused + int64_t(order[k]) * HW;
  float s = 0.f, a = 0.f;
  for (int64_t i = threadIdx.x; i < HW; i += blockDim.x)
    if (id[i] == k) s += 1.f / (1.f + expf(-f[i])), a += 1.f;
  s = block_sum(s, sh);
  a = block_sum(a, sh);
  if (threadIdx.x == 0) mscore[k] = s / (a + 1e-6f), area[k] = int(a);
}

// --use_fast_math 에서 sinf·powf 는 근사 함수라 배정밀도로 계산해 float 로 맞춘다(PyTorch 는 정확한 float 함수).
__global__ void k_sine(int H, int W, int T, const float* add, float* out) {
  const float two_pi = 6.283185307179586f;
  int Tn = T > 0 ? T : 1;
  int64_t n = int64_t(Tn) * H * W * 256;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int ch = int(i % 256);
    int64_t pt = i / 256;
    int p = int(pt % (int64_t(H) * W)), t = int(pt / (int64_t(H) * W));
    int yy = p / W, xx = p % W;
    int k = ch & 127;
    float dim_t = float(pow(10000.0, double(float(2 * (k / 2)) / 128.f)));
    float e = ch < 128 ? __fmul_rn(__fdiv_rn(float(yy + 1), __fadd_rn(float(H), 1e-6f)), two_pi)
                       : __fmul_rn(__fdiv_rn(float(xx + 1), __fadd_rn(float(W), 1e-6f)), two_pi);
    float a = __fdiv_rn(e, dim_t);
    float v = (k & 1) ? float(cos(double(a))) : float(sin(double(a)));
    if (add) v += add[ch];
    if (T > 0) {
      float z = __fmul_rn(__fdiv_rn(float(t + 1), __fadd_rn(float(T), 1e-6f)), two_pi);
      float d = float(pow(10000.0, double(float(2 * (ch / 2)) / 256.f)));
      float zz = __fdiv_rn(z, d);
      v += (ch & 1) ? float(cos(double(zz))) : float(sin(double(zz)));
    }
    out[i] = v;
  }
}

__global__ void k_rank(const float* cls, int Q, float* p, int* order) {
  __shared__ float sp[1024];
  int q = threadIdx.x;
  if (q < Q) {
    float a = cls[q * 2], b = cls[q * 2 + 1], m = fmaxf(a, b);
    float e0 = expf(a - m), e1 = expf(b - m);
    sp[q] = e0 / (e0 + e1);
    p[q] = sp[q];
  }
  __syncthreads();
  if (q < Q) {
    int r = 0;
    for (int o = 0; o < Q; o++) r += sp[o] > sp[q] || (sp[o] == sp[q] && o < q);
    order[r] = q;
  }
}
__global__ void k_final(const float* p, const int* order, const float* mscore, int Q, float* score) {
  int k = threadIdx.x;
  if (k < Q) score[k] = p[order[k]] * mscore[k];
}

}  // namespace

// ================================================================ 호스트 쪽
void sine_pos(int H, int W, int T, const float* add, float* out, cudaStream_t st) {
  k_sine<<<nblk(int64_t(T > 0 ? T : 1) * H * W * 256), kT, 0, st>>>(H, W, T, add, out);
}
void rank_scores(const float* cls, int Q, float* p, int* order, cudaStream_t st) {
  FSO_ASSERT(Q <= 1024, "Q %d", Q);
  k_rank<<<1, 1024, 0, st>>>(cls, Q, p, order);
}
void final_scores(const float* p, const int* order, const float* mscore, int Q, float* score, cudaStream_t st) {
  k_final<<<1, 1024, 0, st>>>(p, order, mscore, Q, score);
}

template <class A, class TY>
void gemm_nt(cublasHandle_t h, int64_t M, int64_t N, int64_t K, const A* X, int ldx, const A* W, int ldw, TY* Y, int ldy,
             float alpha, float beta, int batch, int64_t sx, int64_t sw, int64_t sy) {
  // 열 우선: Yᵀ[N,M] = W[N,K] · Xᵀ[K,M]
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, int(N), int(M), int(K), &alpha, W, blas_t<A>(), ldw, sw,
                                          X, blas_t<A>(), ldx, sx, &beta, Y, blas_t<TY>(), ldy, sy, batch, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT));
}
template <class A, class TY>
void gemm_nn(cublasHandle_t h, int64_t M, int64_t N, int64_t K, const A* X, int ldx, const A* W, int ldw, TY* Y, int ldy,
             float alpha, float beta, int batch, int64_t sx, int64_t sw, int64_t sy) {
  // 열 우선: Yᵀ[N,M] = Wᵀ[N,K] · Xᵀ[K,M], W 메모리(행 우선 [K,N]) = 열 우선 [N,K]
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, int(N), int(M), int(K), &alpha, W, blas_t<A>(), ldw, sw,
                                          X, blas_t<A>(), ldx, sx, &beta, Y, blas_t<TY>(), ldy, sy, batch, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT));
}

void LtCtx::init() {
  CUBLAS_CHECK(cublasLtCreate(&h));
  CUDA_CHECK(cudaMalloc(&ws, wsz));
}
template <class A, class TY>
void lt_gemm(LtCtx& lt, int64_t M, int64_t N, int64_t K, const A* X, int ldx, const A* W, int ldw, TY* Y, int ldy,
             const TY* bias, Epi epi, float beta, cudaStream_t st) {
  // 열 우선: Dᵀ[N,M] = W[N,K]·Xᵀ[K,M] (+ bias 행마다 = 출력 채널마다)
  auto key = std::make_tuple(M, N, K, ldx, ldw, ldy, int(blas_t<A>()), int(blas_t<TY>()), int(epi), int(beta != 0.f));
  auto it = lt.plans.find(key);
  if (it == lt.plans.end()) {
    LtCtx::Plan p;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    cublasLtEpilogue_t e = epi == kEpiBias ? CUBLASLT_EPILOGUE_BIAS
                           : epi == kEpiReluBias ? CUBLASLT_EPILOGUE_RELU_BIAS
                                                 : CUBLASLT_EPILOGUE_DEFAULT;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
    if (epi != kEpiNone) {
      cudaDataType_t bt = blas_t<TY>();
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
      CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.a, blas_t<A>(), K, N, ldw));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.b, blas_t<A>(), K, M, ldx));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.c, blas_t<TY>(), N, M, ldy));
    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt.wsz, sizeof(lt.wsz)));
    cublasLtMatmulHeuristicResult_t r;
    int n = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt.h, p.op, p.a, p.b, p.c, p.c, pref, 1, &r, &n);
    FSO_ASSERT(hs == CUBLAS_STATUS_SUCCESS, "cuBLASLt heuristic %d: M%lld N%lld K%lld ld %d %d %d A%d Y%d epi %d beta %g", int(hs),
               (long long)M, (long long)N, (long long)K, ldx, ldw, ldy, int(blas_t<A>()), int(blas_t<TY>()), int(epi), beta);
    FSO_ASSERT(n > 0, "no cuBLASLt algo M%lld N%lld K%lld epi %d", (long long)M, (long long)N, (long long)K, int(epi));
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    p.algo = r.algo;
    it = lt.plans.emplace(key, p).first;
  }
  LtCtx::Plan& p = it->second;
  if (epi != kEpiNone) CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  float alpha = 1.f;
  CUBLAS_CHECK(cublasLtMatmul(lt.h, p.op, &alpha, W, p.a, X, p.b, &beta, Y, p.c, Y, p.c, &p.algo, lt.ws, lt.wsz, st));
}

template <class TI, class TO>
void layer_norm(Mat<const TI> x, Mat<TO> y, int64_t rows, int C, const float* g, const float* b, float eps, cudaStream_t st,
                float scale, const float* add) {
  int nv = (C / 8 + 31) / 32;
  if (vec_ok(x, C) && vec_ok(y, C) && nv <= 12 && (!add || al16(add))) {
    unsigned nb = nblk(rows * 32);
#define LN_V(N) k_ln_v<TI, TO, N><<<nb, kT, 0, st>>>(x, y, rows, C, g, b, eps, scale, add)
    if (nv <= 1) LN_V(1);
    else if (nv <= 2) LN_V(2);
    else if (nv <= 3) LN_V(3);
    else if (nv <= 6) LN_V(6);
    else LN_V(12);
#undef LN_V
  } else {
    k_ln<TI, TO><<<nblk(rows * 32), kT, 0, st>>>(x, y, rows, C, g, b, eps, scale, add);
  }
}
template <class TI, class TO>
void bias_act(Mat<const TI> x, Mat<TO> y, int64_t rows, int C, const float* bias, Act act, cudaStream_t st, float scale) {
  if (vec_ok(x, C) && vec_ok(y, C) && (!bias || al16(bias)))
    k_bias_act_v<TI, TO><<<nblk(rows * C / 8), kT, 0, st>>>(x, y, rows, C, bias, int(act), scale);
  else
    k_bias_act<TI, TO><<<nblk(rows * C), kT, 0, st>>>(x, y, rows, C, bias, int(act), scale);
}
template <class TI, class A>
void bias_mul(Mat<const TI> x, const float* bias, Mat<const A> m, Mat<A> y, int64_t rows, int C, cudaStream_t st) {
  if (vec_ok(x, C) && vec_ok(m, C) && vec_ok(y, C) && (!bias || al16(bias)))
    k_bias_mul_v<TI, A><<<nblk(rows * C / 8), kT, 0, st>>>(x, bias, m, y, rows, C);
  else
    k_bias_mul<TI, A><<<nblk(rows * C), kT, 0, st>>>(x, bias, m, y, rows, C);
}
void res_add(Mat<float> r, Mat<const float> x, const float* bias, const float* gamma, int64_t rows, int C, cudaStream_t st) {
  if (vec_ok(r, C) && vec_ok(x, C) && al16(bias) && al16(gamma))
    k_res_add_v<<<nblk(rows * C / 8), kT, 0, st>>>(r, x, bias, gamma, rows, C);
  else
    k_res_add<<<nblk(rows * C), kT, 0, st>>>(r, x, bias, gamma, rows, C);
}
template <class A>
void add_ln(Mat<float> r, Mat<const float> x, const float* bias, const float* g, const float* b, float eps, Mat<A> ya,
            int64_t rows, int C, cudaStream_t st) {
  FSO_ASSERT(C <= 512, "add_ln C %d", C);
  k_add_ln<A><<<nblk(rows * 32), kT, 0, st>>>(r, x, bias, g, b, eps, ya, rows, C);
}
template <class TI, class A>
void add_rows(Mat<const TI> x, Mat<const float> y, int64_t rows_y, Mat<A> out, int64_t rows, int C, cudaStream_t st) {
  if (vec_ok(x, C) && vec_ok(y, C) && vec_ok(out, C))
    k_add_rows_v<TI, A><<<nblk(rows * C / 8), kT, 0, st>>>(x, y, rows_y, out, rows, C);
  else
    k_add_rows<TI, A><<<nblk(rows * C), kT, 0, st>>>(x, y, rows_y, out, rows, C);
}
template <class A> void patchify(const A* x, int B, int H, int W, int C, int k, A* y, cudaStream_t st) {
  k_patchify<A><<<nblk(int64_t(B) * H * W * C), kT, 0, st>>>(x, B, H, W, C, k, y);
}
template <class A>
void dwconv(Mat<const A> x, Mat<A> y, int ostep, int B, int H, int W, int C, int K, const float* w, const float* bias,
            float scale, cudaStream_t st) {
  int64_t n = int64_t(B) * H * W * C;
  bool v = vec_ok(x, C) && al16(w) && (!bias || al16(bias)) && (ostep != 1 || vec_ok(y, C));
  if (K == 7 && v) k_dw_v<7, A><<<nblk(n / 8), kT, 0, st>>>(x, y, ostep, B, H, W, C, w, bias, scale);
  else if (K == 3 && v) k_dw_v<3, A><<<nblk(n / 8), kT, 0, st>>>(x, y, ostep, B, H, W, C, w, bias, scale);
  else if (K == 7) k_dw<7, A><<<nblk(n), kT, 0, st>>>(x, y, ostep, B, H, W, C, w, bias, scale);
  else if (K == 3) k_dw<3, A><<<nblk(n), kT, 0, st>>>(x, y, ostep, B, H, W, C, w, bias, scale);
  else FSO_ASSERT(false, "dwconv K %d", K);
}
template <class TI, class A>
void group_norm(const TI* x, float* yf, A* ya, int B, int64_t HW, int C, int G, const float* g, const float* b, float eps,
                bool relu, float* stats, cudaStream_t st) {
  k_gn_stats<TI><<<dim3(G, B), 1024, 0, st>>>(x, HW, C, G, eps, stats);
  k_gn_apply<TI, A><<<nblk(int64_t(B) * HW * C), kT, 0, st>>>(x, yf, ya, B, HW, C, G, g, b, relu, stats);
}
template <class TI, class A>
void resize_bilinear(const TI* x, int B, int Hi, int Wi, int C, A* y, int Ho, int Wo, const A* add, cudaStream_t st) {
  k_resize<TI, A><<<nblk(int64_t(B) * Ho * Wo * C), kT, 0, st>>>(x, B, Hi, Wi, C, y, Ho, Wo, add);
}

void GlobalFilterPlan::init(int B_, int H_, int W_, int C2_, cudaStream_t st) {
  B = B_, H = H_, W = W_, C2 = C2_;
  int n[2] = {H, W};
  size_t w1 = 0, w2 = 0;
  // 채널마다 연속 [B·C2][H][W] → [B·C2][H][W/2+1]
  CUFFT_CHECK(cufftCreate(&r2c));
  CUFFT_CHECK(cufftCreate(&c2r));
  CUFFT_CHECK(cufftSetAutoAllocation(r2c, 0));
  CUFFT_CHECK(cufftSetAutoAllocation(c2r, 0));
  CUFFT_CHECK(cufftMakePlanMany(r2c, 2, n, nullptr, 1, H * W, nullptr, 1, H * (W / 2 + 1), CUFFT_R2C, B * C2, &w1));
  CUFFT_CHECK(cufftMakePlanMany(c2r, 2, n, nullptr, 1, H * (W / 2 + 1), nullptr, 1, H * W, CUFFT_C2R, B * C2, &w2));
  work = std::max(w1, w2);
  CUFFT_CHECK(cufftSetStream(r2c, st));
  CUFFT_CHECK(cufftSetStream(c2r, st));
}
template <class A>
void global_filter(const GlobalFilterPlan& p, Mat<const A> x, Mat<A> y, int ostep, const float2* weight, int wh, int ww,
                   float* buf, float2* spec, void* work, cudaStream_t st) {
  CUFFT_CHECK(cufftSetWorkArea(p.r2c, work));
  CUFFT_CHECK(cufftSetWorkArea(p.c2r, work));
  int64_t HW = int64_t(p.H) * p.W, HWr = int64_t(p.H) * (p.W / 2 + 1);
  dim3 grid(unsigned((HW + 31) / 32), unsigned((p.C2 + 31) / 32), unsigned(p.B)), blk(32, 8);
  k_gf_gather<A><<<grid, blk, 0, st>>>(x, buf, p.B, HW, p.C2);
  CUFFT_CHECK(cufftExecR2C(p.r2c, buf, reinterpret_cast<cufftComplex*>(spec)));
  int Wr = p.W / 2 + 1;
  float sy = p.H > 1 ? float(wh - 1) / float(p.H - 1) : 0.f, sx = Wr > 1 ? float(ww - 1) / float(Wr - 1) : 0.f;
  k_gf_mul<<<nblk(int64_t(p.B) * p.C2 * HWr), kT, 0, st>>>(spec, weight, p.B, p.H, Wr, p.C2, wh, ww, sy, sx,
                                                           1.f / float(p.H * p.W));
  CUFFT_CHECK(cufftExecC2R(p.c2r, reinterpret_cast<cufftComplex*>(spec), buf));
  k_gf_scatter<A><<<grid, blk, 0, st>>>(buf, y, ostep, p.B, HW, p.C2);
}
template <class A>
void ms_deform_sample(const A* value, const float* off, const float* off_bias, const float* aw, const float* aw_bias, A* out,
                      int B, int L, const DeformLevels& lv, cudaStream_t st) {
  k_deform<A><<<nblk(int64_t(B) * L * 8 * 32), kT, 0, st>>>(value, off, off_bias, aw, aw_bias, out, B, L, lv);
}
template <class A>
void masked_softmax(const float* S, const float* mask_logit, A* P, int heads, int64_t rows, int n, cudaStream_t st) {
  k_masked_softmax<A><<<unsigned(heads * rows), 256, 0, st>>>(S, mask_logit, P, rows, n);
}
void fuse_views(const float* logit, int Q, const ViewGeom& g, float* fused, float* cnt, cudaStream_t st) {
  int64_t OHW = int64_t(g.OH) * g.OW;
  CUDA_CHECK(cudaMemsetAsync(fused, 0, sizeof(float) * Q * OHW, st));
  // CropFormer 순서: 자른 화면들을 먼저 더하고 전체 화면을 마지막에.
  for (int v = 1; v <= g.T; v++) {
    int vv = v % g.T;
    int64_t n = int64_t(Q) * g.oh[vv] * g.ow[vv];
    k_fuse_view<<<nblk(n), kT, 0, st>>>(logit, Q, g.T, g.Hm, g.Wm, g.Hp, g.Wp, vv, g.sh[vv], g.sw[vv], g.oh[vv], g.ow[vv],
                                        g.x0[vv], g.y0[vv], g.OW, OHW, fused);
  }
  k_count<<<nblk(OHW), kT, 0, st>>>(cnt, g.OW, OHW, g);
  k_fuse_div<<<nblk(int64_t(Q) * OHW), kT, 0, st>>>(fused, cnt, Q, OHW);
}
void assign_ids(const float* fused, const int* order, int Q, int64_t HW, int16_t* id, float* mscore, int* area, cudaStream_t st) {
  k_assign<<<nblk(HW), kT, 0, st>>>(fused, order, Q, HW, id);
  k_mscore<<<Q, 1024, 0, st>>>(fused, order, id, HW, mscore, area);
}

// ---------------------------------------------------------------- 명시적 인스턴스
#define I_A(A)                                                                                                            \
  template void gemm_nt<A, float>(cublasHandle_t, int64_t, int64_t, int64_t, const A*, int, const A*, int, float*, int, float, \
                                  float, int, int64_t, int64_t, int64_t);                                                  \
  template void gemm_nn<A, float>(cublasHandle_t, int64_t, int64_t, int64_t, const A*, int, const A*, int, float*, int, float, \
                                  float, int, int64_t, int64_t, int64_t);                                                  \
  template void add_ln<A>(Mat<float>, Mat<const float>, const float*, const float*, const float*, float, Mat<A>, int64_t, int, \
                          cudaStream_t);                                                                                   \
  template void patchify<A>(const A*, int, int, int, int, int, A*, cudaStream_t);                                          \
  template void dwconv<A>(Mat<const A>, Mat<A>, int, int, int, int, int, int, const float*, const float*, float, cudaStream_t); \
  template void global_filter<A>(const GlobalFilterPlan&, Mat<const A>, Mat<A>, int, const float2*, int, int, float*, float2*, \
                                 void*, cudaStream_t);                                                                   \
  template void ms_deform_sample<A>(const A*, const float*, const float*, const float*, const float*, A*, int, int,         \
                                    const DeformLevels&, cudaStream_t);                                                    \
  template void masked_softmax<A>(const float*, const float*, A*, int, int64_t, int, cudaStream_t);
I_A(float)
I_A(__half)
template void lt_gemm<float, float>(LtCtx&, int64_t, int64_t, int64_t, const float*, int, const float*, int, float*, int,
                                    const float*, Epi, float, cudaStream_t);
template void lt_gemm<__half, float>(LtCtx&, int64_t, int64_t, int64_t, const __half*, int, const __half*, int, float*, int,
                                     const float*, Epi, float, cudaStream_t);
template void lt_gemm<__half, __half>(LtCtx&, int64_t, int64_t, int64_t, const __half*, int, const __half*, int, __half*, int,
                                      const __half*, Epi, float, cudaStream_t);
template void gemm_nt<__half, __half>(cublasHandle_t, int64_t, int64_t, int64_t, const __half*, int, const __half*, int, __half*,
                                      int, float, float, int, int64_t, int64_t, int64_t);
template void gemm_nn<__half, __half>(cublasHandle_t, int64_t, int64_t, int64_t, const __half*, int, const __half*, int, __half*,
                                      int, float, float, int, int64_t, int64_t, int64_t);
#define I_TI_A(TI, A)                                                                                                     \
  template void bias_mul<TI, A>(Mat<const TI>, const float*, Mat<const A>, Mat<A>, int64_t, int, cudaStream_t);            \
  template void add_rows<TI, A>(Mat<const TI>, Mat<const float>, int64_t, Mat<A>, int64_t, int, cudaStream_t);             \
  template void group_norm<TI, A>(const TI*, float*, A*, int, int64_t, int, int, const float*, const float*, float, bool,  \
                                  float*, cudaStream_t);                                                                   \
  template void resize_bilinear<TI, A>(const TI*, int, int, int, int, A*, int, int, const A*, cudaStream_t);               \
  template void layer_norm<TI, A>(Mat<const TI>, Mat<A>, int64_t, int, const float*, const float*, float, cudaStream_t,     \
                                  float, const float*);                                                                                \
  template void bias_act<TI, A>(Mat<const TI>, Mat<A>, int64_t, int, const float*, Act, cudaStream_t, float);
I_TI_A(float, float)
I_TI_A(float, __half)
I_TI_A(__half, __half)
I_TI_A(__half, float)
