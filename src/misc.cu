#include "misc.cuh"

namespace {
constexpr int kT = 256;
int grid_for(int64_t n) {
  int64_t b = cdiv(n, kT);
  return int(b < 65535 * 8 ? b : 65535 * 8);
}

// ---- 8 채널(16 B) 읽기·쓰기
__device__ __forceinline__ void ld8(const float* p, float o[8]) {
  float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
__device__ __forceinline__ void st8(float* p, const float o[8]) {
  *reinterpret_cast<float4*>(p) = make_float4(o[0], o[1], o[2], o[3]);
  *reinterpret_cast<float4*>(p + 4) = make_float4(o[4], o[5], o[6], o[7]);
}
__device__ __forceinline__ void ld8(const __half* p, float o[8]) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const __half2* h = reinterpret_cast<const __half2*>(&u);
  for (int i = 0; i < 4; i++) { float2 f = __half22float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}
__device__ __forceinline__ void st8(__half* p, const float o[8]) {
  uint4 u;
  __half2* h = reinterpret_cast<__half2*>(&u);
  for (int i = 0; i < 4; i++) h[i] = __floats2half2_rn(o[2 * i], o[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}

// 한 단계 maxpool: src 슬라이스 → dst 슬라이스(같은 cat 버퍼, 채널 ctot). 스레드 하나 = 화소 하나의 8 채널.
__global__ void k_pool5(act_t* cat, int N, int H, int W, int c, int ctot, int si, int di) {
  const int V = c / 8, n = N * H * W * V;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int ch = (i % V) * 8, pix = i / V;
    int w = pix % W, t = pix / W, h = t % H, b = t / H;
    float m[8], v[8];
    for (int k = 0; k < 8; k++) m[k] = -INFINITY;
    for (int y = max(h - 2, 0); y <= min(h + 2, H - 1); y++)
      for (int x = max(w - 2, 0); x <= min(w + 2, W - 1); x++) {
        ld8(cat + int64_t((b * H + y) * W + x) * ctot + si * c + ch, v);
        for (int k = 0; k < 8; k++) m[k] = v[k] > m[k] ? v[k] : m[k];
      }
    st8(cat + int64_t(pix) * ctot + di * c + ch, m);
  }
}

// maxpool 역: 출력 위치마다 창의 첫 최댓값(행 우선, PyTorch 와 같은 규칙)에 기울기를 더한다.
// g_in: 이 단계 출력 기울기 = dcat[di] (+ g_extra), out: fp32 누적(입력 슬라이스 위치). 스레드 하나 = 화소 하나의 8 채널.
__global__ void k_unpool5(const act_t* cat, const act_t* dcat, const float* g_extra, int N, int H, int W, int c,
                          int ctot, int si, int di, float* out) {
  const int V = c / 8, n = N * H * W * V;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int ch = (i % V) * 8, pix = i / V;
    float g[8], v[8], m[8];
    int arg[8];
    ld8(dcat + int64_t(pix) * ctot + di * c + ch, g);
    if (g_extra) {
      ld8(g_extra + int64_t(pix) * c + ch, v);
      for (int k = 0; k < 8; k++) g[k] += v[k];
    }
    bool any = false;
    for (int k = 0; k < 8; k++) { any |= g[k] != 0.f; m[k] = -INFINITY; arg[k] = -1; }
    if (!any) continue;
    int w = pix % W, t = pix / W, h = t % H, b = t / H;
    for (int y = max(h - 2, 0); y <= min(h + 2, H - 1); y++)
      for (int x = max(w - 2, 0); x <= min(w + 2, W - 1); x++) {
        int q = (b * H + y) * W + x;
        ld8(cat + int64_t(q) * ctot + si * c + ch, v);
        for (int k = 0; k < 8; k++)
          if (v[k] > m[k]) { m[k] = v[k]; arg[k] = q; }
      }
    for (int k = 0; k < 8; k++)
      if (g[k] != 0.f) atomicAdd(&out[int64_t(arg[k]) * c + ch + k], g[k]);
  }
}

__global__ void k_add_into_slice(const float* src, act_t* dcat, int npix, int c, int ctot, int si) {
  const int V = c / 8, n = npix * V;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int ch = (i % V) * 8, pix = i / V;
    float a[8], d[8];
    act_t* p = dcat + int64_t(pix) * ctot + si * c + ch;
    ld8(p, d);
    ld8(src + int64_t(pix) * c + ch, a);
    for (int k = 0; k < 8; k++) d[k] += a[k];
    st8(p, d);
  }
}

__global__ void k_chan_partial(const act_t* __restrict__ x, int64_t M, int C, int rows, float* __restrict__ part) {
  int64_t r0 = int64_t(blockIdx.x) * rows;
  int64_t r1 = r0 + rows < M ? r0 + rows : M;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float acc = 0.f;
    for (int64_t r = r0; r < r1; r++) acc += to_f(x[r * C + c]);
    part[int64_t(blockIdx.x) * C + c] = acc;
  }
}
// 블록 하나 = 채널 하나: 부분합 nb 개를 스레드들이 나눠 double 로 더한다.
__global__ void k_chan_final(const float* part, int nb, int C, float* out, bool accum) {
  const int c = blockIdx.x;
  double s = 0;
  for (int b = threadIdx.x; b < nb; b += blockDim.x) s += part[int64_t(b) * C + c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  __shared__ double sm[32];
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x) return;
  s = 0;
  for (int w = 0; w < int(blockDim.x >> 5); w++) s += sm[w];
  out[c] = accum ? out[c] + float(s) : float(s);
}
int chan_rows(int64_t M) { int64_t r = M / 2048; return int(r < 64 ? 64 : r); }

__global__ void k_add_bias(act_t* y, int64_t M, int C, const float* bias) {
  int64_t n = M * C;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    y[i] = from_f<act_t>(to_f(y[i]) + bias[i % C]);
}
// C 가 8 의 배수: 스레드 하나 = 8 채널
__global__ void k_add_bias8(act_t* y, int64_t M, int C, const float* bias) {
  const int V = C / 8;
  const int64_t n = M * V;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int ch = int(i % V) * 8;
    float v[8], b[8];
    act_t* p = y + i * 8;
    ld8(p, v);
    for (int k = 0; k < 8; k++) b[k] = bias[ch + k];  // 파라미터 배열 안 위치라 16 B 정렬이 아닐 수 있다
    for (int k = 0; k < 8; k++) v[k] += b[k];
    st8(p, v);
  }
}
__global__ void k_f32_to_act(const float* s, act_t* d, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    d[i] = from_f<act_t>(s[i]);
}
__global__ void k_act_accum(const act_t* s, float* d, int64_t n, bool accum) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    d[i] = accum ? d[i] + to_f(s[i]) : to_f(s[i]);
}
__global__ void k_fill(float* p, float v, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    p[i] = v;
}
}  // namespace

void sppf_pool_fwd(act_t* cat, int N, int H, int W, int c, cudaStream_t st) {
  int64_t n = int64_t(N) * H * W * c;
  FSO_ASSERT(c % 8 == 0, "SPPF channels %d", c);
  for (int k = 0; k < 3; k++) k_pool5<<<grid_for(n / 8), kT, 0, st>>>(cat, N, H, W, c, 4 * c, k, k + 1);
}

void sppf_pool_bwd(const act_t* cat, act_t* dcat, int N, int H, int W, int c, float* ta, float* tb,
                   cudaStream_t st) {
  int64_t npix = int64_t(N) * H * W, n = npix * c;
  // t = unpool(슬라이스2 → 3 의 기울기)  ⇒ 슬라이스2 위치 누적
  CUDA_CHECK(cudaMemsetAsync(ta, 0, n * sizeof(float), st));
  k_unpool5<<<grid_for(n / 8), kT, 0, st>>>(cat, dcat, nullptr, N, H, W, c, 4 * c, 2, 3, ta);
  CUDA_CHECK(cudaMemsetAsync(tb, 0, n * sizeof(float), st));
  k_unpool5<<<grid_for(n / 8), kT, 0, st>>>(cat, dcat, ta, N, H, W, c, 4 * c, 1, 2, tb);
  CUDA_CHECK(cudaMemsetAsync(ta, 0, n * sizeof(float), st));
  k_unpool5<<<grid_for(n / 8), kT, 0, st>>>(cat, dcat, tb, N, H, W, c, 4 * c, 0, 1, ta);
  k_add_into_slice<<<grid_for(n / 8), kT, 0, st>>>(ta, dcat, int(npix), c, 4 * c, 0);
}

int chan_sum_blocks(int64_t M, int C) { (void)C; return cdiv(M, chan_rows(M)); }
void chan_sum(const act_t* x, int64_t M, int C, float* partial, float* out, bool accum, cudaStream_t st) {
  int rows = chan_rows(M), nb = cdiv(M, rows);
  k_chan_partial<<<nb, C < 256 ? (C + 31) / 32 * 32 : 256, 0, st>>>(x, M, C, rows, partial);
  k_chan_final<<<C, 256, 0, st>>>(partial, nb, C, out, accum);
}
void add_bias(act_t* y, int64_t M, int C, const float* bias, cudaStream_t st) {
  if (C % 8 == 0) k_add_bias8<<<grid_for(M * C / 8), kT, 0, st>>>(y, M, C, bias);
  else k_add_bias<<<grid_for(M * C), kT, 0, st>>>(y, M, C, bias);
}
void f32_to_act(const float* s, act_t* d, int64_t n, cudaStream_t st) { k_f32_to_act<<<grid_for(n), kT, 0, st>>>(s, d, n); }
void act_accum_f32(const act_t* s, float* d, int64_t n, bool accum, cudaStream_t st) {
  k_act_accum<<<grid_for(n), kT, 0, st>>>(s, d, n, accum);
}
void fill_f32(float* p, float v, int64_t n, cudaStream_t st) { k_fill<<<grid_for(n), kT, 0, st>>>(p, v, n); }

namespace {
__global__ void k_accum_items(const float* src, float* acc, int n) {
  int i = threadIdx.x;
  if (i < n) acc[i] += src[i];
  if (i == 0) acc[n] += 1.f;
}
}  // namespace
void accum_items(const float* src, float* acc, int n, cudaStream_t st) { k_accum_items<<<1, 32, 0, st>>>(src, acc, n); }

// ---------------------------------------------------------------- 첫 층 wgrad(입력 3 채널, 3×3 s2 p1, 출력 32 채널)
// cuDNN 은 C=3 에서 텐서코어를 못 써 1.2 ms(4.7 TFLOPS) 걸린다. dW[k][r][s][c] = Σ_pix dz[pix][k]·x[patch(pix)][r,s,c] 를
// 출력 화소 64 개씩 [dz 64×32]ᵀ × [patch 64×32(27 + 0 다섯)] 로 WMMA(fp16 입력, fp32 누적)에 태운다.
#if !defined(FSO_FP32)
#include <mma.h>
namespace {
constexpr int kWgTile = 64;  // 타일 하나 = 출력 한 행의 화소 64 개
__global__ void __launch_bounds__(256) k_wgrad_rgb_s2(const __half* __restrict__ x, const __half* __restrict__ dz,
                                                      int N, int H, int W, int P, int Q, float* __restrict__ dw) {
  using namespace nvcuda;
  __shared__ __align__(32) __half sdz[kWgTile * 32];
  __shared__ __align__(32) __half spt[kWgTile * 32];
  __shared__ __align__(32) float sc[8][16 * 16];
  const int warp = threadIdx.x >> 5, mc = warp >> 1, kb = warp & 1;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
  wmma::fill_fragment(acc[0], 0.f);
  wmma::fill_fragment(acc[1], 0.f);
  const int tq = Q / kWgTile, ntile = N * P * tq;
  for (int t = blockIdx.x; t < ntile; t += gridDim.x) {
    const int q0 = (t % tq) * kWgTile, np = t / tq, p = np % P, n = np / P;
    // dz: 화소 64 개 × 32 채널 = 연속 4 KB
    const int64_t pix0 = (int64_t(n) * P + p) * Q + q0;
    reinterpret_cast<uint4*>(sdz)[threadIdx.x] = reinterpret_cast<const uint4*>(dz + pix0 * 32)[threadIdx.x];
    // patch: 스레드 하나가 한 화소의 8 칸
    {
      const int m = threadIdx.x >> 2, j0 = (threadIdx.x & 3) * 8, q = q0 + m;
      __half v[8];
      for (int u = 0; u < 8; u++) {
        int j = j0 + u;
        __half h = __float2half(0.f);
        if (j < 27) {
          int r = j / 9, s = (j / 3) % 3, c = j % 3;
          int yy = 2 * p - 1 + r, xx = 2 * q - 1 + s;
          if (yy >= 0 && yy < H && xx >= 0 && xx < W) h = x[((int64_t(n) * H + yy) * W + xx) * 3 + c];
        }
        v[u] = h;
      }
      *reinterpret_cast<uint4*>(spt + m * 32 + j0) = *reinterpret_cast<const uint4*>(v);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> fa;
    wmma::load_matrix_sync(fa, sdz + mc * 16 * 32 + kb * 16, 32);
    for (int jb = 0; jb < 2; jb++) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
      wmma::load_matrix_sync(fb, spt + mc * 16 * 32 + jb * 16, 32);
      wmma::mma_sync(acc[jb], fa, fb, acc[jb]);
    }
    __syncthreads();
  }
  // 같은 k 블록의 네 워프(화소 조각)를 더해 전역 fp32 에 atomicAdd
  for (int jb = 0; jb < 2; jb++) {
    wmma::store_matrix_sync(sc[warp], acc[jb], 16, wmma::mem_row_major);
    __syncthreads();
    for (int e = threadIdx.x; e < 2 * 256; e += blockDim.x) {
      int b = e >> 8, i = e & 255, k = b * 16 + (i >> 4), j = jb * 16 + (i & 15);
      float v = sc[b][i] + sc[b + 2][i] + sc[b + 4][i] + sc[b + 6][i];
      if (j < 27) atomicAdd(&dw[k * 27 + j], v);
    }
    __syncthreads();
  }
}
__global__ void k_f32_to_half(const float* s, __half* d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = __float2half(s[i]);
}
}  // namespace
#endif

bool wgrad_rgb_s2(const act_t* x, const act_t* dz, int N, int H, int W, int K, float* tmp, act_t* dw, cudaStream_t st) {
#if defined(FSO_FP32)
  (void)x; (void)dz; (void)N; (void)H; (void)W; (void)K; (void)tmp; (void)dw; (void)st;
  return false;  // 정합성 시험(fp32)은 cuDNN 그대로
#else
  const int P = (H + 1) / 2, Q = (W + 1) / 2;
  if (K != 32 || Q % kWgTile) return false;
  CUDA_CHECK(cudaMemsetAsync(tmp, 0, 32 * 27 * sizeof(float), st));
  k_wgrad_rgb_s2<<<70 * 8, 256, 0, st>>>(x, dz, N, H, W, P, Q, tmp);
  k_f32_to_half<<<cdiv(32 * 27, 256), 256, 0, st>>>(tmp, dw, 32 * 27);
  return true;
#endif
}
