#include "misc.cuh"

namespace {
constexpr int kT = 256;
int grid_for(int64_t n) {
  int64_t b = cdiv(n, kT);
  return int(b < 65535 * 8 ? b : 65535 * 8);
}

// 한 단계 maxpool: src 슬라이스 → dst 슬라이스(같은 cat 버퍼, 채널 ctot).
__global__ void k_pool5(act_t* cat, int N, int H, int W, int c, int ctot, int si, int di) {
  int64_t n = int64_t(N) * H * W * c;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int ch = int(i % c);
    int64_t pix = i / c;
    int w = int(pix % W);
    int64_t t = pix / W;
    int h = int(t % H);
    int64_t b = t / H;
    float m = -INFINITY;
    for (int dy = -2; dy <= 2; dy++) {
      int y = h + dy;
      if (y < 0 || y >= H) continue;
      for (int dx = -2; dx <= 2; dx++) {
        int x = w + dx;
        if (x < 0 || x >= W) continue;
        float v = to_f(cat[((b * H + y) * W + x) * ctot + si * c + ch]);
        m = v > m ? v : m;
      }
    }
    cat[pix * ctot + di * c + ch] = from_f<act_t>(m);
  }
}

// maxpool 역: 출력 위치마다 창의 첫 최댓값(행 우선, PyTorch 와 같은 규칙)에 기울기를 더한다.
// g_in: 이 단계 출력 기울기 = dcat[di] (+ g_extra), out: fp32 누적(입력 슬라이스 위치).
__global__ void k_unpool5(const act_t* cat, const act_t* dcat, const float* g_extra, int N, int H, int W, int c,
                          int ctot, int si, int di, float* out) {
  int64_t n = int64_t(N) * H * W * c;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int ch = int(i % c);
    int64_t pix = i / c;
    float g = to_f(dcat[pix * ctot + di * c + ch]) + (g_extra ? g_extra[i] : 0.f);
    if (g == 0.f) continue;
    int w = int(pix % W);
    int64_t t = pix / W;
    int h = int(t % H);
    int64_t b = t / H;
    float m = -INFINITY;
    int64_t arg = -1;
    for (int dy = -2; dy <= 2; dy++) {
      int y = h + dy;
      if (y < 0 || y >= H) continue;
      for (int dx = -2; dx <= 2; dx++) {
        int x = w + dx;
        if (x < 0 || x >= W) continue;
        int64_t q = (b * H + y) * W + x;
        float v = to_f(cat[q * ctot + si * c + ch]);
        if (v > m) { m = v; arg = q; }
      }
    }
    atomicAdd(&out[arg * c + ch], g);
  }
}

__global__ void k_add_into_slice(const float* src, act_t* dcat, int64_t npix, int c, int ctot, int si) {
  int64_t n = npix * c;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t pix = i / c;
    int ch = int(i - pix * c);
    act_t* p = &dcat[pix * ctot + si * c + ch];
    *p = from_f<act_t>(to_f(*p) + src[i]);
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
__global__ void k_chan_final(const float* part, int nb, int C, float* out, bool accum) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  double s = 0;
  for (int b = 0; b < nb; b++) s += part[int64_t(b) * C + c];
  out[c] = accum ? out[c] + float(s) : float(s);
}
int chan_rows(int64_t M) { int64_t r = M / 2048; return int(r < 64 ? 64 : r); }

__global__ void k_add_bias(act_t* y, int64_t M, int C, const float* bias) {
  int64_t n = M * C;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    y[i] = from_f<act_t>(to_f(y[i]) + bias[i % C]);
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
  for (int k = 0; k < 3; k++) k_pool5<<<grid_for(n), kT, 0, st>>>(cat, N, H, W, c, 4 * c, k, k + 1);
}

void sppf_pool_bwd(const act_t* cat, act_t* dcat, int N, int H, int W, int c, float* ta, float* tb,
                   cudaStream_t st) {
  int64_t npix = int64_t(N) * H * W, n = npix * c;
  // t = unpool(슬라이스2 → 3 의 기울기)  ⇒ 슬라이스2 위치 누적
  CUDA_CHECK(cudaMemsetAsync(ta, 0, n * sizeof(float), st));
  k_unpool5<<<grid_for(n), kT, 0, st>>>(cat, dcat, nullptr, N, H, W, c, 4 * c, 2, 3, ta);
  CUDA_CHECK(cudaMemsetAsync(tb, 0, n * sizeof(float), st));
  k_unpool5<<<grid_for(n), kT, 0, st>>>(cat, dcat, ta, N, H, W, c, 4 * c, 1, 2, tb);
  CUDA_CHECK(cudaMemsetAsync(ta, 0, n * sizeof(float), st));
  k_unpool5<<<grid_for(n), kT, 0, st>>>(cat, dcat, tb, N, H, W, c, 4 * c, 0, 1, ta);
  k_add_into_slice<<<grid_for(n), kT, 0, st>>>(ta, dcat, npix, c, 4 * c, 0);
}

int chan_sum_blocks(int64_t M, int C) { (void)C; return cdiv(M, chan_rows(M)); }
void chan_sum(const act_t* x, int64_t M, int C, float* partial, float* out, bool accum, cudaStream_t st) {
  int rows = chan_rows(M), nb = cdiv(M, rows);
  k_chan_partial<<<nb, C < 256 ? (C + 31) / 32 * 32 : 256, 0, st>>>(x, M, C, rows, partial);
  k_chan_final<<<cdiv(C, 128), 128, 0, st>>>(partial, nb, C, out, accum);
}
void add_bias(act_t* y, int64_t M, int C, const float* bias, cudaStream_t st) {
  k_add_bias<<<grid_for(M * C), kT, 0, st>>>(y, M, C, bias);
}
void f32_to_act(const float* s, act_t* d, int64_t n, cudaStream_t st) { k_f32_to_act<<<grid_for(n), kT, 0, st>>>(s, d, n); }
void act_accum_f32(const act_t* s, float* d, int64_t n, bool accum, cudaStream_t st) {
  k_act_accum<<<grid_for(n), kT, 0, st>>>(s, d, n, accum);
}
void fill_f32(float* p, float v, int64_t n, cudaStream_t st) { k_fill<<<grid_for(n), kT, 0, st>>>(p, v, n); }
