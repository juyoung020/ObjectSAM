// BN+SiLU 커널. 메모리 대역폭이 한계라 다음을 지킨다:
//  - 스레드 하나가 한 화소의 8 채널(16 B)을 한 번에 읽고 쓴다(채널 수·슬라이스 경계는 모두 8 의 배수).
//  - 워프 안 스레드는 같은 화소의 연속 채널 → 다음 화소 순서라 접근이 이어진다.
//  - 통계는 블록별 부분합 → 마지막 블록이 마무리(커널 하나). 분산은 running mean 을 기준점으로 빼서 fp32 로 상쇄 없이.
#include "bn.cuh"

namespace {

constexpr int kT = 256;
constexpr int kV = 8;  // 벡터 폭(채널)

// ---- 8 채널 읽기·쓰기
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
__device__ __forceinline__ void ld8(const __nv_bfloat16* p, float o[8]) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
  for (int i = 0; i < 4; i++) { float2 f = __bfloat1622float2(h[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}
__device__ __forceinline__ void st8(__nv_bfloat16* p, const float o[8]) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
  for (int i = 0; i < 4; i++) h[i] = __floats2bfloat162_rn(o[2 * i], o[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}
__device__ __forceinline__ void ld8f(const float* p, float o[8]) { ld8(p, o); }

__device__ __forceinline__ float sigm(float y) { return 1.f / (1.f + __expf(-y)); }

// 화소 row(n,h,w) 의 업샘플 목적지 첫 화소 번호
__device__ __forceinline__ int up_base(int row, int H, int W) {
  int hw = H * W;
  int n = row / hw, rem = row - n * hw, h = rem / W, w = rem - h * W;
  return (n * 2 * H + 2 * h) * (2 * W) + 2 * w;
}

// dA 모으기(채널 c..c+7)
__device__ __forceinline__ void gather8(const DestList& dl, int row, int c, int H, int W, float dA[8]) {
  for (int i = 0; i < 8; i++) dA[i] = 0.f;
#pragma unroll
  for (int k = 0; k < kMaxDest; k++) {
    if (k >= dl.n) break;
    const Dest& d = dl.d[k];
    if (c < d.c0 || c >= d.c1) continue;
    int cc = d.choff + c - d.c0;
    float t[8];
    if (d.up == 1) {
      ld8(d.g + int64_t(row) * d.ctot + cc, t);
      for (int i = 0; i < 8; i++) dA[i] += t[i];
    } else {
      int b = up_base(row, H, W), W2 = 2 * W;
      int q[4] = {b, b + 1, b + W2, b + W2 + 1};
      for (int u = 0; u < 4; u++) {
        ld8(d.g + int64_t(q[u]) * d.ctot + cc, t);
        for (int i = 0; i < 8; i++) dA[i] += t[i];
      }
    }
  }
}

struct Geo {
  int V, rowlanes, nblk, rows_per_blk;
};
Geo geo(const BnShape& s) {
  Geo g;
  g.V = s.C / kV;
  g.rowlanes = kT / g.V;
  int64_t M = s.M();
  int target = 560;  // 70 SM × 8
  int64_t r = cdiv(M, target);
  if (r < 16) r = 16;
  g.rows_per_blk = int(r);
  g.nblk = cdiv(M, r);
  return g;
}

struct RedArgs {
  const act_t* z;
  BnShape s;
  Geo g;
  // MODE 0 (정방향 통계)
  const float* gamma;
  const float* beta;
  float* run_mean;
  float* run_var;
  float mom, eps;
  float* mean;
  float* istd;
  float* scale;
  float* shift;
  // MODE 1 (역방향)
  DestList dl;
  bool act;
  float* sums;
  float* gw_gamma;
  float* gw_beta;
  bool accum;
  // 공통
  float2* part;
  unsigned* counter;
};

// MODE 0: 기준점 K=running_mean 을 뺀 Σ(z−K), Σ(z−K)²  → 마지막 블록이 평균·분산·scale/shift·running 갱신
// MODE 1: Σg, Σg·x̂ (g = dA·silu'(y))                → 마지막 블록이 dβ, dγ, sums(=평균들)
template <int MODE>
__global__ void __launch_bounds__(kT) k_reduce(RedArgs P) {
  const BnShape s = P.s;
  const int V = P.g.V, rl = threadIdx.x / V, vl = threadIdx.x - rl * V, c = vl * kV;
  const int M = int(s.M());
  const int r0 = blockIdx.x * P.g.rows_per_blk;
  const int r1 = min(r0 + P.g.rows_per_blk, M);
  float a[8], b[8], K[8], sc[8], sh[8], mu[8], is[8];
  for (int i = 0; i < 8; i++) { a[i] = 0.f; b[i] = 0.f; }
  const bool active = rl < P.g.rowlanes;
  if (active) {
    if (MODE == 0) {
      ld8f(P.run_mean + c, K);
    } else {
      ld8f(P.scale + c, sc); ld8f(P.shift + c, sh); ld8f(P.mean + c, mu); ld8f(P.istd + c, is);
    }
    for (int r = r0 + rl; r < r1; r += P.g.rowlanes) {
      float zv[8];
      ld8(P.z + int64_t(r) * s.C + c, zv);
      if (MODE == 0) {
        for (int i = 0; i < 8; i++) { float d = zv[i] - K[i]; a[i] += d; b[i] += d * d; }
      } else {
        float dA[8];
        gather8(P.dl, r, c, s.H, s.W, dA);
        for (int i = 0; i < 8; i++) {
          float g = dA[i];
          if (P.act) { float y = sc[i] * zv[i] + sh[i], sg = sigm(y); g *= sg * (1.f + y * (1.f - sg)); }
          a[i] += g;
          b[i] += g * (zv[i] - mu[i]) * is[i];
        }
      }
    }
  }
  // 블록 안 합치기: [rowlanes][C]
  extern __shared__ float2 sm[];
  if (active)
    for (int i = 0; i < 8; i++) sm[rl * s.C + c + i] = make_float2(a[i], b[i]);
  __syncthreads();
  for (int ch = threadIdx.x; ch < s.C; ch += blockDim.x) {
    float2 acc = make_float2(0.f, 0.f);
    for (int k = 0; k < P.g.rowlanes; k++) { float2 v = sm[k * s.C + ch]; acc.x += v.x; acc.y += v.y; }
    P.part[int64_t(blockIdx.x) * s.C + ch] = acc;
  }
  // 마지막 블록이 마무리
  __shared__ bool last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(P.counter, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const double Md = double(M);
  for (int ch = threadIdx.x; ch < s.C; ch += blockDim.x) {
    float sx = 0.f, sy = 0.f, cx = 0.f, cy = 0.f;  // Kahan
    for (int k = 0; k < gridDim.x; k++) {
      float2 v = __ldcg(&P.part[int64_t(k) * s.C + ch]);
      float yx = v.x - cx, tx = sx + yx; cx = (tx - sx) - yx; sx = tx;
      float yy = v.y - cy, ty = sy + yy; cy = (ty - sy) - yy; sy = ty;
    }
    if (MODE == 0) {
      float Kc = P.run_mean[ch];
      float dm = float(double(sx) / Md);
      float var = float(double(sy) / Md - double(dm) * dm);
      if (var < 0.f) var = 0.f;
      float m = Kc + dm;
      float isd = rsqrtf(var + P.eps);
      P.mean[ch] = m;
      P.istd[ch] = isd;
      float scv = P.gamma[ch] * isd;
      P.scale[ch] = scv;
      P.shift[ch] = P.beta[ch] - m * scv;
      if (P.mom > 0.f) {
        P.run_mean[ch] = (1.f - P.mom) * Kc + P.mom * m;
        P.run_var[ch] = (1.f - P.mom) * P.run_var[ch] + P.mom * float(double(var) * Md / (Md - 1.0));
      }
    } else {
      P.sums[ch] = float(double(sx) / Md);
      P.sums[s.C + ch] = float(double(sy) / Md);
      if (P.accum) { P.gw_beta[ch] += sx; P.gw_gamma[ch] += sy; }
      else { P.gw_beta[ch] = sx; P.gw_gamma[ch] = sy; }
    }
  }
  if (threadIdx.x == 0) *P.counter = 0u;
}

__global__ void __launch_bounds__(kT) k_apply(const act_t* __restrict__ z, BnShape s, const float* __restrict__ scale,
                                              const float* __restrict__ shift, const act_t* __restrict__ res,
                                              DestList dl, bool act) {
  const int V = s.C / kV;
  const int n = int(s.M()) * V;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int row = i / V, c = (i - row * V) * kV;
    float v[8], sc[8], sh[8];
    ld8(z + int64_t(row) * s.C + c, v);
    ld8f(scale + c, sc);
    ld8f(shift + c, sh);
    for (int k = 0; k < 8; k++) { float y = sc[k] * v[k] + sh[k]; v[k] = act ? y * sigm(y) : y; }
    if (res) {
      float r[8];
      ld8(res + int64_t(row) * s.C + c, r);
      for (int k = 0; k < 8; k++) v[k] += r[k];
    }
#pragma unroll
    for (int k = 0; k < kMaxDest; k++) {
      if (k >= dl.n) break;
      const Dest& d = dl.d[k];
      if (c < d.c0 || c >= d.c1) continue;
      int cc = d.choff + c - d.c0;
      if (d.up == 1) {
        st8(d.p + int64_t(row) * d.ctot + cc, v);
      } else {
        int b = up_base(row, s.H, s.W), W2 = 2 * s.W;
        st8(d.p + int64_t(b) * d.ctot + cc, v);
        st8(d.p + int64_t(b + 1) * d.ctot + cc, v);
        st8(d.p + int64_t(b + W2) * d.ctot + cc, v);
        st8(d.p + int64_t(b + W2 + 1) * d.ctot + cc, v);
      }
    }
  }
}

__global__ void __launch_bounds__(kT) k_bwd_apply(const act_t* __restrict__ z, BnShape s, const float* __restrict__ mean,
                                                  const float* __restrict__ istd, const float* __restrict__ scale,
                                                  const float* __restrict__ shift, const float* __restrict__ sums,
                                                  DestList dl, bool act, act_t* __restrict__ dz,
                                                  act_t* __restrict__ res_grad, bool res_accum) {
  const int V = s.C / kV;
  const int n = int(s.M()) * V;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    int row = i / V, c = (i - row * V) * kV;
    int64_t off = int64_t(row) * s.C + c;
    float v[8], dA[8], sc[8], sh[8], mu[8], is[8], m1[8], m2[8];
    ld8(z + off, v);
    gather8(dl, row, c, s.H, s.W, dA);
    if (res_grad) {
      float r[8];
      if (res_accum) { ld8(res_grad + off, r); for (int k = 0; k < 8; k++) r[k] += dA[k]; }
      else for (int k = 0; k < 8; k++) r[k] = dA[k];
      st8(res_grad + off, r);
    }
    ld8f(scale + c, sc); ld8f(shift + c, sh); ld8f(mean + c, mu); ld8f(istd + c, is);
    ld8f(sums + c, m1); ld8f(sums + s.C + c, m2);
    float o[8];
    for (int k = 0; k < 8; k++) {
      float g = dA[k];
      if (act) { float y = sc[k] * v[k] + sh[k], sg = sigm(y); g *= sg * (1.f + y * (1.f - sg)); }
      float xh = (v[k] - mu[k]) * is[k];
      o[k] = sc[k] * (g - m1[k] - xh * m2[k]);
    }
    st8(dz + off, o);
  }
}

int grid_for(int64_t n) {
  int64_t b = cdiv(n, kT);
  return int(b < 70 * 32 ? b : 70 * 32);
}

__global__ void k_eval_scale(int C, const float* g, const float* b, const float* rm, const float* rv, float eps,
                             float* scale, float* shift) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float sc = g[c] * rsqrtf(rv[c] + eps);
  scale[c] = sc;
  shift[c] = b[c] - rm[c] * sc;
}

unsigned* counter() {  // 마지막 블록 판정용(커널이 끝에 0 으로 되돌린다; 한 스트림에서 차례로 쓴다)
  static unsigned* c = nullptr;
  if (!c) { CUDA_CHECK(cudaMalloc(&c, sizeof(unsigned))); CUDA_CHECK(cudaMemset(c, 0, sizeof(unsigned))); }
  return c;
}

}  // namespace

void bn_init() { counter(); }

int bn_partial_blocks(const BnShape& s) { return geo(s).nblk; }

void bn_eval_scale(int C, const float* gamma, const float* beta, const float* run_mean, const float* run_var, float eps,
                   float* scale, float* shift, cudaStream_t st) {
  k_eval_scale<<<cdiv(C, 128), 128, 0, st>>>(C, gamma, beta, run_mean, run_var, eps, scale, shift);
}

void bn_forward_stats(const act_t* z, const BnShape& s, const float* gamma, const float* beta, float* run_mean,
                      float* run_var, float momentum, float eps, float2* partial, float* mean, float* istd,
                      float* scale, float* shift, cudaStream_t st) {
  FSO_ASSERT(s.C % kV == 0 && s.C / kV <= kT, "BN channels %d", s.C);
  RedArgs P{};
  P.z = z; P.s = s; P.g = geo(s);
  P.gamma = gamma; P.beta = beta; P.run_mean = run_mean; P.run_var = run_var; P.mom = momentum; P.eps = eps;
  P.mean = mean; P.istd = istd; P.scale = scale; P.shift = shift;
  P.part = partial; P.counter = counter();
  size_t shm = size_t(P.g.rowlanes) * s.C * sizeof(float2);
  k_reduce<0><<<P.g.nblk, kT, shm, st>>>(P);
}

void bn_forward_apply(const act_t* z, const BnShape& s, const float* scale, const float* shift, const act_t* res,
                      const DestList& dl, bool act, cudaStream_t st) {
  k_apply<<<grid_for(s.M() * s.C / kV), kT, 0, st>>>(z, s, scale, shift, res, dl, act);
}

void bn_backward(const act_t* z, const BnShape& s, const float* gamma, const float* mean, const float* istd,
                 const float* scale, const float* shift, const DestList& dl, bool act, float2* partial, float* sums,
                 act_t* dz, float* gw_gamma, float* gw_beta, bool accum, act_t* res_grad, bool res_accum,
                 cudaStream_t st) {
  (void)gamma;
  RedArgs P{};
  P.z = z; P.s = s; P.g = geo(s);
  P.scale = const_cast<float*>(scale); P.shift = const_cast<float*>(shift);
  P.mean = const_cast<float*>(mean); P.istd = const_cast<float*>(istd);
  P.dl = dl; P.act = act; P.sums = sums; P.gw_gamma = gw_gamma; P.gw_beta = gw_beta; P.accum = accum;
  P.part = partial; P.counter = counter();
  size_t shm = size_t(P.g.rowlanes) * s.C * sizeof(float2);
  k_reduce<1><<<P.g.nblk, kT, shm, st>>>(P);
  k_bwd_apply<<<grid_for(s.M() * s.C / kV), kT, 0, st>>>(z, s, mean, istd, scale, shift, sums, dl, act, dz, res_grad,
                                                         res_accum);
}
