#include "optim.cuh"

#include <algorithm>
#include <cmath>

namespace {
constexpr int kT = 256;
constexpr int kNormBlocks = 1024;

struct Arenas {
  float* g[3];
  int64_t n[3];
};

__global__ void k_sqsum(Arenas A, const OptState* st, float* part) {
  float inv = 1.f / st->scale;
  float acc = 0.f;
  for (int k = 0; k < 3; k++) {
    const float* g = A.g[k];
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < A.n[k]; i += int64_t(gridDim.x) * blockDim.x) {
      float v = g[i] * inv;
      acc += v * v;  // inf·nan 은 그대로 퍼진다
    }
  }
  __shared__ float red[kT / 32];
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int w = 0; w < kT / 32; w++) s += red[w];
    part[blockIdx.x] = s;
  }
}

struct Sched {
  float lr0, lrf, momentum, wmom, wbias;
  int cos_lr, epochs;
  int64_t nb, nw;
  float clip, scale_init;
  int growth;
  int scaling;
};

__device__ float lf(const Sched& s, int64_t epoch) {
  float x = float(epoch);
  if (s.cos_lr) return ((1.f - cosf(x * 3.14159265358979f / s.epochs)) / 2.f) * (s.lrf - 1.f) + 1.f;
  return fmaxf(1.f - x / s.epochs, 0.f) * (1.f - s.lrf) + s.lrf;
}

// 노름 합산 → inf 검사 → 자르기 계수 → lr·momentum 계산(한 스레드)
__global__ void k_prep(const float* part, int nb, OptState* st, Sched s) {
  double sum = 0;
  for (int i = 0; i < nb; i++) sum += part[i];
  float norm = float(sqrt(sum));
  st->grad_norm = norm;
  st->found_inf = s.scaling ? !isfinite(norm) : 0;
  int64_t ni = st->ni;
  int64_t epoch = ni / s.nb;
  float base = s.lr0 * lf(s, epoch);
  if (ni <= s.nw && s.nw > 0) {
    float t = float(ni) / float(s.nw);
    st->lr[0] = t * base;                          // weight
    st->lr[1] = t * base;                          // bn
    st->lr[2] = s.wbias + t * (base - s.wbias);    // bias
    st->mom = s.wmom + t * (s.momentum - s.wmom);
  } else {
    st->lr[0] = st->lr[1] = st->lr[2] = base;
    st->mom = s.momentum;
  }
}

struct SgdArgs {
  float* p[3];
  float* g[3];
  float* m[3];
  int64_t n[3];
  float wd[3];
  act_t* w_act;  // 무리 0 의 act_t 사본
  float clip;
};

__global__ void k_sgd(SgdArgs A, const OptState* st) {
  if (st->found_inf) return;
  float inv = 1.f / st->scale;
  float norm = st->grad_norm;
  float cc = fminf(A.clip / (norm + 1e-6f), 1.f) * inv;
  float mu = st->mom;
  for (int k = 0; k < 3; k++) {
    float lr = st->lr[k], wd = A.wd[k];
    float *p = A.p[k], *g = A.g[k], *m = A.m[k];
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < A.n[k]; i += int64_t(gridDim.x) * blockDim.x) {
      float pv = p[i];
      float gv = g[i] * cc + wd * pv;
      float b = mu * m[i] + gv;  // torch: 첫 스텝 buf = g (0 으로 시작하면 같다)
      m[i] = b;
      gv = gv + mu * b;          // nesterov
      pv -= lr * gv;
      p[i] = pv;
      if (k == 0) A.w_act[i] = from_f<act_t>(pv);
    }
  }
}

struct EmaArgs {
  float* e[P_NKIND];
  const float* p[P_NKIND];
  int64_t n[P_NKIND];
  float decay, tau;
};
__global__ void k_ema(EmaArgs A, const OptState* st) {
  float upd = float(st->updates + 1);
  float d = A.decay * (1.f - expf(-upd / A.tau));
  for (int k = 0; k < P_NKIND; k++) {
    float* e = A.e[k];
    const float* p = A.p[k];
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < A.n[k]; i += int64_t(gridDim.x) * blockDim.x)
      e[i] = e[i] + (p[i] - e[i]) * (1.f - d);  // lerp(e, p, 1-d) = d·e + (1-d)·p
  }
}

__global__ void k_post(OptState* st, Sched s) {
  st->updates += 1;
  if (!s.scaling) return;
  if (st->found_inf) {
    st->scale *= 0.5f;
    st->good = 0;
    st->skipped += 1;
  } else if (++st->good >= s.growth) {
    st->scale *= 2.f;
    st->good = 0;
  }
}

__global__ void k_tick(OptState* st) { st->ni += 1; }

Sched sched_of(const OptCfg& c, int64_t nw) {
  Sched s;
  s.lr0 = c.lr0; s.lrf = c.lrf; s.momentum = c.momentum; s.wmom = c.warmup_momentum; s.wbias = c.warmup_bias_lr;
  s.cos_lr = c.cos_lr; s.epochs = c.epochs; s.nb = c.nb; s.nw = nw; s.clip = c.clip; s.scale_init = c.scale_init;
  s.growth = c.growth_interval; s.scaling = c.loss_scaling;
  return s;
}
}  // namespace

void Optimizer::init(const OptCfg& c, Params* p, cudaStream_t st) {
  cfg = c;
  p_ = p;
  accum_ = std::max(int(std::lround(double(c.nbs) / c.batch)), 1);
  wd_ = c.weight_decay * c.batch * accum_ / c.nbs;
  // _get_warmup_iterations: round(min(warmup_epochs, epochs-1) × nb)
  float we = std::min(c.warmup_epochs, float(std::max(c.epochs - 1, 0)));
  nw_ = we > 0 ? int64_t(std::llround(we * c.nb)) : 0;
  for (int k = 0; k < 3; k++) {
    mbuf_[k] = dalloc<float>(p->size[k]);
    CUDA_CHECK(cudaMemsetAsync(mbuf_[k], 0, p->size[k] * 4, st));
  }
  for (int k = 0; k < P_NKIND; k++) {
    ema_[k] = dalloc<float>(p->size[k]);
    CUDA_CHECK(cudaMemcpyAsync(ema_[k], p->val[k], p->size[k] * 4, cudaMemcpyDeviceToDevice, st));
  }
  part_ = dalloc<float>(kNormBlocks);
  state_ = dalloc<OptState>(1);
  OptState s0{};
  s0.scale = c.loss_scaling ? c.scale_init : 1.f;
  CUDA_CHECK(cudaMemcpyAsync(state_, &s0, sizeof(s0), cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
}

void Optimizer::step(cudaStream_t st) {
  Sched s = sched_of(cfg, nw_);
  Arenas A;
  for (int k = 0; k < 3; k++) { A.g[k] = p_->grad[k]; A.n[k] = p_->size[k]; }
  k_sqsum<<<kNormBlocks, kT, 0, st>>>(A, state_, part_);
  k_prep<<<1, 1, 0, st>>>(part_, kNormBlocks, state_, s);
  SgdArgs S;
  for (int k = 0; k < 3; k++) { S.p[k] = p_->val[k]; S.g[k] = p_->grad[k]; S.m[k] = mbuf_[k]; S.n[k] = p_->size[k]; }
  S.wd[0] = wd_; S.wd[1] = 0.f; S.wd[2] = 0.f;
  S.w_act = p_->w_act;
  S.clip = cfg.clip;
  k_sgd<<<1024, kT, 0, st>>>(S, state_);
  EmaArgs E;
  for (int k = 0; k < P_NKIND; k++) { E.e[k] = ema_[k]; E.p[k] = p_->val[k]; E.n[k] = p_->size[k]; }
  E.decay = cfg.ema_decay; E.tau = cfg.ema_tau;
  k_ema<<<1024, kT, 0, st>>>(E, state_);
  k_post<<<1, 1, 0, st>>>(state_, s);
}

void Optimizer::tick(cudaStream_t st) { k_tick<<<1, 1, 0, st>>>(state_); }
