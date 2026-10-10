// 옵티마이저(Ultralytics 8.4 trainer 와 같은 규칙), 전부 장치에서 — 호스트 동기·인자 갱신 없이 CUDA graph 로 돈다.
//  - SGD nesterov, 파라미터 무리 셋: conv 가중치(감쇠 있음) / BN γ(감쇠 없음) / BN β·conv bias(감쇠 없음, 워밍업 lr 따로)
//  - 워밍업: ni ≤ nw 동안 lr 을 (bias: warmup_bias_lr, 그 밖: 0) → lr0·lf(epoch), momentum 을 warmup_momentum → momentum
//  - lf(epoch): 선형 (1 − e/E)(1 − lrf) + lrf, 또는 cos
//  - 기울기: loss scale 로 나눈 뒤 전체 L2 노름 10 으로 자름. inf/nan 이면 그 스텝을 건너뜀(GradScaler)
//  - loss scale: 시작 65536, 연속 2000 스텝 정상이면 ×2, inf 면 ×0.5
//  - EMA: d = 0.9999·(1 − exp(−updates/2000)), 파라미터와 BN running 통계 모두. 건너뛴 스텝에도 갱신(Ultralytics 와 같음)
#pragma once
#include "model.h"

struct OptCfg {
  float lr0 = 0.01f, lrf = 0.01f, momentum = 0.937f, weight_decay = 5e-4f;
  float warmup_epochs = 3.f, warmup_momentum = 0.8f, warmup_bias_lr = 0.1f;
  bool cos_lr = false;
  int epochs = 100;
  int64_t nb = 1;         // epoch 당 반복(마이크로 배치) 수
  int batch = 16, nbs = 64;
  float clip = 10.f;
  float ema_decay = 0.9999f, ema_tau = 2000.f;
  float scale_init = 65536.f;
  int growth_interval = 2000;
  bool loss_scaling = true;  // fp32 판에서는 끈다
};

// 장치 상태(학습 재개 때 저장·복원)
struct OptState {
  int64_t ni;           // 지금 반복 번호(마이크로 배치)
  int64_t updates;      // EMA 갱신 수
  int64_t skipped;      // inf 로 건너뛴 스텝 수
  float scale;          // loss scale
  int good;             // 연속 정상 스텝
  int found_inf;        // 마지막 스텝
  float grad_norm;      // 마지막 스텝(스케일 해제 뒤, 자르기 전)
  float lr[3];          // 마지막 스텝의 무리별 lr
  float mom;
};

class Optimizer {
 public:
  void init(const OptCfg& c, Params* p, cudaStream_t st);
  // 한 번의 옵티마이저 스텝(기울기 → 파라미터, act_t 사본, EMA, loss scale).
  void step(cudaStream_t st);
  // 마이크로 배치 하나가 끝날 때(ni += 1). 그래프 안에 넣는다.
  void tick(cudaStream_t st);
  const float* scale() const { return &state_->scale; }
  OptState* state() const { return state_; }
  float* ema(int k) const { return ema_[k]; }
  float* mom_buf(int k) const { return mbuf_[k]; }
  int accumulate() const { return accum_; }
  float wd_eff() const { return wd_; }
  OptCfg cfg;

 private:
  Params* p_ = nullptr;
  OptState* state_ = nullptr;
  float* mbuf_[3] = {};
  float* ema_[P_NKIND] = {};
  float* part_ = nullptr;
  int accum_ = 1;
  float wd_ = 0.f;
  int64_t nw_ = 0;
};
