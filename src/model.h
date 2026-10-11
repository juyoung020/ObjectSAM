// FastSAM-s(YOLOv8s-seg, 클래스 1) 신경망: 파라미터, 활성 버퍼, 정방향·역방향.
//
// 활성값은 NHWC packed act_t. conv 는 cuDNN, 나머지는 우리 커널. concat 은 따로 없다 — 생산 층의 BN+SiLU 가
// concat 버퍼의 자기 채널 자리에 바로 쓴다(업샘플 목적지는 2×2 로 펼쳐 쓴다).
// 기울기: 활성 버퍼마다 같은 모양의 기울기 버퍼. 역방향 순서를 시작 때 한 번 흉내 내어 각 쓰기가 덮어쓰기인지
// 더하기인지 정해 둔다(매 스텝 0 으로 지우는 일이 없다).
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "bn.cuh"
#include "conv.h"
#include "fsow.h"

// 파라미터 종류별 연속 저장(옵티마이저가 종류마다 한 번에 돈다).
enum PKind { P_W = 0, P_GAMMA = 1, P_BIAS = 2, P_RUN = 3, P_NKIND };

struct ParamRef {
  PKind kind;
  int64_t off, n;
  std::vector<int64_t> torch_shape;  // PyTorch 모양(저장·불러오기용)
  int layout;                        // 0 그대로, 1 conv OIHW→KRSC, 2 convT (Cin,Cout,kh,kw)→(Cin,kh,kw,Cout)
  int cpad = 0;                      // layout 1: 저장 채널 수(입력 3 채널을 8 로 — 텐서코어용, 나머지는 0)
};

struct Params {
  std::map<std::string, ParamRef> refs;
  std::vector<std::string> order;
  int64_t size[P_NKIND] = {0, 0, 0, 0};
  float* val[P_NKIND] = {};   // fp32 값(running 통계 포함)
  float* grad[P_NKIND] = {};  // fp32 기울기(P_RUN 없음)
  act_t* w_act = nullptr;     // P_W 의 act_t 사본(cuDNN 용)
  act_t* gw_act = nullptr;    // cuDNN wgrad 출력(act_t), 스텝마다 grad[P_W] 로 모은다
  ParamRef& add(const std::string& name, PKind k, std::vector<int64_t> shape, int layout = 0, int cpad = 0);
  void allocate();
  float* ptr(const std::string& name) { auto& r = refs.at(name); return val[r.kind] + r.off; }
  float* gptr(const std::string& name) { auto& r = refs.at(name); return grad[r.kind] + r.off; }
  act_t* wact(const std::string& name) { auto& r = refs.at(name); return w_act + r.off; }
  act_t* gwact(const std::string& name) { auto& r = refs.at(name); return gw_act + r.off; }
  void load(const TensorMap& m, const std::string& prefix = "");   // PyTorch 이름·모양
  TensorMap save(const std::string& prefix = "", bool grads = false) const;
};

constexpr int kInC = 3;  // 입력 영상 채널(RGB 3 + 0 다섯)

struct Act {
  std::string name;
  int C = 0, H = 0, W = 0;
  act_t* p = nullptr;
  act_t* g = nullptr;
  bool need_grad = true;
};

struct DestSpec {
  Act* a;
  int choff, c0, c1, up;
};

struct Node {
  enum Kind { CONVBN, HEADCONV, CONVT, POOL } kind;
  std::string name;  // PyTorch 접두사 (예: model.2.m.0.cv1)
  Act* in = nullptr;
  ConvOp conv;
  std::vector<std::string> wnames;   // conv 가중치 이름들(합친 conv 는 여럿 — 연속 저장)
  std::vector<std::string> bnnames;  // BN 접두사들
  int K = 0;
  // CONVBN
  std::vector<DestSpec> dests;
  Act* res = nullptr;
  act_t* z = nullptr;
  float *mean = nullptr, *istd = nullptr, *scale = nullptr, *shift = nullptr;
  // HEADCONV / CONVT
  Act* out = nullptr;
  std::string bias;
  // 역방향 덮어쓰기/더하기
  bool dgrad_accum = false, res_accum = false, w_accum = false;
  // POOL
  Act* cat = nullptr;
  int pc = 0;
};

// 가중치 묶음에서 reg_max 알아내기(상자 가지 마지막 conv 출력 = 4·reg_max)
inline int reg_max_of(const TensorMap& w, const std::string& prefix = "") {
  auto it = w.find(prefix + "model.22.cv2.0.2.weight");
  FSO_ASSERT(it != w.end(), "no box head in weights");
  return int(it->second.shape[0] / 4);
}

struct ModelCfg {
  int batch = 2, imgsz = 256;
  int reg_max = 16;   // DFL 칸 수(FastSAM 논문 1024 설정은 26). 상자 가지 폭 = max(16, 128/4, 4·reg_max)
  float bn_momentum = 0.03f, bn_eps = 1e-3f;
};

class Model {
 public:
  void build(const ModelCfg& cfg, cudnnHandle_t h, cudaStream_t st);
  void forward(bool train);            // in: input()->p 에 영상(NHWC, 0..1)
  // 출력 기울기(head_*->g, proto->g) → 파라미터 기울기. accum: 마이크로 배치 누적(첫 배치는 false)
  void backward(bool accum = false);
  void sync_weights();                 // fp32 → act_t
  Act* input() { return acts_.at("in").get(); }
  Act* head(int kind, int lvl) { return head_[kind][lvl]; }   // kind 0 box(64) 1 cls(1) 2 coef(32)
  Act* proto() { return proto_; }
  Act* act(const std::string& n) { return acts_.at(n).get(); }
  bool has_act(const std::string& n) const { return acts_.count(n) > 0; }
  Params params;
  ModelCfg cfg;
  size_t ws_bytes = 0;
  void print_plan() const;
  void print_convs() const;  // 층별 conv 시간(재기 결과)

 private:
  Act* new_act(const std::string& n, int C, int H, int W);
  void conv_bn(const std::string& name, Act* in, int K, int k, int s, std::vector<DestSpec> dests, Act* res = nullptr,
               std::vector<std::string> fused = {}, std::vector<int> fused_k = {});
  Act* c2f(const std::string& name, Act* in, int c2, int n, bool shortcut, std::vector<DestSpec> out_dests);
  void head_conv(const std::string& name, Act* in, int K, Act* out);
  void plan_backward();
  cudnnHandle_t h_ = nullptr;
  cudaStream_t st_ = nullptr;
  std::map<std::string, std::unique_ptr<Act>> acts_;
  std::vector<std::unique_ptr<Node>> nodes_;
  Act* head_[3][3] = {};
  Act* proto_ = nullptr;
  void* ws_ = nullptr;
  float* sums_ = nullptr;
  float *pool_a_ = nullptr, *pool_b_ = nullptr;
  // 역방향: wgrad 는 곁 스트림에서 다음 층의 BN 역방향·dgrad 와 겹쳐 돈다(계산 한계 conv 와 대역폭 한계 BN 이
  // 함께 돈다). dz 는 두 벌을 번갈아 쓰고, 곁 스트림은 작업 공간도 따로.
  act_t* dz_[2] = {};
  void* ws_side_ = nullptr;
  cudaStream_t side_ = nullptr;
  cudaEvent_t ev_dz_[2] = {}, ev_wg_[2] = {}, ev_fork_ = nullptr, ev_join_ = nullptr;
  void wgrad_side(Node& nd, const act_t* dy, int buf);
  float* chpart_ = nullptr;
  float* rgb_tmp_ = nullptr;  // 첫 층 wgrad 누적(fp32)
};
