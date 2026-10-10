// v8 분할 손실(Ultralytics 8.4 v8SegmentationLoss, overlap_mask=False, 클래스 1)의 GPU 구현.
//
// 모든 버퍼는 고정 크기, 개수(정답 수·양성 수)는 장치 메모리에서 읽는다 → 호스트 동기 없음, CUDA graph 가능.
// 순서: 해독(DFL 기대값·점수) → 정답마다 top-10 후보 → 앵커별 충돌 해소(겹침 최대) → 정답별 최대값 →
//       목표 점수 합 → 점수·상자·DFL 손실과 기울기(전 앵커) → 마스크 손실(16×16 타일, 자른 상자 안만) → 계수 기울기.
#pragma once
#include "core.h"

struct LossCfg {
  int B = 2;
  int imgsz = 256;
  int lvl_h[3], lvl_w[3];   // 단계별 격자(입력/8, /16, /32)
  int stride[3] = {8, 16, 32};
  int Hp, Wp;               // proto 해상도 = 입력/4
  int G = 256;              // 영상당 정답 상한
  int reg_max = 16;         // DFL 칸 수
  float box_gain = 7.5f, cls_gain = 0.5f, dfl_gain = 1.5f;
  int topk = 10;
  int A() const { return lvl_h[0] * lvl_w[0] + lvl_h[1] * lvl_w[1] + lvl_h[2] * lvl_w[2]; }
};

// 정답(장치): 영상 b 의 정답 수 n_gt[b], 상자 [B][G][4] xyxy 화소, 마스크 비트 [B][G][Hp*Wp/32] (proto 해상도).
// ignore: [B][Hp*Wp/32] 무시 영역 비트(없으면 nullptr). 그 안에 중심이 있는 음성 앵커는 점수 손실을 받지 않는다
// (물체라고도 배경이라고도 가르치지 않음; docs/LABELS.md 3 절).
struct GtBatch {
  int* n_gt;
  float4* box;
  uint32_t* mask;
  uint32_t* ignore = nullptr;
};

// 머리 출력·기울기(장치, NHWC act_t). box[l]: [B,H,W,4·reg_max], cls[l]: [B,H,W,1], coef[l]: [B,H,W,32], proto: [B,Hp,Wp,32]
struct HeadIO {
  const act_t* box[3];
  const act_t* cls[3];
  const act_t* coef[3];
  const act_t* proto;
  act_t* dbox[3];
  act_t* dcls[3];
  act_t* dcoef[3];
  act_t* dproto;
};

class SegLoss {
 public:
  void init(const LossCfg& c);
  // 결과: items() = [box, seg, cls, dfl] (이득 곱, 배치 곱 전 — Ultralytics loss_items 와 같은 값), 합계 = Σ×B.
  // 기울기는 (Σ items)×B 에 대한 것.
  // scale: 장치 float 하나(동적 loss scale). 기울기에만 곱한다(items 는 그대로).
  void run(const HeadIO& io, const GtBatch& gt, const float* scale, cudaStream_t st);
  const float* items() const { return items_; }   // 장치 [4] + [4]=양성 수 [5]=목표 점수 합
  // 디버그: 앵커별 할당 정답(−1 없음)과 목표 점수
  const int* assigned_gt() const { return agt_; }
  const float* target_score() const { return tsc_; }
  LossCfg cfg;

 private:
  float4* pbox_ = nullptr;   // [B*A] 예측 상자(격자 단위 xyxy)
  float* score_ = nullptr;   // [B*A] sigmoid 점수
  int* cnt_ = nullptr;       // [B*A] top-k 에 뽑힌 횟수(해소 때 0 으로 되돌림)
  int* who_ = nullptr;       // [B*A] 뽑은 정답 하나
  int* agt_ = nullptr;       // [B*A] 최종 정답
  float* amet_ = nullptr;    // [B*A] 정렬 지표
  float* aovl_ = nullptr;    // [B*A] 겹침
  float* tsc_ = nullptr;     // [B*A] 목표 점수
  float* galign_ = nullptr;  // [B*G] 양성 정렬 지표 최대
  float* govl_ = nullptr;    // [B*G] 양성 겹침 최대
  int* pos_ = nullptr;       // [B][A] 영상별 양성 앵커 목록
  int* npos_ = nullptr;      // [B]
  float* items_ = nullptr;   // [8]
  int A_ = 0;
};
