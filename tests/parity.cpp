// 정합성: pytools/ref_step.py 가 남긴 Ultralytics fp32 값과 우리 학습기의 정방향·역방향을 층마다 비교한다.
//   build/parity_fp32 weights/FastSAM-s.fsow data/parity/ref.fsow
#include <cmath>
#include <algorithm>
#include <cstring>

#include "../src/loss.cuh"
#include "../src/model.h"

namespace {
struct Cmp {
  double max_abs = 0, ref_max = 0, l2d = 0, l2r = 0;
  void add(double r, double v) {
    double d = std::fabs(r - v);
    max_abs = std::max(max_abs, d);
    ref_max = std::max(ref_max, std::fabs(r));
    l2d += d * d;
    l2r += r * r;
  }
  double rel() const { return std::sqrt(l2d / std::max(l2r, 1e-30)); }
};
int g_fail = 0;
double g_tol = 1e-3;
void report(const std::string& what, const Cmp& c) {
  bool bad = !(c.rel() <= g_tol);
  if (bad) g_fail++;
  printf("%-48s relL2 %.2e  maxabs %.2e  (ref max %.2e)%s\n", what.c_str(), c.rel(), c.max_abs, c.ref_max,
         bad ? "  <-- FAIL" : "");
}
std::vector<float> d2h(const act_t* p, int64_t n) {
  std::vector<act_t> t(n);
  CUDA_CHECK(cudaMemcpy(t.data(), p, n * sizeof(act_t), cudaMemcpyDeviceToHost));
  std::vector<float> o(n);
  for (int64_t i = 0; i < n; i++) o[i] = to_f(t[i]);
  return o;
}
void h2d(act_t* p, const std::vector<float>& v) {
  std::vector<act_t> t(v.size());
  for (size_t i = 0; i < v.size(); i++) t[i] = from_f<act_t>(v[i]);
  CUDA_CHECK(cudaMemcpy(p, t.data(), t.size() * sizeof(act_t), cudaMemcpyHostToDevice));
}
// 참조 NCHW (B,C,H,W) 와 우리 NHWC 버퍼의 채널 [choff, choff+C) 비교.
Cmp cmp_nchw(const HostTensor& ref, const Act* a, int choff, int up = 1) {
  int64_t B = ref.shape[0], C = ref.shape[1], H = ref.shape[2], W = ref.shape[3];
  auto v = d2h(a->p, B * a->H * a->W * a->C);
  Cmp c;
  const float* r = ref.f32();
  for (int64_t b = 0; b < B; b++)
    for (int64_t ch = 0; ch < C; ch++)
      for (int64_t y = 0; y < H; y++)
        for (int64_t x = 0; x < W; x++) {
          int64_t yy = y * up, xx = x * up;
          c.add(r[((b * C + ch) * H + y) * W + x], v[((b * a->H + yy) * a->W + xx) * a->C + choff + ch]);
        }
  return c;
}
}  // namespace

int main(int argc, char** argv) {
  FSO_ASSERT(argc >= 3, "usage: parity <weights.fsow> <ref.fsow> [tol]");
  if (argc > 3) g_tol = atof(argv[3]);
  float lscale = argc > 4 ? atof(argv[4]) : 1.f;   // fp16: 기울기 밑넘침 방지(Ultralytics GradScaler 와 같은 뜻)
  auto W = fsow_load(argv[1]);
  auto R = fsow_load(argv[2]);
  auto& img = R.at("in.img");
  int B = img.shape[0], S = img.shape[2];
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreate(&st));
  cudnnHandle_t h;
  CUDNN_CHECK(cudnnCreate(&h));
  CUDNN_CHECK(cudnnSetStream(h, st));
  Model m;
  m.build(ModelCfg{B, S, reg_max_of(W)}, h, st);
  m.print_plan();
  m.params.load(W);
  m.sync_weights();
  {  // 영상 NCHW → NHWC
    std::vector<float> v(int64_t(B) * S * S * kInC, 0.f);
    const float* p = img.f32();
    for (int b = 0; b < B; b++)
      for (int c = 0; c < 3; c++)
        for (int i = 0; i < S * S; i++) v[(int64_t(b) * S * S + i) * kInC + c] = p[(int64_t(b) * 3 + c) * S * S + i];
    h2d(m.input()->p, v);
  }
  m.forward(true);
  CUDA_CHECK(cudaStreamSynchronize(st));

  printf("== forward (layer outputs)\n");
  struct L { int i; const char* act; int choff; int up; };
  L layers[] = {{0, "a0", 0, 1},     {1, "a1", 0, 1},      {2, "a2", 0, 1},     {3, "a3", 0, 1},
                {4, "a4", 0, 1},     {5, "a5", 0, 1},      {6, "a6", 0, 1},     {7, "a7", 0, 1},
                {8, "a8", 0, 1},     {9, "cat20", 256, 1}, {11, "cat11", 0, 1}, {12, "cat17", 128, 1},
                {14, "cat14", 0, 1}, {15, "p3", 0, 1},     {16, "cat17", 0, 1}, {17, "cat17", 0, 1},
                {18, "p4", 0, 1},    {19, "cat20", 0, 1},  {20, "cat20", 0, 1}, {21, "p5", 0, 1}};
  for (auto& l : layers) report("layer " + std::to_string(l.i) + " (" + l.act + ")",
                                cmp_nchw(R.at("act." + std::to_string(l.i)), m.act(l.act), l.choff, l.up));
  // 머리 출력: 참조 (B, ch, A), A = 단계 0,1,2 를 이은 것
  const char* hk[3] = {"boxes", "scores", "mask_coefficient"};
  for (int k = 0; k < 3; k++) {
    auto& ref = R.at(std::string("out.") + hk[k]);
    int64_t C = ref.shape[1], A = ref.shape[2];
    Cmp c;
    int64_t a0 = 0;
    for (int l = 0; l < 3; l++) {
      Act* a = m.head(k, l);
      auto v = d2h(a->p, int64_t(B) * a->H * a->W * a->C);
      int64_t hw = a->H * a->W;
      for (int b = 0; b < B; b++)
        for (int64_t ch = 0; ch < C; ch++)
          for (int64_t i = 0; i < hw; i++) c.add(ref.f32()[(b * C + ch) * A + a0 + i], v[(b * hw + i) * C + ch]);
      a0 += hw;
    }
    report(std::string("out.") + hk[k], c);
  }
  report("out.proto", cmp_nchw(R.at("out.proto"), m.proto(), 0));
  {  // BN running 통계(정방향 한 번 뒤)
    auto P = m.params.save();
    Cmp c;
    for (auto& [k, t] : P)
      if (k.find("running") != std::string::npos) {
        auto& r = R.at("post." + k);
        for (int64_t i = 0; i < t.numel(); i++) c.add(r.f32()[i], t.f32()[i]);
      }
    report("bn running stats after step", c);
  }

  printf("== loss\n");
  LossCfg lc;
  lc.B = B; lc.imgsz = S;
  for (int l = 0; l < 3; l++) { lc.lvl_h[l] = m.head(0, l)->H; lc.lvl_w[l] = m.head(0, l)->W; }
  lc.Hp = m.proto()->H; lc.Wp = m.proto()->W; lc.G = 32; lc.reg_max = m.cfg.reg_max;
  SegLoss sl;
  sl.init(lc);
  GtBatch gt;
  {
    auto& bi = R.at("in.batch_idx");
    auto& bb = R.at("in.bboxes");
    auto& mk = R.at("in.masks");
    int n = bi.shape[0];
    std::vector<int> ng(B, 0);
    std::vector<float4> box(B * lc.G, make_float4(0, 0, 0, 0));
    int words = lc.Hp * lc.Wp / 32;
    std::vector<uint32_t> bits(int64_t(B) * lc.G * words, 0);
    FSO_ASSERT(mk.shape[1] == lc.Hp && mk.shape[2] == lc.Wp, "mask res");
    for (int i = 0; i < n; i++) {
      int b = bi.i32()[i], j = ng[b]++;
      const float* x = bb.f32() + 4 * i;
      box[b * lc.G + j] = make_float4((x[0] - x[2] / 2) * S, (x[1] - x[3] / 2) * S, (x[0] + x[2] / 2) * S, (x[1] + x[3] / 2) * S);
      for (int p = 0; p < lc.Hp * lc.Wp; p++)
        if (mk.u8()[int64_t(i) * lc.Hp * lc.Wp + p]) bits[(int64_t(b) * lc.G + j) * words + p / 32] |= 1u << (p % 32);
    }
    gt.n_gt = dalloc<int>(B); gt.box = dalloc<float4>(B * lc.G); gt.mask = dalloc<uint32_t>(bits.size());
    CUDA_CHECK(cudaMemcpy(gt.n_gt, ng.data(), B * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gt.box, box.data(), box.size() * 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gt.mask, bits.data(), bits.size() * 4, cudaMemcpyHostToDevice));
  }
  HeadIO io;
  for (int l = 0; l < 3; l++) {
    io.box[l] = m.head(0, l)->p; io.cls[l] = m.head(1, l)->p; io.coef[l] = m.head(2, l)->p;
    io.dbox[l] = m.head(0, l)->g; io.dcls[l] = m.head(1, l)->g; io.dcoef[l] = m.head(2, l)->g;
  }
  io.proto = m.proto()->p; io.dproto = m.proto()->g;
  float* dscale = dalloc<float>(1);
  CUDA_CHECK(cudaMemcpy(dscale, &lscale, 4, cudaMemcpyHostToDevice));
  sl.run(io, gt, dscale, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  float it[16];
  CUDA_CHECK(cudaMemcpy(it, sl.items(), sizeof(it), cudaMemcpyDeviceToHost));
  auto& ri = R.at("loss.items");
  const char* ln[4] = {"box", "seg", "cls", "dfl"};
  for (int k = 0; k < 4; k++) {
    Cmp c; c.add(ri.f32()[k], it[k]);
    report(std::string("loss ") + ln[k] + " (" + std::to_string(ri.f32()[k]) + " vs " + std::to_string(it[k]) + ")", c);
  }
  {
    int A = lc.A();
    std::vector<int> agt(int64_t(B) * A);
    std::vector<float> ts(int64_t(B) * A);
    CUDA_CHECK(cudaMemcpy(agt.data(), sl.assigned_gt(), agt.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ts.data(), sl.target_score(), ts.size() * 4, cudaMemcpyDeviceToHost));
    auto& rf = R.at("asg.fg_mask");
    auto& rg = R.at("asg.target_gt_idx");
    auto& rs = R.at("asg.target_scores");
    int fg_ref = 0, fg_us = 0, same = 0, gsame = 0;
    Cmp cs;
    for (int64_t q = 0; q < int64_t(B) * A; q++) {
      bool f1 = rf.u8()[q], f2 = agt[q] >= 0;
      fg_ref += f1; fg_us += f2; same += f1 == f2;
      if (f1 && f2) gsame += rg.i32()[q] == agt[q];
      cs.add(rs.f32()[q], f2 ? ts[q] : 0.f);
    }
    printf("assign: fg ref %d ours %d, fg agree %d/%ld, same gt %d, items fg %.0f tss %.4f\n", fg_ref, fg_us, same,
           int64_t(B) * A, gsame, it[4], it[5]);
    report("target scores", cs);
  }
  // 머리 출력 기울기 비교
  for (int k = 0; k < 3; k++) {
    auto& ref = R.at(std::string("dout.") + hk[k]);
    int64_t C = ref.shape[1], A = ref.shape[2];
    Cmp c;
    int64_t a0 = 0;
    for (int l = 0; l < 3; l++) {
      Act* a = m.head(k, l);
      auto v = d2h(a->g, int64_t(B) * a->H * a->W * a->C);
      int64_t hw = a->H * a->W;
      for (int b = 0; b < B; b++)
        for (int64_t ch = 0; ch < C; ch++)
          for (int64_t i = 0; i < hw; i++) c.add(ref.f32()[(b * C + ch) * A + a0 + i], v[(b * hw + i) * C + ch] / lscale);
      a0 += hw;
    }
    report(std::string("dout.") + hk[k], c);
  }
  {
    auto& ref = R.at("dout.proto");
    Act* a = m.proto();
    int64_t C = 32, H = a->H, Wd = a->W;
    auto v = d2h(a->g, int64_t(B) * H * Wd * C);
    Cmp c;
    for (int b = 0; b < B; b++)
      for (int64_t ch = 0; ch < C; ch++)
        for (int64_t i = 0; i < H * Wd; i++) c.add(ref.f32()[(b * C + ch) * H * Wd + i], v[(b * H * Wd + i) * C + ch] / lscale);
    report("dout.proto", c);
  }
  printf("== backward (end to end, from our loss)\n");
  m.backward();
  CUDA_CHECK(cudaStreamSynchronize(st));
  {
    auto G = m.params.save("", true);
    Cmp all;
    for (auto& [k, t] : G) {
      auto itr = R.find("grad." + k);
      if (itr == R.end()) continue;
      for (int64_t i = 0; i < t.numel(); i++) all.add(itr->second.f32()[i], t.f32()[i] / lscale);
    }
    report("all parameter grads (end to end)", all);
  }

  printf("== backward (from reference output grads)\n");
  for (int k = 0; k < 3; k++) {
    auto& ref = R.at(std::string("dout.") + hk[k]);
    int64_t C = ref.shape[1], A = ref.shape[2];
    int64_t a0 = 0;
    for (int l = 0; l < 3; l++) {
      Act* a = m.head(k, l);
      int64_t hw = a->H * a->W;
      std::vector<float> v(int64_t(B) * hw * C);
      for (int b = 0; b < B; b++)
        for (int64_t ch = 0; ch < C; ch++)
          for (int64_t i = 0; i < hw; i++) v[(b * hw + i) * C + ch] = ref.f32()[(b * C + ch) * A + a0 + i] * lscale;
      h2d(a->g, v);
      a0 += hw;
    }
  }
  {
    auto& ref = R.at("dout.proto");
    Act* a = m.proto();
    int64_t C = 32, H = a->H, Wd = a->W;
    std::vector<float> v(int64_t(B) * H * Wd * C);
    for (int b = 0; b < B; b++)
      for (int64_t ch = 0; ch < C; ch++)
        for (int64_t i = 0; i < H * Wd; i++) v[(b * H * Wd + i) * C + ch] = ref.f32()[(b * C + ch) * H * Wd + i] * lscale;
    h2d(a->g, v);
  }
  m.backward();
  CUDA_CHECK(cudaStreamSynchronize(st));
  auto G = m.params.save("", true);
  Cmp all;
  int nmiss = 0;
  std::vector<std::pair<double, std::string>> worst;
  for (auto& [k, t] : G) {
    auto it = R.find("grad." + k);
    if (it == R.end()) { nmiss++; printf("  (no ref grad for %s)\n", k.c_str()); continue; }
    Cmp c;
    for (int64_t i = 0; i < t.numel(); i++) { c.add(it->second.f32()[i], t.f32()[i] / lscale); all.add(it->second.f32()[i], t.f32()[i] / lscale); }
    worst.push_back({c.rel(), k});
  }
  std::sort(worst.rbegin(), worst.rend());
  for (int i = 0; i < 8 && i < (int)worst.size(); i++) printf("  worst grad %-44s relL2 %.2e\n", worst[i].second.c_str(), worst[i].first);
  report("all parameter grads", all);
  // 개별 그라디언트가 크게 틀린 것
  int nbad = 0;
  for (auto& [r, k] : worst) if (r > g_tol * 10) nbad++;
  printf("params with relL2 > %.0e: %d / %zu\n", g_tol * 10, nbad, worst.size());
  printf("%s (%d checks failed)\n", g_fail ? "PARITY FAIL" : "PARITY OK", g_fail);
  return g_fail ? 1 : 0;
}
