// 학습 스텝 속도·메모리: 합성 배치로 정방향·손실·역방향·옵티마이저를 돌린다(그래프 있음/없음).
//   build/bench_train weights/FastSAM-s.fsow [batch 16] [imgsz 1024] [iters 30]
#include <cuda_profiler_api.h>

#include <chrono>
#include <cmath>
#include <random>

#include "../src/loss.cuh"
#include "../src/model.h"
#include "../src/optim.cuh"

int main(int argc, char** argv) {
  FSO_ASSERT(argc >= 2, "usage: bench_train <weights.fsow> [batch] [imgsz] [iters]");
  int B = argc > 2 ? atoi(argv[2]) : 16, S = argc > 3 ? atoi(argv[3]) : 1024, iters = argc > 4 ? atoi(argv[4]) : 30;
  cudaStream_t st;
  int prio_lo, prio_hi;  // train 과 같이 계산 스트림은 높은 우선순위
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
  CUDA_CHECK(cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, prio_hi));
  cudnnHandle_t h;
  CUDNN_CHECK(cudnnCreate(&h));
  CUDNN_CHECK(cudnnSetStream(h, st));
  auto t0 = std::chrono::steady_clock::now();
  Model m;
  auto W0 = fsow_load(argv[1]);
  m.build(ModelCfg{B, S, reg_max_of(W0)}, h, st);
  auto t1 = std::chrono::steady_clock::now();
  m.params.load(W0);
  m.sync_weights();
  LossCfg lc;
  lc.B = B; lc.imgsz = S; lc.G = 256; lc.reg_max = m.cfg.reg_max;
  for (int l = 0; l < 3; l++) { lc.lvl_h[l] = m.head(0, l)->H; lc.lvl_w[l] = m.head(0, l)->W; }
  lc.Hp = m.proto()->H; lc.Wp = m.proto()->W;
  SegLoss sl;
  sl.init(lc);
  OptCfg oc;
  oc.batch = B; oc.nb = 1000; oc.epochs = 10;
#ifdef FSO_FP32
  oc.loss_scaling = false;
#endif
  Optimizer opt;
  opt.init(oc, &m.params, st);

  // 합성 정답: 영상마다 정답 40 개(타원), 영상은 무늬 + 도형
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> U(0.f, 1.f);
  int ngt = 40, words = lc.Hp * lc.Wp / 32;
  std::vector<int> ng(B, ngt);
  std::vector<float4> box(int64_t(B) * lc.G);
  std::vector<uint32_t> bits(int64_t(B) * lc.G * words, 0);
  for (int b = 0; b < B; b++)
    for (int j = 0; j < ngt; j++) {
      float cx = U(rng), cy = U(rng), rw = 0.02f + 0.15f * U(rng), rh = 0.02f + 0.15f * U(rng);
      box[b * lc.G + j] = make_float4(std::max(0.f, cx - rw) * S, std::max(0.f, cy - rh) * S, std::min(1.f, cx + rw) * S,
                                      std::min(1.f, cy + rh) * S);
      for (int y = 0; y < lc.Hp; y++)
        for (int x = 0; x < lc.Wp; x++) {
          float dx = ((x + 0.5f) / lc.Wp - cx) / rw, dy = ((y + 0.5f) / lc.Hp - cy) / rh;
          if (dx * dx + dy * dy <= 1.f) {
            int p = y * lc.Wp + x;
            bits[(int64_t(b) * lc.G + j) * words + p / 32] |= 1u << (p % 32);
          }
        }
    }
  GtBatch gt;
  gt.n_gt = dalloc<int>(B); gt.box = dalloc<float4>(box.size()); gt.mask = dalloc<uint32_t>(bits.size());
  CUDA_CHECK(cudaMemcpy(gt.n_gt, ng.data(), B * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gt.box, box.data(), box.size() * 16, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gt.mask, bits.data(), bits.size() * 4, cudaMemcpyHostToDevice));
  {
    std::vector<act_t> img(int64_t(B) * S * S * kInC);
    for (size_t i = 0; i < img.size(); i++) img[i] = from_f<act_t>(i % kInC < 3 ? U(rng) : 0.f);
    CUDA_CHECK(cudaMemcpy(m.input()->p, img.data(), img.size() * sizeof(act_t), cudaMemcpyHostToDevice));
  }
  HeadIO io;
  for (int l = 0; l < 3; l++) {
    io.box[l] = m.head(0, l)->p; io.cls[l] = m.head(1, l)->p; io.coef[l] = m.head(2, l)->p;
    io.dbox[l] = m.head(0, l)->g; io.dcls[l] = m.head(1, l)->g; io.dcoef[l] = m.head(2, l)->g;
  }
  io.proto = m.proto()->p; io.dproto = m.proto()->g;
  m.print_plan();
  if (getenv("FSO_CONVS")) m.print_convs();
  size_t fr, tot;
  CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
  printf("build %.1f s; GPU used %.2f GB of %.2f GB (ours %.2f GB)\n",
         std::chrono::duration<double>(t1 - t0).count(), (tot - fr) / 1e9, tot / 1e9, DevMem::total() / 1e9);

  auto step = [&]() {
    m.forward(true);
    sl.run(io, gt, opt.scale(), st);
    m.backward(false);
    opt.step(st);
    opt.tick(st);
  };
  // 단계별 시간(이벤트)
  cudaEvent_t e[5];
  for (auto& x : e) CUDA_CHECK(cudaEventCreate(&x));
  for (int i = 0; i < 3; i++) step();
  float acc[4] = {0, 0, 0, 0};
  int nprof = 5;
  for (int i = 0; i < nprof; i++) {
    CUDA_CHECK(cudaEventRecord(e[0], st));
    m.forward(true);
    CUDA_CHECK(cudaEventRecord(e[1], st));
    sl.run(io, gt, opt.scale(), st);
    CUDA_CHECK(cudaEventRecord(e[2], st));
    m.backward(false);
    CUDA_CHECK(cudaEventRecord(e[3], st));
    opt.step(st);
    opt.tick(st);
    CUDA_CHECK(cudaEventRecord(e[4], st));
    CUDA_CHECK(cudaEventSynchronize(e[4]));
    for (int k = 0; k < 4; k++) { float ms; cudaEventElapsedTime(&ms, e[k], e[k + 1]); acc[k] += ms; }
  }
  printf("eager per step: forward %.2f  loss %.2f  backward %.2f  optim %.2f  ms\n", acc[0] / nprof, acc[1] / nprof,
         acc[2] / nprof, acc[3] / nprof);
  // CUDA graph
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  step();
  CUDA_CHECK(cudaStreamEndCapture(st, &g));
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  size_t nn = 0;
  CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &nn));
  CUDA_CHECK(cudaGraphLaunch(ge, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaProfilerStart());
  CUDA_CHECK(cudaEventRecord(e[0], st));
  for (int i = 0; i < iters; i++) CUDA_CHECK(cudaGraphLaunch(ge, st));
  CUDA_CHECK(cudaEventRecord(e[1], st));
  CUDA_CHECK(cudaEventSynchronize(e[1]));
  CUDA_CHECK(cudaProfilerStop());
  float ms;
  cudaEventElapsedTime(&ms, e[0], e[1]);
  float it[16];
  OptState os;
  CUDA_CHECK(cudaMemcpy(it, sl.items(), sizeof(it), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&os, opt.state(), sizeof(os), cudaMemcpyDeviceToHost));
  printf("graph (%zu nodes): %.2f ms/step, %.1f img/s  | loss box %.3f seg %.3f cls %.3f dfl %.3f fg %.0f | scale %.0f "
         "skipped %ld norm %.3f lr %.5f\n",
         nn, ms / iters, B * iters * 1000.f / ms, it[0], it[1], it[2], it[3], it[4], os.scale, os.skipped, os.grad_norm,
         os.lr[0]);
  return 0;
}
