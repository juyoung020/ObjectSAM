// FastSAM-s 학습기(순수 C++/CUDA).
//
//   build/train --init weights/yolov8s-seg-nc1.fsow --shards 'data/shards/sa_0000*.bin' --out runs/r1 [--epochs 20 ...]
//
// 흐름(배치마다, 호스트는 GPU 를 기다리지 않는다):
//   로더 스레드: 샤드 → 원본 디코딩 → pinned 버퍼(여럿 돌려 씀)
//   복사 스트림: pinned → 장치 슬롯(둘) — 앞 배치 계산과 겹친다
//   계산 스트림: CUDA graph [증강 → 정방향 → 손실 → 역방향(누적) → ni+1] (+ 누적이 차면 옵티마이저 graph)
// 손실 값은 pinned 고리 버퍼로 비동기 복사해 몇 스텝 뒤에 읽는다.
#include <glob.h>
#include <signal.h>
#include <sys/stat.h>

#include <chrono>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <map>
#include <string>

#include "augment.cuh"
#include "loss.cuh"
#include "misc.cuh"
#include "model.h"
#include "optim.cuh"

namespace {

volatile sig_atomic_t g_stop = 0;
void on_sig(int) { g_stop = 1; }

struct Args {
  std::string init = "weights/yolov8s-seg-nc1.fsow", shards = "data/shards/*.bin", out = "runs/run", resume;
  int epochs = 20, batch = 16, imgsz = 1024, workers = 24, log_every = 50, G = 256;
  float lr0 = 0.002f, lrf = 0.01f;
  float warmup_epochs = 3.f;  // Ultralytics 기본 3; 공식 COCO 파인튜닝은 약 1
  bool cos_lr = false;
  int64_t max_steps = -1;   // 시험용: 이만큼 하고 멈춤
  uint64_t seed = 0;
  float mosaic = 1.f;
  // 선생 라벨에서 바로(docs/LABELS.md). teacher 가 있으면 --shards 대신.
  std::string teacher, plan, siglip, exclude = "data/coco_shards/val_image_ids.json";
  float score_thr = 0.5f, siglip_thr = 0.8f;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto v = [&]() { FSO_ASSERT(i + 1 < argc, "missing value for %s", k.c_str()); return std::string(argv[++i]); };
    if (k == "--init") a.init = v();
    else if (k == "--shards") a.shards = v();
    else if (k == "--out") a.out = v();
    else if (k == "--resume") a.resume = v();
    else if (k == "--epochs") a.epochs = std::stoi(v());
    else if (k == "--batch") a.batch = std::stoi(v());
    else if (k == "--imgsz") a.imgsz = std::stoi(v());
    else if (k == "--workers") a.workers = std::stoi(v());
    else if (k == "--log-every") a.log_every = std::stoi(v());
    else if (k == "--lr0") a.lr0 = std::stof(v());
    else if (k == "--lrf") a.lrf = std::stof(v());
    else if (k == "--warmup-epochs") a.warmup_epochs = std::stof(v());
    else if (k == "--cos-lr") a.cos_lr = true;
    else if (k == "--max-steps") a.max_steps = std::stoll(v());
    else if (k == "--seed") a.seed = std::stoull(v());
    else if (k == "--mosaic") a.mosaic = std::stof(v());
    else if (k == "--teacher") a.teacher = v();
    else if (k == "--plan") a.plan = v();
    else if (k == "--siglip") a.siglip = v();
    else if (k == "--exclude") a.exclude = v();
    else if (k == "--score-thr") a.score_thr = std::stof(v());
    else if (k == "--siglip-thr") a.siglip_thr = std::stof(v());
    else FSO_ASSERT(false, "unknown arg %s", k.c_str());
  }
  return a;
}

std::vector<std::string> glob_files(const std::string& pat) {
  glob_t g;
  std::vector<std::string> out;
  if (glob(pat.c_str(), 0, nullptr, &g) == 0)
    for (size_t i = 0; i < g.gl_pathc; i++) out.push_back(g.gl_pathv[i]);
  globfree(&g);
  return out;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// 장치 배열 ↔ FSOW 텐서
HostTensor d2t(const float* d, int64_t n) {
  HostTensor t = HostTensor::make_f32({n}, nullptr);
  CUDA_CHECK(cudaMemcpy(t.data.data(), d, n * 4, cudaMemcpyDeviceToHost));
  return t;
}
void t2d(const HostTensor& t, float* d, int64_t n) {
  FSO_ASSERT(t.numel() == n, "state size %ld vs %ld", t.numel(), n);
  CUDA_CHECK(cudaMemcpy(d, t.f32(), n * 4, cudaMemcpyHostToDevice));
}

// EMA 값을 PyTorch 이름·모양으로(Params::save 를 EMA 배열로 잠시 돌려 쓴다)
TensorMap ema_tensors(Params& P, Optimizer& opt) {
  float* keep[P_NKIND];
  for (int k = 0; k < P_NKIND; k++) { keep[k] = P.val[k]; P.val[k] = opt.ema(k); }
  TensorMap m = P.save();
  for (int k = 0; k < P_NKIND; k++) P.val[k] = keep[k];
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);
  std::filesystem::create_directories(a.out);
  auto shards = a.teacher.empty() ? glob_files(a.shards) : std::vector<std::string>{};
  FSO_ASSERT(!a.teacher.empty() || !shards.empty(), "no shards: %s", a.shards.c_str());
  const int B = a.batch, S = a.imgsz;

  cudaStream_t cs, xs;  // 계산, 복사
  // 계산 스트림은 높은 우선순위: 모델의 곁 스트림(wgrad)이 남는 SM 을 채운다(Model::backward)
  int prio_lo, prio_hi;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
  CUDA_CHECK(cudaStreamCreateWithPriority(&cs, cudaStreamNonBlocking, prio_hi));
  CUDA_CHECK(cudaStreamCreateWithFlags(&xs, cudaStreamNonBlocking));
  cudnnHandle_t h;
  CUDNN_CHECK(cudnnCreate(&h));
  CUDNN_CHECK(cudnnSetStream(h, cs));

  // ---- 데이터
  DataCfg dc;
  dc.shards = shards;
  dc.batch = B; dc.imgsz = S; dc.workers = a.workers; dc.seed = a.seed; dc.mosaic = a.mosaic;
  dc.teacher_ftl = a.teacher; dc.teacher_plan = a.plan; dc.teacher_siglip = a.siglip; dc.exclude_ids = a.exclude;
  dc.score_thr = a.score_thr; dc.siglip_thr = a.siglip_thr;
  Dataset ds;
  ds.open(dc);
  const int64_t nb = ds.batches_per_epoch();

  // ---- 모델·손실·옵티마이저
  Model m;
  auto W0 = fsow_load(a.init);
  m.build(ModelCfg{B, S, reg_max_of(W0)}, h, cs);
  m.params.load(W0);
  W0.clear();
  m.sync_weights();
  LossCfg lc;
  lc.B = B; lc.imgsz = S; lc.G = a.G; lc.reg_max = m.cfg.reg_max;
  for (int l = 0; l < 3; l++) { lc.lvl_h[l] = m.head(0, l)->H; lc.lvl_w[l] = m.head(0, l)->W; }
  lc.Hp = m.proto()->H; lc.Wp = m.proto()->W;
  SegLoss sl;
  sl.init(lc);
  OptCfg oc;
  oc.lr0 = a.lr0; oc.lrf = a.lrf; oc.warmup_epochs = a.warmup_epochs; oc.cos_lr = a.cos_lr; oc.epochs = a.epochs; oc.nb = nb; oc.batch = B;
#ifdef FSO_FP32
  oc.loss_scaling = false;
#endif
  Optimizer opt;
  opt.init(oc, &m.params, cs);
  int64_t ni0 = 0;  // 이어 하기: 시작 반복 번호
  int64_t last_opt = -1;
  if (!a.resume.empty()) {
    auto R = fsow_load(a.resume);
    m.params.load(R, "");
    for (int k = 0; k < P_NKIND; k++) t2d(R.at("ema." + std::to_string(k)), opt.ema(k), m.params.size[k]);
    for (int k = 0; k < 3; k++) t2d(R.at("mom." + std::to_string(k)), opt.mom_buf(k), m.params.size[k]);
    auto& st = R.at("opt.state");
    CUDA_CHECK(cudaMemcpy(opt.state(), st.data.data(), sizeof(OptState), cudaMemcpyHostToDevice));
    OptState os;
    memcpy(&os, st.data.data(), sizeof(os));
    ni0 = os.ni;
    last_opt = int64_t(R.at("host.last_opt").f32()[0]);
    m.sync_weights();
    printf("resumed %s at iteration %ld (epoch %ld)\n", a.resume.c_str(), ni0, ni0 / nb);
  }

  // ---- 장치 배치 슬롯 둘과 증강 출력
  ds.set_epoch(ni0);  // 배치 번호 = 반복 번호
  ds.start(4);
  BatchBuf dev[4];
  AugOut ao;
  ao.G = a.G; ao.Hp = lc.Hp; ao.Wp = lc.Wp;
  ao.img = m.input()->p;
  ao.gt.n_gt = dalloc<int>(B);
  ao.gt.box = dalloc<float4>(B * a.G);
  ao.gt.mask = dalloc<uint32_t>(int64_t(B) * a.G * lc.Hp * lc.Wp / 32);
  ao.gt.ignore = dalloc<uint32_t>(int64_t(B) * lc.Hp * lc.Wp / 32);
  ao.gt_inst = dalloc<int>(B * a.G);
  HeadIO io;
  for (int l = 0; l < 3; l++) {
    io.box[l] = m.head(0, l)->p; io.cls[l] = m.head(1, l)->p; io.coef[l] = m.head(2, l)->p;
    io.dbox[l] = m.head(0, l)->g; io.dcls[l] = m.head(1, l)->g; io.dcoef[l] = m.head(2, l)->g;
  }
  io.proto = m.proto()->p; io.dproto = m.proto()->g;
  m.print_plan();

  // ---- graph: 옵티마이저 스텝 하나(마이크로 배치 L 개 + 옵티마이저 + 손실 기록 합)를 graph 하나로.
  // 장치 입력 슬롯 4 개: 이번 스텝의 배치 둘을 쓰는 동안 다음 스텝의 배치 둘을 복사한다.
  constexpr int kSlots = 4;
  std::map<int, cudaGraphExec_t> g_group;  // 키: 시작 슬롯·길이·옵티마이저 여부
  auto capture = [&](auto body) {
    cudaGraph_t g;
    cudaGraphExec_t e;
    CUDA_CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
    body();
    CUDA_CHECK(cudaStreamEndCapture(cs, &g));
    CUDA_CHECK(cudaGraphInstantiate(&e, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return e;
  };
  float* logacc = dalloc<float>(8);  // 손실 items[0..5] 합, [6] 마이크로 배치 수
  CUDA_CHECK(cudaMemset(logacc, 0, 8 * sizeof(float)));
  auto micro = [&](int s, bool accum) {
    augment(dev[s], B, S, ao, cs);
    m.forward(true);
    sl.run(io, ao.gt, opt.scale(), cs);
    m.backward(accum);
    opt.tick(cs);
    accum_items(sl.items(), logacc, 6, cs);
  };

  // ---- 기록
  float* logh;
  CUDA_CHECK(cudaHostAlloc(&logh, 8 * sizeof(float), cudaHostAllocDefault));
  OptState* ost;
  CUDA_CHECK(cudaHostAlloc(&ost, sizeof(OptState), cudaHostAllocDefault));
  cudaEvent_t ev_copied[kSlots], ev_used[kSlots];
  for (int k = 0; k < kSlots; k++) {
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_copied[k], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_used[k], cudaEventDisableTiming));
  }
  FILE* flog = fopen((a.out + "/train.log").c_str(), "a");

  auto save_ckpt = [&](const std::string& path, int64_t ni_done, int epoch_done = -1) {
    CUDA_CHECK(cudaStreamSynchronize(cs));
    TensorMap T = m.params.save();
    for (int k = 0; k < P_NKIND; k++) T["ema." + std::to_string(k)] = d2t(opt.ema(k), m.params.size[k]);
    for (int k = 0; k < 3; k++) T["mom." + std::to_string(k)] = d2t(opt.mom_buf(k), m.params.size[k]);
    HostTensor st;
    st.shape = {int64_t(sizeof(OptState))}; st.dtype = 2; st.data.resize(sizeof(OptState));
    CUDA_CHECK(cudaMemcpy(st.data.data(), opt.state(), sizeof(OptState), cudaMemcpyDeviceToHost));
    T["opt.state"] = st;
    float lo = float(last_opt);
    T["host.last_opt"] = HostTensor::make_f32({1}, &lo);
    fsow_save(path, T);
    TensorMap E = ema_tensors(m.params, opt);  // 추론·내보내기용(EMA, PyTorch 이름)
    fsow_save(a.out + "/ema_last.fsow", E);
    if (epoch_done > 0) fsow_save(a.out + "/ema_ep" + std::to_string(epoch_done) + ".fsow", E);  // epoch 마다(검증용)
    printf("saved %s (iteration %ld)\n", path.c_str(), ni_done);
  };

  // ---- 학습 고리: 옵티마이저 스텝 단위
  const int64_t total = a.max_steps > 0 ? std::min<int64_t>(ni0 + a.max_steps, int64_t(a.epochs) * nb)
                                        : int64_t(a.epochs) * nb;
  const int64_t nw = std::llround(std::min(double(a.warmup_epochs), double(std::max(a.epochs - 1, 0))) * nb);
  printf("training: %d epochs × %ld batches (batch %d, nbs 64 → accumulate %d), warmup %ld iterations, lr0 %g\n",
         a.epochs, nb, B, opt.accumulate(), nw, a.lr0);
  // 누적 수: Ultralytics 와 같이 워밍업 동안 1 → nbs/batch. 이터레이션 ni 에서 ni − last_opt ≥ acc(ni) 이면 스텝.
  auto acc_at = [&](int64_t i) {
    double f = i <= nw && nw > 0 ? 1.0 + (64.0 / B - 1.0) * double(i) / double(nw) : 64.0 / B;
    return std::max(1, int(std::lround(f)));
  };
  double t0 = now_s(), tlog = t0;
  int64_t ni = ni0, next_log = (ni0 / a.log_every + 1) * a.log_every, logged_at = ni0;
  std::vector<BatchBuf*> host_slot(kSlots, nullptr);
  while (ni < total && !g_stop) {
    // 이번 그룹의 길이(마지막 그룹은 짧을 수 있고, 그러면 옵티마이저 스텝 없이 기울기만 쌓는다 — 원래 고리와 같음)
    int L = 1;
    while (ni + L - 1 - last_opt < acc_at(ni + L - 1)) L++;
    bool step = ni + L <= total;
    if (!step) L = int(total - ni);
    int s0 = int(ni % kSlots);
    // 앞 그룹 배치들의 복사가 끝났으면 pinned 버퍼를 로더에 돌려준다(쥐고 있으면 로더 버퍼 4 개가 바닥나 멈춘다)
    for (int k = 0; k < kSlots; k++)
      if (host_slot[k]) {
        CUDA_CHECK(cudaEventSynchronize(ev_copied[k]));
        ds.release(host_slot[k]);
        host_slot[k] = nullptr;
      }
    for (int k = 0; k < L; k++) {
      int s = int((ni + k) % kSlots);
      int64_t bi;
      BatchBuf* hb = ds.acquire(bi);
      FSO_ASSERT(bi == ni + k, "batch order %ld vs %ld", bi, ni + k);
      if (!dev[s].hdr) dev[s].alloc_dev(*hb);
      host_slot[s] = hb;
      CUDA_CHECK(cudaStreamWaitEvent(xs, ev_used[s], 0));  // 장치 슬롯이 앞 사용에서 풀렸을 때
      CUDA_CHECK(cudaMemcpyAsync(dev[s].hdr, hb->hdr, hb->meta_bytes(), cudaMemcpyHostToDevice, xs));
      CUDA_CHECK(cudaMemcpyAsync(dev[s].img, hb->img, hb->hdr->img_bytes, cudaMemcpyHostToDevice, xs));
      CUDA_CHECK(cudaMemcpyAsync(dev[s].mask, hb->mask, hb->hdr->mask_words * 4, cudaMemcpyHostToDevice, xs));
      CUDA_CHECK(cudaEventRecord(ev_copied[s], xs));
      CUDA_CHECK(cudaStreamWaitEvent(cs, ev_copied[s], 0));
    }
    int key = (s0 * 8 + L) * 2 + step;
    bool first = ni - last_opt == 1;  // 옵티마이저 스텝 바로 뒤: 첫 마이크로 배치가 기울기를 덮어쓴다
    FSO_ASSERT(first, "group must start right after an optimizer step");
    if (!g_group.count(key))
      g_group[key] = capture([&]() {
        for (int k = 0; k < L; k++) micro(int((s0 + k) % kSlots), k > 0);
        if (step) opt.step(cs);
      });
    CUDA_CHECK(cudaGraphLaunch(g_group[key], cs));
    for (int k = 0; k < L; k++) CUDA_CHECK(cudaEventRecord(ev_used[(s0 + k) % kSlots], cs));
    ni += L;
    if (step) last_opt = ni - 1;
    if (ni >= next_log) {
      CUDA_CHECK(cudaMemcpyAsync(logh, logacc, 8 * sizeof(float), cudaMemcpyDeviceToHost, cs));
      CUDA_CHECK(cudaMemcpyAsync(ost, opt.state(), sizeof(OptState), cudaMemcpyDeviceToHost, cs));
      CUDA_CHECK(cudaMemsetAsync(logacc, 0, 8 * sizeof(float), cs));
      CUDA_CHECK(cudaStreamSynchronize(cs));
      double t = now_s();
      double ips = double(ni - logged_at) * B / (t - tlog);
      tlog = t;
      double eta = (total - ni) * B / ips / 3600.0;
      float n = std::max(logh[6], 1.f);
      char line[512];
      snprintf(line, sizeof(line),
               "ep %ld it %ld/%ld | box %.4f seg %.4f cls %.4f dfl %.4f | fg %.0f | lr %.5f mom %.3f scale %.0f skip %ld "
               "norm %.2f | %.1f img/s, data wait %.1f ms/it, eta %.2f h",
               (ni - 1) / nb, ni, total, logh[0] / n, logh[1] / n, logh[2] / n, logh[3] / n, logh[4] / n, ost->lr[0],
               ost->mom, ost->scale, ost->skipped, ost->grad_norm, ips, ds.wait_ms / double(ni - logged_at), eta);
      ds.wait_ms = 0;
      logged_at = ni;
      next_log = (ni / a.log_every + 1) * a.log_every;
      puts(line);
      fflush(stdout);
      if (flog) { fprintf(flog, "%s\n", line); fflush(flog); }
    }
    if (ni / nb != (ni - L) / nb) save_ckpt(a.out + "/last.fsow", ni, int(ni / nb));  // epoch 경계를 넘은 그룹 뒤
  }
  if (ni != ni0) save_ckpt(a.out + "/last.fsow", ni);
  ds.stop();
  printf("done: %ld iterations in %.1f min\n", ni - ni0, (now_s() - t0) / 60);
  return 0;
}
