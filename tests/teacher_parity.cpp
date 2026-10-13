// 선생(CropFormer) C++ 포팅 정합성: 단계마다 PyTorch 기준 입력을 넣고 출력을 비교한다(pytools/teacher_dump.py).
//   teacher_parity WEIGHTS REF [fp32|fast|hhf|half] [tol]
// fp32 = 모두 fp32(정합성), fast = 백본 fp16 + 나머지 fp32, hhf = 백본·픽셀 디코더 fp16, half = 모두 fp16.
#include <chrono>
#include <cmath>

#include "prep.h"
#include "tnet.h"

namespace {
// (B, C, H, W) → (B, H, W, C)
std::vector<float> nhwc(const HostTensor& t) {
  int64_t B = t.shape[0], C = t.shape[1], HW = t.numel() / (B * C);
  std::vector<float> o(t.numel());
  for (int64_t b = 0; b < B; b++)
    for (int64_t c = 0; c < C; c++)
      for (int64_t p = 0; p < HW; p++) o[(b * HW + p) * C + c] = t.f32()[(b * C + c) * HW + p];
  return o;
}
template <class T> std::vector<float> down(const T* d, int64_t n) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  std::vector<float> o(n);
  for (int64_t i = 0; i < n; i++) o[i] = to_f(h[i]);
  return o;
}
template <class T> void up(T* d, const std::vector<float>& v) {
  std::vector<T> h(v.size());
  for (size_t i = 0; i < v.size(); i++) h[i] = from_f<T>(v[i]);
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
}
bool report(const char* name, const std::vector<float>& a, const std::vector<float>& r, double tol) {
  FSO_ASSERT(a.size() == r.size(), "%s size %zu vs %zu", name, a.size(), r.size());
  double mx = 0, mr = 0, se = 0, sr = 0;
  for (size_t i = 0; i < a.size(); i++) {
    double d = std::fabs(double(a[i]) - r[i]);
    if (!(d <= mx)) mx = d;  // NaN 도 잡는다
    mr = std::max(mr, std::fabs(double(r[i])));
    se += d * d;
    sr += double(r[i]) * r[i];
  }
  double rel = mx / (mr + 1e-12), rms = std::sqrt(se / (sr + 1e-30));
  bool ok = rel < tol;
  printf("  %-18s max|d| %.3e  max|ref| %.3e  rel %.2e  rms %.2e  %s\n", name, mx, mr, rel, rms, ok ? "ok" : "FAIL");
  return ok;
}
}  // namespace

template <class Net, class AB>
int run(const char* weights, const TensorMap& ref, double tol) {
  Net net;
  net.load(weights);
  const HostTensor& in = ref.at("input");
  int T = in.shape[0], Hp = in.shape[2], Wp = in.shape[3];
  net.set_shape(T, Hp, Wp);
  printf("T %d, %dx%d\n", T, Hp, Wp);
  bool ok = true;
  // 전처리: 기준 사진(cv2 BGR)으로 C++ 전처리를 돌려 기준 입력과 비교(비트 단위로 같아야 한다)
  {
    auto& img = ref.at("image_bgr");  // (h, w, 3) u8
    int h = img.shape[0], w = img.shape[1];
    std::vector<uint8_t> rgb(size_t(h) * w * 3);
    for (size_t i = 0; i < size_t(h) * w; i++)
      for (int c = 0; c < 3; c++) rgb[i * 3 + c] = img.u8()[i * 3 + 2 - c];
    Prepared pr;
    prepare(rgb.data(), h, w, pr);
    auto rin = nhwc(in);
    double mx = 0;
    if (pr.Hp == Hp && pr.Wp == Wp)
      for (size_t i = 0; i < rin.size(); i++) mx = std::max(mx, double(std::fabs(rin[i] - pr.x[i])));
    auto& cr = ref.at("crop_regions");
    auto& is = ref.at("image_sizes");
    bool geom = pr.Hp == Hp && pr.Wp == Wp;
    for (int v = 0; v < T; v++) {
      geom &= pr.g.sh[v] == is.i32()[v * 2] && pr.g.sw[v] == is.i32()[v * 2 + 1];
      if (v) geom &= pr.g.x0[v] == cr.i32()[(v - 1) * 4] && pr.g.y0[v] == cr.i32()[(v - 1) * 4 + 1];
    }
    printf("  prepare: %dx%d geometry %s, max|d| %.3g %s\n", pr.Wp, pr.Hp, geom ? "same" : "DIFFERENT", mx,
           geom && mx == 0 ? "ok" : "FAIL");
    ok &= geom && mx == 0;
  }
  AB* x = dalloc<AB>(in.numel());
  up(x, nhwc(in));

  auto t0 = std::chrono::steady_clock::now();
  auto lap = [&](const char* what) {
    CUDA_CHECK(cudaStreamSynchronize(net.st));
    auto t1 = std::chrono::steady_clock::now();
    printf("%s: %.1f ms\n", what, std::chrono::duration<double, std::milli>(t1 - t0).count());
    t0 = t1;
  };
  net.backbone(x);
  lap("backbone");
  for (int s = 0; s < 4; s++) {
    auto& r = ref.at("bb.res" + std::to_string(s + 2));
    ok &= report(("res" + std::to_string(s + 2)).c_str(), down(net.res[s], r.numel()), nhwc(r), tol);
  }
  // 픽셀 디코더: 기준 백본 출력을 넣는다
  for (int s = 0; s < 4; s++) up(net.res[s], nhwc(ref.at("bb.res" + std::to_string(s + 2))));
  t0 = std::chrono::steady_clock::now();
  net.pixel_decoder();
  lap("pixel decoder");
  for (int l = 0; l < 3; l++) {
    auto& r = ref.at("pd.ms" + std::to_string(l));
    ok &= report(("ms" + std::to_string(l)).c_str(), down(net.ms[l], r.numel()), nhwc(r), tol);
  }
  auto& rmf = ref.at("pd.mask_features");
  ok &= report("mask_features", down(net.mf, rmf.numel()), nhwc(rmf), tol);
  // 디코더: 기준 픽셀 디코더 출력을 넣는다
  for (int l = 0; l < 3; l++) up(net.ms[l], nhwc(ref.at("pd.ms" + std::to_string(l))));
  up(net.mf, nhwc(rmf));
  t0 = std::chrono::steady_clock::now();
  net.decoder();
  lap("decoder");
  if (getenv("TWICE")) {
    auto a = down(net.q3d, 200 * 256);
    net.decoder();
    CUDA_CHECK(cudaStreamSynchronize(net.st));
    report("make3d run2 vs 1", down(net.q3d, 200 * 256), a, tol);
  }
  {
    auto& r = ref.at("dec.logits2d");  // (T, Q, 2)
    ok &= report("logits2d", down(net.cls2d, r.numel()), std::vector<float>(r.f32(), r.f32() + r.numel()), tol);
    auto& m3 = ref.at("dec.make3d");  // (Q, 1, 256)
    ok &= report("make3d", down(net.q3d, m3.numel()), std::vector<float>(m3.f32(), m3.f32() + m3.numel()), tol);
    auto& c3 = ref.at("dec.logits3d");  // (1, Q, 2)
    ok &= report("logits3d", down(net.cls3d, c3.numel()), std::vector<float>(c3.f32(), c3.f32() + c3.numel()), tol);
    auto& k3 = ref.at("dec.masks3d");  // (1, Q, T, H, W)
    ok &= report("masks3d", down(net.mask3d, k3.numel()), std::vector<float>(k3.f32(), k3.f32() + k3.numel()), tol);
  }
  // 후처리: 기준 3d 출력을 넣는다
  {
    auto& k3 = ref.at("dec.masks3d");
    auto& c3 = ref.at("dec.logits3d");
    up(net.mask3d, std::vector<float>(k3.f32(), k3.f32() + k3.numel()));
    up(net.cls3d, std::vector<float>(c3.f32(), c3.f32() + c3.numel()));
    auto& cr = ref.at("crop_regions");
    auto& is = ref.at("image_sizes");
    auto& vh = ref.at("views_hw");
    ViewGeom g{};
    g.T = T, g.Hm = net.lh[0], g.Wm = net.lw[0], g.Hp = Hp, g.Wp = Wp, g.OH = vh.i32()[0], g.OW = vh.i32()[1];
    for (int v = 0; v < T; v++) {
      g.sh[v] = is.i32()[v * 2], g.sw[v] = is.i32()[v * 2 + 1];
      g.oh[v] = vh.i32()[v * 2], g.ow[v] = vh.i32()[v * 2 + 1];
      g.x0[v] = v ? cr.i32()[(v - 1) * 4] : 0;
      g.y0[v] = v ? cr.i32()[(v - 1) * 4 + 1] : 0;
    }
    TeacherResult res;
    t0 = std::chrono::steady_clock::now();
    net.postprocess(g, res);
    lap("postprocess");
    auto& rm = ref.at("out.masks");  // (Q, OH, OW) u8, 순위 순
    auto& rs = ref.at("out.scores");
    int Q = rm.shape[0];
    int64_t HW = int64_t(g.OH) * g.OW, diff = 0, tot = 0;
    for (int k = 0; k < Q; k++)
      for (int64_t p = 0; p < HW; p++) {
        bool a = res.id[p] == k, b = rm.u8()[k * HW + p] != 0;
        diff += a != b;
        tot += b;
      }
    printf("  mine: ");
    for (int k = 0; k < 6; k++) printf("(%.4f, %d) ", res.score[k], res.area[k]);
    printf("\n  ref:  ");
    for (int k = 0; k < 6; k++) {
      int a = 0;
      for (int64_t p = 0; p < HW; p++) a += rm.u8()[k * HW + p];
      printf("(%.4f, %d) ", rs.f32()[k], a);
    }
    printf("\n");
    {
      int shown = 0;
      for (int k = 0; k < Q && shown < 8; k++)
        if (std::fabs(res.score[k] - rs.f32()[k]) > 1e-3f) {
          int a = 0;
          for (int64_t p = 0; p < HW; p++) a += rm.u8()[k * HW + p];
          printf("  rank %d: mine (%.5f, %d) ref (%.5f, %d)\n", k, res.score[k], res.area[k], rs.f32()[k], a);
          shown++;
        }
    }
    printf("  masks: %lld / %lld mask pixels differ\n", (long long)diff, (long long)tot);
    ok &= report("scores", res.score, std::vector<float>(rs.f32(), rs.f32() + rs.numel()), tol);
    ok &= diff <= tot / 10000;
  }
  // 실험: 디코더 입력을 fp16 으로 반올림해 fp32 디코더에 넣으면 최종 마스크가 얼마나 바뀌나
  if (getenv("HALFIN")) {
    auto rnd = [](std::vector<float> v) {
      for (auto& x : v) x = __half2float(__float2half(x));
      return v;
    };
    for (int l = 0; l < 3; l++) up(net.ms[l], rnd(nhwc(ref.at("pd.ms" + std::to_string(l)))));
    up(net.mf, rnd(nhwc(rmf)));
    net.decoder();
    CUDA_CHECK(cudaStreamSynchronize(net.st));
    auto& k3 = ref.at("dec.masks3d");
    report("masks3d(half in)", down(net.mask3d, k3.numel()), std::vector<float>(k3.f32(), k3.f32() + k3.numel()), tol);
  }
  // 처음부터 끝까지(기준값 주입 없이): 점수 ≥ 0.5 마스크(라벨로 쓸 것)를 기준과 맞춰 본다.
  // INNOISE=e: 입력에 ±e 균일 잡음(정규화 단위; 픽셀값 1 단계 ≈ 0.017) — 모델 자체의 민감도 재기.
  {
    auto xin = nhwc(in);
    if (getenv("INNOISE")) {
      float e = atof(getenv("INNOISE"));
      uint32_t s = 12345;
      for (auto& v : xin) {
        s = s * 1664525u + 1013904223u;
        v += e * (2.f * (s >> 8) / float(1 << 24) - 1.f);
      }
    }
    up(x, xin);
    t0 = std::chrono::steady_clock::now();
    net.backbone(x);
    net.pixel_decoder();
    net.decoder();
    lap("end-to-end net");
    if (getenv("BENCH")) {  // 따뜻한 상태에서 단계별 시간
      int n = atoi(getenv("BENCH"));
      double tb = 0, tp = 0, td = 0;
      for (int i = 0; i < n; i++) {
        auto a = std::chrono::steady_clock::now();
        net.backbone(x);
        CUDA_CHECK(cudaStreamSynchronize(net.st));
        auto b = std::chrono::steady_clock::now();
        net.pixel_decoder();
        CUDA_CHECK(cudaStreamSynchronize(net.st));
        auto c = std::chrono::steady_clock::now();
        net.decoder();
        CUDA_CHECK(cudaStreamSynchronize(net.st));
        auto d = std::chrono::steady_clock::now();
        tb += std::chrono::duration<double, std::milli>(b - a).count();
        tp += std::chrono::duration<double, std::milli>(c - b).count();
        td += std::chrono::duration<double, std::milli>(d - c).count();
      }
      printf("bench x%d: backbone %.1f ms, pixel decoder %.1f ms, decoder %.1f ms, total %.1f ms\n", n, tb / n, tp / n, td / n,
             (tb + tp + td) / n);
    }
    auto& cr = ref.at("crop_regions");
    auto& is = ref.at("image_sizes");
    auto& vh = ref.at("views_hw");
    ViewGeom g{};
    g.T = T, g.Hm = net.lh[0], g.Wm = net.lw[0], g.Hp = Hp, g.Wp = Wp, g.OH = vh.i32()[0], g.OW = vh.i32()[1];
    for (int v = 0; v < T; v++) {
      g.sh[v] = is.i32()[v * 2], g.sw[v] = is.i32()[v * 2 + 1];
      g.oh[v] = vh.i32()[v * 2], g.ow[v] = vh.i32()[v * 2 + 1];
      g.x0[v] = v ? cr.i32()[(v - 1) * 4] : 0;
      g.y0[v] = v ? cr.i32()[(v - 1) * 4 + 1] : 0;
    }
    TeacherResult res;
    net.postprocess(g, res);
    auto& rm = ref.at("out.masks");
    auto& rs = ref.at("out.scores");
    int Q = rm.shape[0];
    int64_t HW = int64_t(g.OH) * g.OW;
    std::vector<int> mine, theirs;
    for (int k = 0; k < Q; k++) {
      if (res.score[k] >= 0.5f) mine.push_back(k);
      if (rs.f32()[k] >= 0.5f) theirs.push_back(k);
    }
    // 기준 마스크마다 내 마스크와의 최대 IoU
    std::vector<int> lab(HW, -1);
    for (size_t i = 0; i < mine.size(); i++)
      for (int64_t p = 0; p < HW; p++)
        if (res.id[p] == mine[i]) lab[p] = int(i);
    double sum_iou = 0;
    int n90 = 0, n50 = 0;
    for (int k : theirs) {
      std::vector<int64_t> inter(mine.size(), 0);
      int64_t a = 0;
      for (int64_t p = 0; p < HW; p++)
        if (rm.u8()[k * HW + p]) {
          a++;
          if (lab[p] >= 0) inter[lab[p]]++;
        }
      double best = 0;
      for (size_t i = 0; i < mine.size(); i++) {
        int64_t b = res.area[mine[i]];
        best = std::max(best, double(inter[i]) / double(a + b - inter[i]));
      }
      sum_iou += best;
      n90 += best >= 0.9;
      n50 += best >= 0.5;
    }
    printf("  e2e: kept masks mine %zu ref %zu; per ref mask best IoU mean %.4f, >=0.9 %d, >=0.5 %d\n", mine.size(),
           theirs.size(), theirs.empty() ? 0.0 : sum_iou / theirs.size(), n90, n50);
  }
  printf(ok ? "PARITY OK\n" : "PARITY FAIL\n");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  FSO_ASSERT(argc >= 3, "usage: teacher_parity WEIGHTS REF [fp32|fast|hhf|half] [tol]");
  std::string prec = argc > 3 ? argv[3] : "fp32";
  double tol = argc > 4 ? atof(argv[4]) : 1e-3;
  TensorMap ref = fsow_load(argv[2]);
  if (prec == "fp32") return run<TeacherNetT<float, float, float>, float>(argv[1], ref, tol);
  if (prec == "fast") return run<TeacherNetT<__half, float, float>, __half>(argv[1], ref, tol);
  if (prec == "hhf") return run<TeacherNetT<__half, __half, float>, __half>(argv[1], ref, tol);
  if (prec == "half") return run<TeacherNetT<__half, __half, __half>, __half>(argv[1], ref, tol);
  FSO_ASSERT(false, "prec %s", prec.c_str());
}
