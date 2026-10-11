#include "conv.h"

#include <algorithm>
#include <vector>

namespace {
void nhwc(cudnnTensorDescriptor_t d, cudnnDataType_t dt, int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NHWC, dt, n, c, h, w));
}

// ---- backend(graph) API 도우미
cudnnBackendDescriptor_t be_make(cudnnBackendDescriptorType_t t) {
  cudnnBackendDescriptor_t d;
  CUDNN_CHECK(cudnnBackendCreateDescriptor(t, &d));
  return d;
}
template <class T>
void be_set(cudnnBackendDescriptor_t d, cudnnBackendAttributeName_t a, cudnnBackendAttributeType_t t, int64_t n,
            const T* v) {
  CUDNN_CHECK(cudnnBackendSetAttribute(d, a, t, n, v));
}
// NHWC packed 텐서(논리 차원 n, c, h, w). 필터는 KRSC = (k, c, r, s) 를 같은 꼴로.
cudnnBackendDescriptor_t be_tensor(int64_t uid, cudnnDataType_t dt, int64_t n, int64_t c, int64_t h, int64_t w) {
  auto d = be_make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  int64_t dims[4] = {n, c, h, w}, str[4] = {h * w * c, 1, w * c, c}, al = 16;
  be_set(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt);
  be_set(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims);
  be_set(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, str);
  be_set(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
  be_set(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &al);
  CUDNN_CHECK(cudnnBackendFinalize(d));
  return d;
}
// 엔진의 수치 성질이 legacy 경로와 같은가: fp32 누적·변환 없음, FFT/Winograd 아님, fp32 는 텐서코어(TF32)도 아님.
bool be_numerics_ok(cudnnBackendDescriptor_t cfg, cudnnDataType_t dt) {
  cudnnBackendDescriptor_t eng = be_make(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
  int64_t n = 0;
  bool ok = cudnnBackendGetAttribute(cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &n, &eng) ==
            CUDNN_STATUS_SUCCESS;
  cudnnBackendNumericalNote_t notes[CUDNN_NUMERICAL_NOTE_TYPE_COUNT];
  n = 0;
  if (ok)
    ok = cudnnBackendGetAttribute(eng, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE,
                                  CUDNN_NUMERICAL_NOTE_TYPE_COUNT, &n, notes) == CUDNN_STATUS_SUCCESS;
  for (int64_t i = 0; ok && i < n; i++) {
    switch (notes[i]) {
      case CUDNN_NUMERICAL_NOTE_TENSOR_CORE: ok = dt != CUDNN_DATA_FLOAT; break;
      case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:
      case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
      case CUDNN_NUMERICAL_NOTE_FFT:
      case CUDNN_NUMERICAL_NOTE_WINOGRAD:
      case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4:
      case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6:
      case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13: ok = false; break;
      default: break;
    }
  }
  cudnnBackendDestroyDescriptor(eng);
  return ok;
}
constexpr int64_t kUidIn = 1, kUidOut = 2, kUidW = 3;
}  // namespace

void ConvOp::init(cudnnHandle_t h, const ConvGeom& g, bool transposed, cudnnDataType_t dt) {
  h_ = h;
  dt_ = dt;
  g_ = g;
  tr_ = transposed;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&xd_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&yd_));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&wd_));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&cd_));
  if (!tr_) {
    nhwc(xd_, dt_, g.N, g.C, g.H, g.W);
    nhwc(yd_, dt_, g.N, g.K, g.P(), g.Q());
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wd_, dt_, CUDNN_TENSOR_NHWC, g.K, g.C, g.R, g.S));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(cd_, g.pad, g.pad, g.stride, g.stride, 1, 1, CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
  } else {
    FSO_ASSERT(g.pad == 0 && g.R == g.stride && g.S == g.stride, "transposed conv: k == stride only");
    // 내부 conv: big(N, K, H*s, W*s) → small(N, C, H, W), 필터 (C, K, R, S).
    nhwc(xd_, dt_, g.N, g.C, g.H, g.W);                                  // small = 이 층 입력
    nhwc(yd_, dt_, g.N, g.K, g.H * g.stride, g.W * g.stride);            // big = 이 층 출력
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wd_, dt_, CUDNN_TENSOR_NHWC, g.C, g.K, g.R, g.S));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(cd_, 0, 0, g.stride, g.stride, 1, 1, CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
  }
  // fp32 는 정합성 시험·정밀 경로: TF32 금지
  CUDNN_CHECK(cudnnSetConvolutionMathType(cd_, dt_ == CUDNN_DATA_FLOAT ? CUDNN_FMA_MATH : CUDNN_TENSOR_OP_MATH));
}

int64_t ConvOp::out_elems() const {
  return tr_ ? int64_t(g_.N) * g_.K * g_.H * g_.stride * g_.W * g_.stride : int64_t(g_.N) * g_.K * g_.P() * g_.Q();
}

size_t ConvOp::tune(act_t* x, act_t* w, act_t* y, act_t* dx, act_t* dw, act_t* dy, void* ws, size_t ws_limit,
                    float dgrad_beta) {
  int n = 0;
  size_t need = 0;
  // 내부 conv 의 (입력, 출력) 디스크립터: 보통은 (x, y), 전치면 (y=big, x=small).
  cudnnTensorDescriptor_t ind = tr_ ? yd_ : xd_, outd = tr_ ? xd_ : yd_;
  act_t* in = tr_ ? y : x;   // 내부 conv 입력 버퍼
  act_t* out = tr_ ? x : y;  // 내부 conv 출력 버퍼
  act_t* din = tr_ ? dy : dx;
  act_t* dout = tr_ ? dx : dy;
  float ms[3];  // 내부 fwd, bwd-data, bwd-filter
  {
    cudnnConvolutionFwdAlgoPerf_t p[16];
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(h_, ind, in, wd_, w, cd_, outd, out, 16, &n, p, ws, ws_limit));
    int i = 0;
    while (i < n && p[i].status != CUDNN_STATUS_SUCCESS) i++;
    FSO_ASSERT(i < n, "no fwd algo");
    fa_ = p[i].algo;
    need = std::max(need, p[i].memory);
    ms[0] = p[i].time;
  }
  {
    cudnnConvolutionBwdDataAlgoPerf_t p[16];
    CUDNN_CHECK(
        cudnnFindConvolutionBackwardDataAlgorithmEx(h_, wd_, w, outd, dout, cd_, ind, din, 16, &n, p, ws, ws_limit));
    int i = 0;
    while (i < n && p[i].status != CUDNN_STATUS_SUCCESS) i++;
    FSO_ASSERT(i < n, "no bwd-data algo");
    da_ = p[i].algo;
    need = std::max(need, p[i].memory);
    ms[1] = p[i].time;
  }
  {
    cudnnConvolutionBwdFilterAlgoPerf_t p[16];
    // 전치: dW = BwdFilter(내부 입력 = 이 층 출력 기울기 dy(big), 내부 출력 기울기 = 이 층 입력 x(small)).
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(h_, ind, tr_ ? dy : x, outd, tr_ ? x : dy, cd_, wd_, dw,
                                                              16, &n, p, ws, ws_limit));
    int i = 0;
    while (i < n && p[i].status != CUDNN_STATUS_SUCCESS) i++;
    FSO_ASSERT(i < n, "no bwd-filter algo");
    wa_ = p[i].algo;
    need = std::max(need, p[i].memory);
    ms[2] = p[i].time;
  }
  // backend 엔진: 내부 op 마다 beta 는 실제 호출과 같게(전치면 이 층 dgrad = 내부 정방향).
  const float beta[3] = {tr_ ? dgrad_beta : 0.f, tr_ ? 0.f : dgrad_beta, 0.f};
  void* bin[3] = {in, din, tr_ ? dy : x};
  void* bout[3] = {out, dout, tr_ ? x : dy};
  void* bw[3] = {w, w, dw};
  for (int op = 0; op < 3; op++) {
    size_t wneed = 0;
    float t = be_tune(op, bin[op], bout[op], bw[op], beta[op], ws, ws_limit, wneed);
    if (plan_[op] && t < ms[op]) {
      ms[op] = t;
      need = std::max(need, wneed);
    } else if (plan_[op]) {
      cudnnBackendDestroyDescriptor(plan_[op]);
      plan_[op] = nullptr;
    }
  }
  // tuned_ms: 이 층 기준 [fwd, dgrad, wgrad]
  tuned_ms[0] = ms[tr_ ? 1 : 0];
  tuned_ms[1] = ms[tr_ ? 0 : 1];
  tuned_ms[2] = ms[2];
  return need;
}

// 내부 op 하나(0 fwd, 1 bwd-data, 2 bwd-filter)를 heuristics(mode A) 엔진 설정마다 재서 가장 빠른 실행 계획을
// plan_[op] 에 둔다. 돌려주는 값은 그 시간(ms, 없으면 무한대).
float ConvOp::be_tune(int op, void* in, void* out, void* w, float beta, void* ws, size_t ws_limit, size_t& need) {
  const ConvGeom& g = g_;
  const int64_t s = g.stride;
  // 내부 conv: 입력 (N, ci, hi, wi) → 출력 (N, co, ho, wo), 필터 (co, ci, R, S)
  int64_t ci, hi, wi, co, ho, wo, pad;
  if (!tr_) { ci = g.C; hi = g.H; wi = g.W; co = g.K; ho = g.P(); wo = g.Q(); pad = g.pad; }
  else { ci = g.K; hi = g.H * s; wi = g.W * s; co = g.C; ho = g.H; wo = g.W; pad = 0; }
  auto IN = be_tensor(kUidIn, dt_, g.N, ci, hi, wi), OUT = be_tensor(kUidOut, dt_, g.N, co, ho, wo),
       W = be_tensor(kUidW, dt_, co, ci, g.R, g.S);
  auto cv = be_make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
  {
    int64_t sd = 2, dil[2] = {1, 1}, fs[2] = {s, s}, pp[2] = {pad, pad};
    cudnnDataType_t ct = CUDNN_DATA_FLOAT;
    cudnnConvolutionMode_t cm = CUDNN_CROSS_CORRELATION;
    be_set(cv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &sd);
    be_set(cv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &ct);
    be_set(cv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &cm);
    be_set(cv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dil);
    be_set(cv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, fs);
    be_set(cv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pp);
    be_set(cv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pp);
    CUDNN_CHECK(cudnnBackendFinalize(cv));
  }
  const cudnnBackendDescriptorType_t ot[3] = {CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                              CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                                              CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR};
  // 속성 이름: [op][x, w, y, conv, alpha, beta]
  const cudnnBackendAttributeName_t an[3][6] = {
      {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA},
      {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA},
      {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
       CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA}};
  auto od = be_make(ot[op]);
  const float alpha = 1.f;
  be_set(od, an[op][0], CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &IN);
  be_set(od, an[op][1], CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &W);
  be_set(od, an[op][2], CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &OUT);
  be_set(od, an[op][3], CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cv);
  be_set(od, an[op][4], CUDNN_TYPE_FLOAT, 1, &alpha);
  be_set(od, an[op][5], CUDNN_TYPE_FLOAT, 1, &beta);
  CUDNN_CHECK(cudnnBackendFinalize(od));
  auto og = be_make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  be_set(og, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &od);
  be_set(og, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &h_);
  CUDNN_CHECK(cudnnBackendFinalize(og));
  std::vector<cudnnBackendDescriptor_t> cfgs;
  {
    auto he = be_make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    cudnnBackendHeurMode_t hm = CUDNN_HEUR_MODE_A;
    be_set(he, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &og);
    be_set(he, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &hm);
    if (cudnnBackendFinalize(he) == CUDNN_STATUS_SUCCESS) {
      int64_t n = 0;
      CUDNN_CHECK(cudnnBackendGetAttribute(he, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &n, nullptr));
      cfgs.resize(n);
      for (auto& d : cfgs) d = be_make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
      CUDNN_CHECK(cudnnBackendGetAttribute(he, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, n, &n,
                                           cfgs.data()));
      cfgs.resize(n);
    }
    cudnnBackendDestroyDescriptor(he);
  }
  cudaStream_t st;
  CUDNN_CHECK(cudnnGetStream(h_, &st));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  auto vpack = [&](void* wsp) {
    auto vp = be_make(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    void* ptrs[3] = {in, out, w};
    int64_t uids[3] = {kUidIn, kUidOut, kUidW};
    be_set(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs);
    be_set(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
    be_set(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &wsp);
    CUDNN_CHECK(cudnnBackendFinalize(vp));
    return vp;
  };
  auto vp = vpack(ws);
  auto time_plan = [&](cudnnBackendDescriptor_t pl, int reps) {
    CUDA_CHECK(cudaEventRecord(e0, st));
    for (int r = 0; r < reps; r++) CUDNN_CHECK(cudnnBackendExecute(h_, pl, vp));
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float t;
    CUDA_CHECK(cudaEventElapsedTime(&t, e0, e1));
    return t / reps;
  };
  // 한 번씩 재고(첫 실행은 버림), 빠른 넷을 다시 여러 번 재서 고른다.
  std::vector<std::pair<float, cudnnBackendDescriptor_t>> cand;
  for (auto cfg : cfgs) {
    if (!be_numerics_ok(cfg, dt_)) continue;
    auto pl = be_make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    be_set(pl, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &h_);
    be_set(pl, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfg);
    int64_t wsz = 0, got = 0;
    if (cudnnBackendFinalize(pl) != CUDNN_STATUS_SUCCESS ||
        cudnnBackendGetAttribute(pl, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &got, &wsz) !=
            CUDNN_STATUS_SUCCESS ||
        size_t(wsz) > ws_limit || cudnnBackendExecute(h_, pl, vp) != CUDNN_STATUS_SUCCESS) {
      cudnnBackendDestroyDescriptor(pl);
      continue;
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    cand.push_back({time_plan(pl, 1), pl});
  }
  std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first < b.first; });
  float best = 1e30f;
  for (size_t i = 0; i < cand.size(); i++) {
    if (i < 4) {
      float t = time_plan(cand[i].second, 5);
      if (t < best) {
        best = t;
        if (plan_[op]) cudnnBackendDestroyDescriptor(plan_[op]);
        plan_[op] = cand[i].second;
        continue;
      }
    }
    cudnnBackendDestroyDescriptor(cand[i].second);
  }
  if (plan_[op]) {
    int64_t wsz = 0, got = 0;
    CUDNN_CHECK(cudnnBackendGetAttribute(plan_[op], CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &got,
                                         &wsz));
    need = size_t(wsz);
    plan_beta_[op] = beta;
  }
  cudnnBackendDestroyDescriptor(vp);
  for (auto d : cfgs) cudnnBackendDestroyDescriptor(d);
  for (auto d : {og, od, cv, IN, OUT, W}) cudnnBackendDestroyDescriptor(d);
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  return best;
}

void ConvOp::be_exec(int op, const void* in, const void* out, const void* w, void* ws, float beta) const {
  FSO_ASSERT(beta == plan_beta_[op], "conv backend plan beta %g, called with %g", plan_beta_[op], beta);
  auto vp = be_make(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
  void* ptrs[3] = {const_cast<void*>(in), const_cast<void*>(out), const_cast<void*>(w)};
  int64_t uids[3] = {kUidIn, kUidOut, kUidW};
  be_set(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs);
  be_set(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
  be_set(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws);
  CUDNN_CHECK(cudnnBackendFinalize(vp));
  CUDNN_CHECK(cudnnBackendExecute(h_, plan_[op], vp));
  cudnnBackendDestroyDescriptor(vp);
}

size_t ConvOp::tune_fwd(void* x, void* w, void* y, void* ws, size_t ws_limit) {
  FSO_ASSERT(!tr_, "tune_fwd: transposed");
  int n = 0;
  cudnnConvolutionFwdAlgoPerf_t p[16];
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(h_, xd_, x, wd_, w, cd_, yd_, y, 16, &n, p, ws, ws_limit));
  int i = 0;
  while (i < n && p[i].status != CUDNN_STATUS_SUCCESS) i++;
  FSO_ASSERT(i < n, "no fwd algo");
  fa_ = p[i].algo;
  tuned_ms[0] = p[i].time;
  return p[i].memory;
}

void ConvOp::fwd_v(const void* x, const void* w, void* y, void* ws, size_t wsz) const {
  float a = 1.f, b = 0.f;
  CUDNN_CHECK(cudnnConvolutionForward(h_, &a, xd_, x, wd_, w, cd_, fa_, ws, wsz, &b, yd_, y));
}

void ConvOp::fwd(const act_t* x, const act_t* w, act_t* y, void* ws, size_t wsz, float beta) const {
  float a = 1.f;
  if (!tr_) {
    if (plan_[0]) return be_exec(0, x, y, w, ws, beta);
    CUDNN_CHECK(cudnnConvolutionForward(h_, &a, xd_, x, wd_, w, cd_, fa_, ws, wsz, &beta, yd_, y));
  } else {  // y(big) = 내부 conv 의 입력 기울기 = BackwardData(x(small))
    if (plan_[1]) return be_exec(1, y, x, w, ws, beta);
    CUDNN_CHECK(cudnnConvolutionBackwardData(h_, &a, wd_, w, xd_, x, cd_, da_, ws, wsz, &beta, yd_, y));
  }
}

void ConvOp::dgrad(const act_t* dy, const act_t* w, act_t* dx, void* ws, size_t wsz, float beta) const {
  float a = 1.f;
  if (!tr_) {
    if (plan_[1]) return be_exec(1, dx, dy, w, ws, beta);
    CUDNN_CHECK(cudnnConvolutionBackwardData(h_, &a, wd_, w, yd_, dy, cd_, da_, ws, wsz, &beta, xd_, dx));
  } else {  // dx(small) = 내부 conv 정방향(dy(big))
    if (plan_[0]) return be_exec(0, dy, dx, w, ws, beta);
    CUDNN_CHECK(cudnnConvolutionForward(h_, &a, yd_, dy, wd_, w, cd_, fa_, ws, wsz, &beta, xd_, dx));
  }
}

void ConvOp::wgrad(const act_t* x, const act_t* dy, act_t* dw, void* ws, size_t wsz, float beta) const {
  float a = 1.f;
  if (!tr_) {
    if (plan_[2]) return be_exec(2, x, dy, dw, ws, beta);
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(h_, &a, xd_, x, yd_, dy, cd_, wa_, ws, wsz, &beta, wd_, dw));
  } else {
    if (plan_[2]) return be_exec(2, dy, x, dw, ws, beta);
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(h_, &a, yd_, dy, xd_, x, cd_, wa_, ws, wsz, &beta, wd_, dw));
  }
}
