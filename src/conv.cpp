#include "conv.h"

#include <algorithm>

namespace {
void nhwc(cudnnTensorDescriptor_t d, cudnnDataType_t dt, int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NHWC, dt, n, c, h, w));
}
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

size_t ConvOp::tune(act_t* x, act_t* w, act_t* y, act_t* dx, act_t* dw, act_t* dy, void* ws, size_t ws_limit) {
  int n = 0;
  size_t need = 0;
  // 내부 conv 의 (입력, 출력) 디스크립터: 보통은 (x, y), 전치면 (y=big, x=small).
  cudnnTensorDescriptor_t ind = tr_ ? yd_ : xd_, outd = tr_ ? xd_ : yd_;
  act_t* in = tr_ ? y : x;   // 내부 conv 입력 버퍼
  act_t* out = tr_ ? x : y;  // 내부 conv 출력 버퍼
  act_t* din = tr_ ? dy : dx;
  act_t* dout = tr_ ? dx : dy;
  {
    cudnnConvolutionFwdAlgoPerf_t p[16];
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(h_, ind, in, wd_, w, cd_, outd, out, 16, &n, p, ws, ws_limit));
    int i = 0;
    while (i < n && p[i].status != CUDNN_STATUS_SUCCESS) i++;
    FSO_ASSERT(i < n, "no fwd algo");
    fa_ = p[i].algo;
    need = std::max(need, p[i].memory);
    tuned_ms[tr_ ? 1 : 0] = p[i].time;
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
    tuned_ms[tr_ ? 0 : 1] = p[i].time;
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
    tuned_ms[2] = p[i].time;
  }
  return need;
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
  if (!tr_)
    CUDNN_CHECK(cudnnConvolutionForward(h_, &a, xd_, x, wd_, w, cd_, fa_, ws, wsz, &beta, yd_, y));
  else  // y(big) = 내부 conv 의 입력 기울기 = BackwardData(x(small))
    CUDNN_CHECK(cudnnConvolutionBackwardData(h_, &a, wd_, w, xd_, x, cd_, da_, ws, wsz, &beta, yd_, y));
}

void ConvOp::dgrad(const act_t* dy, const act_t* w, act_t* dx, void* ws, size_t wsz, float beta) const {
  float a = 1.f;
  if (!tr_)
    CUDNN_CHECK(cudnnConvolutionBackwardData(h_, &a, wd_, w, yd_, dy, cd_, da_, ws, wsz, &beta, xd_, dx));
  else  // dx(small) = 내부 conv 정방향(dy(big))
    CUDNN_CHECK(cudnnConvolutionForward(h_, &a, yd_, dy, wd_, w, cd_, fa_, ws, wsz, &beta, xd_, dx));
}

void ConvOp::wgrad(const act_t* x, const act_t* dy, act_t* dw, void* ws, size_t wsz, float beta) const {
  float a = 1.f;
  if (!tr_)
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(h_, &a, xd_, x, yd_, dy, cd_, wa_, ws, wsz, &beta, wd_, dw));
  else
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(h_, &a, yd_, dy, xd_, x, cd_, wa_, ws, wsz, &beta, wd_, dw));
}
