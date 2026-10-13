#include "augment.cuh"

namespace {

constexpr int kT = 256;

// 캔버스(2S×2S) 화소: 타일 안이면 원본 화소, 밖이면 114(Ultralytics mosaic 채움값과 warpAffine borderValue)
__device__ __forceinline__ void canvas_px(const SampleInfo& sm, const SrcInfo* src, const uint8_t* img, int S, int px,
                                          int py, float c[3]) {
  c[0] = c[1] = c[2] = 114.f;
  if (px < 0 || py < 0 || px >= 2 * S || py >= 2 * S) return;
#pragma unroll
  for (int t = 0; t < 4; t++) {
    int s = sm.src[t];
    if (s < 0) continue;
    if (px >= sm.ta[t][0] && px < sm.ta[t][2] && py >= sm.ta[t][1] && py < sm.ta[t][3]) {
      const SrcInfo& si = src[s];
      int x = px - sm.ta[t][0] + sm.tb[t][0], y = py - sm.ta[t][1] + sm.tb[t][1];
      const uint8_t* p = img + si.img_off + (int64_t(y) * si.iw + x) * 3;
      c[0] = p[0]; c[1] = p[1]; c[2] = p[2];
      return;
    }
  }
}

__device__ __forceinline__ int rnd_u8(float v) { return min(255, max(0, __float2int_rn(v))); }

// OpenCV RGB(u8) → HSV(u8, H 0..179)
__device__ void rgb2hsv(int r, int g, int b, int& h, int& s, int& v) {
  int mx = max(r, max(g, b)), mn = min(r, min(g, b));
  v = mx;
  int d = mx - mn;
  s = mx ? __float2int_rn(255.f * d / mx) : 0;
  float hh = 0.f;
  if (d) {
    if (mx == r) hh = 60.f * (g - b) / d;
    else if (mx == g) hh = 120.f + 60.f * (b - r) / d;
    else hh = 240.f + 60.f * (r - g) / d;
    if (hh < 0) hh += 360.f;
  }
  h = __float2int_rn(hh * 0.5f);
  if (h >= 180) h -= 180;
}

// OpenCV HSV(u8) → RGB(u8)
__device__ void hsv2rgb(int h, int s, int v, int& r, int& g, int& b) {
  float H = h * 2.f / 60.f, S_ = s / 255.f, V = float(v);
  int i = int(floorf(H)) % 6;
  float f = H - floorf(H);
  float p = V * (1 - S_), q = V * (1 - S_ * f), t = V * (1 - S_ * (1 - f));
  float R, G, B;
  switch (i) {
    case 0: R = V; G = t; B = p; break;
    case 1: R = q; G = V; B = p; break;
    case 2: R = p; G = V; B = t; break;
    case 3: R = p; G = q; B = V; break;
    case 4: R = t; G = p; B = V; break;
    default: R = V; G = p; B = q; break;
  }
  r = rnd_u8(R); g = rnd_u8(G); b = rnd_u8(B);
}

__global__ void k_image(BatchBuf d, int S, act_t* out) {
  int i = blockIdx.y;
  int64_t q = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= int64_t(S) * S) return;
  int x = int(q % S), y = int(q / S);
  const SampleInfo sm = d.smp[i];
  int xs = sm.flip ? S - 1 - x : x;
  // warpAffine: dst(x,y) = src(M⁻¹·(x,y)), 쌍선형, 화소 중심 = 정수 좌표
  float u = (xs - sm.tx) / sm.s, v = (y - sm.ty) / sm.s;
  int x0 = int(floorf(u)), y0 = int(floorf(v));
  float fx = u - x0, fy = v - y0;
  float a[3], b[3], c[3], e[3];
  canvas_px(sm, d.src, d.img, S, x0, y0, a);
  canvas_px(sm, d.src, d.img, S, x0 + 1, y0, b);
  canvas_px(sm, d.src, d.img, S, x0, y0 + 1, c);
  canvas_px(sm, d.src, d.img, S, x0 + 1, y0 + 1, e);
  int rgb[3];
  for (int k = 0; k < 3; k++)
    rgb[k] = rnd_u8((a[k] * (1 - fx) + b[k] * fx) * (1 - fy) + (c[k] * (1 - fx) + e[k] * fx) * fy);
  // RandomHSV: LUT hue = (x·r0) % 180, sat = clip(x·r1), val = clip(x·r2) (u8 로 자름)
  int h, s, vv;
  rgb2hsv(rgb[0], rgb[1], rgb[2], h, s, vv);
  h = int(fmodf(h * sm.hsv[0], 180.f));
  s = int(fminf(fmaxf(s * sm.hsv[1], 0.f), 255.f));
  vv = int(fminf(fmaxf(vv * sm.hsv[2], 0.f), 255.f));
  hsv2rgb(h, s, vv, rgb[0], rgb[1], rgb[2]);
  act_t* o = out + ((int64_t(i) * S + y) * S + x) * 3;
  o[0] = from_f<act_t>(rgb[0] / 255.f);
  o[1] = from_f<act_t>(rgb[1] / 255.f);
  o[2] = from_f<act_t>(rgb[2] / 255.f);
}

// 블록 안 순서 보존 압축: keep 인 스레드의 (블록 앞 스레드들 중 keep 수), 전체 keep 수. blockDim 은 32 의 배수.
__device__ __forceinline__ int block_rank(bool keep, int& total) {
  __shared__ int wsum[32];
  unsigned m = __ballot_sync(~0u, keep);
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) wsum[w] = __popc(m);
  __syncthreads();
  int off = 0;
  total = 0;
  for (int q = 0; q < int(blockDim.x >> 5); q++) {
    if (q < w) off += wsum[q];
    total += wsum[q];
  }
  __syncthreads();
  return off + __popc(m & ((1u << lane) - 1u));
}

// 표본마다 정답 상자: 원본 → 캔버스(타일로 자름) → affine → 출력(0..S 로 자름) → 뒤집기 → 후보 거르기.
// 블록 하나 = 표본 하나. 인스턴스를 blockDim 개씩 나눠 보고, 원래 순서대로 앞에서 G 개까지.
__global__ void k_targets(BatchBuf d, int S, AugOut o) {
  const int i = blockIdx.x;
  const SampleInfo sm = d.smp[i];
  const int ninst = d.hdr->ninst;
  int n = 0;
  for (int k0 = 0; k0 < ninst && n < o.G; k0 += blockDim.x) {
    const int k = k0 + threadIdx.x;
    bool keep = false;
    float X1 = 0.f, Y1 = 0.f, X2 = 0.f, Y2 = 0.f;
    if (k < ninst) {
      const InstInfo in = d.inst[k];
      if (in.sample == i && !in.ignore) {
        int t = in.tile;
        float dx = float(sm.ta[t][0] - sm.tb[t][0]), dy = float(sm.ta[t][1] - sm.tb[t][1]);
        float x1 = fmaxf(in.box[0] + dx, sm.ta[t][0]), y1 = fmaxf(in.box[1] + dy, sm.ta[t][1]);
        float x2 = fminf(in.box[2] + dx, sm.ta[t][2]), y2 = fminf(in.box[3] + dy, sm.ta[t][3]);
        if (x2 > x1 && y2 > y1) {
          float w1 = (x2 - x1) * sm.s, h1 = (y2 - y1) * sm.s;
          X1 = fminf(fmaxf(sm.s * x1 + sm.tx, 0.f), float(S)); X2 = fminf(fmaxf(sm.s * x2 + sm.tx, 0.f), float(S));
          Y1 = fminf(fmaxf(sm.s * y1 + sm.ty, 0.f), float(S)); Y2 = fminf(fmaxf(sm.s * y2 + sm.ty, 0.f), float(S));
          if (sm.flip) { float a = S - X2, b = S - X1; X1 = a; X2 = b; }
          float w2 = X2 - X1, h2 = Y2 - Y1;
          float ar = fmaxf(w2 / (h2 + 1e-16f), h2 / (w2 + 1e-16f));
          keep = w2 > 2.f && h2 > 2.f && w2 * h2 / (w1 * h1 + 1e-16f) > 0.1f && ar < 100.f;
        }
      }
    }
    int total;
    int slot = n + block_rank(keep, total);
    if (keep && slot < o.G) {
      o.gt.box[i * o.G + slot] = make_float4(X1, Y1, X2, Y2);
      o.gt_inst[i * o.G + slot] = k;
    }
    n += total;
  }
  if (threadIdx.x == 0) o.gt.n_gt[i] = min(n, o.G);
}

// 정답 마스크 비트(Hp×Wp, 칸 = 출력 4×4 화소): 칸 중심을 거꾸로 따라가 원본 마스크에서 집어 온다.
// 블록 하나 = (표본, 정답) 하나, 스레드 하나 = 한 행의 u32 한 낱말(32 칸).
__global__ void k_gtmask(BatchBuf d, int S, AugOut o) {
  int i = blockIdx.y, j = blockIdx.x;
  if (j >= o.gt.n_gt[i]) return;
  const SampleInfo sm = d.smp[i];
  const InstInfo in = d.inst[o.gt_inst[i * o.G + j]];
  const SrcInfo si = d.src[in.src];
  int t = in.tile;
  float4 bx = o.gt.box[i * o.G + j];
  float cs = float(S) / o.Wp;  // 칸 크기(화소)
  int c0 = max(0, int(floorf(bx.x / cs)) - 1), c1 = min(o.Wp, int(ceilf(bx.z / cs)) + 1);
  int r0 = max(0, int(floorf(bx.y / cs)) - 1), r1 = min(o.Hp, int(ceilf(bx.w / cs)) + 1);
  int w0 = c0 >> 5, w1 = (c1 + 31) >> 5, nw = w1 - w0;
  int words = o.Hp * o.Wp / 32, wpr = o.Wp / 32;
  uint32_t* dst = o.gt.mask + (int64_t(i) * o.G + j) * words;
  float kx = float(si.mw) / si.iw, ky = float(si.mh) / si.ih;
  for (int q = threadIdx.x; q < nw * (r1 - r0); q += blockDim.x) {
    int r = r0 + q / nw, w = w0 + q % nw;
    uint32_t bits = 0;
    for (int b = 0; b < 32; b++) {
      int c = w * 32 + b;
      if (c < c0 || c >= c1) continue;
      float x = (c + 0.5f) * cs - 0.5f, y = (r + 0.5f) * cs - 0.5f;   // 출력 화소 좌표(칸 중심)
      float xs = sm.flip ? S - 1 - x : x;
      float u = (xs - sm.tx) / sm.s, v = (y - sm.ty) / sm.s;          // 캔버스
      if (u < sm.ta[t][0] || u >= sm.ta[t][2] || v < sm.ta[t][1] || v >= sm.ta[t][3]) continue;  // 타일 밖
      float sx = u - sm.ta[t][0] + sm.tb[t][0], sy = v - sm.ta[t][1] + sm.tb[t][1];             // 원본 화소
      int mx = int(floorf((sx + 0.5f) * kx)) - in.mx0, my = int(floorf((sy + 0.5f) * ky)) - in.my0;
      if (mx < 0 || my < 0 || mx >= in.mcw || my >= in.mch) continue;
      uint32_t wd = d.mask[in.mask_off + int64_t(my) * in.wpr + (mx >> 5)];
      if (wd >> (mx & 31) & 1u) bits |= 1u << b;
    }
    dst[r * wpr + w] = bits;
  }
}

// 무시 영역 비트(Hp×Wp): 표본마다 무시 인스턴스를 정답 마스크와 같은 규칙으로 옮긴다. 블록 하나 = 표본 하나.
// 이 표본의 무시 인스턴스를 blockDim 개씩 공유 메모리 목록으로 모은 뒤 하나씩 블록 전체가 칸을 나눠 본다.
__global__ void k_ignore(BatchBuf d, int S, AugOut o) {
  int i = blockIdx.x;
  const SampleInfo sm = d.smp[i];
  int ninst = d.hdr->ninst, words = o.Hp * o.Wp / 32;
  float cs = float(S) / o.Wp;
  uint32_t* dst = o.gt.ignore + int64_t(i) * words;
  __shared__ int lst[1024];
  for (int k0 = 0; k0 < ninst; k0 += blockDim.x) {
    const int kk = k0 + threadIdx.x;
    bool hit = false;
    if (kk < ninst) {
      const InstInfo& in = d.inst[kk];
      hit = in.sample == i && in.ignore;
    }
    int nl;
    int slot = block_rank(hit, nl);
    if (hit) lst[slot] = kk;
    __syncthreads();
    for (int li = 0; li < nl; li++) {
      const InstInfo in = d.inst[lst[li]];
      const SrcInfo si = d.src[in.src];
      int t = in.tile;
      float dx = float(sm.ta[t][0] - sm.tb[t][0]), dy = float(sm.ta[t][1] - sm.tb[t][1]);
      float x1 = fmaxf(in.box[0] + dx, sm.ta[t][0]), y1 = fmaxf(in.box[1] + dy, sm.ta[t][1]);
      float x2 = fminf(in.box[2] + dx, sm.ta[t][2]), y2 = fminf(in.box[3] + dy, sm.ta[t][3]);
      if (x2 <= x1 || y2 <= y1) continue;
      float X1 = fminf(fmaxf(sm.s * x1 + sm.tx, 0.f), float(S)), X2 = fminf(fmaxf(sm.s * x2 + sm.tx, 0.f), float(S));
      float Y1 = fminf(fmaxf(sm.s * y1 + sm.ty, 0.f), float(S)), Y2 = fminf(fmaxf(sm.s * y2 + sm.ty, 0.f), float(S));
      if (sm.flip) { float a = S - X2, b = S - X1; X1 = a; X2 = b; }
      int c0 = max(0, int(floorf(X1 / cs)) - 1), c1 = min(o.Wp, int(ceilf(X2 / cs)) + 1);
      int r0 = max(0, int(floorf(Y1 / cs)) - 1), r1 = min(o.Hp, int(ceilf(Y2 / cs)) + 1);
      float kx = float(si.mw) / si.iw, ky = float(si.mh) / si.ih;
      for (int q = threadIdx.x; q < (c1 - c0) * (r1 - r0); q += blockDim.x) {
        int r = r0 + q / (c1 - c0), c = c0 + q % (c1 - c0);
        float x = (c + 0.5f) * cs - 0.5f, y = (r + 0.5f) * cs - 0.5f;
        float xs = sm.flip ? S - 1 - x : x;
        float u = (xs - sm.tx) / sm.s, v = (y - sm.ty) / sm.s;
        if (u < sm.ta[t][0] || u >= sm.ta[t][2] || v < sm.ta[t][1] || v >= sm.ta[t][3]) continue;
        float sx = u - sm.ta[t][0] + sm.tb[t][0], sy = v - sm.ta[t][1] + sm.tb[t][1];
        int mx = int(floorf((sx + 0.5f) * kx)) - in.mx0, my = int(floorf((sy + 0.5f) * ky)) - in.my0;
        if (mx < 0 || my < 0 || mx >= in.mcw || my >= in.mch) continue;
        if (d.mask[in.mask_off + int64_t(my) * in.wpr + (mx >> 5)] >> (mx & 31) & 1u) {
          int64_t bit = int64_t(r) * o.Wp + c;
          atomicOr(&dst[bit >> 5], 1u << (bit & 31));
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace

void augment(const BatchBuf& dev, int B, int S, const AugOut& o, cudaStream_t st) {
  CUDA_CHECK(cudaMemsetAsync(o.gt.mask, 0, size_t(B) * o.G * o.Hp * o.Wp / 8, st));
  k_image<<<dim3(cdiv(int64_t(S) * S, kT), B), kT, 0, st>>>(dev, S, o.img);
  k_targets<<<B, 256, 0, st>>>(dev, S, o);
  k_gtmask<<<dim3(o.G, B), 128, 0, st>>>(dev, S, o);
  if (o.gt.ignore) {
    CUDA_CHECK(cudaMemsetAsync(o.gt.ignore, 0, size_t(B) * o.Hp * o.Wp / 8, st));
    k_ignore<<<B, 256, 0, st>>>(dev, S, o);
  }
}
