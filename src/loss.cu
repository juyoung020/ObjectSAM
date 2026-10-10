#include "loss.cuh"

namespace {

constexpr int kT = 256;
constexpr float kCiouEps = 1e-7f;  // bbox_iou eps
constexpr float kTalEps = 1e-9f;

struct Lv {
  int off[4];
  int h[3], w[3], s[3];
};

__device__ __forceinline__ void anchor_of(const Lv& L, int a, int& l, int& h, int& w) {
  l = a < L.off[1] ? 0 : (a < L.off[2] ? 1 : 2);
  int r = a - L.off[l];
  h = r / L.w[l];
  w = r - h * L.w[l];
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float bce_logits(float x, float y) { return fmaxf(x, 0.f) - x * y + log1pf(expf(-fabsf(x))); }

// Ultralytics bbox_iou(box1, box2, xywh=False, CIoU=True). alpha 는 상수(no_grad) 취급.
__device__ float ciou(float4 b1, float4 b2) {
  float w1 = b1.z - b1.x, h1 = b1.w - b1.y + kCiouEps, w2 = b2.z - b2.x, h2 = b2.w - b2.y + kCiouEps;
  float iw = fmaxf(fminf(b1.z, b2.z) - fmaxf(b1.x, b2.x), 0.f), ih = fmaxf(fminf(b1.w, b2.w) - fmaxf(b1.y, b2.y), 0.f);
  float inter = iw * ih, uni = w1 * h1 + w2 * h2 - inter + kCiouEps, iou = inter / uni;
  float cw = fmaxf(b1.z, b2.z) - fminf(b1.x, b2.x), ch = fmaxf(b1.w, b2.w) - fminf(b1.y, b2.y);
  float c2 = cw * cw + ch * ch + kCiouEps;
  float sx = b2.x + b2.z - b1.x - b1.z, sy = b2.y + b2.w - b1.y - b1.w;
  float rho2 = (sx * sx + sy * sy) * 0.25f;
  float dv = atanf(w2 / h2) - atanf(w1 / h1);
  float v = 0.40528473456935108f * dv * dv;  // 4/π²
  float alpha = v / (1.f - iou + v + kCiouEps);
  return iou - (rho2 / c2 + v * alpha);
}

// 위와 같은 값 + d ciou / d box1 (box1 = 예측).
__device__ float ciou_grad(float4 b1, float4 b2, float4& g) {
  float w1 = b1.z - b1.x, h1 = b1.w - b1.y + kCiouEps, w2 = b2.z - b2.x, h2 = b2.w - b2.y + kCiouEps;
  float ix2 = fminf(b1.z, b2.z), ix1 = fmaxf(b1.x, b2.x), iy2 = fminf(b1.w, b2.w), iy1 = fmaxf(b1.y, b2.y);
  float iw_ = ix2 - ix1, ih_ = iy2 - iy1;
  float iw = fmaxf(iw_, 0.f), ih = fmaxf(ih_, 0.f);
  float inter = iw * ih, uni = w1 * h1 + w2 * h2 - inter + kCiouEps, iou = inter / uni;
  float cx2 = fmaxf(b1.z, b2.z), cx1 = fminf(b1.x, b2.x), cy2 = fmaxf(b1.w, b2.w), cy1 = fminf(b1.y, b2.y);
  float cw = cx2 - cx1, ch = cy2 - cy1;
  float c2 = cw * cw + ch * ch + kCiouEps;
  float sx = b2.x + b2.z - b1.x - b1.z, sy = b2.y + b2.w - b1.y - b1.w;
  float rho2 = (sx * sx + sy * sy) * 0.25f;
  float r1 = w1 / h1;
  float dv = atanf(w2 / h2) - atanf(r1);
  const float k4 = 0.40528473456935108f;
  float v = k4 * dv * dv;
  float alpha = v / (1.f - iou + v + kCiouEps);
  float out = iou - (rho2 / c2 + v * alpha);
  // 역전파 (d out = 1)
  float dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0, dw1 = 0, dh1 = 0;
  // iou = inter / (A1 + A2 - inter + eps)
  float dinter = (uni + inter) / (uni * uni);
  float dA1 = -inter / (uni * uni);
  dw1 += dA1 * h1;
  dh1 += dA1 * w1;
  float diw = iw_ > 0.f ? dinter * ih : 0.f, dih = ih_ > 0.f ? dinter * iw : 0.f;
  if (b1.z < b2.z) dx2 += diw;
  if (b1.x > b2.x) dx1 -= diw;
  if (b1.w < b2.w) dy2 += dih;
  if (b1.y > b2.y) dy1 -= dih;
  // - rho2 / c2
  float drho2 = -1.f / c2, dc2 = rho2 / (c2 * c2);
  dx1 += drho2 * (-sx * 0.5f);
  dx2 += drho2 * (-sx * 0.5f);
  dy1 += drho2 * (-sy * 0.5f);
  dy2 += drho2 * (-sy * 0.5f);
  float dcw = dc2 * 2.f * cw, dch = dc2 * 2.f * ch;
  if (b1.z > b2.z) dx2 += dcw;
  if (b1.x < b2.x) dx1 -= dcw;
  if (b1.w > b2.w) dy2 += dch;
  if (b1.y < b2.y) dy1 -= dch;
  // - alpha * v,  v = k4 (atan(w2/h2) - atan(w1/h1))²
  float dr1 = -alpha * k4 * 2.f * dv * (-1.f / (1.f + r1 * r1));
  dw1 += dr1 / h1;
  dh1 += dr1 * (-w1 / (h1 * h1));
  dx2 += dw1;
  dx1 -= dw1;
  dy2 += dh1;
  dy1 -= dh1;
  g = make_float4(dx1, dy1, dx2, dy2);
  return out;
}

// 1) 해독: 예측 상자(격자 단위)와 점수
struct DecArgs {
  const act_t* box[3];
  const act_t* cls[3];
};
template <int kReg>  // DFL 칸 수(reg_max)
__global__ void k_decode(Lv L, int A, int B, DecArgs io, float4* pbox, float* score) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= B * A) return;
  int b = q / A, a = q - b * A, l, h, w;
  anchor_of(L, a, l, h, w);
  int64_t pix = (int64_t(b) * L.h[l] + h) * L.w[l] + w;
  const act_t* p = io.box[l] + pix * 4 * kReg;
  float d[4];
  for (int s = 0; s < 4; s++) {
    float m = -INFINITY, v[kReg];
    for (int k = 0; k < kReg; k++) { v[k] = to_f(p[s * kReg + k]); m = fmaxf(m, v[k]); }
    float z = 0, e = 0;
    for (int k = 0; k < kReg; k++) { float t = expf(v[k] - m); z += t; e += t * k; }
    d[s] = e / z;
  }
  float ax = w + 0.5f, ay = h + 0.5f;
  pbox[q] = make_float4(ax - d[0], ay - d[1], ax + d[2], ay + d[3]);
  score[q] = sigm(to_f(io.cls[l][pix]));
}

__device__ __forceinline__ float4 enlarge(float4 g) {  // select_candidates_in_gts: 변이 16(stride[1]) 미만이면 키움
  float cx = (g.x + g.z) * 0.5f, cy = (g.y + g.w) * 0.5f, w = g.z - g.x, h = g.w - g.y;
  if (w < 16.f) w = 16.f;
  if (h < 16.f) h = 16.f;
  return make_float4(cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f);
}
__device__ __forceinline__ bool inside(float4 e, float px, float py) {
  return px - e.x > kTalEps && py - e.y > kTalEps && e.z - px > kTalEps && e.w - py > kTalEps;
}
__device__ __forceinline__ float metric_of(float score, float ov) {
  float o2 = ov * ov;
  return sqrtf(score) * o2 * o2 * o2;
}

// 2) 정답마다 상자 안 앵커 중 정렬 지표 top-k
__global__ void k_topk(Lv L, int A, int G, int topk, const int* n_gt, const float4* gtb, const float4* pbox,
                       const float* score, int* cnt, int* who, float* galign, float* govl, int* npos) {
  int b = blockIdx.x / G, j = blockIdx.x - b * G;
  if (threadIdx.x == 0) {
    galign[blockIdx.x] = 0.f;
    govl[blockIdx.x] = 0.f;
    if (j == 0) npos[b] = 0;
  }
  if (j >= n_gt[b]) return;
  float4 g = gtb[blockIdx.x];
  float4 e = enlarge(g);
  constexpr int K = 10;
  float tv[K];
  int ti[K];
  for (int k = 0; k < K; k++) { tv[k] = -1.f; ti[k] = -1; }
  for (int l = 0; l < 3; l++) {
    float s = L.s[l];
    int w0 = max(0, int(floorf(e.x / s - 0.5f))), w1 = min(L.w[l] - 1, int(ceilf(e.z / s - 0.5f)));
    int h0 = max(0, int(floorf(e.y / s - 0.5f))), h1 = min(L.h[l] - 1, int(ceilf(e.w / s - 0.5f)));
    int nw = w1 - w0 + 1, nh = h1 - h0 + 1;
    if (nw <= 0 || nh <= 0) continue;
    for (int t = threadIdx.x; t < nw * nh; t += blockDim.x) {
      int hh = h0 + t / nw, ww = w0 + t % nw;
      if (!inside(e, (ww + 0.5f) * s, (hh + 0.5f) * s)) continue;
      int a = L.off[l] + hh * L.w[l] + ww;
      int q = b * A + a;
      float4 pb = pbox[q];
      float4 pp = make_float4(pb.x * s, pb.y * s, pb.z * s, pb.w * s);
      float ov = fmaxf(ciou(g, pp), 0.f);
      float m = metric_of(score[q], ov);
      // 지표 내림차순, 같으면 앵커 번호 오름차순
      if (m > tv[K - 1] || (m == tv[K - 1] && (ti[K - 1] < 0 || a < ti[K - 1]))) {
        int k = K - 1;
        while (k > 0 && (m > tv[k - 1] || (m == tv[k - 1] && a < ti[k - 1]))) { tv[k] = tv[k - 1]; ti[k] = ti[k - 1]; k--; }
        tv[k] = m;
        ti[k] = a;
      }
    }
  }
  // 블록 병합: topk 번 최댓값 뽑기
  __shared__ float sv[kT];
  __shared__ int si[kT], st_[kT];
  int head = 0;
  for (int r = 0; r < topk; r++) {
    float v = head < K ? tv[head] : -1.f;
    int i = head < K ? ti[head] : -1;
    sv[threadIdx.x] = i < 0 ? -1.f : v;
    si[threadIdx.x] = i;
    st_[threadIdx.x] = threadIdx.x;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
      if (threadIdx.x < o) {
        float v2 = sv[threadIdx.x + o];
        int i2 = si[threadIdx.x + o];
        bool take = i2 >= 0 && (si[threadIdx.x] < 0 || v2 > sv[threadIdx.x] || (v2 == sv[threadIdx.x] && i2 < si[threadIdx.x]));
        if (take) { sv[threadIdx.x] = v2; si[threadIdx.x] = i2; st_[threadIdx.x] = st_[threadIdx.x + o]; }
      }
      __syncthreads();
    }
    int win = si[0], wt = st_[0];
    __syncthreads();
    if (win < 0) break;
    if (threadIdx.x == wt) head++;
    if (threadIdx.x == 0) {
      atomicAdd(&cnt[b * A + win], 1);
      who[b * A + win] = j;
    }
  }
}

// 3) 앵커별 충돌 해소 → 최종 정답, 정렬 지표, 겹침; 정답별 최대; 양성 목록
__global__ void k_resolve(Lv L, int A, int B, int G, const int* n_gt, const float4* gtb, const float4* pbox,
                          const float* score, int* cnt, const int* who, int* agt, float* amet, float* aovl,
                          float* galign, float* govl, int* pos, int* npos, float* items) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= B * A) return;
  int c = cnt[q];
  if (c == 0) { agt[q] = -1; return; }
  cnt[q] = 0;
  int b = q / A, a = q - b * A, l, h, w;
  anchor_of(L, a, l, h, w);
  float s = L.s[l], px = (w + 0.5f) * s, py = (h + 0.5f) * s;
  float4 pb = pbox[q];
  float4 pp = make_float4(pb.x * s, pb.y * s, pb.z * s, pb.w * s);
  int j = who[q];
  int ng = n_gt[b];
  if (c > 1) {  // 겹침이 가장 큰 정답(첫 최대). 상자 밖은 겹침 0.
    float best = -1.f;
    for (int k = 0; k < ng; k++) {
      float4 g = gtb[b * G + k];
      float ov = inside(enlarge(g), px, py) ? fmaxf(ciou(g, pp), 0.f) : 0.f;
      if (ov > best) { best = ov; j = k; }
    }
  }
  float4 g = gtb[b * G + j];
  float ov = 0.f, m = 0.f;
  if (inside(enlarge(g), px, py)) {
    ov = fmaxf(ciou(g, pp), 0.f);
    m = metric_of(score[q], ov);
  }
  agt[q] = j;
  amet[q] = m;
  aovl[q] = ov;
  atomicMax(reinterpret_cast<int*>(&galign[b * G + j]), __float_as_int(m));
  atomicMax(reinterpret_cast<int*>(&govl[b * G + j]), __float_as_int(ov));
  int k = atomicAdd(&npos[b], 1);
  pos[b * A + k] = a;
  atomicAdd(&items[4], 1.f);
}

// 4a) 양성 앵커의 목표 점수와 합
__global__ void k_tsum(int A, int B, int G, const int* npos, const int* pos, const int* agt, const float* amet,
                       const float* galign, const float* govl, float* tsc, float* items) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  float t = 0.f;
  if (q < B * A) {
    int b = q / A, i = q - b * A;
    if (i < npos[b]) {
      int a = pos[b * A + i], qa = b * A + a, j = agt[qa];
      t = amet[qa] * govl[b * G + j] / (galign[b * G + j] + kTalEps);
      tsc[qa] = t;
    }
  }
  for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
  if ((threadIdx.x & 31) == 0 && t != 0.f) atomicAdd(&items[5], t);
}

__device__ __forceinline__ void block_add(float v, float* dst) {
  __shared__ float red[kT / 32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0 && v != 0.f) atomicAdd(dst, v);
  }
  __syncthreads();
}

// 4b) 점수 BCE(전 앵커), 상자 CIoU·DFL(양성) — 손실과 머리 기울기
struct DetArgs {
  Lv L;
  int A, B, G, imgsz;
  const act_t* box[3];
  const act_t* cls[3];
  act_t* dbox[3];
  act_t* dcls[3];
  act_t* dcoef[3];
  const int* agt;
  const float* tsc;
  const float4* gtb;
  float* items;
  const float* scale;
  float box_gain, cls_gain, dfl_gain;
};
template <int kReg>
__global__ void k_det(DetArgs P) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  float lcls = 0, liou = 0, ldfl = 0;
  if (q < P.B * P.A) {
    int b = q / P.A, a = q - b * P.A, l, h, w;
    anchor_of(P.L, a, l, h, w);
    int64_t pix = (int64_t(b) * P.L.h[l] + h) * P.L.w[l] + w;
    float tss = fmaxf(P.items[5], 1.f);
    float sc = float(P.B) / tss * *P.scale;
    int j = P.agt[q];
    float t = j >= 0 ? P.tsc[q] : 0.f;
    float x = to_f(P.cls[l][pix]);
    lcls = bce_logits(x, t);
    P.dcls[l][pix] = from_f<act_t>((sigm(x) - t) * P.cls_gain * sc);
    act_t* db = P.dbox[l] + pix * 4 * kReg;
    if (j < 0) {
      for (int k = 0; k < 4 * kReg; k++) db[k] = from_f<act_t>(0.f);
      act_t* dc = P.dcoef[l] + pix * 32;
      for (int k = 0; k < 32; k++) dc[k] = from_f<act_t>(0.f);
    } else {
      const act_t* bl = P.box[l] + pix * 4 * kReg;
      float pr[4][kReg], dist[4];
      for (int s = 0; s < 4; s++) {
        float m = -INFINITY;
        for (int k = 0; k < kReg; k++) { pr[s][k] = to_f(bl[s * kReg + k]); m = fmaxf(m, pr[s][k]); }
        float z = 0;
        for (int k = 0; k < kReg; k++) { pr[s][k] = expf(pr[s][k] - m); z += pr[s][k]; }
        float e = 0;
        for (int k = 0; k < kReg; k++) { pr[s][k] /= z; e += pr[s][k] * k; }
        dist[s] = e;
      }
      float s_ = float(P.L.s[l]);
      float ax = w + 0.5f, ay = h + 0.5f;
      float4 pb = make_float4(ax - dist[0], ay - dist[1], ax + dist[2], ay + dist[3]);
      float4 tg = P.gtb[b * P.G + j];
      float4 tb = make_float4(tg.x / s_, tg.y / s_, tg.z / s_, tg.w / s_);
      float4 gb;
      float ci = ciou_grad(pb, tb, gb);
      liou = (1.f - ci) * t;
      float gi = -t * P.box_gain * sc;  // d(가중 손실)/d ciou
      float dd[4] = {-gb.x * gi, -gb.y * gi, gb.z * gi, gb.w * gi};  // d / d dist (lt, rb)
      float tl[4] = {ax - tb.x, ay - tb.y, tb.z - ax, tb.w - ay};
      float gd = t * 0.25f * P.dfl_gain * sc;
      for (int s = 0; s < 4; s++) {
        float tt = fminf(fmaxf(tl[s], 0.f), kReg - 1 - 0.01f);
        int il = int(tt), ir = il + 1;
        float wl = ir - tt, wr = 1.f - wl;
        ldfl += -(logf(pr[s][il]) * wl + logf(pr[s][ir]) * wr) * 0.25f * t;
        for (int k = 0; k < kReg; k++) {
          float p = pr[s][k];
          float g = dd[s] * p * (k - dist[s]) + gd * (p - (k == il ? wl : 0.f) - (k == ir ? wr : 0.f));
          db[s * kReg + k] = from_f<act_t>(g);
        }
      }
    }
  }
  block_add(lcls, &P.items[10]);
  block_add(liou, &P.items[8]);
  block_add(ldfl, &P.items[11]);
}

// 5) 마스크 손실: 영상 b, 16×16 타일. 양성마다 자른 상자와 타일이 겹치면 화소별 BCE 와 d proto.
struct MaskArgs {
  Lv L;
  int A, B, G, imgsz, Hp, Wp;
  const act_t* coef[3];
  const act_t* proto;
  act_t* dproto;
  act_t* dcoef[3];
  const int* npos;
  const int* pos;
  const int* agt;
  const float4* gtb;
  const uint32_t* gmask;
  float* items;
  const float* scale;
  float box_gain;
};
__device__ __forceinline__ void crop_of(const MaskArgs& P, float4 tg, float& x1, float& y1, float& x2, float& y2,
                                        float& area) {
  float s = float(P.imgsz);
  float nx1 = tg.x / s, ny1 = tg.y / s, nx2 = tg.z / s, ny2 = tg.w / s;
  area = (nx2 - nx1) * (ny2 - ny1);
  x1 = nx1 * P.Wp; x2 = nx2 * P.Wp; y1 = ny1 * P.Hp; y2 = ny2 * P.Hp;
}
__global__ void k_mask_tile(MaskArgs P) {
  int b = blockIdx.y;
  int tx = blockIdx.x % (P.Wp / 16), ty = blockIdx.x / (P.Wp / 16);
  int px = tx * 16 + (threadIdx.x & 15), py = ty * 16 + (threadIdx.x >> 4);
  int64_t pix = (int64_t(b) * P.Hp + py) * P.Wp + px;
  float pr[32], acc[32];
  const act_t* pp = P.proto + pix * 32;
  for (int k = 0; k < 32; k++) { pr[k] = to_f(pp[k]); acc[k] = 0.f; }
  float fg = fmaxf(P.items[4], 1.f);
  float hw = float(P.Hp) * P.Wp;
  float lsum = 0.f;
  int np = P.npos[b];
  int words = P.Hp * P.Wp / 32;
  __shared__ float cf[32];
  for (int i = 0; i < np; i++) {
    int a = P.pos[b * P.A + i];
    int j = P.agt[b * P.A + a];
    float4 tg = P.gtb[b * P.G + j];
    float x1, y1, x2, y2, area;
    crop_of(P, tg, x1, y1, x2, y2, area);
    // 타일 [tx*16, tx*16+16) 와 [x1, x2) 가 겹치나(화소 번호 r 은 r>=x1 && r<x2)
    if (!(tx * 16 + 15 >= x1 && tx * 16 < x2 && ty * 16 + 15 >= y1 && ty * 16 < y2)) continue;
    __syncthreads();
    if (threadIdx.x < 32) {
      int l, h, w;
      anchor_of(P.L, a, l, h, w);
      cf[threadIdx.x] = to_f(P.coef[l][((int64_t(b) * P.L.h[l] + h) * P.L.w[l] + w) * 32 + threadIdx.x]);
    }
    __syncthreads();
    if (px >= x1 && px < x2 && py >= y1 && py < y2) {
      float z = 0.f;
      for (int k = 0; k < 32; k++) z += cf[k] * pr[k];
      int pi = py * P.Wp + px;
      float y = (P.gmask[(int64_t(b) * P.G + j) * words + (pi >> 5)] >> (pi & 31)) & 1u ? 1.f : 0.f;
      float inv = 1.f / (hw * area);
      lsum += bce_logits(z, y) * inv;
      float g = (sigm(z) - y) * inv * P.box_gain * float(P.B) / fg * *P.scale;
      for (int k = 0; k < 32; k++) acc[k] += g * cf[k];
    }
  }
  act_t* dp = P.dproto + pix * 32;
  for (int k = 0; k < 32; k++) dp[k] = from_f<act_t>(acc[k]);
  block_add(lsum, &P.items[9]);
}

// 6) 마스크 계수 기울기: 양성마다 자른 상자 화소를 돌며 Σ g·proto.
__global__ void k_mask_coef(MaskArgs P) {
  __shared__ float red[kT / 32][32];
  __shared__ float cf[32];
  float fg = fmaxf(P.items[4], 1.f);
  float hw = float(P.Hp) * P.Wp;
  int words = P.Hp * P.Wp / 32;
  for (int b = 0; b < P.B; b++) {
    int np = P.npos[b];
    for (int i = blockIdx.x; i < np; i += gridDim.x) {
      int a = P.pos[b * P.A + i];
      int j = P.agt[b * P.A + a];
      int l, h, w;
      anchor_of(P.L, a, l, h, w);
      int64_t apix = (int64_t(b) * P.L.h[l] + h) * P.L.w[l] + w;
      __syncthreads();
      if (threadIdx.x < 32) cf[threadIdx.x] = to_f(P.coef[l][apix * 32 + threadIdx.x]);
      __syncthreads();
      float4 tg = P.gtb[b * P.G + j];
      float x1, y1, x2, y2, area;
      crop_of(P, tg, x1, y1, x2, y2, area);
      int c0 = max(0, int(ceilf(x1))), c1 = min(P.Wp, int(ceilf(x2)));
      int r0 = max(0, int(ceilf(y1))), r1 = min(P.Hp, int(ceilf(y2)));
      float acc[32];
      for (int k = 0; k < 32; k++) acc[k] = 0.f;
      int nw = c1 - c0, n = nw > 0 && r1 > r0 ? nw * (r1 - r0) : 0;
      float sc = P.box_gain * float(P.B) / (fg * hw * area) * *P.scale;
      for (int t = threadIdx.x; t < n; t += blockDim.x) {
        int py = r0 + t / nw, px = c0 + t % nw;
        const act_t* pp = P.proto + ((int64_t(b) * P.Hp + py) * P.Wp + px) * 32;
        float pr[32], z = 0.f;
        for (int k = 0; k < 32; k++) { pr[k] = to_f(pp[k]); z += cf[k] * pr[k]; }
        int pi = py * P.Wp + px;
        float y = (P.gmask[(int64_t(b) * P.G + j) * words + (pi >> 5)] >> (pi & 31)) & 1u ? 1.f : 0.f;
        float g = (sigm(z) - y) * sc;
        for (int k = 0; k < 32; k++) acc[k] += g * pr[k];
      }
      for (int k = 0; k < 32; k++) {
        float v = acc[k];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5][k] = v;
      }
      __syncthreads();
      if (threadIdx.x < 32) {
        float v = 0.f;
        for (int wp = 0; wp < int(blockDim.x / 32); wp++) v += red[wp][threadIdx.x];
        P.dcoef[l][apix * 32 + threadIdx.x] = from_f<act_t>(v);
      }
    }
  }
}

__global__ void k_final(float* items, float box_gain, float cls_gain, float dfl_gain) {
  float tss = fmaxf(items[5], 1.f), fg = items[4];
  items[0] = items[8] / tss * box_gain;
  items[1] = fg > 0.f ? items[9] / fg * box_gain : 0.f;
  items[2] = items[10] / tss * cls_gain;
  items[3] = items[11] / tss * dfl_gain;
}

Lv make_lv(const LossCfg& c) {
  Lv L;
  L.off[0] = 0;
  for (int l = 0; l < 3; l++) {
    L.h[l] = c.lvl_h[l];
    L.w[l] = c.lvl_w[l];
    L.s[l] = c.stride[l];
    L.off[l + 1] = L.off[l] + L.h[l] * L.w[l];
  }
  return L;
}
}  // namespace

void SegLoss::init(const LossCfg& c) {
  cfg = c;
  A_ = c.A();
  FSO_ASSERT(c.topk <= 10, "topk ≤ 10");
  FSO_ASSERT(c.reg_max == 16 || c.reg_max == 26, "reg_max 16 or 26");
  FSO_ASSERT(c.Hp % 16 == 0 && c.Wp % 16 == 0 && (c.Hp * c.Wp) % 32 == 0, "proto size");
  int64_t BA = int64_t(c.B) * A_;
  pbox_ = dalloc<float4>(BA);
  score_ = dalloc<float>(BA);
  cnt_ = dalloc<int>(BA);
  CUDA_CHECK(cudaMemset(cnt_, 0, BA * sizeof(int)));
  who_ = dalloc<int>(BA);
  agt_ = dalloc<int>(BA);
  amet_ = dalloc<float>(BA);
  aovl_ = dalloc<float>(BA);
  tsc_ = dalloc<float>(BA);
  galign_ = dalloc<float>(int64_t(c.B) * c.G);
  govl_ = dalloc<float>(int64_t(c.B) * c.G);
  pos_ = dalloc<int>(BA);
  npos_ = dalloc<int>(c.B);
  items_ = dalloc<float>(16);
}

void SegLoss::run(const HeadIO& io, const GtBatch& gt, const float* scale, cudaStream_t st) {
  Lv L = make_lv(cfg);
  int B = cfg.B, A = A_;
  int nq = cdiv(int64_t(B) * A, kT);
  CUDA_CHECK(cudaMemsetAsync(items_, 0, 16 * sizeof(float), st));
  DetArgs D;
  D.L = L; D.A = A; D.B = B; D.G = cfg.G; D.imgsz = cfg.imgsz;
  for (int l = 0; l < 3; l++) {
    D.box[l] = io.box[l]; D.cls[l] = io.cls[l]; D.dbox[l] = io.dbox[l]; D.dcls[l] = io.dcls[l]; D.dcoef[l] = io.dcoef[l];
  }
  D.agt = agt_; D.tsc = tsc_; D.gtb = gt.box; D.items = items_; D.scale = scale;
  D.box_gain = cfg.box_gain; D.cls_gain = cfg.cls_gain; D.dfl_gain = cfg.dfl_gain;
  DecArgs da;
  for (int l = 0; l < 3; l++) { da.box[l] = io.box[l]; da.cls[l] = io.cls[l]; }
  if (cfg.reg_max == 16) k_decode<16><<<nq, kT, 0, st>>>(L, A, B, da, pbox_, score_);
  else k_decode<26><<<nq, kT, 0, st>>>(L, A, B, da, pbox_, score_);
  k_topk<<<B * cfg.G, kT, 0, st>>>(L, A, cfg.G, cfg.topk, gt.n_gt, gt.box, pbox_, score_, cnt_, who_, galign_, govl_,
                                   npos_);
  k_resolve<<<nq, kT, 0, st>>>(L, A, B, cfg.G, gt.n_gt, gt.box, pbox_, score_, cnt_, who_, agt_, amet_, aovl_, galign_,
                               govl_, pos_, npos_, items_);
  k_tsum<<<nq, kT, 0, st>>>(A, B, cfg.G, npos_, pos_, agt_, amet_, galign_, govl_, tsc_, items_);
  if (cfg.reg_max == 16) k_det<16><<<nq, kT, 0, st>>>(D);
  else k_det<26><<<nq, kT, 0, st>>>(D);
  MaskArgs M;
  M.L = L; M.A = A; M.B = B; M.G = cfg.G; M.imgsz = cfg.imgsz; M.Hp = cfg.Hp; M.Wp = cfg.Wp;
  for (int l = 0; l < 3; l++) { M.coef[l] = io.coef[l]; M.dcoef[l] = io.dcoef[l]; }
  M.proto = io.proto; M.dproto = io.dproto; M.npos = npos_; M.pos = pos_; M.agt = agt_; M.gtb = gt.box;
  M.gmask = gt.mask; M.items = items_; M.scale = scale; M.box_gain = cfg.box_gain;
  k_mask_tile<<<dim3((cfg.Hp / 16) * (cfg.Wp / 16), B), kT, 0, st>>>(M);
  k_mask_coef<<<280, kT, 0, st>>>(M);
  k_final<<<1, 1, 0, st>>>(items_, cfg.box_gain, cfg.cls_gain, cfg.dfl_gain);
}
