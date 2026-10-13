#include "prep.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace {
constexpr int kPrecisionBits = 32 - 8 - 2;  // Pillow PRECISION_BITS
constexpr int kShort = 800, kMax = 1333;
constexpr double kCropRatio = 0.7, kStrideRatio = 0.6;
constexpr int kCrops = 4;
const float kMean[3] = {123.675f, 116.28f, 103.53f};
const float kStd[3] = {58.395f, 57.120f, 57.375f};

double bilinear_filter(double x) {
  if (x < 0.0) x = -x;
  if (x < 1.0) return 1.0 - x;
  return 0.0;
}

// Pillow precompute_coeffs + normalize_coeffs_8bpc. bounds[2·xx] = xmin, bounds[2·xx+1] = 개수.
int coeffs(int in_size, double in0, double in1, int out_size, std::vector<int>& bounds, std::vector<int32_t>& kk) {
  double scale = (in1 - in0) / out_size, filterscale = scale;
  if (filterscale < 1.0) filterscale = 1.0;
  double support = 1.0 * filterscale;
  int ksize = int(std::ceil(support)) * 2 + 1;
  std::vector<double> pre(size_t(out_size) * ksize, 0.0);
  bounds.assign(size_t(out_size) * 2, 0);
  for (int xx = 0; xx < out_size; xx++) {
    double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0, ss = 1.0 / filterscale;
    int xmin = int(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = int(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double* k = &pre[size_t(xx) * ksize];
    for (int x = 0; x < xmax; x++) {
      double w = bilinear_filter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; x++)
      if (ww != 0.0) k[x] /= ww;
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  kk.resize(pre.size());
  for (size_t i = 0; i < pre.size(); i++)
    kk[i] = pre[i] < 0 ? int32_t(-0.5 + pre[i] * (1 << kPrecisionBits)) : int32_t(0.5 + pre[i] * (1 << kPrecisionBits));
  return ksize;
}
inline uint8_t clip8(int64_t in) {
  if (in >= (int64_t(1) << kPrecisionBits << 8)) return 255;
  if (in <= 0) return 0;
  return uint8_t(in >> kPrecisionBits);
}

// detectron2 ResizeShortestEdge.get_output_shape
void out_shape(int h, int w, int& nh, int& nw) {
  double size = kShort, scale = size / std::min(h, w);
  double newh, neww;
  if (h < w) newh = size, neww = scale * w;
  else newh = scale * h, neww = size;
  if (std::max(newh, neww) > kMax) {
    scale = kMax * 1.0 / std::max(newh, neww);
    newh = newh * scale;
    neww = neww * scale;
  }
  nw = int(neww + 0.5);
  nh = int(newh + 0.5);
}
}  // namespace

void pil_resize_rgb(const uint8_t* src, int sw, int sh, int sstride, uint8_t* dst, int dw, int dh) {
  std::vector<int> bh, bv;
  std::vector<int32_t> kh, kv;
  int ksh = coeffs(sw, 0.0, double(sw), dw, bh, kh);
  int ksv = coeffs(sh, 0.0, double(sh), dh, bv, kv);
  bool need_h = dw != sw, need_v = dh != sh;
  int y_first = bv[0], y_last = bv[(dh - 1) * 2] + bv[(dh - 1) * 2 + 1];
  // 가로: 세로에 쓰일 행 [y_first, y_last) 만
  std::vector<uint8_t> tmp;
  const uint8_t* mid = src;
  int mid_stride = sstride, mid_y0 = 0;
  if (need_h) {
    int rows = y_last - y_first;
    tmp.resize(size_t(rows) * dw * 3);
    for (int y = 0; y < rows; y++) {
      const uint8_t* in = src + size_t(y + y_first) * sstride;
      uint8_t* out = &tmp[size_t(y) * dw * 3];
      for (int xx = 0; xx < dw; xx++) {
        int xmin = bh[xx * 2], xmax = bh[xx * 2 + 1];
        const int32_t* k = &kh[size_t(xx) * ksh];
        int64_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
        for (int x = 0; x < xmax; x++) {
          const uint8_t* p = in + (x + xmin) * 3;
          s0 += int64_t(p[0]) * k[x];
          s1 += int64_t(p[1]) * k[x];
          s2 += int64_t(p[2]) * k[x];
        }
        out[xx * 3] = clip8(s0), out[xx * 3 + 1] = clip8(s1), out[xx * 3 + 2] = clip8(s2);
      }
    }
    mid = tmp.data();
    mid_stride = dw * 3;
    mid_y0 = y_first;
  }
  int mw = need_h ? dw : sw;
  if (need_v) {
    for (int yy = 0; yy < dh; yy++) {
      int ymin = bv[yy * 2] - mid_y0, ymax = bv[yy * 2 + 1];
      const int32_t* k = &kv[size_t(yy) * ksv];
      uint8_t* out = dst + size_t(yy) * dw * 3;
      for (int xx = 0; xx < mw; xx++) {
        int64_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
        for (int y = 0; y < ymax; y++) {
          const uint8_t* p = mid + size_t(y + ymin) * mid_stride + xx * 3;
          s0 += int64_t(p[0]) * k[y];
          s1 += int64_t(p[1]) * k[y];
          s2 += int64_t(p[2]) * k[y];
        }
        out[xx * 3] = clip8(s0), out[xx * 3 + 1] = clip8(s1), out[xx * 3 + 2] = clip8(s2);
      }
    }
  } else {
    for (int y = 0; y < dh; y++)
      std::copy(mid + size_t(y + (need_h ? 0 : 0)) * mid_stride, mid + size_t(y) * mid_stride + size_t(mw) * 3,
                dst + size_t(y) * dw * 3);
  }
}

void prepare(const uint8_t* rgb, int h, int w, Prepared& out) {
  // 자르기 영역(EntityCrop.get_crop_axises, 시험 때는 앞 네 개)
  int crop_w = int(kCropRatio * w), crop_h = int(kCropRatio * h);
  int stride_w = int(kStrideRatio * w), stride_h = int(kStrideRatio * h);
  std::vector<std::array<int, 4>> crops;
  for (int sh0 = 0; sh0 < h; sh0 += stride_h)
    for (int sw0 = 0; sw0 < w; sw0 += stride_w) {
      int eh = std::min(sh0 + crop_h, h), ew = std::min(sw0 + crop_w, w);
      crops.push_back({ew - crop_w, eh - crop_h, ew, eh});
    }
  FSO_ASSERT(int(crops.size()) == kCrops, "crop count %zu for %dx%d", crops.size(), w, h);
  const int T = 1 + kCrops;
  // 화면별 리사이즈
  std::vector<std::vector<uint8_t>> views(T);
  int vh[8], vw[8];
  for (int v = 0; v < T; v++) {
    int x0 = v ? crops[v - 1][0] : 0, y0 = v ? crops[v - 1][1] : 0;
    int cw = v ? crop_w : w, ch = v ? crop_h : h;
    out_shape(ch, cw, vh[v], vw[v]);
    views[v].resize(size_t(vh[v]) * vw[v] * 3);
    pil_resize_rgb(rgb + (size_t(y0) * w + x0) * 3, cw, ch, w * 3, views[v].data(), vw[v], vh[v]);
    out.g.sh[v] = vh[v], out.g.sw[v] = vw[v];
    out.g.oh[v] = ch, out.g.ow[v] = cw;
    out.g.x0[v] = x0, out.g.y0[v] = y0;
  }
  int Hp = *std::max_element(vh, vh + T), Wp = *std::max_element(vw, vw + T);
  Hp = (Hp + 31) / 32 * 32, Wp = (Wp + 31) / 32 * 32;
  out.T = T, out.Hp = Hp, out.Wp = Wp;
  out.g.T = T, out.g.Hm = Hp / 4, out.g.Wm = Wp / 4, out.g.Hp = Hp, out.g.Wp = Wp, out.g.OH = h, out.g.OW = w;
  out.x.assign(size_t(T) * Hp * Wp * 3, 0.f);
  for (int v = 0; v < T; v++)
    for (int y = 0; y < vh[v]; y++)
      for (int x = 0; x < vw[v]; x++)
        for (int c = 0; c < 3; c++)
          out.x[((size_t(v) * Hp + y) * Wp + x) * 3 + c] =
              (float(views[v][(size_t(y) * vw[v] + x) * 3 + c]) - kMean[c]) / kStd[c];
}
