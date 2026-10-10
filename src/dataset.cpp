#include "dataset.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <turbojpeg.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>

namespace {

template <class T> T rd(const uint8_t*& p) {
  T v;
  memcpy(&v, p, sizeof(T));
  p += sizeof(T);
  return v;
}

struct RecHead {
  int iw, ih, mw, mh, n;
  const uint8_t* jpeg;
  uint32_t jpeg_len;
  const uint8_t* masks;  // 첫 마스크 항목
};

RecHead parse_head(const uint8_t* p) {
  FSO_ASSERT(!memcmp(p, "FSR2", 4), "bad record magic");
  p += 4;
  RecHead h;
  h.iw = rd<uint16_t>(p); h.ih = rd<uint16_t>(p); h.mw = rd<uint16_t>(p); h.mh = rd<uint16_t>(p);
  h.jpeg_len = rd<uint32_t>(p);
  h.n = rd<uint16_t>(p);
  rd<uint16_t>(p);
  h.jpeg = p;
  h.masks = p + h.jpeg_len;
  return h;
}

// 마스크 항목 하나 읽기(포인터를 다음 항목으로)
struct MaskEnt {
  float box[4];
  uint32_t area;
  uint8_t frac[5];
  int np;
  const uint16_t* parents;  // 정렬 안 됨일 수 있어 memcpy 로 읽는다
  uint32_t rle_len;
  const uint8_t* rle;
};
MaskEnt next_mask(const uint8_t*& p) {
  MaskEnt m;
  for (float& b : m.box) b = rd<float>(p);
  m.area = rd<uint32_t>(p);
  memcpy(m.frac, p, 5);
  p += 6;
  m.np = rd<uint16_t>(p);
  m.parents = reinterpret_cast<const uint16_t*>(p);
  p += 2 * m.np;
  m.rle_len = rd<uint32_t>(p);
  m.rle = p;
  p += m.rle_len;
  return m;
}

// LEB128 → 교대 길이(0 부터). fn(start, len) 을 1 인 구간마다 부른다.
template <class F> void for_ones(const uint8_t* p, uint32_t n, F fn) {
  const uint8_t* e = p + n;
  int64_t pos = 0;
  bool one = false;
  while (p < e) {
    uint32_t v = 0;
    int sh = 0;
    uint8_t b;
    do { b = *p++; v |= uint32_t(b & 0x7F) << sh; sh += 7; } while (b & 0x80);
    if (one && v) fn(pos, int64_t(v));
    pos += v;
    one = !one;
  }
}

// 비트 [c0, c1) 를 행 words 에 켠다
inline void set_bits(uint32_t* row, int c0, int c1) {
  while (c0 < c1) {
    int w = c0 >> 5, b = c0 & 31;
    int n = std::min(32 - b, c1 - c0);
    uint32_t m = n == 32 ? 0xFFFFFFFFu : (((1u << n) - 1u) << b);
    row[w] |= m;
    c0 += n;
  }
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

// ---------------------------------------------------------------- 버퍼

void BatchBuf::alloc_host(int B, int S, int mips) {
  max_src = 4 * B;
  max_inst = max_src * mips;
  img_cap = int64_t(max_src) * S * S * 3;
  mask_cap = int64_t(16) << 20;  // u32 1600 만 개(64 MB) — 상자 영역만 저장하므로 보통 수 MB
  CUDA_CHECK(cudaHostAlloc(&hdr, meta_bytes(), cudaHostAllocDefault));
  smp = reinterpret_cast<SampleInfo*>(hdr + 1);
  src = reinterpret_cast<SrcInfo*>(smp + B);
  inst = reinterpret_cast<InstInfo*>(src + max_src);
  CUDA_CHECK(cudaHostAlloc(&img, img_cap, cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc(&mask, mask_cap * 4, cudaHostAllocDefault));
  hdr->B = B;
  hdr->S = S;
}

size_t BatchBuf::meta_bytes() const {
  int B = max_src / 4;
  return sizeof(BatchHeader) + sizeof(SampleInfo) * B + sizeof(SrcInfo) * max_src + sizeof(InstInfo) * max_inst;
}

void BatchBuf::alloc_dev(const BatchBuf& like) {
  max_src = like.max_src;
  max_inst = like.max_inst;
  img_cap = like.img_cap;
  mask_cap = like.mask_cap;
  int B = max_src / 4;
  hdr = static_cast<BatchHeader*>(DevMem::alloc(meta_bytes()));
  smp = reinterpret_cast<SampleInfo*>(hdr + 1);
  src = reinterpret_cast<SrcInfo*>(smp + B);
  inst = reinterpret_cast<InstInfo*>(src + max_src);
  img = dalloc<uint8_t>(img_cap);
  mask = dalloc<uint32_t>(mask_cap);
}

// ---------------------------------------------------------------- 열기

void Dataset::open(const DataCfg& c) {
  cfg = c;
  for (size_t si = 0; si < c.shards.size(); si++) {
    const std::string& bin = c.shards[si];
    std::string idx = bin.substr(0, bin.size() - 4) + ".idx";
    Shard sh;
    sh.fd = ::open(bin.c_str(), O_RDONLY);
    FSO_ASSERT(sh.fd >= 0, "open %s", bin.c_str());
    struct stat st;
    fstat(sh.fd, &st);
    sh.len = st.st_size;
    void* m = mmap(nullptr, sh.len, PROT_READ, MAP_SHARED, sh.fd, 0);
    FSO_ASSERT(m != MAP_FAILED, "mmap %s", bin.c_str());
    madvise(m, sh.len, MADV_RANDOM);
    sh.base = static_cast<const uint8_t*>(m);
    std::ifstream f(idx, std::ios::binary | std::ios::ate);
    FSO_ASSERT(f.good(), "open %s", idx.c_str());
    size_t n = size_t(f.tellg()) / 8;
    std::vector<uint64_t> off(n);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(off.data()), n * 8);
    FSO_ASSERT(off.back() == sh.len, "%s: idx/bin size mismatch", bin.c_str());
    size_t nrec = n - 1, ho = std::min<size_t>(c.hold_per_shard, nrec / 10);
    for (size_t i = 0; i < nrec; i++)
      (i + ho >= nrec ? val_ : train_).push_back({int(si), int64_t(off[i]), int64_t(off[i + 1] - off[i])});
    shards_.push_back(sh);
  }
  if (c.train_limit > 0 && int64_t(train_.size()) > c.train_limit) train_.resize(c.train_limit);
  printf("dataset: %zu shards, %zu train, %zu held out, %ld batches/epoch\n", shards_.size(), train_.size(),
         val_.size(), batches_per_epoch());
}

std::vector<int64_t> Dataset::perm(int64_t epoch) {
  std::vector<int64_t> p(train_.size());
  for (size_t i = 0; i < p.size(); i++) p[i] = int64_t(i);
  std::mt19937_64 r(cfg.seed * 1000003ull + uint64_t(epoch));
  std::shuffle(p.begin(), p.end(), r);
  return p;
}

// ---------------------------------------------------------------- 원본 하나 풀기

void Dataset::decode_src(const Rec& r, BatchBuf& b, int si, int sample, int tile, int64_t img_off,
                         int64_t& mask_cursor, std::mutex& mu) {
  const uint8_t* p = shards_[r.shard].base + r.off;
  RecHead h = parse_head(p);
  // JPEG → RGB
  thread_local tjhandle tj = tjInitDecompress();
  int w, hh, sub, cs;
  FSO_ASSERT(tjDecompressHeader3(tj, h.jpeg, h.jpeg_len, &w, &hh, &sub, &cs) == 0, "jpeg header");
  FSO_ASSERT(w == h.iw && hh == h.ih, "jpeg size %dx%d vs %dx%d", w, hh, h.iw, h.ih);
  FSO_ASSERT(tjDecompress2(tj, h.jpeg, h.jpeg_len, b.img + img_off, w, w * 3, hh, TJPF_RGB, 0) == 0, "jpeg decode: %s",
             tjGetErrorStr2(tj));
  // 규칙: 배경 그룹 비율 합 ≥ bg_frac → 버림; 남긴 더 큰 마스크 안 80 % 이상(부모) → 부분으로 버림(문·창·계단 빼고)
  const uint8_t* q = h.masks;
  std::vector<uint8_t> kept(h.n, 0);
  struct Out { MaskEnt e; int x0, y0, cw, ch; };
  std::vector<Out> outs;
  int bgthr = int(cfg.bg_frac * 255.f + 0.5f), dwsthr = int(cfg.dws_frac * 255.f + 0.5f);
  for (int i = 0; i < h.n; i++) {
    MaskEnt e = next_mask(q);
    int bg = 0;
    for (int g = 0; g < 4; g++)
      if (cfg.bg_groups >> g & 1) bg += e.frac[g];
    if (bg >= bgthr) continue;
    if (e.frac[4] < dwsthr) {
      bool part = false;
      for (int k = 0; k < e.np && !part; k++) {
        uint16_t par;
        memcpy(&par, e.parents + k, 2);
        part = kept[par];
      }
      if (part) continue;
    }
    kept[i] = 1;
    if (int(outs.size()) < cfg.max_inst_per_src) outs.push_back({e, 0, 0, 0, 0});
  }
  // 마스크 상자(RLE 에서 정확히) → 상자 영역 비트맵
  std::vector<uint32_t> bits;
  std::vector<int64_t> local_off;
  for (auto& o : outs) {
    int x0 = h.mw, y0 = h.mh, x1 = -1, y1 = -1;
    for_ones(o.e.rle, o.e.rle_len, [&](int64_t s, int64_t n) {
      int r0 = int(s / h.mw), r1 = int((s + n - 1) / h.mw);
      y0 = std::min(y0, r0); y1 = std::max(y1, r1);
      if (r0 == r1) { x0 = std::min(x0, int(s % h.mw)); x1 = std::max(x1, int((s + n - 1) % h.mw)); }
      else { x0 = 0; x1 = h.mw - 1; }
    });
    if (x1 < 0) { o.cw = 0; local_off.push_back(-1); continue; }
    o.x0 = x0; o.y0 = y0; o.cw = x1 - x0 + 1; o.ch = y1 - y0 + 1;
    int wpr = (o.cw + 31) / 32;
    int64_t base = int64_t(bits.size());
    bits.resize(bits.size() + size_t(wpr) * o.ch, 0u);
    local_off.push_back(base);
    for_ones(o.e.rle, o.e.rle_len, [&](int64_t s, int64_t n) {
      int64_t e = s + n;
      while (s < e) {
        int r = int(s / h.mw), c = int(s % h.mw);
        int c1 = int(std::min<int64_t>(h.mw, c + (e - s)));
        int a = std::max(c, o.x0) - o.x0, z = std::min(c1, o.x0 + o.cw) - o.x0;
        if (z > a) set_bits(&bits[base + int64_t(r - o.y0) * wpr], a, z);
        s += c1 - c;
      }
    });
  }
  // 배치 버퍼에 자리 잡기(여러 스레드가 함께 쓴다)
  int64_t moff, i0;
  {
    std::lock_guard<std::mutex> lk(mu);
    moff = mask_cursor;
    if (moff + int64_t(bits.size()) > b.mask_cap) { n_dropped_cap += int64_t(outs.size()); return; }
    mask_cursor += int64_t(bits.size());
    i0 = b.hdr->ninst;
    int nn = 0;
    for (auto& o : outs) nn += o.cw > 0;
    if (i0 + nn > b.max_inst) { n_dropped_cap += nn; mask_cursor -= int64_t(bits.size()); return; }
    b.hdr->ninst += nn;
  }
  memcpy(b.mask + moff, bits.data(), bits.size() * 4);
  int k = 0;
  for (size_t j = 0; j < outs.size(); j++) {
    auto& o = outs[j];
    if (o.cw == 0) continue;
    InstInfo& in = b.inst[i0 + k++];
    in.src = si; in.sample = sample; in.tile = tile;
    memcpy(in.box, o.e.box, sizeof(in.box));
    in.mx0 = o.x0; in.my0 = o.y0; in.mcw = o.cw; in.mch = o.ch;
    in.wpr = (o.cw + 31) / 32;
    in.mask_off = moff + local_off[j];
  }
  n_inst_total += k;
}

// ---------------------------------------------------------------- 배치 하나 채우기

void Dataset::fill(BatchBuf& b, int64_t bi) {
  const int B = cfg.batch, S = cfg.imgsz;
  int64_t nb = batches_per_epoch();
  int64_t epoch = bi / nb, j = bi % nb;
  {
    std::lock_guard<std::mutex> lk(perm_mu_);
    if (perm_for_epoch_ != epoch) { perm_epoch_ = perm(epoch); perm_for_epoch_ = epoch; }
  }
  std::mt19937_64 rng(cfg.seed * 0x9E3779B97F4A7C15ull + uint64_t(bi) * 0xBF58476D1CE4E5B9ull + 1);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  auto unif = [&](double a, double z) { return a + (z - a) * U(rng); };
  b.hdr->B = B; b.hdr->S = S; b.hdr->nsrc = 0; b.hdr->ninst = 0;
  struct Job { Rec r; int si, sample, tile; int64_t img_off; };
  std::vector<Job> jobs;
  int64_t img_cursor = 0;
  for (int i = 0; i < B; i++) {
    SampleInfo& sm = b.smp[i];
    int64_t main_idx;
    {
      std::lock_guard<std::mutex> lk(perm_mu_);
      main_idx = perm_epoch_[j * B + i];
    }
    int ntile = U(rng) < cfg.mosaic ? 4 : 1;
    int64_t idx[4] = {main_idx, -1, -1, -1};
    for (int t = 1; t < ntile; t++) idx[t] = int64_t(U(rng) * double(train_.size())) % int64_t(train_.size());
    // mosaic 중심: uniform(S/2, 3S/2) 정수(Ultralytics Mosaic._mosaic4)
    int xc = int(unif(S / 2, 1.5 * S)), yc = int(unif(S / 2, 1.5 * S));
    for (int t = 0; t < 4; t++) sm.src[t] = -1;
    for (int t = 0; t < ntile; t++) {
      const Rec& r = train_[idx[t]];
      RecHead h = parse_head(shards_[r.shard].base + r.off);
      int w = h.iw, hh = h.ih;
      int x1a, y1a, x2a, y2a, x1b, y1b;
      if (ntile == 1) {  // mosaic 없음: 캔버스 가운데에 놓는다(RandomPerspective 가 border 없이 받는 것과 같게)
        x1a = (2 * S - w) / 2; y1a = (2 * S - hh) / 2; x2a = x1a + w; y2a = y1a + hh; x1b = 0; y1b = 0;
      } else if (t == 0) {
        x1a = std::max(xc - w, 0); y1a = std::max(yc - hh, 0); x2a = xc; y2a = yc;
        x1b = w - (x2a - x1a); y1b = hh - (y2a - y1a);
      } else if (t == 1) {
        x1a = xc; y1a = std::max(yc - hh, 0); x2a = std::min(xc + w, 2 * S); y2a = yc;
        x1b = 0; y1b = hh - (y2a - y1a);
      } else if (t == 2) {
        x1a = std::max(xc - w, 0); y1a = yc; x2a = xc; y2a = std::min(2 * S, yc + hh);
        x1b = w - (x2a - x1a); y1b = 0;
      } else {
        x1a = xc; y1a = yc; x2a = std::min(xc + w, 2 * S); y2a = std::min(2 * S, yc + hh);
        x1b = 0; y1b = 0;
      }
      int si = b.hdr->nsrc++;
      SrcInfo& s = b.src[si];
      s.img_off = img_cursor; s.iw = w; s.ih = hh; s.mw = h.mw; s.mh = h.mh;
      img_cursor += int64_t(w) * hh * 3;
      FSO_ASSERT(img_cursor <= b.img_cap, "image buffer");
      sm.src[t] = si;
      sm.ta[t][0] = x1a; sm.ta[t][1] = y1a; sm.ta[t][2] = x2a; sm.ta[t][3] = y2a;
      sm.tb[t][0] = x1b; sm.tb[t][1] = y1b;
      jobs.push_back({r, si, i, t, s.img_off});
    }
    // RandomPerspective(degrees 0, shear 0, perspective 0): M = T·S·C, C = −(2S)/2, T = uniform(.5±translate)·S
    double sc = unif(1.0 - cfg.scale, 1.0 + cfg.scale);
    double tx = unif(0.5 - cfg.translate, 0.5 + cfg.translate) * S, ty = unif(0.5 - cfg.translate, 0.5 + cfg.translate) * S;
    sm.s = float(sc);
    sm.tx = float(tx - sc * S);
    sm.ty = float(ty - sc * S);
    sm.hsv[0] = float(unif(-1, 1) * cfg.hsv_h + 1);
    sm.hsv[1] = float(unif(-1, 1) * cfg.hsv_s + 1);
    sm.hsv[2] = float(unif(-1, 1) * cfg.hsv_v + 1);
    sm.flip = U(rng) < cfg.fliplr;
    sm.inst0 = 0; sm.ninst = 0;
  }
  b.hdr->img_bytes = img_cursor;
  // 원본 풀기: 이 스레드와 도우미들이 나눠서
  int64_t mask_cursor = 0;
  std::mutex mu;
  std::atomic<int> next{0};
  auto work = [&]() {
    for (int k; (k = next.fetch_add(1)) < int(jobs.size());) {
      auto& jb = jobs[k];
      decode_src(jb.r, b, jb.si, jb.sample, jb.tile, jb.img_off, mask_cursor, mu);
    }
  };
  int helpers = std::max(1, std::min(cfg.workers / 2, int(jobs.size())) - 1);
  std::vector<std::thread> th;
  for (int k = 0; k < helpers; k++) th.emplace_back(work);
  work();
  for (auto& t : th) t.join();
  b.hdr->mask_words = mask_cursor;
}

// ---------------------------------------------------------------- 백그라운드

void Dataset::start(int slots) {
  bufs_.resize(slots);
  for (auto& b : bufs_) b.alloc_host(cfg.batch, cfg.imgsz, cfg.max_inst_per_src);
  state_.assign(slots, 0);
  buf_batch_.assign(slots, -1);
  int fillers = 2;  // 배치 둘을 함께 채운다(배치 안에서는 도우미 스레드로 나눠 푼다)
  for (int f = 0; f < fillers; f++)
    threads_.emplace_back([this]() {
      for (;;) {
        int slot = -1;
        int64_t bi;
        {
          std::unique_lock<std::mutex> lk(mu_);
          cv_.wait(lk, [&]() {
            if (stop_) return true;
            for (size_t s = 0; s < state_.size(); s++)
              if (state_[s] == 0) { slot = int(s); return true; }
            return false;
          });
          if (stop_) return;
          state_[slot] = 1;
          bi = next_fill_++;
          buf_batch_[slot] = bi;
        }
        fill(bufs_[slot], bi);
        {
          std::lock_guard<std::mutex> lk(mu_);
          state_[slot] = 2;
        }
        cv_.notify_all();
      }
    });
}

BatchBuf* Dataset::acquire(int64_t& bi) {
  double t0 = now_ms();
  std::unique_lock<std::mutex> lk(mu_);
  int slot = -1;
  cv_.wait(lk, [&]() {
    for (size_t s = 0; s < state_.size(); s++)
      if (state_[s] == 2 && buf_batch_[s] == next_take_) { slot = int(s); return true; }
    return false;
  });
  state_[slot] = 3;
  bi = next_take_++;
  wait_ms += now_ms() - t0;
  return &bufs_[slot];
}

void Dataset::release(BatchBuf* b) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    state_[b - bufs_.data()] = 0;
  }
  cv_.notify_all();
}

void Dataset::set_epoch(int64_t first_batch) {
  std::lock_guard<std::mutex> lk(mu_);
  FSO_ASSERT(threads_.empty(), "set_epoch before start");
  next_fill_ = next_take_ = first_batch;
}

void Dataset::stop() {
  stop_ = true;
  cv_.notify_all();
  for (auto& t : threads_) t.join();
  threads_.clear();
}
