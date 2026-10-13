// 선생(CropFormer Hornet-L)으로 사진마다 겹치지 않는 개체 마스크를 만든다.
//   teacher_label --weights W.fsow --list LIST --out OUT.ftl [--workers 8] [--prec half|fp32] [--limit N]
// LIST: 줄마다 "image_id 경로". 이미 OUT 에 있는 image_id 는 건너뛴다(이어 하기). 끝이 잘린 레코드는 잘라 낸다.
//
// 레코드(리틀 엔디언): 'FTL1', u32 image_id, u16 H, u16 W, u16 n, u16 0,
//   n × [f32 점수(물체 확률 × 마스크 점수), u32 넓이] (점수 내림차순 순위 순, 넓이 > 0 인 것만),
//   u32 RLE 바이트 수, RLE: 행 우선으로 (값, 길이) 쌍을 LEB128 로. 값 0 = 마스크 없음, k = k-1 번 마스크.
// 점수 기준은 나중에 정한다(여기서는 넓이 > 0 이면 모두 남긴다).
//
// --parts P (기본 꺼짐; FTL1 출력은 그대로): 겹치지 않는 id 지도에서 대부분 가려진 확신 있는 마스크(부품 후보)를 P 에 더 쓴다.
//   후보 조건(tk.cuh PartCfg): 온전한(펴기 전, fused > 0.5) 마스크 점수 ≥ --part-score(0.5; 점수 = 물체 확률 × 온전한 마스크
//   안 평균 sigmoid(확률), FTL 점수와 같은 식을 온전한 마스크에), 온전한 넓이 ≥ --part-min-area(100), 보이는 비율 < --part-max-vis
//   (0.5), 온전한 마스크의 ≥ --part-inside(0.8) 가 앞 순위 마스크 하나(부모: FTL 점수 ≥ --parent-score 0.5)의 보이는 영역 안,
//   부품 넓이 / 부모 보이는 넓이 ∈ [--part-rel-lo 0.01, --part-rel-hi 0.7]. 사진당 순위 순으로 최대 64 개.
//   레코드(리틀 엔디언, 사진마다 하나, 후보 0 개여도): 'FTP2', u32 image_id, u16 H, u16 W, u16 n, u16 0,
//   n × [u16 부모(FTL1 마스크 번호 = 넓이 > 0 순위 번호), u16 질의 순위(0..199, FTL1 에 없을 수 있음), f32 온전한 점수,
//        u32 온전한 넓이, u32 보이는 넓이, f32 부모 안 비율, u32 RLE 바이트 수, RLE(부품의 온전한 마스크, 사진 해상도;
//        pytools/coco_build.py 와 같음: 행 우선, 0 부터 교대 길이, LEB128)].
// --lowmem: 백본을 화면 하나씩 돌리고 픽셀 디코더 뒤에 앞 단계 작업 버퍼를 놓는다(다른 학습과 GPU 를 나눠 쓸 때).
//   GEMM 모양이 달라 결과가 비트 단위로 같지는 않다(fp16 흔들림 범위, docs/TEACHER.md).
//   이어 하기는 P 를 기준으로 한다(--parts 를 줄 때 OUT 은 버리는 파일로 쓴다; pytools/ftl.py read_parts 로 읽는다).
#include <turbojpeg.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

#include "prep.h"
#include "tnet.h"

namespace {

struct Item {
  uint32_t id;
  std::string path;
};
// 고정(pinned) 입력 슬롯: 읽기 스레드가 채우고 GPU 스레드가 올린다. ev: 올리기가 끝난 때.
struct InSlot {
  __half* x = nullptr;
  size_t cap = 0;
  cudaEvent_t ev;
};
// 고정 출력 슬롯: GPU 가 결과를 내려 쓰고 쓰기 스레드가 읽는다. ev: 내려 쓰기가 끝난 때.
struct OutSlot {
  int16_t* id = nullptr;
  float* score = nullptr;
  int* area = nullptr;
  uint32_t image_id = 0;
  int OH = 0, OW = 0;
  PartInfo* pinfo = nullptr;
  int* pn = nullptr;
  unsigned long long* pbits = nullptr;
  cudaEvent_t ev;
};
struct Job {
  uint32_t id;
  int slot;
  Prepared pr;  // 모양만(x 는 비움)
};
constexpr size_t kMaxIn = size_t(5) * 1344 * 1344 * 3;  // 화면 5 장 × 최대 패딩 크기
constexpr size_t kMaxHW = size_t(1024) * 1024;           // 원본 사진 화소 수 상한(COCO ≤ 640×640)

template <class T>
class Queue {
 public:
  explicit Queue(size_t cap) : cap_(cap) {}
  void push(T v) {
    std::unique_lock<std::mutex> l(m_);
    nf_.wait(l, [&] { return q_.size() < cap_; });
    q_.push_back(std::move(v));
    ne_.notify_one();
  }
  bool pop(T& v) {
    std::unique_lock<std::mutex> l(m_);
    ne_.wait(l, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;
    v = std::move(q_.front());
    q_.pop_front();
    nf_.notify_one();
    return true;
  }
  void close() {
    std::lock_guard<std::mutex> l(m_);
    closed_ = true;
    ne_.notify_all();
  }

 private:
  std::mutex m_;
  std::condition_variable ne_, nf_;
  std::deque<T> q_;
  size_t cap_;
  bool closed_ = false;
};

void leb(std::string& o, uint64_t v) {
  do {
    uint8_t b = v & 0x7F;
    v >>= 7;
    o.push_back(char(b | (v ? 0x80 : 0)));
  } while (v);
}

// 넓이 > 0 인 순위만 남기고 id 지도를 RLE 로.
std::string encode(uint32_t id, const OutSlot& r) {
  const int Q = TeacherNetT<__half, __half, __half>::kQ;
  std::vector<int> remap(Q, -1);
  std::string head, body;
  int n = 0;
  for (int k = 0; k < Q; k++)
    if (r.area[k] > 0) remap[k] = n++;
  head.append("FTL1", 4);
  auto put = [&](std::string& o, const void* p, size_t b) { o.append(static_cast<const char*>(p), b); };
  uint16_t H = r.OH, W = r.OW, nn = n, z = 0;
  (void)id;
  id = r.image_id;
  put(head, &id, 4), put(head, &H, 2), put(head, &W, 2), put(head, &nn, 2), put(head, &z, 2);
  for (int k = 0; k < Q; k++)
    if (remap[k] >= 0) {
      float s = r.score[k];
      uint32_t a = r.area[k];
      put(head, &s, 4), put(head, &a, 4);
    }
  size_t HW = size_t(H) * W;
  size_t i = 0;
  while (i < HW) {
    int v = r.id[i] < 0 ? 0 : remap[r.id[i]] + 1;
    size_t j = i + 1;
    while (j < HW && (r.id[j] < 0 ? 0 : remap[r.id[j]] + 1) == v) j++;
    leb(body, uint64_t(v));
    leb(body, j - i);
    i = j;
  }
  uint32_t nb = body.size();
  put(head, &nb, 4);
  return head + body;
}

// 부품 후보 레코드(머리말 --parts).
std::string encode_parts(const OutSlot& r) {
  const int Q = TeacherNetT<__half, __half, __half>::kQ;
  std::vector<int> remap(Q, -1);
  for (int k = 0, n = 0; k < Q; k++)
    if (r.area[k] > 0) remap[k] = n++;
  std::string o("FTP2", 4);
  auto put = [&](const void* p, size_t b) { o.append(static_cast<const char*>(p), b); };
  uint32_t id = r.image_id;
  uint16_t H = r.OH, W = r.OW, n = uint16_t(*r.pn), z = 0;
  put(&id, 4), put(&H, 2), put(&W, 2), put(&n, 2), put(&z, 2);
  size_t HW = size_t(H) * W;
  for (int c = 0; c < n; c++) {
    const PartInfo& pi = r.pinfo[c];
    FSO_ASSERT(remap[pi.parent] >= 0, "parent without area");
    uint16_t par = remap[pi.parent], rk = pi.rank;
    uint32_t full = pi.full, vis = pi.vis;
    put(&par, 2), put(&rk, 2), put(&pi.score, 4), put(&full, 4), put(&vis, 4), put(&pi.inside, 4);
    std::string body;
    unsigned long long m = 1ull << c;
    bool cur = false;
    size_t i = 0;
    while (i < HW) {  // 0 부터 교대 길이
      size_t j = i;
      while (j < HW && bool(r.pbits[j] & m) == cur) j++;
      leb(body, j - i);
      cur = !cur;
      i = j;
    }
    uint32_t nb = body.size();
    put(&nb, 4);
    o += body;
  }
  return o;
}

std::set<uint32_t> scan_parts(const std::string& path) {
  std::set<uint32_t> done;
  std::ifstream f(path, std::ios::binary);
  if (!f) return done;
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
  size_t p = 0, good = 0;
  while (p + 16 <= buf.size()) {
    if (memcmp(&buf[p], "FTP2", 4)) break;
    uint32_t id;
    uint16_t n;
    memcpy(&id, &buf[p + 4], 4);
    memcpy(&n, &buf[p + 12], 2);
    size_t q = p + 16;
    bool ok = true;
    for (int c = 0; c < n && ok; c++) {
      if (q + 24 > buf.size()) { ok = false; break; }
      uint32_t nb;
      memcpy(&nb, &buf[q + 20], 4);
      q += 24 + nb;
      if (q > buf.size()) ok = false;
    }
    if (!ok) break;
    done.insert(id);
    p = good = q;
  }
  f.close();
  if (good != buf.size()) {
    fprintf(stderr, "truncating %s: %zu → %zu bytes\n", path.c_str(), buf.size(), good);
    FSO_ASSERT(truncate(path.c_str(), off_t(good)) == 0, "truncate");
  }
  return done;
}

// 이어 하기: 온전한 레코드의 image_id 를 모으고, 끝이 잘렸으면 그 앞까지로 자른다.
std::set<uint32_t> scan(const std::string& path) {
  std::set<uint32_t> done;
  std::ifstream f(path, std::ios::binary);
  if (!f) return done;
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
  size_t p = 0, good = 0;
  while (p + 16 <= buf.size()) {
    if (memcmp(&buf[p], "FTL1", 4)) break;
    uint32_t id;
    uint16_t n;
    memcpy(&id, &buf[p + 4], 4);
    memcpy(&n, &buf[p + 12], 2);
    size_t q = p + 16 + size_t(n) * 8;
    if (q + 4 > buf.size()) break;
    uint32_t nb;
    memcpy(&nb, &buf[q], 4);
    if (q + 4 + nb > buf.size()) break;
    done.insert(id);
    p = good = q + 4 + nb;
  }
  f.close();
  if (good != buf.size()) {
    fprintf(stderr, "truncating %s: %zu → %zu bytes\n", path.c_str(), buf.size(), good);
    FSO_ASSERT(truncate(path.c_str(), off_t(good)) == 0, "truncate");
  }
  return done;
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights = "weights/teacher/cropformer_hornet_l.fsow", list, out, prec = "half", parts;
  PartCfg pc;
  int workers = 8;
  bool lowmem = false;
  long limit = -1;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&] {
      FSO_ASSERT(i + 1 < argc, "missing value for %s", a.c_str());
      return std::string(argv[++i]);
    };
    if (a == "--weights") weights = next();
    else if (a == "--list") list = next();
    else if (a == "--out") out = next();
    else if (a == "--workers") workers = std::stoi(next());
    else if (a == "--prec") prec = next();
    else if (a == "--limit") limit = std::stol(next());
    else if (a == "--parts") parts = next();
    else if (a == "--lowmem") lowmem = true;
    else if (a == "--part-score") pc.score = std::stof(next());
    else if (a == "--part-min-area") pc.min_area = std::stoi(next());
    else if (a == "--part-max-vis") pc.max_vis = std::stof(next());
    else if (a == "--part-inside") pc.inside = std::stof(next());
    else if (a == "--part-rel-lo") pc.rel_lo = std::stof(next());
    else if (a == "--part-rel-hi") pc.rel_hi = std::stof(next());
    else if (a == "--parent-score") pc.parent_score = std::stof(next());
    else FSO_ASSERT(false, "unknown arg %s", a.c_str());
  }
  FSO_ASSERT(!list.empty() && !out.empty(), "need --list and --out");
  FSO_ASSERT(prec == "half", "only --prec half is built into teacher_label");
  std::vector<Item> items;
  {
    std::ifstream f(list);
    FSO_ASSERT(f.good(), "open %s", list.c_str());
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream ss(line);
      Item it;
      if (ss >> it.id >> it.path) items.push_back(it);
    }
  }
  auto done = scan(out);
  if (!parts.empty()) done = scan_parts(parts);  // 부품 파일이 기준(OUT 은 버리는 파일)
  std::vector<Item> todo;
  for (auto& it : items)
    if (!done.count(it.id)) todo.push_back(it);
  if (limit >= 0 && long(todo.size()) > limit) todo.resize(limit);
  fprintf(stderr, "%zu listed, %zu done, %zu to do\n", items.size(), done.size(), todo.size());
  if (todo.empty()) return 0;

  using Net = TeacherNetT<__half, __half, __half>;
  Net net;
  net.load(weights);

  // 고정 메모리 슬롯
  constexpr int kIn = 4, kOut = 4;
  std::vector<InSlot> ins(kIn);
  std::vector<OutSlot> outs(kOut);
  for (auto& s : ins) {
    CUDA_CHECK(cudaMallocHost(&s.x, kMaxIn * sizeof(__half)));
    s.cap = kMaxIn;
    CUDA_CHECK(cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming));
  }
  for (auto& s : outs) {
    CUDA_CHECK(cudaMallocHost(&s.id, kMaxHW * sizeof(int16_t)));
    CUDA_CHECK(cudaMallocHost(&s.score, Net::kQ * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&s.area, Net::kQ * sizeof(int)));
    if (!parts.empty()) {
      CUDA_CHECK(cudaMallocHost(&s.pinfo, kMaxParts * sizeof(PartInfo)));
      CUDA_CHECK(cudaMallocHost(&s.pn, sizeof(int)));
      CUDA_CHECK(cudaMallocHost(&s.pbits, kMaxHW * sizeof(unsigned long long)));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming));
  }
  Queue<int> free_in(kIn), free_out(kOut), written(kOut);
  for (int i = 0; i < kIn; i++) free_in.push(i);
  for (int i = 0; i < kOut; i++) free_out.push(i);
  Queue<Job> jobs(kIn);

  std::atomic<size_t> next{0};
  std::atomic<int> readers_left{workers};
  std::mutex fail_m;
  std::vector<std::thread> rd;
  for (int w = 0; w < workers; w++)
    rd.emplace_back([&] {
      tjhandle tj = tjInitDecompress();
      std::vector<unsigned char> jpg, rgb;
      for (size_t k; (k = next.fetch_add(1)) < todo.size();) {
        Job j;
        j.id = todo[k].id;
        std::ifstream f(todo[k].path, std::ios::binary);
        jpg.assign(std::istreambuf_iterator<char>(f), {});
        int w_ = 0, h_ = 0, ss = 0, cs = 0;
        bool ppm = jpg.size() > 2 && jpg[0] == 'P' && jpg[1] == '6';  // JPEG 이 아닌 사진(COCO 에 PNG 1 장)은 PPM 으로 바꿔 넣는다
        bool fail = false;
        if (ppm) {
          int maxv = 0, off = 0;
          fail = sscanf(reinterpret_cast<const char*>(jpg.data()), "P6 %d %d %d%n", &w_, &h_, &maxv, &off) != 3 || maxv != 255 ||
                 jpg.size() < size_t(off + 1) + size_t(w_) * h_ * 3;
          if (!fail) rgb.assign(jpg.begin() + off + 1, jpg.begin() + off + 1 + size_t(w_) * h_ * 3);
        } else {
          fail = jpg.empty() || tjDecompressHeader3(tj, jpg.data(), jpg.size(), &w_, &h_, &ss, &cs) != 0;
        }
        if (!fail && !ppm) {
          rgb.resize(size_t(w_) * h_ * 3);
          fail = tjDecompress2(tj, jpg.data(), jpg.size(), rgb.data(), w_, 0, h_, TJPF_RGB, 0) != 0;
        }
        if (fail || size_t(w_) * h_ > kMaxHW) {
          std::lock_guard<std::mutex> l(fail_m);
          fprintf(stderr, "skip: %u %s (%s)\n", j.id, todo[k].path.c_str(), fail ? tjGetErrorStr2(tj) : "too large");
          continue;
        }
        prepare(rgb.data(), h_, w_, j.pr);
        FSO_ASSERT(j.pr.x.size() <= kMaxIn, "input too large %zu", j.pr.x.size());
        free_in.pop(j.slot);
        InSlot& s = ins[j.slot];
        CUDA_CHECK(cudaEventSynchronize(s.ev));  // 앞서 이 슬롯을 올리던 복사가 끝날 때까지
        for (size_t i = 0; i < j.pr.x.size(); i++) s.x[i] = __float2half(j.pr.x[i]);
        j.pr.x = std::vector<float>();
        jobs.push(std::move(j));
      }
      tjDestroy(tj);
      if (--readers_left == 0) jobs.close();
    });

  std::thread writer([&] {
    std::ofstream f(out, std::ios::binary | std::ios::app);
    FSO_ASSERT(f.good(), "write %s", out.c_str());
    std::ofstream fp;
    if (!parts.empty()) {
      fp.open(parts, std::ios::binary | std::ios::app);
      FSO_ASSERT(fp.good(), "write %s", parts.c_str());
    }
    size_t nparts = 0;
    int si;
    size_t n = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (written.pop(si)) {
      OutSlot& s = outs[si];
      CUDA_CHECK(cudaEventSynchronize(s.ev));
      std::string rec = encode(s.image_id, s);
      std::string prec_;
      if (!parts.empty()) prec_ = encode_parts(s), nparts += *s.pn;
      free_out.push(si);
      f.write(rec.data(), rec.size());
      if (!parts.empty()) fp.write(prec_.data(), prec_.size());
      if (++n % 100 == 0) {
        f.flush();
        if (!parts.empty()) fp.flush();
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "%zu / %zu  %.3f s/img  eta %.1f h  parts %.2f/img\n", n, todo.size(), sec / n,
                sec / n * (todo.size() - n) / 3600, double(nparts) / n);
      }
    }
    f.flush();
    if (!parts.empty()) fp.flush();
    fprintf(stderr, "wrote %zu records, %zu parts\n", n, nparts);
  });

  // GPU: 사진 하나씩 스트림에 넣기만 하고(호스트와 맞추지 않음) 다음으로 넘어간다.
  __half* xd = nullptr;
  CUDA_CHECK(cudaMalloc(&xd, kMaxIn * sizeof(__half)));
  Job j;
  while (jobs.pop(j)) {
    InSlot& in = ins[j.slot];
    net.set_shape(j.pr.T, j.pr.Hp, j.pr.Wp);
    size_t n = size_t(j.pr.T) * j.pr.Hp * j.pr.Wp * 3;
    CUDA_CHECK(cudaMemcpyAsync(xd, in.x, n * sizeof(__half), cudaMemcpyHostToDevice, net.st));
    CUDA_CHECK(cudaEventRecord(in.ev, net.st));
    free_in.push(j.slot);
    if (lowmem) net.backbone_per_view(xd);
    else net.backbone(xd);
    net.pixel_decoder();
    if (lowmem) net.release_scratch();
    net.decoder();
    if (lowmem) net.release_scratch(true);
    int oi;
    free_out.pop(oi);
    OutSlot& o = outs[oi];
    o.image_id = j.id, o.OH = j.pr.g.OH, o.OW = j.pr.g.OW;
    FSO_ASSERT(size_t(o.OH) * o.OW <= kMaxHW, "image too large");
    net.postprocess_async(j.pr.g, o.id, o.score, o.area);
    if (!parts.empty()) net.parts_async(j.pr.g, pc, o.pinfo, o.pn, o.pbits);
    CUDA_CHECK(cudaEventRecord(o.ev, net.st));
    written.push(oi);
  }
  written.close();
  for (auto& t : rd) t.join();
  writer.join();
  return 0;
}
