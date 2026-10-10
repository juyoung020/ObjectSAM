// 학습 데이터: FSR3 샤드(정답 마스크·상자만, 변환기가 배경을 이미 뺐다) → 배치 준비(CPU 스레드) → pinned 버퍼.
//
// 배치마다 출력 영상 B 장, 장마다 원본 4 장 mosaic(Ultralytics Mosaic + RandomPerspective 와 같은 난수 규칙).
// CPU 는 원본 디코딩(JPEG→RGB, RLE→상자 영역 비트맵)과 난수 매개변수만 만든다. 화소 일은 GPU(augment.cu).
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core.h"

// ---- 배치 버퍼 배치(호스트 pinned 와 장치에 같은 모양으로 둔다)
struct SrcInfo {        // 원본 영상 하나
  int64_t img_off;      // 영상 덩어리 안 바이트 위치(RGB u8, 행 iw*3)
  int iw, ih;           // 영상 크기(긴 변 1024)
  int mw, mh;           // 마스크 해상도(긴 변 512)
};
struct InstInfo {       // 정답 후보 하나(원본 기준)
  int src;              // SrcInfo 번호
  int sample;           // 출력 영상 번호
  int tile;             // 0..3
  float box[4];         // 원본 영상 화소 xyxy
  int mx0, my0, mcw, mch;  // 마스크 상자(마스크 화소): 시작, 크기
  int wpr;              // 비트맵 한 행의 u32 수
  int64_t mask_off;     // 마스크 덩어리 안 u32 위치
};
struct SampleInfo {     // 출력 영상 하나
  int src[4];           // 타일별 원본(없으면 −1)
  int ta[4][4];         // 캔버스 위 타일 사각형 x1a,y1a,x2a,y2a
  int tb[4][2];         // 원본 안 시작 x1b,y1b
  float s, tx, ty;      // 캔버스(2S) → 출력(S): out = s·u + tx
  float hsv[3];         // 이득
  int flip;             // 좌우 뒤집기
  int inst0, ninst;     // InstInfo 범위
};
struct BatchHeader {
  int B, S;
  int nsrc, ninst;
  int64_t img_bytes, mask_words;
};

struct DataCfg {
  std::vector<std::string> shards;  // .bin 경로들(.idx 같은 이름)
  int batch = 16, imgsz = 1024;
  int workers = 24;
  int max_inst_per_src = 256;       // 원본 한 장에서 쓰는 정답 수 상한(넓이 큰 순)
  float mosaic = 1.f, scale = 0.5f, translate = 0.1f, fliplr = 0.5f;
  float hsv_h = 0.015f, hsv_s = 0.7f, hsv_v = 0.4f;
  uint64_t seed = 0;
};

// 배치 버퍼 하나(pinned). 장치 쪽은 같은 크기로 학습기가 잡는다.
struct BatchBuf {
  BatchHeader* hdr = nullptr;
  SampleInfo* smp = nullptr;
  SrcInfo* src = nullptr;
  InstInfo* inst = nullptr;
  uint8_t* img = nullptr;
  uint32_t* mask = nullptr;
  // 용량
  int max_src = 0, max_inst = 0;
  int64_t img_cap = 0, mask_cap = 0;
  void alloc_host(int B, int S, int max_inst_per_src);
  void alloc_dev(const BatchBuf& like);
  size_t meta_bytes() const;
};

class Dataset {
 public:
  void open(const DataCfg& c);
  int64_t size() const { return int64_t(train_.size()); }   // 학습 영상 수
  int64_t batches_per_epoch() const { return size() / cfg.batch; }
  // 백그라운드 준비 시작. 버퍼 slots 개를 돌려 쓴다.
  void start(int slots);
  // 다음 배치(차례대로). 준비될 때까지 기다린다. 다 쓰면 release.
  BatchBuf* acquire(int64_t& batch_index);
  void release(BatchBuf* b);
  void stop();
  void set_epoch(int64_t first_batch);  // 학습 재개: 이 배치 번호부터
  DataCfg cfg;
  // 통계
  std::atomic<int64_t> n_inst_total{0}, n_dropped_cap{0};
  double wait_ms = 0;

 private:
  struct Rec { int shard; int64_t off, len; };
  struct Shard { const uint8_t* base = nullptr; size_t len = 0; int fd = -1; };
  std::vector<Shard> shards_;
  std::vector<Rec> train_;
  // 배치 준비
  void fill(BatchBuf& b, int64_t batch_index);
  void decode_src(const Rec& r, BatchBuf& b, int si, int sample, int tile, int64_t img_off, int64_t& mask_cursor,
                  std::mutex& mu);
  std::vector<BatchBuf> bufs_;
  std::vector<int64_t> buf_batch_;
  std::vector<int> state_;  // 0 빔, 1 준비 중, 2 준비됨, 3 사용 중
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::thread> threads_;
  std::atomic<bool> stop_{false};
  int64_t next_fill_ = 0, next_take_ = 0;
  std::vector<int64_t> perm_epoch_;  // 현재 epoch 의 순서
  int64_t perm_for_epoch_ = -1;
  std::vector<int64_t> perm(int64_t epoch);
  std::mutex perm_mu_;
};
