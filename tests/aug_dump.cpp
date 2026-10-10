// 증강 확인: 배치 하나를 로더 + GPU 증강으로 만들어 앞 n 장의 영상·정답 상자·정답 마스크를 FSOW 로 남긴다.
//   build/aug_dump out.fsow shard1.bin [shard2.bin ...]   (그림: pytools/view_aug.py)
#include "../src/augment.cuh"
#include "../src/fsow.h"

int main(int argc, char** argv) {
  FSO_ASSERT(argc >= 3, "usage: aug_dump out.fsow shards...");
  DataCfg dc;
  for (int i = 2; i < argc; i++) dc.shards.push_back(argv[i]);
  dc.batch = 8;
  dc.seed = 1;
  Dataset ds;
  ds.open(dc);
  ds.start(2);
  int64_t bi;
  BatchBuf* hb = ds.acquire(bi);
  const int B = dc.batch, S = dc.imgsz, G = 256, Hp = S / 4, Wp = S / 4;
  BatchBuf db;
  db.alloc_dev(*hb);
  CUDA_CHECK(cudaMemcpy(db.hdr, hb->hdr, hb->meta_bytes(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db.img, hb->img, hb->hdr->img_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db.mask, hb->mask, hb->hdr->mask_words * 4, cudaMemcpyHostToDevice));
  printf("batch %ld: %d sources, %d instances, img %.1f MB, masks %.1f MB, dropped %ld\n", bi, hb->hdr->nsrc,
         hb->hdr->ninst, hb->hdr->img_bytes / 1e6, hb->hdr->mask_words * 4 / 1e6, ds.n_dropped_cap.load());
  AugOut o;
  o.G = G; o.Hp = Hp; o.Wp = Wp;
  o.img = dalloc<act_t>(int64_t(B) * S * S * 3);
  o.gt.n_gt = dalloc<int>(B);
  o.gt.box = dalloc<float4>(B * G);
  o.gt.mask = dalloc<uint32_t>(int64_t(B) * G * Hp * Wp / 32);
  o.gt_inst = dalloc<int>(B * G);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  augment(db, B, S, o, 0);
  cudaEventRecord(e0);
  for (int k = 0; k < 10; k++) augment(db, B, S, o, 0);
  cudaEventRecord(e1);
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  printf("augment: %.3f ms per batch of %d\n", ms / 10, B);
  int n = std::min(B, 4);
  std::vector<act_t> img(int64_t(n) * S * S * 3);
  CUDA_CHECK(cudaMemcpy(img.data(), o.img, img.size() * sizeof(act_t), cudaMemcpyDeviceToHost));
  std::vector<int> ng(B);
  CUDA_CHECK(cudaMemcpy(ng.data(), o.gt.n_gt, B * 4, cudaMemcpyDeviceToHost));
  std::vector<float4> box(B * G);
  CUDA_CHECK(cudaMemcpy(box.data(), o.gt.box, box.size() * 16, cudaMemcpyDeviceToHost));
  std::vector<uint32_t> bits(int64_t(n) * G * Hp * Wp / 32);
  CUDA_CHECK(cudaMemcpy(bits.data(), o.gt.mask, bits.size() * 4, cudaMemcpyDeviceToHost));
  TensorMap out;
  HostTensor ti;
  ti.shape = {n, S, S, 3}; ti.dtype = 2; ti.data.resize(img.size());
  for (size_t k = 0; k < img.size(); k++) ti.data[k] = uint8_t(std::min(255.f, std::max(0.f, to_f(img[k]) * 255.f + 0.5f)));
  out["img"] = ti;
  for (int i = 0; i < n; i++) {
    printf("sample %d: %d gt\n", i, ng[i]);
    std::vector<float> bx(ng[i] * 4);
    for (int j = 0; j < ng[i]; j++) { float4 b = box[i * G + j]; bx[4 * j] = b.x; bx[4 * j + 1] = b.y; bx[4 * j + 2] = b.z; bx[4 * j + 3] = b.w; }
    out["box" + std::to_string(i)] = HostTensor::make_f32({ng[i], 4}, bx.data());
    HostTensor tm;
    tm.shape = {ng[i], Hp, Wp}; tm.dtype = 2; tm.data.resize(size_t(ng[i]) * Hp * Wp);
    for (int j = 0; j < ng[i]; j++)
      for (int p = 0; p < Hp * Wp; p++)
        tm.data[size_t(j) * Hp * Wp + p] = bits[(int64_t(i) * G + j) * (Hp * Wp / 32) + p / 32] >> (p % 32) & 1u;
    out["mask" + std::to_string(i)] = tm;
  }
  fsow_save(argv[1], out);
  ds.release(hb);
  ds.stop();
  printf("wrote %s\n", argv[1]);
  return 0;
}
