// 공통: 오류 검사, 정밀도(활성값 형), 장치 버퍼.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                                                    \
  do {                                                                                                   \
    cudaError_t e_ = (x);                                                                                \
    if (e_ != cudaSuccess) {                                                                             \
      fprintf(stderr, "CUDA %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));            \
      std::abort();                                                                                      \
    }                                                                                                    \
  } while (0)
#define CUDNN_CHECK(x)                                                                                   \
  do {                                                                                                   \
    cudnnStatus_t e_ = (x);                                                                              \
    if (e_ != CUDNN_STATUS_SUCCESS) {                                                                    \
      fprintf(stderr, "cuDNN %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudnnGetErrorString(e_));          \
      std::abort();                                                                                      \
    }                                                                                                    \
  } while (0)
#define FSO_ASSERT(c, ...)                                                                               \
  do {                                                                                                   \
    if (!(c)) {                                                                                          \
      fprintf(stderr, "assert %s:%d %s: ", __FILE__, __LINE__, #c);                                      \
      fprintf(stderr, __VA_ARGS__);                                                                      \
      fprintf(stderr, "\n");                                                                             \
      std::abort();                                                                                      \
    }                                                                                                    \
  } while (0)

// 활성값·conv 입출력 형. 학습은 fp16(+동적 loss scaling, Ultralytics AMP 와 같음), 정합성 시험은 fp32(-DFSO_FP32).
// (bf16 은 가수 7 비트라 BN 직전 conv 출력 정밀도가 모자라 쓰지 않는다 — PyTorch bf16 AMP 도 같은 오차.)
#if defined(FSO_FP32)
using act_t = float;
constexpr cudnnDataType_t kActDnn = CUDNN_DATA_FLOAT;
#else
using act_t = __half;
constexpr cudnnDataType_t kActDnn = CUDNN_DATA_HALF;
#endif

__host__ __device__ inline float to_f(float x) { return x; }
__host__ __device__ inline float to_f(__half x) { return __half2float(x); }
template <class T> __host__ __device__ inline T from_f(float x);
template <> __host__ __device__ inline float from_f<float>(float x) { return x; }
template <> __host__ __device__ inline __half from_f<__half>(float x) { return __float2half(x); }

// 장치 메모리: 시작할 때 한 번 잡고 학습 중에는 할당하지 않는다.
struct DevMem {
  size_t bytes = 0;
  static size_t& total() { static size_t t = 0; return t; }
  static void* alloc(size_t n) {
    void* p = nullptr;
    if (n) CUDA_CHECK(cudaMalloc(&p, (n + 255) & ~size_t(255)));
    total() += n;
    return p;
  }
};
template <class T> T* dalloc(size_t n) { return static_cast<T*>(DevMem::alloc(n * sizeof(T))); }

inline int cdiv(int64_t a, int64_t b) { return int((a + b - 1) / b); }
