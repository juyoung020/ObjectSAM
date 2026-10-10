// FSOW 텐서 묶음 읽기·쓰기(pytools/fsow.py 와 같은 형식).
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "core.h"

struct HostTensor {
  std::vector<int64_t> shape;
  uint8_t dtype = 0;  // 0 f32, 1 i32, 2 u8
  std::vector<uint8_t> data;
  int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
  const float* f32() const { FSO_ASSERT(dtype == 0, "not f32"); return reinterpret_cast<const float*>(data.data()); }
  const int32_t* i32() const { FSO_ASSERT(dtype == 1, "not i32"); return reinterpret_cast<const int32_t*>(data.data()); }
  const uint8_t* u8() const { FSO_ASSERT(dtype == 2, "not u8"); return data.data(); }
  static HostTensor make_f32(std::vector<int64_t> s, const float* p) {
    HostTensor t; t.shape = std::move(s); t.dtype = 0; t.data.resize(t.numel() * 4);
    if (p) memcpy(t.data.data(), p, t.data.size());
    return t;
  }
};
using TensorMap = std::map<std::string, HostTensor>;

inline TensorMap fsow_load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  FSO_ASSERT(f.good(), "open %s", path.c_str());
  char magic[8];
  f.read(magic, 8);
  FSO_ASSERT(!memcmp(magic, "FSOW0001", 8), "bad magic %s", path.c_str());
  uint32_t n; f.read(reinterpret_cast<char*>(&n), 4);
  TensorMap out;
  static const int isz[3] = {4, 4, 1};
  for (uint32_t i = 0; i < n; i++) {
    uint16_t ln; f.read(reinterpret_cast<char*>(&ln), 2);
    std::string name(ln, '\0'); f.read(name.data(), ln);
    uint8_t nd; f.read(reinterpret_cast<char*>(&nd), 1);
    HostTensor t; t.shape.resize(nd);
    f.read(reinterpret_cast<char*>(t.shape.data()), 8 * nd);
    f.read(reinterpret_cast<char*>(&t.dtype), 1);
    t.data.resize(size_t(t.numel()) * isz[t.dtype]);
    f.read(reinterpret_cast<char*>(t.data.data()), t.data.size());
    out.emplace(std::move(name), std::move(t));
  }
  FSO_ASSERT(f.good(), "truncated %s", path.c_str());
  return out;
}

inline void fsow_save(const std::string& path, const TensorMap& m) {
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    FSO_ASSERT(f.good(), "write %s", tmp.c_str());
    f.write("FSOW0001", 8);
    uint32_t n = m.size(); f.write(reinterpret_cast<char*>(&n), 4);
    for (auto& [name, t] : m) {
      uint16_t ln = name.size(); f.write(reinterpret_cast<char*>(&ln), 2); f.write(name.data(), ln);
      uint8_t nd = t.shape.size(); f.write(reinterpret_cast<char*>(&nd), 1);
      f.write(reinterpret_cast<const char*>(t.shape.data()), 8 * nd);
      f.write(reinterpret_cast<const char*>(&t.dtype), 1);
      f.write(reinterpret_cast<const char*>(t.data.data()), t.data.size());
    }
  }
  FSO_ASSERT(rename(tmp.c_str(), path.c_str()) == 0, "rename %s", path.c_str());
}
