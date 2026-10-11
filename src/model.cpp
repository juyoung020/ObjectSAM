#include "model.h"

#include <algorithm>
#include <set>

#include "misc.cuh"

// ---------------------------------------------------------------- 파라미터

ParamRef& Params::add(const std::string& name, PKind k, std::vector<int64_t> shape, int layout, int cpad) {
  FSO_ASSERT(!refs.count(name), "dup param %s", name.c_str());
  int64_t n = 1;
  for (auto s : shape) n *= s;
  if (cpad) n = n / shape[1] * cpad;
  ParamRef r{k, size[k], n, shape, layout, cpad};
  size[k] += n;
  order.push_back(name);
  return refs[name] = r;
}

void Params::allocate() {
  for (int k = 0; k < P_NKIND; k++) {
    val[k] = dalloc<float>(size[k]);
    if (k != P_RUN) grad[k] = dalloc<float>(size[k]);
  }
  w_act = dalloc<act_t>(size[P_W]);
  gw_act = dalloc<act_t>(size[P_W]);
}

namespace {
// conv OIHW (K,C,R,S) ↔ KRSC(채널 Cp ≥ C, 남는 채널은 0). convT (Cin,Cout,kh,kw) 도 같은 식(K=Cin, C=Cout).
void to_krsc(const float* src, float* dst, const std::vector<int64_t>& s, int64_t Cp) {
  int64_t K = s[0], C = s[1], R = s[2], S = s[3];
  for (int64_t k = 0; k < K; k++)
    for (int64_t c = 0; c < Cp; c++)
      for (int64_t r = 0; r < R; r++)
        for (int64_t q = 0; q < S; q++)
          dst[((k * R + r) * S + q) * Cp + c] = c < C ? src[((k * C + c) * R + r) * S + q] : 0.f;
}
void from_krsc(const float* src, float* dst, const std::vector<int64_t>& s, int64_t Cp) {
  int64_t K = s[0], C = s[1], R = s[2], S = s[3];
  for (int64_t k = 0; k < K; k++)
    for (int64_t c = 0; c < C; c++)
      for (int64_t r = 0; r < R; r++)
        for (int64_t q = 0; q < S; q++) dst[((k * C + c) * R + r) * S + q] = src[((k * R + r) * S + q) * Cp + c];
}
}  // namespace

void Params::load(const TensorMap& m, const std::string& prefix) {
  std::vector<float> buf;
  for (auto& name : order) {
    auto& r = refs.at(name);
    auto it = m.find(prefix + name);
    FSO_ASSERT(it != m.end(), "missing %s%s", prefix.c_str(), name.c_str());
    FSO_ASSERT(it->second.shape == r.torch_shape, "shape %s", name.c_str());
    const float* src = it->second.f32();
    buf.assign(src, src + it->second.numel());
    buf.resize(r.n);
    if (r.layout) to_krsc(src, buf.data(), r.torch_shape, r.cpad ? r.cpad : r.torch_shape[1]);
    CUDA_CHECK(cudaMemcpy(val[r.kind] + r.off, buf.data(), r.n * 4, cudaMemcpyHostToDevice));
  }
}

TensorMap Params::save(const std::string& prefix, bool grads) const {
  TensorMap out;
  std::vector<float> buf, tmp;
  for (auto& name : order) {
    auto& r = refs.at(name);
    if (grads && r.kind == P_RUN) continue;
    buf.resize(r.n);
    CUDA_CHECK(cudaMemcpy(buf.data(), (grads ? grad[r.kind] : val[r.kind]) + r.off, r.n * 4, cudaMemcpyDeviceToHost));
    if (r.layout) {
      tmp = buf;
      int64_t n = 1;
      for (auto d : r.torch_shape) n *= d;
      buf.resize(n);
      from_krsc(tmp.data(), buf.data(), r.torch_shape, r.cpad ? r.cpad : r.torch_shape[1]);
    }
    out[prefix + name] = HostTensor::make_f32(r.torch_shape, buf.data());
  }
  return out;
}

// ---------------------------------------------------------------- 그래프 만들기

Act* Model::new_act(const std::string& n, int C, int H, int W) {
  FSO_ASSERT(!acts_.count(n), "dup act %s", n.c_str());
  auto a = std::make_unique<Act>();
  a->name = n; a->C = C; a->H = H; a->W = W;
  Act* p = a.get();
  acts_[n] = std::move(a);
  return p;
}

void Model::conv_bn(const std::string& name, Act* in, int K, int k, int s, std::vector<DestSpec> dests, Act* res,
                    std::vector<std::string> fused, std::vector<int> fused_k) {
  auto nd = std::make_unique<Node>();
  nd->kind = Node::CONVBN;
  nd->name = name;
  nd->in = in;
  nd->K = K;
  std::vector<std::string> pre = fused.empty() ? std::vector<std::string>{name} : fused;
  std::vector<int> ks;
  if (fused.empty()) ks = {K};
  else {
    // 합친 머리 conv: 상자 c2, 점수 128, 마스크 계수 32 (YOLOv8s-seg 머리)
    ks = fused_k;
    FSO_ASSERT(ks.size() == pre.size() && ks[0] + ks[1] + ks[2] == K, "fused head K");
  }
  int tc = in->name == "in" ? 3 : in->C;  // 입력 영상은 3 채널을 8 로 채워 둔다
  for (size_t i = 0; i < pre.size(); i++) {
    params.add(pre[i] + ".conv.weight", P_W, {ks[i], tc, k, k}, 1, tc != in->C ? in->C : 0);
    nd->wnames.push_back(pre[i] + ".conv.weight");
    nd->bnnames.push_back(pre[i] + ".bn");
  }
  for (auto& p : pre) params.add(p + ".bn.weight", P_GAMMA, {(int64_t)ks[&p - &pre[0]]});
  for (auto& p : pre) params.add(p + ".bn.bias", P_BIAS, {(int64_t)ks[&p - &pre[0]]});
  for (auto& p : pre) params.add(p + ".bn.running_mean", P_RUN, {(int64_t)ks[&p - &pre[0]]});
  for (auto& p : pre) params.add(p + ".bn.running_var", P_RUN, {(int64_t)ks[&p - &pre[0]]});
  ConvGeom g{cfg.batch, in->H, in->W, in->C, K, k, k, s, k / 2};
  nd->conv.init(h_, g);
  for (auto& d : dests) {
    if (d.c1 == 0) d.c1 = K;
    int P = g.P() * d.up, Q = g.Q() * d.up;
    FSO_ASSERT(d.a->H == P && d.a->W == Q, "%s → %s: shape %dx%d vs %dx%d", name.c_str(), d.a->name.c_str(), P, Q,
               d.a->H, d.a->W);
    FSO_ASSERT(d.choff + d.c1 - d.c0 <= d.a->C, "%s dest channels", name.c_str());
  }
  nd->dests = dests;
  nd->res = res;
  nodes_.push_back(std::move(nd));
}

Act* Model::c2f(const std::string& name, Act* in, int c2, int n, bool shortcut, std::vector<DestSpec> out_dests) {
  int c = c2 / 2, H = in->H, W = in->W;
  Act* cat = new_act(name + ".cat", (2 + n) * c, H, W);
  Act* bin = new_act(name + ".m0.in", c, H, W);
  conv_bn(name + ".cv1", in, 2 * c, 1, 1, {{cat, 0, 0, 2 * c, 1}, {bin, 0, c, 2 * c, 1}});
  for (int j = 0; j < n; j++) {
    std::string mj = name + ".m." + std::to_string(j);
    Act* t = new_act(mj + ".t", c, H, W);
    conv_bn(mj + ".cv1", bin, c, 3, 1, {{t, 0, 0, c, 1}});
    std::vector<DestSpec> d = {{cat, (2 + j) * c, 0, c, 1}};
    Act* next = nullptr;
    if (j + 1 < n) { next = new_act(name + ".m" + std::to_string(j + 1) + ".in", c, H, W); d.push_back({next, 0, 0, c, 1}); }
    conv_bn(mj + ".cv2", t, c, 3, 1, d, shortcut ? bin : nullptr);
    bin = next;
  }
  conv_bn(name + ".cv2", cat, c2, 1, 1, out_dests);
  return cat;
}

void Model::head_conv(const std::string& name, Act* in, int K, Act* out) {
  auto nd = std::make_unique<Node>();
  nd->kind = Node::HEADCONV;
  nd->name = name;
  nd->in = in;
  nd->K = K;
  nd->out = out;
  params.add(name + ".weight", P_W, {K, in->C, 1, 1}, 1);
  params.add(name + ".bias", P_BIAS, {K});
  nd->wnames = {name + ".weight"};
  nd->bias = name + ".bias";
  nd->conv.init(h_, ConvGeom{cfg.batch, in->H, in->W, in->C, K, 1, 1, 1, 0});
  nodes_.push_back(std::move(nd));
}

void Model::build(const ModelCfg& c, cudnnHandle_t h, cudaStream_t st) {
  cfg = c;
  h_ = h;
  st_ = st;
  bn_init();
  const int S = cfg.imgsz;
  FSO_ASSERT(S % 32 == 0, "imgsz must be a multiple of 32");
  auto M = [](int i) { return "model." + std::to_string(i); };
  Act* in = new_act("in", kInC, S, S);  // RGB + 0 채움(텐서코어 정렬)
  in->need_grad = false;
  // 백본
  Act* a0 = new_act("a0", 32, S / 2, S / 2);
  conv_bn(M(0), in, 32, 3, 2, {{a0, 0, 0, 0, 1}});
  Act* a1 = new_act("a1", 64, S / 4, S / 4);
  conv_bn(M(1), a0, 64, 3, 2, {{a1, 0, 0, 0, 1}});
  Act* a2 = new_act("a2", 64, S / 4, S / 4);
  c2f(M(2), a1, 64, 1, true, {{a2, 0, 0, 0, 1}});
  Act* a3 = new_act("a3", 128, S / 8, S / 8);
  conv_bn(M(3), a2, 128, 3, 2, {{a3, 0, 0, 0, 1}});
  // 목 버퍼(concat): cat11 = [up(9) 512, 6 256] @S/16, cat14 = [up(12) 256, 4 128] @S/8,
  //                  cat17 = [16 128, 12 256] @S/16, cat20 = [19 256, 9 512] @S/32
  Act* cat11 = new_act("cat11", 768, S / 16, S / 16);
  Act* cat14 = new_act("cat14", 384, S / 8, S / 8);
  Act* cat17 = new_act("cat17", 384, S / 16, S / 16);
  Act* cat20 = new_act("cat20", 768, S / 32, S / 32);
  Act* a4 = new_act("a4", 128, S / 8, S / 8);
  c2f(M(4), a3, 128, 2, true, {{a4, 0, 0, 0, 1}, {cat14, 256, 0, 0, 1}});
  Act* a5 = new_act("a5", 256, S / 16, S / 16);
  conv_bn(M(5), a4, 256, 3, 2, {{a5, 0, 0, 0, 1}});
  Act* a6 = new_act("a6", 256, S / 16, S / 16);
  c2f(M(6), a5, 256, 2, true, {{a6, 0, 0, 0, 1}, {cat11, 512, 0, 0, 1}});
  Act* a7 = new_act("a7", 512, S / 32, S / 32);
  conv_bn(M(7), a6, 512, 3, 2, {{a7, 0, 0, 0, 1}});
  Act* a8 = new_act("a8", 512, S / 32, S / 32);
  c2f(M(8), a7, 512, 1, true, {{a8, 0, 0, 0, 1}});
  // SPPF
  Act* sp = new_act(M(9) + ".cat", 1024, S / 32, S / 32);
  conv_bn(M(9) + ".cv1", a8, 256, 1, 1, {{sp, 0, 0, 0, 1}});
  {
    auto nd = std::make_unique<Node>();
    nd->kind = Node::POOL;
    nd->name = M(9) + ".m";
    nd->cat = sp;
    nd->pc = 256;
    nodes_.push_back(std::move(nd));
  }
  conv_bn(M(9) + ".cv2", sp, 512, 1, 1, {{cat11, 0, 0, 0, 2}, {cat20, 256, 0, 0, 1}});
  // 목
  c2f(M(12), cat11, 256, 1, false, {{cat14, 0, 0, 0, 2}, {cat17, 128, 0, 0, 1}});
  Act* p3 = new_act("p3", 128, S / 8, S / 8);
  c2f(M(15), cat14, 128, 1, false, {{p3, 0, 0, 0, 1}});
  conv_bn(M(16), p3, 128, 3, 2, {{cat17, 0, 0, 0, 1}});
  Act* p4 = new_act("p4", 256, S / 16, S / 16);
  c2f(M(18), cat17, 256, 1, false, {{p4, 0, 0, 0, 1}});
  conv_bn(M(19), p4, 256, 3, 2, {{cat20, 0, 0, 0, 1}});
  Act* p5 = new_act("p5", 512, S / 32, S / 32);
  c2f(M(21), cat20, 512, 1, false, {{p5, 0, 0, 0, 1}});
  // 머리: 단계마다 첫 3×3 conv 셋(상자 c2·점수 128·계수 32)을 하나로 합친다(같은 입력, BN 은 채널별).
  // Ultralytics Detect: c2 = max(16, ch0/4, 4·reg_max) (ch0 = P3 채널 128), c3 = max(ch0, min(nc, 100)) = 128.
  const int R4 = 4 * cfg.reg_max;
  const int c2 = std::max(16, std::max(128 / 4, R4));
  const int KH = c2 + 128 + 32;
  Act* P[3] = {p3, p4, p5};
  const std::string hd = M(22);
  for (int l = 0; l < 3; l++) {
    std::string L = std::to_string(l);
    int H = P[l]->H, W = P[l]->W;
    Act* hb = new_act("h" + L + ".b1", c2, H, W);
    Act* hc = new_act("h" + L + ".c1", 128, H, W);
    Act* hm = new_act("h" + L + ".m1", 32, H, W);
    conv_bn(hd + ".h" + L + ".0", P[l], KH, 3, 1,
            {{hb, 0, 0, c2, 1}, {hc, 0, c2, c2 + 128, 1}, {hm, 0, c2 + 128, KH, 1}}, nullptr,
            {hd + ".cv2." + L + ".0", hd + ".cv3." + L + ".0", hd + ".cv4." + L + ".0"}, {c2, 128, 32});
    Act* hb2 = new_act("h" + L + ".b2", c2, H, W);
    Act* hc2 = new_act("h" + L + ".c2", 128, H, W);
    Act* hm2 = new_act("h" + L + ".m2", 32, H, W);
    conv_bn(hd + ".cv2." + L + ".1", hb, c2, 3, 1, {{hb2, 0, 0, 0, 1}});
    conv_bn(hd + ".cv3." + L + ".1", hc, 128, 3, 1, {{hc2, 0, 0, 0, 1}});
    conv_bn(hd + ".cv4." + L + ".1", hm, 32, 3, 1, {{hm2, 0, 0, 0, 1}});
    head_[0][l] = new_act("out.box" + L, R4, H, W);
    head_[1][l] = new_act("out.cls" + L, 1, H, W);
    head_[2][l] = new_act("out.coef" + L, 32, H, W);
    head_conv(hd + ".cv2." + L + ".2", hb2, R4, head_[0][l]);
    head_conv(hd + ".cv3." + L + ".2", hc2, 1, head_[1][l]);
    head_conv(hd + ".cv4." + L + ".2", hm2, 32, head_[2][l]);
  }
  // proto: Conv3×3 → ConvTranspose 2×2 s2 (+bias) → Conv3×3 → Conv1×1(32)
  Act* pr1 = new_act("proto.1", 128, S / 8, S / 8);
  conv_bn(hd + ".proto.cv1", p3, 128, 3, 1, {{pr1, 0, 0, 0, 1}});
  Act* pru = new_act("proto.up", 128, S / 4, S / 4);
  {
    auto nd = std::make_unique<Node>();
    nd->kind = Node::CONVT;
    nd->name = hd + ".proto.upsample";
    nd->in = pr1;
    nd->out = pru;
    nd->K = 128;
    params.add(nd->name + ".weight", P_W, {128, 128, 2, 2}, 2);
    params.add(nd->name + ".bias", P_BIAS, {128});
    nd->wnames = {nd->name + ".weight"};
    nd->bias = nd->name + ".bias";
    nd->conv.init(h_, ConvGeom{cfg.batch, pr1->H, pr1->W, 128, 128, 2, 2, 2, 0}, true);
    nodes_.push_back(std::move(nd));
  }
  Act* pr2 = new_act("proto.2", 128, S / 4, S / 4);
  conv_bn(hd + ".proto.cv2", pru, 128, 3, 1, {{pr2, 0, 0, 0, 1}});
  proto_ = new_act("out.proto", 32, S / 4, S / 4);
  conv_bn(hd + ".proto.cv3", pr2, 32, 1, 1, {{proto_, 0, 0, 0, 1}});

  // ---- 할당
  params.allocate();
  const int64_t N = cfg.batch;
  int64_t max_z = 0, max_part = 0, max_c = 0;
  for (auto& [n, a] : acts_) {
    int64_t e = N * a->H * a->W * a->C;
    a->p = dalloc<act_t>(e);
    if (a->need_grad) a->g = dalloc<act_t>(e);
  }
  for (auto& nd : nodes_) {
    if (nd->kind == Node::CONVBN) {
      int64_t e = nd->conv.out_elems();
      nd->z = dalloc<act_t>(e);
      max_z = std::max(max_z, e);
      max_c = std::max<int64_t>(max_c, nd->K);
      nd->mean = dalloc<float>(nd->K);
      nd->istd = dalloc<float>(nd->K);
      nd->scale = dalloc<float>(nd->K);
      nd->shift = dalloc<float>(nd->K);
    } else if (nd->kind == Node::HEADCONV || nd->kind == Node::CONVT) {
      int64_t M = N * nd->out->H * nd->out->W;
      max_part = std::max<int64_t>(max_part, int64_t(chan_sum_blocks(M, nd->K)) * nd->K);
    }
  }
  dz_[0] = dalloc<act_t>(max_z);
  dz_[1] = dalloc<act_t>(max_z);
  chpart_ = dalloc<float>(max_part);
  sums_ = dalloc<float>(2 * max_c);
  {
    int64_t e = N * sp->H * sp->W * 256;
    pool_a_ = dalloc<float>(e);
    pool_b_ = dalloc<float>(e);
  }
  plan_backward();

  // ---- 알고리즘 재기(실제 버퍼로, 한 번)
  size_t lim = size_t(512) << 20, need = 0;  // 알고리즘 재기 작업 공간 상한
  void* tws = nullptr;
  CUDA_CHECK(cudaMalloc(&tws, lim));
  act_t* scratch_dx = nullptr;
  for (auto& nd : nodes_) {
    if (nd->kind == Node::POOL) continue;
    act_t* w = params.wact(nd->wnames[0]);
    act_t* gw = params.gwact(nd->wnames[0]);
    act_t* y = nd->kind == Node::CONVBN ? nd->z : nd->out->p;
    act_t* dy = nd->kind == Node::CONVBN ? dz_[0] : nd->out->g;
    act_t* dx = nd->in->g;
    if (!dx) {
      if (!scratch_dx) CUDA_CHECK(cudaMalloc(&scratch_dx, N * nd->in->H * nd->in->W * nd->in->C * sizeof(act_t)));
      dx = scratch_dx;
    }
    need = std::max(need, nd->conv.tune(nd->in->p, w, y, dx, gw, dy, tws, lim, nd->dgrad_accum ? 1.f : 0.f));
  }
  CUDA_CHECK(cudaFree(tws));
  if (scratch_dx) CUDA_CHECK(cudaFree(scratch_dx));
  ws_bytes = need;
  ws_ = DevMem::alloc(need);
  ws_side_ = DevMem::alloc(need);
  {
    int lo, hi;  // 곁 스트림(wgrad)은 가장 낮은 우선순위: 주 스트림(BN 역방향·dgrad, 임계 경로)이 먼저
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&side_, cudaStreamNonBlocking, lo));
  }
  for (int i = 0; i < 2; i++) {
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_dz_[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_wg_[i], cudaEventDisableTiming));
  }
  CUDA_CHECK(cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming));
}

void Model::plan_backward() {
  std::set<const Act*> written;
  for (int l = 0; l < 3; l++)
    for (int k = 0; k < 3; k++) written.insert(head_[k][l]);
  written.insert(proto_);
  for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
    Node& nd = **it;
    if (nd.kind == Node::POOL) {
      FSO_ASSERT(written.count(nd.cat), "pool grad before cat grad");
      continue;
    }
    if (nd.kind == Node::CONVBN)
      for (auto& d : nd.dests) FSO_ASSERT(written.count(d.a), "%s: grad of %s not ready", nd.name.c_str(), d.a->name.c_str());
    else
      FSO_ASSERT(written.count(nd.out), "%s: out grad not ready", nd.name.c_str());
    if (nd.res) { nd.res_accum = written.count(nd.res) > 0; written.insert(nd.res); }
    if (nd.in->need_grad) { nd.dgrad_accum = written.count(nd.in) > 0; written.insert(nd.in); }
  }
}

void Model::sync_weights() { f32_to_act(params.val[P_W], params.w_act, params.size[P_W], st_); }

namespace {
DestList mk_dl(const Node& nd, bool grads) {
  DestList dl;
  for (auto& d : nd.dests) {
    Dest& o = dl.d[dl.n++];
    o.p = d.a->p; o.g = grads ? d.a->g : nullptr;
    o.ctot = d.a->C; o.choff = d.choff; o.c0 = d.c0; o.c1 = d.c1; o.up = d.up;
  }
  return dl;
}
}  // namespace

void Model::forward(bool train) {
  for (auto& ndp : nodes_) {
    Node& nd = *ndp;
    switch (nd.kind) {
      case Node::CONVBN: {
        nd.conv.fwd(nd.in->p, params.wact(nd.wnames[0]), nd.z, ws_, ws_bytes);
        BnShape s{cfg.batch, nd.conv.geom().P(), nd.conv.geom().Q(), nd.K};
        const std::string& b = nd.bnnames[0];
        float* g = params.ptr(b + ".weight");
        float* be = params.ptr(b + ".bias");
        float* rm = params.ptr(b + ".running_mean");
        float* rv = params.ptr(b + ".running_var");
        if (train)
          bn_forward_stats(nd.z, s, g, be, rm, rv, cfg.bn_momentum, cfg.bn_eps, nd.mean, nd.istd, nd.scale,
                           nd.shift, st_);
        else
          bn_eval_scale(nd.K, g, be, rm, rv, cfg.bn_eps, nd.scale, nd.shift, st_);
        bn_forward_apply(nd.z, s, nd.scale, nd.shift, nd.res ? nd.res->p : nullptr, mk_dl(nd, false), true, st_);
        break;
      }
      case Node::HEADCONV:
      case Node::CONVT: {
        nd.conv.fwd(nd.in->p, params.wact(nd.wnames[0]), nd.out->p, ws_, ws_bytes);
        add_bias(nd.out->p, int64_t(cfg.batch) * nd.out->H * nd.out->W, nd.K, params.ptr(nd.bias), st_);
        break;
      }
      case Node::POOL:
        sppf_pool_fwd(nd.cat->p, cfg.batch, nd.cat->H, nd.cat->W, nd.pc, st_);
        break;
    }
  }
}

// 곁 스트림에서 wgrad: dy(버퍼 buf) 가 준비된 뒤 시작하고, 끝나면 ev_wg_[buf] 로 알린다.
void Model::wgrad_side(Node& nd, const act_t* dy, int buf) {
  CUDA_CHECK(cudaEventRecord(ev_dz_[buf], st_));
  CUDA_CHECK(cudaStreamWaitEvent(side_, ev_dz_[buf], 0));
  CUDNN_CHECK(cudnnSetStream(h_, side_));
  nd.conv.wgrad(nd.in->p, dy, params.gwact(nd.wnames[0]), ws_side_, ws_bytes);
  CUDNN_CHECK(cudnnSetStream(h_, st_));
  CUDA_CHECK(cudaEventRecord(ev_wg_[buf], side_));
}

void Model::backward(bool accum) {
  CUDA_CHECK(cudaEventRecord(ev_fork_, st_));
  CUDA_CHECK(cudaStreamWaitEvent(side_, ev_fork_, 0));
  bool busy[2] = {false, false};  // dz 버퍼를 읽는 wgrad 가 곁 스트림에 걸려 있나
  int buf = 0;
  for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
    Node& nd = **it;
    switch (nd.kind) {
      case Node::CONVBN: {
        BnShape s{cfg.batch, nd.conv.geom().P(), nd.conv.geom().Q(), nd.K};
        const std::string& b = nd.bnnames[0];
        if (busy[buf]) CUDA_CHECK(cudaStreamWaitEvent(st_, ev_wg_[buf], 0));
        bn_backward(nd.z, s, params.ptr(b + ".weight"), nd.mean, nd.istd, nd.scale, nd.shift, mk_dl(nd, true), true,
                    sums_, dz_[buf], params.gptr(b + ".weight"), params.gptr(b + ".bias"), accum,
                    nd.res ? nd.res->g : nullptr, nd.res_accum, st_);
        wgrad_side(nd, dz_[buf], buf);
        busy[buf] = true;
        if (nd.in->need_grad)
          nd.conv.dgrad(dz_[buf], params.wact(nd.wnames[0]), nd.in->g, ws_, ws_bytes, nd.dgrad_accum ? 1.f : 0.f);
        buf ^= 1;
        break;
      }
      case Node::HEADCONV:
      case Node::CONVT: {
        // dy = 출력 기울기 버퍼(이 뒤로 바뀌지 않는다): dz 버퍼 차례와 상관없이 곁 스트림에
        int64_t M = int64_t(cfg.batch) * nd.out->H * nd.out->W;
        chan_sum(nd.out->g, M, nd.K, chpart_, params.gptr(nd.bias), accum, st_);
        CUDA_CHECK(cudaEventRecord(ev_fork_, st_));
        CUDA_CHECK(cudaStreamWaitEvent(side_, ev_fork_, 0));
        CUDNN_CHECK(cudnnSetStream(h_, side_));
        nd.conv.wgrad(nd.in->p, nd.out->g, params.gwact(nd.wnames[0]), ws_side_, ws_bytes);
        CUDNN_CHECK(cudnnSetStream(h_, st_));
        if (nd.in->need_grad)
          nd.conv.dgrad(nd.out->g, params.wact(nd.wnames[0]), nd.in->g, ws_, ws_bytes, nd.dgrad_accum ? 1.f : 0.f);
        break;
      }
      case Node::POOL:
        sppf_pool_bwd(nd.cat->p, nd.cat->g, cfg.batch, nd.cat->H, nd.cat->W, nd.pc, pool_a_, pool_b_, st_);
        break;
    }
  }
  CUDA_CHECK(cudaEventRecord(ev_join_, side_));
  CUDA_CHECK(cudaStreamWaitEvent(st_, ev_join_, 0));
  act_accum_f32(params.gw_act, params.grad[P_W], params.size[P_W], accum, st_);
}

void Model::print_convs() const {
  double tot[3] = {0, 0, 0};
  printf("%-34s %4s %4s %4s %4s %2s %2s | %7s %7s %7s ms | TFLOPS f/d/w\n", "conv", "H", "W", "C", "K", "k", "s", "fwd",
         "dgrad", "wgrad");
  for (auto& n : nodes_) {
    if (n->kind == Node::POOL) continue;
    auto& g = n->conv.geom();
    double fl = 2.0 * g.N * double(n->kind == Node::CONVT ? g.H * g.W : g.P() * g.Q()) * g.K * g.C * g.R * g.S;
    const float* t = n->conv.tuned_ms;
    printf("%-34s %4d %4d %4d %4d %2d %2d | %7.3f %7.3f %7.3f    | %5.1f %5.1f %5.1f\n", n->name.c_str(), g.H, g.W, g.C,
           g.K, g.R, g.stride, t[0], t[1], t[2], fl / t[0] / 1e9, fl / t[1] / 1e9, fl / t[2] / 1e9);
    for (int k = 0; k < 3; k++) tot[k] += t[k];
  }
  printf("total conv: fwd %.2f dgrad %.2f wgrad %.2f ms\n", tot[0], tot[1], tot[2]);
}

void Model::print_plan() const {
  int nconv = 0;
  for (auto& n : nodes_) nconv += n->kind != Node::POOL;
  printf("model: %zu nodes (%d conv), params W %ld γ %ld β/bias %ld run %ld, workspace %.1f MB, device total %.1f MB\n",
         nodes_.size(), nconv, params.size[P_W], params.size[P_GAMMA], params.size[P_BIAS], params.size[P_RUN],
         ws_bytes / 1e6, DevMem::total() / 1e6);
}
