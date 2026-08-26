// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "onnx/onnx_pb.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/logging.h"
#include "onnxoptimizer/passes/string_utils.h"
#include "onnxoptimizer/passes/tensor_content_hash.h"
#include "onnxoptimizer/passes/tensor_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

// Exploratory diagnostic (onnxsim issue #633): splits CSETensorHash/
// CSETensorCompare's cost by which of their two branches did the work --
// raw_data (a cheap byte hash / a single memcmp, the common case for real
// exported models) vs typed-field (a BLAKE3-digest-backed path, rarer). Off
// by default; shares pass.h's SetPassPhaseProfilingEnabled toggle so it
// switches on/off together with the pass-phase timers.
struct CSEHashCompareTiming {
  uint64_t raw_hash_calls = 0;
  double raw_hash_ms = 0.0;
  // Of raw_hash_calls above, how many were served from g_raw_hash_cache
  // below instead of recomputed. On real (raw_data-heavy) exported models,
  // measured 98%+ on onnxsim issue #633's repro -- most initializers are
  // unchanged, unmutated Tensor objects from one OptAndShape round to the
  // next, so their hash need only ever be computed once.
  uint64_t raw_hash_cache_hits = 0;
  double raw_hash_cache_hit_ms = 0.0;
  uint64_t raw_hash_cache_misses = 0;
  double raw_hash_cache_miss_ms = 0.0;
  uint64_t typed_hash_calls = 0;
  double typed_hash_ms = 0.0;
  uint64_t raw_compare_calls = 0;
  double raw_compare_ms = 0.0;
  uint64_t typed_compare_calls = 0;
  double typed_compare_ms = 0.0;
  // CSENodeHash/CSEEqual (below): the whole-node hash/equality machinery
  // that eliminate_common_subexpression's hash_map keys on. Their cost is
  // separate from raw_hash_*/typed_hash_* above -- those only fire for
  // nodes with a tensor-valued (t/ts) attribute (e.g. Constant), while
  // node_hash_ms/node_equal_ms cover every CSE-eligible node.
  uint64_t node_hash_calls = 0;
  double node_hash_ms = 0.0;
  // Of node_hash_ms, time spent specifically in attributeNames() (an
  // allocating std::vector<Symbol> return by value, see ir.h) plus sorting
  // it -- isolated separately since it is the one obviously allocation-heavy
  // step in an otherwise cheap hash.
  uint64_t node_hash_attrsort_calls = 0;
  double node_hash_attrsort_ms = 0.0;
  uint64_t node_equal_calls = 0;
  double node_equal_ms = 0.0;
  // Same attributeNames()+sort isolation as node_hash_attrsort_*, but
  // CSEEqual does it twice per call (once per side).
  uint64_t node_equal_attrsort_calls = 0;
  double node_equal_attrsort_ms = 0.0;
};
inline CSEHashCompareTiming g_cse_hash_compare_timing;
inline void ResetCSEHashCompareTiming() {
  g_cse_hash_compare_timing = {};
}
inline const CSEHashCompareTiming& GetCSEHashCompareTiming() {
  return g_cse_hash_compare_timing;
}

// The actual cache: memoizes CSETensorHash's raw_data-branch seed by
// Tensor::tensor_id() (never reused across distinct content -- fresh on
// every construction/assignment, see tensor.h), so an unmutated tensor's
// hash is computed once and reused on every later lookup instead of
// rescanning its raw bytes from scratch each time. This is
// eliminate_duplicate_initializer's dominant cost on raw_data-heavy models
// (see onnxsim issue #633) -- CSETensorCompare's own raw_data fast path
// (a single memcmp on a hash-bucket hit) was already cheap and is
// unaffected.
//
// Cleared via ClearRawHashCache(), with the same lifetime rules as
// tensor_content_hash.h's ClearTensorContentDigestCache (see that
// function's header comment for the full rationale): no onnx-optimizer
// pass mutates a retained tensor's content in place, so this safely
// outlives a single pass call, up to whatever scope the caller (see
// Optimizer::optimize(Graph&, ...)'s clear_tensor_digest_cache parameter)
// chooses to clear it at.
inline std::unordered_map<uint64_t, std::size_t> g_raw_hash_cache;
inline void ClearRawHashCache() {
  g_raw_hash_cache.clear();
}

// RAII: adds the scope's elapsed wall time to *ms and increments *calls on
// destruction, only when profiling is on -- a no-op pair of branches
// otherwise.
class ScopedCSETiming {
 public:
  ScopedCSETiming(uint64_t* calls, double* ms)
      : enabled_(GetPassPhaseProfilingEnabled()), calls_(calls), ms_(ms) {
    if (enabled_)
      start_ = std::chrono::steady_clock::now();
  }
  ~ScopedCSETiming() {
    if (enabled_) {
      const auto end = std::chrono::steady_clock::now();
      (*calls_)++;
      *ms_ += std::chrono::duration<double, std::milli>(end - start_).count();
    }
  }

 private:
  bool enabled_;
  uint64_t* calls_;
  double* ms_;
  std::chrono::steady_clock::time_point start_;
};

/// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
inline void hash_combine(std::size_t& seed) {}

template <typename Hasher, typename T, typename... Rest>
void hash_combine(std::size_t& seed, const Hasher& hasher, const T& v,
                  Rest... rest) {
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}

// Shared by HashRawDataBounded (raw_data tensors) and HashTypedFieldBounded
// (typed-field tensors, via their ParseTensorData<T>-parsed byte
// representation) below -- takes a generic byte span rather than a
// std::string specifically, since the typed-field caller has no std::string
// to hand it (its bytes live in a std::vector<T>). Cost is bounded by a
// small constant, however large the buffer is -- unlike g_raw_hash_cache
// above, a tensor's *first* hash is not a caching gap: it's paid exactly
// once no matter what (see that cache's own comment), so on raw_data-heavy
// models with large tensors it was this function's previous
// std::hash<std::string>() full-buffer scan, not a cache miss, that
// dominated eliminate_duplicate_initializer's cost (see onnxsim issue #633
// follow-up profiling).
//
// This is safe to make approximate: CSETensorHash only needs to be a good
// *bucketing* key. The sole source of truth for equality is
// CSETensorCompare's raw_data fast path -- a full lhs->raw() == rhs->raw()
// memcmp, run on every hash-bucket hit regardless of how the hash was
// computed -- so a lower-quality/sampled hash can never cause an incorrect
// merge. The only cost of a spurious collision (two distinct tensors that
// happen to match on the sampled bytes) is one extra, already-cheap memcmp
// against a candidate that turns out not to match.
inline std::size_t HashBytesBounded(const char* data, std::size_t n) {
  // Below this size, a full hash is already cheap -- and exact rather than
  // sampled, so small buffers keep today's collision behavior exactly.
  constexpr std::size_t kFullHashLimit = 4096;
  constexpr std::size_t kSampleWindow = 64;
  constexpr std::size_t kMaxSamples = 256;

  if (n <= kFullHashLimit) {
    return std::hash<std::string_view>()(std::string_view(data, n));
  }

  std::size_t seed = 0;
  hash_combine(seed, std::hash<std::size_t>(), n);
  // Evenly spaced windows across the buffer (always including offset 0),
  // capped at kMaxSamples regardless of n -- so this whole function costs
  // at most O(kMaxSamples * kSampleWindow) bytes hashed, a small constant,
  // whether the buffer is 5KB or 500MB.
  const std::size_t stride = std::max(kSampleWindow, n / kMaxSamples);
  for (std::size_t offset = 0; offset < n; offset += stride) {
    const std::size_t len = std::min(kSampleWindow, n - offset);
    hash_combine(seed, std::hash<std::string_view>(),
                 std::string_view(data + offset, len));
  }
  return seed;
}

// CSETensorHash's raw_data path (below): thin wrapper over HashBytesBounded.
inline std::size_t HashRawDataBounded(std::string_view raw) {
  return HashBytesBounded(raw.data(), raw.size());
}

struct SymbolCompare {
  bool operator()(const Symbol& lhs, const Symbol& rhs) {
    return static_cast<uint32_t>(lhs) < static_cast<uint32_t>(rhs);
  }
};

inline bool CSETensorCompare(const Tensor* lhs, const Tensor* rhs) {
  // lhs and rhs maybe nullptr
  if (!lhs) {
    return !rhs;
  } else if (!rhs) {
    return !lhs;
  }
  ONNX_ASSERT(!lhs->is_segment() && !rhs->is_segment());
  if (lhs->elem_type() != rhs->elem_type() || lhs->sizes() != rhs->sizes()) {
    return false;
  }

  if (lhs->is_raw_data() && rhs->is_raw_data()) {
    // Fast path: raw_data is always little-endian on disk regardless of host
    // byte order (unlike ParseTensorData's typed accessors below, which
    // return host-order values), so byte-identical raw_data always implies
    // value-identical data, on any host -- this can never produce a false
    // positive. Skips ParseTensorData<T>'s double copy (one to
    // un-const/byte-swap the string, one to convert it to a typed vector),
    // which otherwise dominates eliminate_duplicate_initializer's cost on
    // models with large raw_data initializers (see onnxsim issue #633).
    // This is already as cheap as a comparison can be (one memcmp via
    // std::string::operator==), so -- deliberately, see
    // tensor_content_hash.h's header comment -- it does NOT go through
    // TensorContentDigest: BLAKE3 would only add cost here, not save any
    // (most real models are raw_data-heavy, so this is the hot path).
    ScopedCSETiming _t(&g_cse_hash_compare_timing.raw_compare_calls,
                       &g_cse_hash_compare_timing.raw_compare_ms);
    return lhs->raw() == rhs->raw();
  }

  if (lhs->elem_type() != ONNX_NAMESPACE::TensorProto_DataType_STRING &&
      GetTrustTensorContentHash()) {
    // Neither side is raw_data (or they mismatch raw_data-ness) here, so
    // this is the typed-field case TensorContentDigest actually targets:
    // trusting digest equality as tensor equality avoids re-deriving
    // (ParseTensorData<T>) the tensor's full contents here, on top of
    // whatever CSETensorHash already did to find this candidate -- and the
    // digest is memoized per tensor pointer for the lifetime of this pass
    // call (see ClearTensorContentDigestCache), so this is normally a cache
    // hit, not a fresh BLAKE3 pass. See GetTrustTensorContentHash's own
    // comment for the (practically negligible) tradeoff, and how to
    // disable it.
    ScopedCSETiming _t(&g_cse_hash_compare_timing.typed_compare_calls,
                       &g_cse_hash_compare_timing.typed_compare_ms);
    return TensorContentDigest(*lhs) == TensorContentDigest(*rhs);
  }

#define DO_CASE(pb_type, cpp_type)                                        \
  case ONNX_NAMESPACE::TensorProto_DataType_##pb_type:                    \
    if (ParseTensorData<cpp_type>(lhs) != ParseTensorData<cpp_type>(rhs)) \
      return false;                                                       \
    break;

  switch (lhs->elem_type()) {
    DO_CASE(BOOL, bool)
    DO_CASE(INT8, int8_t)
    DO_CASE(INT16, int16_t)
    DO_CASE(INT32, int32_t)
    DO_CASE(INT64, int64_t)
    DO_CASE(UINT8, uint8_t)
    DO_CASE(UINT16, uint16_t)
    DO_CASE(UINT32, uint32_t)
    DO_CASE(UINT64, uint64_t)
    DO_CASE(FLOAT, float)
    DO_CASE(DOUBLE, double)
    DO_CASE(COMPLEX64, Complex64)
    DO_CASE(COMPLEX128, Complex128)
    DO_CASE(FLOAT16, Float16)
    DO_CASE(BFLOAT16, BFloat16)

#undef DO_CASE

    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      if (lhs->strings() != rhs->strings())
        return false;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      // tensor is empty
      break;
    default:
      return false;
  }
  return true;
}

inline bool IsSupportedByCSE(const Node* n) {
  if (!n) {
    return false;
  }
  const auto attribute_names = n->attributeNames();
  for (const auto& name : attribute_names) {
    auto kind = n->kindOf(name);
    switch (kind) {
      case AttributeKind::g:
      case AttributeKind::gs:
      case AttributeKind::tp:
      case AttributeKind::tps:
        return false;
      default:
        break;
    }
  }
  return true;
}

template <typename T>
struct CSEContainerHash {
  std::size_t operator()(const std::vector<T>& container) const {
    std::size_t seed = 0;
    hash_combine(seed, std::hash<std::string>(), std::string(typeid(T).name()),
                 std::hash<std::size_t>(), container.size());
    for (const auto& d : container) {
      hash_combine(seed, std::hash<T>(), d);
    }
    return seed;
  }
};

// Local copies of tensor_content_hash.cc's CanonicalizeZero: that file's
// version has internal (anonymous-namespace) linkage and so can't be called
// from this translation unit. See that file's comment on the original
// float/double overloads for why this matters (IEEE754 +0.0/-0.0 must hash
// equal, or CSE silently stops deduplicating Constant/initializer tensors
// that differ only in a zero's sign -- a real regression, see onnxsim PR
// #641). Complex64/Complex128 canonicalize both components.
inline float CanonicalizeZero(float v) {
  return v == 0.0f ? 0.0f : v;
}
inline double CanonicalizeZero(double v) {
  return v == 0.0 ? 0.0 : v;
}
inline Complex64 CanonicalizeZero(Complex64 v) {
  v.real_part = CanonicalizeZero(v.real_part);
  v.imaginary_part = CanonicalizeZero(v.imaginary_part);
  return v;
}
inline Complex128 CanonicalizeZero(Complex128 v) {
  v.real_part = CanonicalizeZero(v.real_part);
  v.imaginary_part = CanonicalizeZero(v.imaginary_part);
  return v;
}

// CSETensorHash's typed-field path (below): mirrors
// ComputeTensorContentDigest's per-dtype ParseTensorData<T> dispatch
// (tensor_content_hash.cc), including the same signed-zero canonicalization
// for FLOAT/DOUBLE/COMPLEX64/COMPLEX128, but hashes each parsed vector's
// bytes with HashBytesBounded instead of a full BLAKE3 pass -- same
// bucketing-only-needs-to-be-approximate reasoning as HashRawDataBounded
// above (CSETensorCompare's typed-field path, or TensorContentDigest
// equality under GetTrustTensorContentHash(), remains the sole source of
// truth for equality on any hash-bucket hit). ParseTensorData<T> itself is
// already cheap here (a std::vector copy, no byte-swapping unlike the
// raw_data branch) -- it was the subsequent full-buffer BLAKE3 hash that
// dominated eliminate_duplicate_initializer's cost on typed-field-heavy
// models (see onnxsim issue #633 follow-up profiling).
inline std::size_t HashTypedFieldBounded(const Tensor& tensor) {
  const int32_t elem_type = tensor.elem_type();
  std::size_t seed = 0;
  hash_combine(seed, std::hash<int32_t>(), elem_type);
  hash_combine(seed, CSEContainerHash<int64_t>(), tensor.sizes());

#define HASH_CASE(pb_type, cpp_type)                                       \
  case ONNX_NAMESPACE::TensorProto_DataType_##pb_type: {                   \
    const auto values = ParseTensorData<cpp_type>(&tensor);                \
    seed ^= HashBytesBounded(reinterpret_cast<const char*>(values.data()), \
                             values.size() * sizeof(cpp_type)) +           \
            0x9e3779b9 + (seed << 6) + (seed >> 2);                        \
    break;                                                                 \
  }

#define HASH_CASE_CANON(pb_type, cpp_type)                                 \
  case ONNX_NAMESPACE::TensorProto_DataType_##pb_type: {                   \
    auto values = ParseTensorData<cpp_type>(&tensor);                      \
    for (auto& v : values)                                                 \
      v = CanonicalizeZero(v);                                             \
    seed ^= HashBytesBounded(reinterpret_cast<const char*>(values.data()), \
                             values.size() * sizeof(cpp_type)) +           \
            0x9e3779b9 + (seed << 6) + (seed >> 2);                        \
    break;                                                                 \
  }

  switch (elem_type) {
    HASH_CASE(INT8, int8_t)
    HASH_CASE(INT16, int16_t)
    HASH_CASE(INT32, int32_t)
    HASH_CASE(INT64, int64_t)
    HASH_CASE(UINT8, uint8_t)
    HASH_CASE(UINT16, uint16_t)
    HASH_CASE(UINT32, uint32_t)
    HASH_CASE(UINT64, uint64_t)
    HASH_CASE(FLOAT16, Float16)
    HASH_CASE(BFLOAT16, BFloat16)

    HASH_CASE_CANON(FLOAT, float)
    HASH_CASE_CANON(DOUBLE, double)
    HASH_CASE_CANON(COMPLEX64, Complex64)
    HASH_CASE_CANON(COMPLEX128, Complex128)

#undef HASH_CASE
#undef HASH_CASE_CANON

    case ONNX_NAMESPACE::TensorProto_DataType_BOOL: {
      // std::vector<bool> is bit-packed, not a contiguous array of
      // byte-sized elements -- pack one byte per value first, matching
      // ComputeTensorContentDigest's BOOL case.
      const auto values = ParseTensorData<bool>(&tensor);
      std::string packed(values.size(), '\0');
      for (std::size_t i = 0; i < values.size(); ++i) {
        packed[i] = values[i] ? 1 : 0;
      }
      seed ^= HashBytesBounded(packed.data(), packed.size()) + 0x9e3779b9 +
              (seed << 6) + (seed >> 2);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      // tensor is empty
      break;
    default:
      throw std::runtime_error(
          Str("HashTypedFieldBounded: no supported data type: ", elem_type));
  }

  return seed;
}

struct CSETensorHash {
  std::size_t operator()(const Tensor* tensor) const {
    /// https://github.com/onnx/onnx/issues/2630
    ONNX_ASSERT(tensor && !tensor->is_segment());
    const auto elem_type = tensor->elem_type();

    if (tensor->is_raw_data()) {
      // Cheap byte hash, matching CSETensorCompare's raw_data fast path
      // above -- no BLAKE3 here, see that comment for why. Memoized by
      // tensor_id() in g_raw_hash_cache: see that cache's own comment for
      // why this is normally a hit, not a fresh scan of tensor->raw().
      const bool profiling = GetPassPhaseProfilingEnabled();
      std::chrono::steady_clock::time_point t0;
      if (profiling)
        t0 = std::chrono::steady_clock::now();

      const uint64_t id = tensor->tensor_id();
      auto cached = g_raw_hash_cache.find(id);
      if (cached != g_raw_hash_cache.end()) {
        if (profiling) {
          const auto t1 = std::chrono::steady_clock::now();
          const double ms =
              std::chrono::duration<double, std::milli>(t1 - t0).count();
          g_cse_hash_compare_timing.raw_hash_calls++;
          g_cse_hash_compare_timing.raw_hash_ms += ms;
          g_cse_hash_compare_timing.raw_hash_cache_hits++;
          g_cse_hash_compare_timing.raw_hash_cache_hit_ms += ms;
        }
        return cached->second;
      }

      std::size_t seed = 0;
      hash_combine(seed, std::hash<int32_t>(), elem_type);
      hash_combine(seed, CSEContainerHash<int64_t>(), tensor->sizes());
      // Bounded-cost sampled hash rather than std::hash<std::string> over
      // the full buffer -- see HashRawDataBounded's own comment for why a
      // sampled hash here is safe, and why the full-buffer scan (paid once
      // per distinct tensor, not something g_raw_hash_cache above can
      // amortize away) was worth cutting.
      seed ^= HashRawDataBounded(tensor->raw()) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
      g_raw_hash_cache.emplace(id, seed);

      if (profiling) {
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        g_cse_hash_compare_timing.raw_hash_calls++;
        g_cse_hash_compare_timing.raw_hash_ms += ms;
        g_cse_hash_compare_timing.raw_hash_cache_misses++;
        g_cse_hash_compare_timing.raw_hash_cache_miss_ms += ms;
      }
      return seed;
    }

    if (elem_type != ONNX_NAMESPACE::TensorProto_DataType_STRING) {
      // Typed-field, non-STRING: bounded-cost sampled hash rather than
      // TensorContentDigest's full BLAKE3 pass over every element -- see
      // HashTypedFieldBounded's own comment for why a sampled hash here is
      // safe (bucketing only; CSETensorCompare's typed-field path, gated by
      // GetTrustTensorContentHash(), is still the sole source of truth for
      // equality either way) and why the full pass was worth cutting.
      ScopedCSETiming _t(&g_cse_hash_compare_timing.typed_hash_calls,
                         &g_cse_hash_compare_timing.typed_hash_ms);
      return HashTypedFieldBounded(*tensor);
    }

    // Only STRING reaches here.
    std::size_t seed = 0;
    hash_combine(seed, std::hash<int32_t>(), elem_type);
    hash_combine(seed, CSEContainerHash<int64_t>(), tensor->sizes());
    hash_combine(seed, CSEContainerHash<std::string>(), tensor->strings());
    return seed;
  }
};

template <>
struct CSEContainerHash<Tensor> {
  std::size_t operator()(const std::vector<Tensor>& container) const {
    std::size_t seed = 0;
    hash_combine(seed, std::hash<std::string>(),
                 std::string(typeid(Tensor).name()), std::hash<std::size_t>(),
                 container.size());
    for (const auto& d : container) {
      hash_combine(seed, CSETensorHash(), &d);
    }
    return seed;
  }
};

struct CSENodeHash {
  std::size_t operator()(const Node* n) const {
    ScopedCSETiming _t(&g_cse_hash_compare_timing.node_hash_calls,
                       &g_cse_hash_compare_timing.node_hash_ms);
    ONNX_ASSERT(n);
    std::size_t seed = 0;
    const auto inputs = n->inputs();
    auto size_t_hasher = std::hash<std::size_t>();
    auto string_hasher = std::hash<std::string>();
    auto sym_hasher = std::hash<Symbol>();
    hash_combine(seed, std::hash<uint32_t>(), static_cast<uint32_t>(n->kind()),
                 size_t_hasher, inputs.size());
    for (const auto& input : inputs) {
      hash_combine(seed, string_hasher, input->uniqueName());
    }
    std::vector<Symbol> attribute_names;
    {
      ScopedCSETiming _attr_t(
          &g_cse_hash_compare_timing.node_hash_attrsort_calls,
          &g_cse_hash_compare_timing.node_hash_attrsort_ms);
      attribute_names = n->attributeNames();
      SymbolCompare cmp;
      std::sort(attribute_names.begin(), attribute_names.end(), cmp);
    }
    for (const auto& name : attribute_names) {
      hash_combine(seed, sym_hasher, name);
      auto kind = n->kindOf(name);
      switch (kind) {
        case ONNX_NAMESPACE::AttributeKind::f:
          hash_combine(seed, std::hash<double>(), n->f(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::fs:
          hash_combine(seed, CSEContainerHash<double>(), n->fs(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::i:
          hash_combine(seed, std::hash<int64_t>(), n->i(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::is:
          hash_combine(seed, CSEContainerHash<int64_t>(), n->is(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::s:
          hash_combine(seed, string_hasher, n->s(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::ss:
          hash_combine(seed, CSEContainerHash<std::string>(), n->ss(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::t:
          hash_combine(seed, CSETensorHash(), &n->t(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::ts:
          hash_combine(seed, CSEContainerHash<Tensor>(), n->ts(name));
          break;
        default:
          throw std::runtime_error(
              Str("no support hash type: ", ONNX_NAMESPACE::toString(kind)));
          break;
      }
    }
    hash_combine(seed, size_t_hasher, n->outputs().size());
    return seed;
  }
};

struct CSEEqual {
  bool operator()(const Node* lhs, const Node* rhs) const {
    ScopedCSETiming _t(&g_cse_hash_compare_timing.node_equal_calls,
                       &g_cse_hash_compare_timing.node_equal_ms);
    if (!lhs) {
      return !rhs;
    } else if (!rhs) {
      return !lhs;
    }

    auto inputs_l = lhs->inputs();
    auto inputs_r = rhs->inputs();
    auto outputs_l = lhs->outputs();
    auto outputs_r = rhs->outputs();
    std::vector<Symbol> attr_names_l;
    std::vector<Symbol> attr_names_r;
    {
      ScopedCSETiming _attr_t(
          &g_cse_hash_compare_timing.node_equal_attrsort_calls,
          &g_cse_hash_compare_timing.node_equal_attrsort_ms);
      attr_names_l = lhs->attributeNames();
      attr_names_r = rhs->attributeNames();
      SymbolCompare cmp;
      std::sort(attr_names_l.begin(), attr_names_l.end(), cmp);
      std::sort(attr_names_r.begin(), attr_names_r.end(), cmp);
    }
    if (lhs->kind() != rhs->kind() || inputs_l.size() != inputs_r.size() ||
        outputs_l.size() != outputs_r.size() || attr_names_l != attr_names_r) {
      return false;
    }
    for (int i = 0; i < inputs_l.size(); ++i) {
      if (inputs_l[i]->uniqueName() != inputs_r[i]->uniqueName()) {
        return false;
      }
    }

    for (int i = 0; i < attr_names_l.size(); ++i) {
      const auto attr_name = attr_names_l[i];
      if (lhs->kindOf(attr_name) != rhs->kindOf(attr_name)) {
        return false;
      }
      switch (lhs->kindOf(attr_name)) {
        case AttributeKind::f:
          if (lhs->f(attr_name) != rhs->f(attr_name))
            return false;
          break;
        case AttributeKind::fs:
          if (lhs->fs(attr_name) != rhs->fs(attr_name))
            return false;
          break;
        case AttributeKind::i:
          if (lhs->i(attr_name) != rhs->i(attr_name))
            return false;
          break;
        case AttributeKind::is:
          if (lhs->is(attr_name) != rhs->is(attr_name))
            return false;
          break;
        case AttributeKind::s:
          if (lhs->s(attr_name) != rhs->s(attr_name))
            return false;
          break;
        case AttributeKind::ss:
          if (lhs->ss(attr_name) != rhs->ss(attr_name))
            return false;
          break;
        case AttributeKind::t:
          if (!CSETensorCompare(&lhs->t(attr_name), &rhs->t(attr_name)))
            return false;
          break;
        case AttributeKind::ts: {
          const auto& lts = lhs->ts(attr_name);
          const auto& rts = rhs->ts(attr_name);
          if (lts.size() != rts.size())
            return false;
          for (std::size_t k = 0; k < lts.size(); ++k) {
            if (!CSETensorCompare(&lts[k], &rts[k])) {
              return false;
            }
          }
          break;
        }
        default:
          return false;
      }
    }
    return true;
  }
};

struct CSETensorEqual {
  bool operator()(const Tensor* lhs, const Tensor* rhs) const {
    return CSETensorCompare(lhs, rhs);
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
