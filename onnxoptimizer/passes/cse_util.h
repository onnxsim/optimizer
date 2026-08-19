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
#include <type_traits>
#include <typeinfo>
#include <unordered_map>

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
      hash_combine(seed, std::hash<std::string>(), tensor->raw());
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
      // Typed-field, non-STRING: TensorContentDigest (tensor_content_hash.h,
      // memoized per tensor pointer for this pass call) already folds
      // dtype + shape + values into one BLAKE3 pass, so hash THAT instead
      // of redoing the work here. This is purely a faster/better bucketing
      // key and holds regardless of GetTrustTensorContentHash():
      // CSETensorCompare (which that toggle does gate) still runs its own
      // check on any bucket collision, so using this hash can never by
      // itself cause two distinct tensors to be merged, whichever way that
      // toggle is set.
      ScopedCSETiming _t(&g_cse_hash_compare_timing.typed_hash_calls,
                         &g_cse_hash_compare_timing.typed_hash_ms);
      return std::hash<std::string>()(TensorContentDigest(*tensor));
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
    auto attribute_names = n->attributeNames();
    SymbolCompare cmp;
    std::sort(attribute_names.begin(), attribute_names.end(), cmp);
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
    if (!lhs) {
      return !rhs;
    } else if (!rhs) {
      return !lhs;
    }

    auto inputs_l = lhs->inputs();
    auto inputs_r = rhs->inputs();
    auto outputs_l = lhs->outputs();
    auto outputs_r = rhs->outputs();
    auto attr_names_l = lhs->attributeNames();
    auto attr_names_r = rhs->attributeNames();
    SymbolCompare cmp;
    std::sort(attr_names_l.begin(), attr_names_l.end(), cmp);
    std::sort(attr_names_r.begin(), attr_names_r.end(), cmp);
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
