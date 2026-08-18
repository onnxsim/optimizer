// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnxoptimizer/passes/tensor_content_hash.h"

#include "blake3/c/blake3.h"
#include "onnxoptimizer/passes/string_utils.h"

namespace ONNX_NAMESPACE {
namespace optimization {

namespace {

bool g_trust_tensor_content_hash = true;

// IEEE754 defines +0.0 == -0.0, and std::hash<float>/std::hash<double> (and
// hence the old, pre-digest CSETensorHash/CSETensorCompare, which hashed and
// compared typed-field tensors via std::hash<T> and operator==) are required
// by the C++ standard to agree: equal values must hash equal. A raw
// byte/bit-pattern digest does not have that property (0x00000000 !=
// 0x80000000), so hashing the typed-field parsed array's bytes verbatim
// silently stopped CSE from deduplicating Constant/initializer tensors that
// differ only in the sign of a zero -- a real, measured regression on
// several transformer models (see onnxsim PR #641 model-regression
// investigation). Canonicalizing to +0.0 before hashing restores the old
// dedup coverage while keeping the digest-based speedup. This only matters
// for the typed-field path below: the raw_data fast path already hashes
// disk bytes verbatim in both the old and new code, so it never treated
// +0.0/-0.0 as equal either way, and Float16/BFloat16 compare and hash by
// bit pattern already (see data_type.h), so they need no adjustment.
inline float CanonicalizeZero(float v) { return v == 0.0f ? 0.0f : v; }
inline double CanonicalizeZero(double v) { return v == 0.0 ? 0.0 : v; }

// dtype/shape/value bytes are hashed in host-native order throughout this
// file: unlike onnxsim's TensorPool content hash (which is exported for
// cross-process/cross-host verification and so must be endian-portable),
// TensorContentDigest never leaves this process -- every digest it produces
// is computed and consumed within the same run on the same host, so a
// consistent-with-itself native-order hash is exactly as safe as an
// explicitly little-endian one, at less code.
template <typename T>
void UpdatePod(blake3_hasher* hasher, const T& v) {
  blake3_hasher_update(hasher, &v, sizeof(T));
}

}  // namespace

std::string TensorContentDigest(const Tensor& tensor) {
  ONNX_ASSERT(!tensor.is_segment());
  ONNX_ASSERT(tensor.elem_type() !=
              ONNX_NAMESPACE::TensorProto_DataType_STRING);

  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  const int32_t elem_type = tensor.elem_type();
  UpdatePod(&hasher, elem_type);
  const auto& sizes = tensor.sizes();
  const uint64_t rank = sizes.size();
  UpdatePod(&hasher, rank);
  for (int64_t d : sizes) {
    UpdatePod(&hasher, d);
  }

  if (tensor.is_raw_data()) {
    const std::string& raw = tensor.raw();
    blake3_hasher_update(&hasher, raw.data(), raw.size());
  } else {
    // Mirrors cse_util.h's CSETensorHash/CSETensorCompare typed-field
    // switch, but hashes each ParseTensorData<T> vector's contiguous bytes
    // directly instead of the caller doing its own element-wise hash/equal
    // -- one parse per tensor regardless of how many comparisons follow
    // (see this file's header comment).
#define DO_CASE(pb_type, cpp_type)                          \
  case ONNX_NAMESPACE::TensorProto_DataType_##pb_type: {    \
    const auto values = ParseTensorData<cpp_type>(&tensor); \
    blake3_hasher_update(&hasher, values.data(),             \
                         values.size() * sizeof(cpp_type)); \
    break;                                                  \
  }

    switch (elem_type) {
      DO_CASE(INT8, int8_t)
      DO_CASE(INT16, int16_t)
      DO_CASE(INT32, int32_t)
      DO_CASE(INT64, int64_t)
      DO_CASE(UINT8, uint8_t)
      DO_CASE(UINT16, uint16_t)
      DO_CASE(UINT32, uint32_t)
      DO_CASE(UINT64, uint64_t)
      DO_CASE(FLOAT16, Float16)
      DO_CASE(BFLOAT16, BFloat16)

#undef DO_CASE

      // FLOAT/DOUBLE/COMPLEX64/COMPLEX128 hash a signed-zero-canonicalized
      // copy instead of the DO_CASE macro's verbatim bytes -- see
      // CanonicalizeZero's comment above for why.
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        auto values = ParseTensorData<float>(&tensor);
        for (auto& v : values) v = CanonicalizeZero(v);
        blake3_hasher_update(&hasher, values.data(),
                             values.size() * sizeof(float));
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        auto values = ParseTensorData<double>(&tensor);
        for (auto& v : values) v = CanonicalizeZero(v);
        blake3_hasher_update(&hasher, values.data(),
                             values.size() * sizeof(double));
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
        auto values = ParseTensorData<Complex64>(&tensor);
        for (auto& v : values) {
          v.real_part = CanonicalizeZero(v.real_part);
          v.imaginary_part = CanonicalizeZero(v.imaginary_part);
        }
        blake3_hasher_update(&hasher, values.data(),
                             values.size() * sizeof(Complex64));
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {
        auto values = ParseTensorData<Complex128>(&tensor);
        for (auto& v : values) {
          v.real_part = CanonicalizeZero(v.real_part);
          v.imaginary_part = CanonicalizeZero(v.imaginary_part);
        }
        blake3_hasher_update(&hasher, values.data(),
                             values.size() * sizeof(Complex128));
        break;
      }

      case ONNX_NAMESPACE::TensorProto_DataType_BOOL: {
        // std::vector<bool> is bit-packed, not a contiguous array of
        // byte-sized elements -- pack one byte per value instead of relying
        // on (nonexistent) contiguous storage.
        const auto values = ParseTensorData<bool>(&tensor);
        std::string packed(values.size(), '\0');
        for (size_t i = 0; i < values.size(); ++i) {
          packed[i] = values[i] ? 1 : 0;
        }
        blake3_hasher_update(&hasher, packed.data(), packed.size());
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
        break;
      default:
        throw std::runtime_error(
            Str("TensorContentDigest: no supported data type: ", elem_type));
    }
  }

  std::string digest(32, '\0');
  blake3_hasher_finalize(&hasher, reinterpret_cast<uint8_t*>(digest.data()),
                         32);
  return digest;
}

void SetTrustTensorContentHash(bool trust) {
  g_trust_tensor_content_hash = trust;
}

bool GetTrustTensorContentHash() { return g_trust_tensor_content_hash; }

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
