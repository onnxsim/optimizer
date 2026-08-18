// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// A cryptographic (BLAKE3) content digest for a Tensor's dtype + shape +
// values, and a global toggle controlling whether cse_util.h's
// CSETensorHash/CSETensorCompare trust digest equality as tensor equality
// instead of hashing/comparing every element.
//
// CSETensorHash's raw_data fast path already hashes the tensor's bytes in
// one pass (via std::hash<std::string>), and CSETensorCompare's matching
// fast path already compares them with a single std::string::operator==
// (effectively an early-exiting memcmp) -- both about as cheap as a
// one-shot comparison can be, so switching *that* path to BLAKE3 buys
// nothing. The real cost this file targets is the *typed-field* path
// (float_data/int64_data/... for a tensor with no raw_data): today, every
// hash AND every equality check independently calls ParseTensorData<T>,
// which copies the tensor's values into a fresh std::vector (see that
// function's own comment on why -- byte-swapping raw_data into host order
// for the typed accessors). A tensor compared against several candidates in
// its hash bucket re-pays that copy on every single comparison.
// TensorContentDigest computes one BLAKE3 hash over the same values (via one
// ParseTensorData<T> call) that CSETensorHash and CSETensorCompare can then
// both consume, cutting an EliminateDuplicateInitializer/CSE pass on a
// typed-field-heavy graph down to one parse per tensor instead of one per
// comparison.
#pragma once

#include <string>

#include "onnx/onnx_pb.h"
#include "onnxoptimizer/passes/tensor_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

// 32-byte BLAKE3 digest of `tensor`'s dtype + shape + values, covering both
// raw_data and typed-field tensors uniformly: two tensors with equal
// TensorContentDigest have identical dtype, shape, and element values with
// overwhelming (cryptographic) probability -- not exactly, the way
// CSETensorCompare's own element-by-element comparison is, so this is only
// used where GetTrustTensorContentHash() says that tradeoff is acceptable
// (see cse_util.h). `tensor` must not be a STRING tensor or a segment
// (ONNX_ASSERT-fails on either -- callers already special-case both before
// reaching this).
std::string TensorContentDigest(const Tensor& tensor);

// See cse_util.h's CSETensorHash/CSETensorCompare for how this is
// consulted. Defaults to true: those hash-bucket lookups and equality
// checks trust TensorContentDigest equality as sufficient proof of tensor
// equality instead of re-deriving/re-comparing every element -- BLAKE3 is a
// cryptographic hash, so a false-positive match is not a practical concern
// (see this header's own top comment for where that actually saves work).
// Set false to always fall back to the exact, pre-hashing comparison
// instead (e.g. if you don't trust the hash, or are debugging a suspected
// collision). Process-global and not thread-safe to toggle concurrently
// with a running pass, matching this file's use from single-threaded
// pass execution.
void SetTrustTensorContentHash(bool trust);
bool GetTrustTensorContentHash();

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
