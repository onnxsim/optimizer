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

// TensorContentDigest is memoized per Tensor pointer (a full BLAKE3 pass is
// too expensive to redo on every hash-bucket lookup and every equality
// check against that bucket's candidates -- see cse_util.h's CSETensorHash/
// CSETensorCompare, both of which call it for the same tensor within a
// single pass invocation). The cache is valid ONLY within one call to
// EliminateDuplicateInitializer::EliminateInitializer or
// EliminateCommonSubexpression::EliminateCommonSubexpressions, since a
// tensor's content is never mutated in place *during* either pass (only
// nodes/edges are rewired, and any tensor a pass drops stays alive,
// unmutated, for the rest of that same call) -- neither pass mutates a
// retained tensor's bytes mid-call, but a tensor pointer CAN be reused by
// an unrelated, differently-contented tensor once freed between calls (a
// later optimizer pass, a later FixedPointFn round, or an entirely
// different graph), so each pass clears this cache at entry rather than
// relying on any cross-call invariant.
void ClearTensorContentDigestCache();

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
