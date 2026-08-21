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

// TensorContentDigest is memoized per Tensor::tensor_id() (a full BLAKE3
// pass is too expensive to redo on every hash-bucket lookup and every
// equality check against that bucket's candidates -- see cse_util.h's
// CSETensorHash/CSETensorCompare, both of which call it for the same tensor
// possibly many times per pass invocation). Keyed by tensor_id() rather than
// by `&tensor`: a `Tensor*` can be freed and its memory reused by an
// unrelated, differently-contented tensor (e.g. after
// Graph::eraseInitializer, or a Node attribute being replaced) within the
// cache's validity window, which would silently alias a stale digest onto
// the new tensor if keyed by address -- tensor_id() can't collide this way,
// since Tensor mints a fresh one on every construction and every
// (re)assignment (see tensor.h), so the cache naturally misses instead of
// aliasing.
//
// Because of that, this cache's validity is NOT scoped to a single pass
// call the way it once was: no onnx-optimizer pass mutates a *retained*
// tensor's content in place (only nodes/edges are rewired, or a tensor is
// dropped and replaced wholesale by a fresh one, which mints its own
// tensor_id() and simply misses the cache) -- so entries for tensors that
// are still alive and unchanged stay valid, and correct, across many pass
// calls and many onnxsim OptAndShape/FixedPointFn rounds within the same
// Optimizer::optimize(Graph&) call. See that function's
// `clear_tensor_digest_cache` parameter for how a caller opts out of the
// default per-call clear to extend this further, across its own repeated
// optimize() calls on one resident Graph.
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
