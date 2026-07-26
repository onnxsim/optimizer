// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Before (torch.onnx export of the "rotate_half" rotary embedding
//   y = x * cos + rotate_half(x) * sin,
//   rotate_half(x) = concat(-x[..., d/2:], x[..., :d/2])):
//   S1 = Slice(X, ..:d/2, axis=-1)
//   S2 = Slice(X, d/2:.., axis=-1)
//   R  = Concat([Neg(S2), S1], axis=-1)
//   Y  = Add(Mul(X, cos), Mul(R, sin))
// After:
//   Y = RotaryEmbedding(X, cos_half, sin_half, num_heads=1, interleaved=0)
//
// Re-fuses the primitive rotate-half sub-graph back into the single ai.onnx
// RotaryEmbedding operator (opset 23). The op consumes half-dimension caches,
// so this rewrite is only sound when the full-dimension cos/sin tensors are
// constants whose two halves are identical (as produced by the standard
// cos = concat(c, c) construction); those constraints are verified before the
// rewrite fires and the half caches are materialised as new initializers.

#include <algorithm>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseRotaryEmbedding final : public PredicateBasedPass {
  explicit FuseRotaryEmbedding()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_rotary_embedding";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "Add", 0, "Mul") && CheckKind(node, "Add", 1, "Mul");
  }

  // Read a single-element Slice operand (starts/ends/axes). Returns false if it
  // is not a constant one-element vector.
  static bool GetScalarSliceInput(const Node* slice, size_t idx, int64_t& val) {
    std::vector<int64_t> v;
    if (!GetValueFromInput(slice, idx, v) || v.size() != 1) {
      return false;
    }
    val = v[0];
    return true;
  }

  // Take the first half along the last axis of a constant tensor whose two
  // halves are identical, materialising it as a new FLOAT initializer.
  static Value* MakeDuplicatedHalfCache(Graph& graph, const Value* full) {
    if (!IsConstantTensor(full)) {
      return nullptr;
    }
    const Tensor* t = FetchConstantTensor(full);
    if (!t || t->elem_type() != TensorProto_DataType_FLOAT) {
      return nullptr;
    }
    std::vector<int64_t> sizes = t->sizes();
    if (sizes.empty() || sizes.back() % 2 != 0) {
      return nullptr;
    }
    const int64_t d = sizes.back();
    const int64_t half = d / 2;
    std::vector<float> data = ParseTensorData<float>(t);
    int64_t rows = 1;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
      rows *= sizes[i];
    }
    if (static_cast<int64_t>(data.size()) != rows * d) {
      return nullptr;
    }
    std::vector<float> half_data;
    half_data.reserve(rows * half);
    for (int64_t r = 0; r < rows; ++r) {
      const float* row = data.data() + r * d;
      for (int64_t k = 0; k < half; ++k) {
        if (row[k] != row[k + half]) {
          return nullptr;  // halves not identical -> not fusible
        }
        half_data.push_back(row[k]);
      }
    }
    Tensor out;
    out.elem_type() = TensorProto_DataType_FLOAT;
    sizes.back() = half;
    out.sizes() = sizes;
    out.floats() = std::move(half_data);
    return graph.addInitializerAndCreateValue(out);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Node* mul0 = n->input(0)->node();
    Node* mul1 = n->input(1)->node();

    // One Mul carries the rotate_half Concat; the other is the plain X * cos.
    Node* mul_rot = nullptr;
    Node* mul_plain = nullptr;
    if (CheckKind(mul0->input(0)->node(), "Concat") ||
        CheckKind(mul0->input(1)->node(), "Concat")) {
      mul_rot = mul0;
      mul_plain = mul1;
    } else if (CheckKind(mul1->input(0)->node(), "Concat") ||
               CheckKind(mul1->input(1)->node(), "Concat")) {
      mul_rot = mul1;
      mul_plain = mul0;
    } else {
      return false;
    }

    const int concat_idx =
        CheckKind(mul_rot->input(0)->node(), "Concat") ? 0 : 1;
    Node* concat = mul_rot->input(concat_idx)->node();
    Value* sin = mul_rot->input(1 - concat_idx);

    // Concat([Neg(Slice hi), Slice lo], axis=-1).
    if (concat->inputs().size() != 2) {
      return false;
    }
    Node* neg = concat->input(0)->node();
    Node* slice_lo = concat->input(1)->node();
    if (!CheckKind(neg, "Neg") || !CheckKind(slice_lo, "Slice")) {
      return false;
    }
    Node* slice_hi = neg->input()->node();
    if (!CheckKind(slice_hi, "Slice")) {
      return false;
    }

    // Both slices operate on the same X, the plain Mul multiplies that X by cos.
    Value* x = slice_lo->input(0);
    if (slice_hi->input(0) != x) {
      return false;
    }
    Value* cos = nullptr;
    if (mul_plain->input(0) == x) {
      cos = mul_plain->input(1);
    } else if (mul_plain->input(1) == x) {
      cos = mul_plain->input(0);
    } else {
      return false;
    }

    // Slice lo = X[.., :half], slice hi = X[.., half:] on the last axis.
    int64_t lo_start, lo_end, hi_start, lo_axis, hi_axis;
    if (!GetScalarSliceInput(slice_lo, 1, lo_start) ||
        !GetScalarSliceInput(slice_lo, 2, lo_end) ||
        !GetScalarSliceInput(slice_lo, 3, lo_axis) ||
        !GetScalarSliceInput(slice_hi, 1, hi_start) ||
        !GetScalarSliceInput(slice_hi, 3, hi_axis)) {
      return false;
    }
    if (lo_start != 0 || lo_axis != hi_axis || lo_end != hi_start) {
      return false;
    }
    // The rotate split must be along the last axis.
    if (x->has_sizes()) {
      const int64_t rank = static_cast<int64_t>(x->sizes().size());
      const int64_t norm_axis = lo_axis < 0 ? lo_axis + rank : lo_axis;
      if (norm_axis != rank - 1) {
        return false;
      }
    }
    const int64_t half = lo_end;

    // Materialise the half-dimension caches (requires duplicated-half consts).
    Value* cos_half = MakeDuplicatedHalfCache(graph, cos);
    Value* sin_half = MakeDuplicatedHalfCache(graph, sin);
    if (!cos_half || !sin_half) {
      return false;
    }
    // Cache half dim must line up with the rotate split.
    if (static_cast<int64_t>(FetchConstantTensor(cos_half)->sizes().back()) !=
        half) {
      return false;
    }

    if (mul_rot->output()->uses().size() != 1 ||
        mul_plain->output()->uses().size() != 1 ||
        concat->output()->uses().size() != 1 ||
        neg->output()->uses().size() != 1 ||
        slice_lo->output()->uses().size() != 1 ||
        slice_hi->output()->uses().size() != 1) {
      return false;
    }

    Node* rope = graph.create(Symbol("RotaryEmbedding"), 1);
    rope->addInput(x);
    rope->addInput(cos_half);
    rope->addInput(sin_half);
    rope->i_(Symbol("num_heads"), 1);
    rope->i_(Symbol("interleaved"), 0);
    rope->output()->copyMetadata(n->output());
    rope->insertBefore(n);

    if (!tryReplacingAllUsesWith(n, rope)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
