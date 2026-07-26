// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Before (numerically-stabilised softmax):
//   M = ReduceMax(X, axes=[axis], keepdims=1)
//   S = Sub(X, M)
//   E = Exp(S)
//   D = ReduceSum(E, axes=[axis], keepdims=1)
//   Y = Div(E, D)
// After:
//   Y = Softmax(X, axis=axis)
//
// Re-fuses the primitive sub-graph a framework emits for a hand-written
// softmax back into the single ai.onnx Softmax operator (axis semantics:
// opset 13). The max-subtraction is a numerically-stable no-op, so the result
// is identical to Softmax(X, axis).

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseSoftmax final : public PredicateBasedPass {
  explicit FuseSoftmax()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_softmax";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "Div", 0, "Exp", 0, "Sub") &&
           CheckKind(node, "Div", 1, "ReduceSum");
  }

  static bool GetSingleAxis(const Node* reduce, int64_t& axis) {
    if (GetValueFromAttrWithDefault<int64_t>(reduce, "keepdims", 1) != 1) {
      return false;
    }
    std::vector<int64_t> axes;
    if (!GetValueFromAttrOrInput(reduce, "axes", 1, axes) || axes.size() != 1) {
      return false;
    }
    axis = axes[0];
    return true;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Node* exp = n->input(0)->node();
    Node* reduce_sum = n->input(1)->node();
    Node* sub = exp->input()->node();
    Node* reduce_max = sub->input(1)->node();
    Value* x = sub->input(0);

    if (!CheckKind(reduce_max, "ReduceMax")) {
      return false;
    }
    // ReduceSum consumes the Exp output; ReduceMax reduces the same X.
    if (reduce_sum->input(0) != exp->output() || reduce_max->input(0) != x) {
      return false;
    }
    // Intermediates must feed only the pattern so DCE removes them. Exp is
    // shared by Div and ReduceSum (exactly two internal uses).
    if (exp->output()->uses().size() != 2 ||
        sub->output()->uses().size() != 1 ||
        reduce_max->output()->uses().size() != 1 ||
        reduce_sum->output()->uses().size() != 1) {
      return false;
    }

    int64_t axis_max = 0, axis_sum = 0;
    if (!GetSingleAxis(reduce_max, axis_max) ||
        !GetSingleAxis(reduce_sum, axis_sum) || axis_max != axis_sum) {
      return false;
    }

    Node* softmax = graph.create(Symbol("Softmax"), 1);
    softmax->addInput(x);
    softmax->i_(Symbol("axis"), axis_sum);
    softmax->output()->copyMetadata(n->output());
    softmax->insertBefore(n);

    if (!tryReplacingAllUsesWith(n, softmax)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
