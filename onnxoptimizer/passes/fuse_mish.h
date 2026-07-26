// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Before:
//   P = Softplus(X)
//   T = Tanh(P)
//   Y = Mul(X, T)      (or Mul(T, X))
// After:
//   Y = Mish(X)
//
// Re-fuses the primitive sub-graph emitted by torch.onnx export of nn.Mish
// (x * tanh(softplus(x))) back into the single ai.onnx Mish operator
// (opset 18).

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseMish final : public PredicateBasedPass {
  explicit FuseMish()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_mish";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "Mul", 0, "Tanh", 0, "Softplus") ||
           CheckKind(node, "Mul", 1, "Tanh", 0, "Softplus");
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    const int tanh_idx = CheckKind(n->input(0)->node(), "Tanh") ? 0 : 1;
    Node* tanh = n->input(tanh_idx)->node();
    Node* softplus = tanh->input()->node();
    Value* x = n->input(1 - tanh_idx);

    // softplus(x) must operate on the same tensor as the outer multiply, and
    // the tanh/softplus intermediates must feed nothing else.
    if (softplus->input() != x || tanh->output()->uses().size() != 1 ||
        softplus->output()->uses().size() != 1) {
      return false;
    }

    Node* mish = graph.create(Symbol("Mish"), 1);
    mish->addInput(x);
    mish->output()->copyMetadata(n->output());
    mish->insertBefore(n);

    if (!tryReplacingAllUsesWith(n, mish)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
