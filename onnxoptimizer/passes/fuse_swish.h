// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Before:
//   S = Sigmoid(X)
//   Y = Mul(X, S)      (or Mul(S, X))
// After:
//   Y = Swish(X)       (alpha = 1.0, i.e. the SiLU activation)
//
// This "re-fuses" the primitive sub-graph that framework exporters (e.g.
// torch.onnx export of nn.SiLU / x * sigmoid(x)) emit back into the single
// ai.onnx Swish operator (opset 22).

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseSwish final : public PredicateBasedPass {
  explicit FuseSwish()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_swish";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "Mul", 0, "Sigmoid") ||
           CheckKind(node, "Mul", 1, "Sigmoid");
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    const int sigmoid_idx = CheckKind(n->input(0)->node(), "Sigmoid") ? 0 : 1;
    Node* sigmoid = n->input(sigmoid_idx)->node();
    Value* x = n->input(1 - sigmoid_idx);

    // Sigmoid must be applied to the same tensor that is multiplied, and it
    // must feed only this Mul so that it is removed after the rewrite.
    if (sigmoid->input() != x || sigmoid->output()->uses().size() != 1) {
      return false;
    }

    Node* swish = graph.create(Symbol("Swish"), 1);
    swish->addInput(x);
    swish->f_(Symbol("alpha"), 1.0);
    swish->output()->copyMetadata(n->output());
    swish->insertBefore(n);

    if (!tryReplacingAllUsesWith(n, swish)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
