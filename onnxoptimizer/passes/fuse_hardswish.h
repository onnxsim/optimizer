// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Before:
//   H = HardSigmoid(X, alpha=1/6, beta=1/2)
//   Y = Mul(X, H)      (or Mul(H, X))
// After:
//   Y = HardSwish(X)
//
// Re-fuses the primitive sub-graph emitted by torch.onnx export of
// nn.Hardswish (x * hardsigmoid(x), with alpha=1/6, beta=1/2) back into the
// single ai.onnx HardSwish operator (opset 14). The alpha/beta of the
// HardSigmoid must match the HardSwish definition, otherwise the numerics
// differ and the fusion is skipped.

#include <cmath>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseHardSwish final : public PredicateBasedPass {
  explicit FuseHardSwish()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_hardswish";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "Mul", 0, "HardSigmoid") ||
           CheckKind(node, "Mul", 1, "HardSigmoid");
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    const int hs_idx = CheckKind(n->input(0)->node(), "HardSigmoid") ? 0 : 1;
    Node* hard_sigmoid = n->input(hs_idx)->node();
    Value* x = n->input(1 - hs_idx);

    if (hard_sigmoid->input() != x ||
        hard_sigmoid->output()->uses().size() != 1) {
      return false;
    }

    // HardSwish(x) = x * HardSigmoid(x, alpha=1/6, beta=1/2).
    const float alpha =
        GetValueFromAttrWithDefault(hard_sigmoid, "alpha", 0.2f);
    const float beta = GetValueFromAttrWithDefault(hard_sigmoid, "beta", 0.5f);
    if (std::abs(alpha - 1.0f / 6.0f) > 1e-4f || std::abs(beta - 0.5f) > 1e-4f) {
      return false;
    }

    Node* hardswish = graph.create(Symbol("HardSwish"), 1);
    hardswish->addInput(x);
    hardswish->output()->copyMetadata(n->output());
    hardswish->insertBefore(n);

    if (!tryReplacingAllUsesWith(n, hardswish)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
