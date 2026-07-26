// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Before:
//   L = ReduceLogSumExp(X, axes=[axis], keepdims=1)
//   Y = Sub(X, L)
// After:
//   Y = LogSoftmax(X, axis=axis)
//
// Re-fuses the primitive sub-graph emitted by a torch.onnx export of
// ``x - x.logsumexp(axis, keepdim=True)`` back into the single ai.onnx
// LogSoftmax operator (axis semantics: opset 13).

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseLogSoftmax final : public PredicateBasedPass {
  explicit FuseLogSoftmax()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_logsoftmax";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "Sub", 1, "ReduceLogSumExp");
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Value* x = n->input(0);
    Node* rlse = n->input(1)->node();

    // ReduceLogSumExp must reduce exactly the tensor that is subtracted, and
    // feed only this Sub.
    if (rlse->input(0) != x || rlse->output()->uses().size() != 1) {
      return false;
    }
    if (GetValueFromAttrWithDefault<int64_t>(rlse, "keepdims", 1) != 1) {
      return false;
    }

    // A single reduction axis maps to LogSoftmax's `axis` attribute. The axes
    // are an attribute (<=opset17) or the optional second input (>=opset18).
    std::vector<int64_t> axes;
    if (!GetValueFromAttrOrInput(rlse, "axes", 1, axes) || axes.size() != 1) {
      return false;
    }

    Node* logsoftmax = graph.create(Symbol("LogSoftmax"), 1);
    logsoftmax->addInput(x);
    logsoftmax->i_(Symbol("axis"), axes[0]);
    logsoftmax->output()->copyMetadata(n->output());
    logsoftmax->insertBefore(n);

    if (!tryReplacingAllUsesWith(n, logsoftmax)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
