// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Before (torch.onnx export of x * rsqrt(mean(x^2) + eps) * weight):
//   M    = ReduceMean(Pow(X, 2), axes)
//   R    = Div(1, Sqrt(Add(M, eps)))     (reciprocal std)
//   Y    = Mul(Mul(X, R), Weight)
// After:
//   Y = RMSNormalization(X, Weight, axis=axis, epsilon=eps)
//
// Re-fuses the primitive root-mean-square normalisation sub-graph (as emitted
// for e.g. LLaMA-style RMSNorm modules) back into the single ai.onnx
// RMSNormalization operator (opset 23).

#include <algorithm>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseRMSNormalization final : public PredicateBasedPass {
  explicit FuseRMSNormalization()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_rms_normalization";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "Mul", 0, "Mul") || CheckKind(node, "Mul", 1, "Mul");
  }

  static bool GetReduceAxis(const Node* reduce, int64_t& axis) {
    if (GetValueFromAttrWithDefault<int64_t>(reduce, "keepdims", 1) != 1) {
      return false;
    }
    std::vector<int64_t> axes;
    if (!GetValueFromAttrOrInput(reduce, "axes", 1, axes) || axes.empty()) {
      return false;
    }
    std::sort(axes.begin(), axes.end());
    for (size_t i = 1; i < axes.size(); ++i) {
      if (axes[i] != axes[i - 1] + 1) {
        return false;
      }
    }
    axis = axes.front();
    return true;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    // Outer Mul: one operand is the inner (X * reciprocal) Mul, the other is
    // the constant weight.
    const int inner_idx = CheckKind(n->input(0)->node(), "Mul") ? 0 : 1;
    Node* inner = n->input(inner_idx)->node();
    Value* weight = n->input(1 - inner_idx);
    if (!IsConstantTensor(weight)) {
      return false;
    }

    // Inner Mul: one operand is the reciprocal std (a Div), the other is X.
    Node* div = nullptr;
    Value* x = nullptr;
    if (CheckKind(inner->input(0)->node(), "Div")) {
      div = inner->input(0)->node();
      x = inner->input(1);
    } else if (CheckKind(inner->input(1)->node(), "Div")) {
      div = inner->input(1)->node();
      x = inner->input(0);
    } else {
      return false;
    }

    // Div must be a reciprocal: numerator == 1.
    float numerator = 0.0f;
    if (!IsConstantTensor(div->input(0)) ||
        !FetchSoleValueOfTensor(div->input(0), numerator) ||
        numerator != 1.0f) {
      return false;
    }
    Node* sqrt = div->input(1)->node();
    if (!CheckKind(sqrt, "Sqrt")) {
      return false;
    }
    Node* add_eps = sqrt->input()->node();
    if (!CheckKind(add_eps, "Add")) {
      return false;
    }
    Node* rmean = add_eps->input(0)->node();
    Value* eps_val = add_eps->input(1);
    if (!CheckKind(rmean, "ReduceMean")) {
      rmean = add_eps->input(1)->node();
      eps_val = add_eps->input(0);
      if (!CheckKind(rmean, "ReduceMean")) {
        return false;
      }
    }
    Node* pow = rmean->input(0)->node();
    if (!CheckKind(pow, "Pow") || pow->input(0) != x) {
      return false;
    }

    float exponent = 0.0f, epsilon = 0.0f;
    if (!IsConstantTensor(pow->input(1)) ||
        !FetchSoleValueOfTensor(pow->input(1), exponent) || exponent != 2.0f) {
      return false;
    }
    if (!IsConstantTensor(eps_val) ||
        !FetchSoleValueOfTensor(eps_val, epsilon)) {
      return false;
    }

    int64_t axis = 0;
    if (!GetReduceAxis(rmean, axis)) {
      return false;
    }

    if (inner->output()->uses().size() != 1 ||
        div->output()->uses().size() != 1 ||
        sqrt->output()->uses().size() != 1 ||
        add_eps->output()->uses().size() != 1 ||
        rmean->output()->uses().size() != 1 ||
        pow->output()->uses().size() != 1) {
      return false;
    }

    Node* rms = graph.create(Symbol("RMSNormalization"), 1);
    rms->addInput(x);
    rms->addInput(weight);
    rms->i_(Symbol("axis"), axis);
    rms->f_(Symbol("epsilon"), epsilon);
    rms->output()->copyMetadata(n->output());
    rms->insertBefore(n);

    if (!tryReplacingAllUsesWith(n, rms)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
