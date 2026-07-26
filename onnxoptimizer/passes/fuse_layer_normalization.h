// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Before (torch.onnx export of nn.LayerNorm, opset < 17):
//   Mean = ReduceMean(X, axes)
//   D    = Sub(X, Mean)
//   V    = ReduceMean(Pow(D, 2), axes)
//   Std  = Sqrt(Add(V, eps))
//   Y    = Add(Mul(Div(D, Std), Scale), Bias)
// After:
//   Y = LayerNormalization(X, Scale, Bias, axis=axis, epsilon=eps)
//
// Re-fuses the primitive mean/variance normalisation sub-graph back into the
// single ai.onnx LayerNormalization operator (opset 17).

#include <algorithm>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseLayerNormalization final : public PredicateBasedPass {
  explicit FuseLayerNormalization()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_layer_normalization";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "Add", 0, "Mul", 0, "Div");
  }

  // Fetch a single contiguous trailing reduction axis set and return its start.
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
        return false;  // not contiguous
      }
    }
    axis = axes.front();
    return true;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Node* mul = n->input(0)->node();
    Value* bias = n->input(1);
    Node* div = mul->input(0)->node();
    Value* scale = mul->input(1);

    if (!IsConstantTensor(bias) || !IsConstantTensor(scale)) {
      return false;
    }

    Node* sub = div->input(0)->node();
    Node* sqrt = div->input(1)->node();
    if (!CheckKind(sub, "Sub") || !CheckKind(sqrt, "Sqrt")) {
      return false;
    }
    Node* add_eps = sqrt->input()->node();
    if (!CheckKind(add_eps, "Add")) {
      return false;
    }
    // Add(variance, eps) in either operand order.
    Node* var_rm = add_eps->input(0)->node();
    Value* eps_val = add_eps->input(1);
    if (!CheckKind(var_rm, "ReduceMean")) {
      var_rm = add_eps->input(1)->node();
      eps_val = add_eps->input(0);
      if (!CheckKind(var_rm, "ReduceMean")) {
        return false;
      }
    }
    Node* pow = var_rm->input(0)->node();
    if (!CheckKind(pow, "Pow")) {
      return false;
    }
    // The squared term must be the same Sub output that the Div normalises.
    if (pow->input(0) != sub->output()) {
      return false;
    }
    Node* mean_rm = sub->input(1)->node();
    Value* x = sub->input(0);
    if (!CheckKind(mean_rm, "ReduceMean") || mean_rm->input(0) != x) {
      return false;
    }

    // Exponent must be 2 and epsilon/scalar constants must be available.
    float exponent = 0.0f, epsilon = 0.0f;
    if (!IsConstantTensor(pow->input(1)) ||
        !FetchSoleValueOfTensor(pow->input(1), exponent) || exponent != 2.0f) {
      return false;
    }
    if (!IsConstantTensor(eps_val) ||
        !FetchSoleValueOfTensor(eps_val, epsilon)) {
      return false;
    }

    int64_t axis_mean = 0, axis_var = 0;
    if (!GetReduceAxis(mean_rm, axis_mean) ||
        !GetReduceAxis(var_rm, axis_var) || axis_mean != axis_var) {
      return false;
    }

    // Ensure every intermediate feeds only this pattern so DCE removes them.
    if (mean_rm->output()->uses().size() != 1 ||
        sub->output()->uses().size() != 2 ||
        pow->output()->uses().size() != 1 ||
        var_rm->output()->uses().size() != 1 ||
        add_eps->output()->uses().size() != 1 ||
        sqrt->output()->uses().size() != 1 ||
        div->output()->uses().size() != 1 ||
        mul->output()->uses().size() != 1) {
      return false;
    }

    Node* ln = graph.create(Symbol("LayerNormalization"), 1);
    ln->addInput(x);
    ln->addInput(scale);
    ln->addInput(bias);
    ln->i_(Symbol("axis"), axis_mean);
    ln->f_(Symbol("epsilon"), epsilon);
    ln->output()->copyMetadata(n->output());
    ln->insertBefore(n);

    if (!tryReplacingAllUsesWith(n, ln)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
