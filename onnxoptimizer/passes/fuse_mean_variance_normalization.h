// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Before (torch.onnx export of (x - mean) / sqrt(var), var biased):
//   Mean = ReduceMean(X, axes)
//   D    = Sub(X, Mean)
//   V    = ReduceMean(Mul(D, D), axes)
//   Y    = Div(D, Sqrt(V))
// After:
//   Y = MeanVarianceNormalization(X, axes=axes)
//
// Re-fuses the primitive mean/variance normalisation sub-graph back into the
// single ai.onnx MeanVarianceNormalization operator (opset 13). The op has no
// epsilon term, so only the eps-free decomposition is matched.

#include <algorithm>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseMeanVarianceNormalization final : public PredicateBasedPass {
  explicit FuseMeanVarianceNormalization()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_mean_variance_normalization";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "Div", 0, "Sub") && CheckKind(node, "Div", 1, "Sqrt");
  }

  static bool GetAxes(const Node* reduce, std::vector<int64_t>& axes) {
    if (GetValueFromAttrWithDefault<int64_t>(reduce, "keepdims", 1) != 1) {
      return false;
    }
    axes.clear();
    return GetValueFromAttrOrInput(reduce, "axes", 1, axes) && !axes.empty();
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Node* sub = n->input(0)->node();
    Node* sqrt = n->input(1)->node();
    Node* var_rm = sqrt->input()->node();
    if (!CheckKind(var_rm, "ReduceMean")) {
      return false;
    }
    Node* sq = var_rm->input(0)->node();
    // Variance term must be the squared deviation D * D.
    if (!CheckKind(sq, "Mul") || sq->input(0) != sub->output() ||
        sq->input(1) != sub->output()) {
      return false;
    }
    Value* x = sub->input(0);
    Node* mean_rm = sub->input(1)->node();
    if (!CheckKind(mean_rm, "ReduceMean") || mean_rm->input(0) != x) {
      return false;
    }

    std::vector<int64_t> axes_mean, axes_var;
    if (!GetAxes(mean_rm, axes_mean) || !GetAxes(var_rm, axes_var)) {
      return false;
    }
    std::sort(axes_mean.begin(), axes_mean.end());
    std::sort(axes_var.begin(), axes_var.end());
    if (axes_mean != axes_var) {
      return false;
    }

    // Every intermediate must feed only this pattern. The deviation D is used
    // twice by the square and once by the final Div.
    if (mean_rm->output()->uses().size() != 1 ||
        sub->output()->uses().size() != 3 ||
        sq->output()->uses().size() != 1 ||
        var_rm->output()->uses().size() != 1 ||
        sqrt->output()->uses().size() != 1) {
      return false;
    }

    Node* mvn = graph.create(Symbol("MeanVarianceNormalization"), 1);
    mvn->addInput(x);
    mvn->is_(Symbol("axes"), std::move(axes_mean));
    mvn->output()->copyMetadata(n->output());
    mvn->insertBefore(n);

    if (!tryReplacingAllUsesWith(n, mvn)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
