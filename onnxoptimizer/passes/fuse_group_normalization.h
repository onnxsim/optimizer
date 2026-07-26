// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Before (torch.onnx export of nn.GroupNorm):
//   R0 = Reshape(X, [0, num_groups, -1])
//   I  = InstanceNormalization(R0, ones(num_groups), zeros(num_groups), eps)
//   R1 = Reshape(I, Shape(X))
//   Y  = Add(Mul(R1, Weight), Bias)          Weight/Bias shaped (C, 1, 1)
// After:
//   Y = GroupNormalization(X, Scale, Bias, num_groups=num_groups, epsilon=eps)
//
// Re-fuses the InstanceNormalization-based decomposition that framework
// exporters emit for group norm back into the single ai.onnx
// GroupNormalization operator (opset 21). The per-group InstanceNorm affine
// must be the identity (scale=1, bias=0); the real affine lives in the trailing
// per-channel Mul/Add, whose (C, 1, 1) constants are flattened to the (C,)
// shape GroupNormalization expects.

#include <algorithm>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseGroupNormalization final : public PredicateBasedPass {
  explicit FuseGroupNormalization()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_group_normalization";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "Add", 0, "Mul", 0, "Reshape", 0,
                     "InstanceNormalization");
  }

  // Flatten a constant tensor (e.g. shape (C,1,1)) to a 1-D (C,) initializer.
  static Value* MakeFlatFloatInitializer(Graph& graph, const Value* src) {
    std::vector<float> data;
    if (!GetValueFromInput(src, data) || data.empty()) {
      return nullptr;
    }
    Tensor t;
    t.sizes().push_back(static_cast<int64_t>(data.size()));
    t.elem_type() = TensorProto_DataType_FLOAT;
    t.floats() = std::move(data);
    return graph.addInitializerAndCreateValue(t);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    Node* mul = n->input(0)->node();
    Value* bias_src = n->input(1);
    Node* reshape_back = mul->input(0)->node();
    Value* weight_src = mul->input(1);
    Node* inst_norm = reshape_back->input(0)->node();

    if (!IsConstantTensor(weight_src) || !IsConstantTensor(bias_src)) {
      return false;
    }

    // Reshape back to the original shape must be driven by Shape(X).
    Node* shape = reshape_back->input(1)->node();
    if (!CheckKind(shape, "Shape")) {
      return false;
    }
    Value* x = shape->input(0);

    // The grouping reshape must be Reshape(X, [0, num_groups, -1]).
    Node* reshape_group = inst_norm->input(0)->node();
    if (!CheckKind(reshape_group, "Reshape") || reshape_group->input(0) != x) {
      return false;
    }
    std::vector<int64_t> group_shape;
    if (!GetValueFromInput(reshape_group->input(1), group_shape) ||
        group_shape.size() != 3 || group_shape[0] != 0 ||
        group_shape[2] != -1) {
      return false;
    }
    const int64_t num_groups = group_shape[1];

    // The per-group InstanceNorm affine must be the identity.
    std::vector<float> in_scale, in_bias;
    if (!GetValueFromInput(inst_norm->input(1), in_scale) ||
        !GetValueFromInput(inst_norm->input(2), in_bias) ||
        static_cast<int64_t>(in_scale.size()) != num_groups ||
        static_cast<int64_t>(in_bias.size()) != num_groups) {
      return false;
    }
    if (std::any_of(in_scale.begin(), in_scale.end(),
                    [](float v) { return v != 1.0f; }) ||
        std::any_of(in_bias.begin(), in_bias.end(),
                    [](float v) { return v != 0.0f; })) {
      return false;
    }

    const float epsilon =
        GetValueFromAttrWithDefault(inst_norm, "epsilon", 1e-5f);

    if (reshape_group->output()->uses().size() != 1 ||
        inst_norm->output()->uses().size() != 1 ||
        shape->output()->uses().size() != 1 ||
        reshape_back->output()->uses().size() != 1 ||
        mul->output()->uses().size() != 1) {
      return false;
    }

    Value* scale = MakeFlatFloatInitializer(graph, weight_src);
    Value* bias = MakeFlatFloatInitializer(graph, bias_src);
    if (!scale || !bias) {
      return false;
    }

    Node* gn = graph.create(Symbol("GroupNormalization"), 1);
    gn->addInput(x);
    gn->addInput(scale);
    gn->addInput(bias);
    gn->i_(Symbol("num_groups"), num_groups);
    gn->f_(Symbol("epsilon"), epsilon);
    gn->output()->copyMetadata(n->output());
    gn->insertBefore(n);

    if (!tryReplacingAllUsesWith(n, gn)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
