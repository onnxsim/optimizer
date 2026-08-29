// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct SetUniqueNameForNodes final : public PredicateBasedPass {
  explicit SetUniqueNameForNodes()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::None) {}

  std::string getPassName() const override {
    return "set_unique_name_for_nodes";
  }

  bool patternMatchPredicate(Node* node) override {
    return !node->has_name();
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    node->setName(nextReservedName(graph));
    destroy_current = NodeDestroyType::DestroyZero;
    return true;
  }

 private:
  // This pass's patternMatchPredicate fires once per unnamed node in the
  // whole graph -- a model exported without per-node names (common) can have
  // as many matches as it has nodes. Each unbatched getNextUniqueName() call
  // pays a full graph (+ subgraph) scan via isNameUnique(), making the pass
  // quadratic in node count. Batch the draws instead, amortizing the scan
  // across kNameBatchSize names at a time (see
  // onnxsim/passes/fuse_bn_into_conv.h's nextReservedName() for the same
  // pattern).
  static constexpr size_t kNameBatchSize = 256;
  std::vector<std::string> reserved_names_;
  size_t reserved_used_ = 0;

  std::string nextReservedName(Graph& graph) {
    if (reserved_used_ >= reserved_names_.size()) {
      reserved_names_ = graph.reserveUniqueNames(kNameBatchSize);
      reserved_used_ = 0;
    }
    return std::move(reserved_names_[reserved_used_++]);
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
