// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopDropout final : public PredicateBasedPass {
  explicit EliminateNopDropout()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_dropout";
  }

  bool patternMatchPredicate(Node* node) override {
    // When ratio is an *attribute* (ONNX opset < 12) the node is the
    // inference-only form of Dropout: at test time its data output equals its
    // input, so it is a no-op regardless of the ratio value. The previous
    // predicate only matched ratio == 0.0 and therefore left real-world
    // inference exports (e.g. GoogLeNet/Inception, which carry ratio=0.5 with an
    // unused mask output) untouched. We now match any attribute-ratio Dropout.
    //
    // Opset >= 12 encodes ratio/training_mode as inputs (no ratio attribute) so
    // this predicate does not match them, leaving training graphs alone.
    if (node->kind() != kDropout || !node->hasAttribute(kratio)) {
      return false;
    }
    // The optional mask output (index 1) is only meaningful for training and
    // cannot be replaced by the input, so only fold when it is unused.
    if (node->outputs().size() > 1 && !node->outputs()[1]->uses().empty()) {
      return false;
    }
    return true;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    // The data output (index 0) is an identity on the input at inference time.
    // Any mask output is unused (guaranteed by the predicate) and disappears
    // with the node.
    const bool replacing_success =
        tryReplacingAllUsesWith(node->outputs()[0], node->input());
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
