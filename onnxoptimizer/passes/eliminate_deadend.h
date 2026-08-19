// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.
#pragma once
#include <chrono>

#include "onnxoptimizer/pass.h"
namespace ONNX_NAMESPACE {
namespace optimization {
struct EliminateDeadEnd final : public FullGraphBasedPass {
  explicit EliminateDeadEnd()
      : FullGraphBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "eliminate_deadend";
  }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::CountBased;
  }
  unsigned int EliminateDead(Graph& graph) {
    const bool profiling = GetPassPhaseProfilingEnabled();
    unsigned int nodes_removed = 0;
    uint64_t nodes_seen = 0;
    double has_uses_ms = 0.0;
    double destroy_ms = 0.0;
    auto nodes = graph.nodes().reverse();
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      nodes_seen++;
      bool has_uses;
      if (profiling) {
        const auto t0 = std::chrono::steady_clock::now();
        has_uses = node->hasUses();
        const auto t1 = std::chrono::steady_clock::now();
        has_uses_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
      } else {
        has_uses = node->hasUses();
      }
      if (!has_uses) {
        nodes_removed++;
        if (profiling) {
          const auto t0 = std::chrono::steady_clock::now();
          it.destroyCurrent();
          const auto t1 = std::chrono::steady_clock::now();
          destroy_ms +=
              std::chrono::duration<double, std::milli>(t1 - t0).count();
        } else {
          it.destroyCurrent();
        }
      }
    }
    if (profiling) {
      RecordDeadendPassTiming(nodes_seen, nodes_removed, has_uses_ms,
                              destroy_ms);
    }
    return nodes_removed;
  }
  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    auto nodes_removed = this->EliminateDead(graph);
    return std::shared_ptr<PostPassAnalysis>(
        new CountBasedPassAnalysis(this, nodes_removed, false, false));
  }
};
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
