// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnxoptimizer/pass.h"

#include <chrono>

#include "onnx/common/assertions.h"

namespace ONNX_NAMESPACE {
namespace optimization {

namespace {
bool g_pass_phase_profiling_enabled = false;
std::unordered_map<std::string, PassPhaseTiming> g_pass_phase_timings;
std::unordered_map<std::string, PassTotalTiming> g_pass_total_timings;
}  // namespace

void SetPassPhaseProfilingEnabled(bool enabled) {
  g_pass_phase_profiling_enabled = enabled;
}

bool GetPassPhaseProfilingEnabled() {
  return g_pass_phase_profiling_enabled;
}

const std::unordered_map<std::string, PassPhaseTiming>& GetPassPhaseTimings() {
  return g_pass_phase_timings;
}

void ResetPassPhaseTimings() {
  g_pass_phase_timings.clear();
}

void RecordPassTotalTime(const std::string& pass_name, double ms) {
  PassTotalTiming& t = g_pass_total_timings[pass_name];
  t.calls++;
  t.total_ms += ms;
}

const std::unordered_map<std::string, PassTotalTiming>& GetPassTotalTimings() {
  return g_pass_total_timings;
}

void ResetPassTotalTimings() {
  g_pass_total_timings.clear();
}

Pass::Pass(PassType pass_type, PassEfficiency pass_efficiency,
           PassOptimizationType pass_optimization_type) {
  this->pass_type = pass_type;
  this->pass_efficiency = pass_efficiency;
  this->pass_optimization_type = pass_optimization_type;
}

Pass::~Pass() {}

unsigned int Pass::DescendOnGraphAttributesAndCount(
    Node* n, std::function<unsigned int(Graph&)> fn) {
  unsigned int num_changes = 0;
  for (auto name : n->attributeNames()) {
    auto kind = n->kindOf(name);
    if (kind == AttributeKind::g) {
      num_changes += fn(*n->g(name));
    }
    if (kind == AttributeKind::gs) {
      for (auto& g : n->gs(name)) {
        num_changes += fn(*g);
      }
    }
  }
  return num_changes;
}

void Pass::DescendOnGraphAttributesUnconstrained(
    Node* n, std::function<void(Graph&)> fn) {
  for (auto name : n->attributeNames()) {
    auto kind = n->kindOf(name);
    if (kind == AttributeKind::g) {
      fn(*n->g(name));
    }
    if (kind == AttributeKind::gs) {
      for (auto& g : n->gs(name)) {
        fn(*g);
      }
    }
  }
}

PredicateBasedPass::~PredicateBasedPass() {}

unsigned int PredicateBasedPass::_runPassInternal(Graph& graph) {
  unsigned int num_changes = false;
  // Only touches g_pass_phase_timings when profiling is on, so the lookup
  // (once per call, not once per node) and the two std::chrono reads per
  // node below are the only cost this diagnostic imposes when enabled.
  const bool profiling = g_pass_phase_profiling_enabled;
  PassPhaseTiming* timing =
      profiling ? &g_pass_phase_timings[this->getPassName()] : nullptr;
  for (auto it = graph.begin(); it != graph.end(); ++it) {
    auto* n = *it;
    num_changes += this->DescendOnGraphAttributesAndCount(
        n, [this](Graph& g) { return _runPassInternal(g); });
    bool matched;
    if (profiling) {
      const auto t0 = std::chrono::steady_clock::now();
      matched = this->patternMatchPredicate(n);
      const auto t1 = std::chrono::steady_clock::now();
      timing->match_calls++;
      timing->match_ms +=
          std::chrono::duration<double, std::milli>(t1 - t0).count();
    } else {
      matched = this->patternMatchPredicate(n);
    }
    if (matched) {
      NodeDestroyType destroy_type = NodeDestroyType::DestroyZero;
      if (profiling) {
        const auto t0 = std::chrono::steady_clock::now();
        num_changes += this->runTransform(n, graph, destroy_type);
        const auto t1 = std::chrono::steady_clock::now();
        timing->transform_calls++;
        timing->transform_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
      } else {
        num_changes += this->runTransform(n, graph, destroy_type);
      }

      if (destroy_type == NodeDestroyType::DestroyOne) {
        it.destroyCurrent();
      }
    }
  }
  return num_changes;
}

PassAnalysisType PredicateBasedPass::getPassAnalysisType() const {
  return PassAnalysisType::CountBased;
}

std::shared_ptr<PostPassAnalysis> PredicateBasedPass::runPass(Graph& graph) {
  bool initialized_pass = this->initializePass(graph);
  unsigned int touched_optimizations = this->_runPassInternal(graph);
  bool finalized_pass = this->finalizePass(graph);

  return std::shared_ptr<PostPassAnalysis>(new CountBasedPassAnalysis(
      this, touched_optimizations, initialized_pass, finalized_pass));
}

CountBasedPassAnalysis::CountBasedPassAnalysis(
    Pass* pass, unsigned int num_positive_transforms, bool initialization_done,
    bool finalization_done) {
  this->pass = pass;
  this->num_positive_transforms = num_positive_transforms;
  this->initialization_done = initialization_done;
  this->finalization_done = finalization_done;
}

FullGraphBasedPass::~FullGraphBasedPass() {}

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
