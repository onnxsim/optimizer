// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnxoptimizer/pass_manager.h"

#include <chrono>

#include "onnxoptimizer/passes/logging.h"

namespace ONNX_NAMESPACE {
namespace optimization {

namespace {
// Times a single pass->runPass(graph) call when pass-phase profiling
// (pass.h's SetPassPhaseProfilingEnabled) is on; a plain passthrough
// otherwise. See PassTotalTiming's comment for what this captures beyond
// PredicateBasedPass's own matching/modifying timers.
std::shared_ptr<PostPassAnalysis> RunPassTimed(
    const std::shared_ptr<Pass>& pass, Graph& graph) {
  if (!GetPassPhaseProfilingEnabled()) {
    return pass->runPass(graph);
  }
  const auto t0 = std::chrono::steady_clock::now();
  auto analysis = pass->runPass(graph);
  const auto t1 = std::chrono::steady_clock::now();
  RecordPassTotalTime(
      pass->getPassName(),
      std::chrono::duration<double, std::milli>(t1 - t0).count());
  return analysis;
}
}  // namespace

PassManager::PassManager() {}
PassManager::~PassManager() {}

GeneralPassManager::~GeneralPassManager() {
  this->passes.clear();
}
void GeneralPassManager::add(std::shared_ptr<Pass> pass) {
  this->passes.push_back(std::move(pass));
}

std::shared_ptr<PassManagerAnalysis> GeneralPassManager::run(Graph& graph) {
  auto report = std::make_shared<PassManagerAnalysis>();
  for (const std::shared_ptr<Pass>& pass : this->passes) {
    std::shared_ptr<PostPassAnalysis> analysis = pass->runPass(graph);
    if (pass->getPassAnalysisType() == PassAnalysisType::Empty) {
      continue;
    }
    std::shared_ptr<CountBasedPassAnalysis> count_analysis =
        std::static_pointer_cast<CountBasedPassAnalysis>(analysis);
    report->transform_counts[pass->getPassName()] +=
        count_analysis->num_positive_transforms;
  }
  return report;
}

std::shared_ptr<PassManagerAnalysis> FixedPointPassManager::run(Graph& graph) {
  bool fixed_point_optimization_done;
  auto report = std::make_shared<PassManagerAnalysis>();

  do {
    fixed_point_optimization_done = false;
    for (const std::shared_ptr<Pass>& pass : this->passes) {
      std::shared_ptr<PostPassAnalysis> analysis = RunPassTimed(pass, graph);
      if (pass->getPassAnalysisType() == PassAnalysisType::Empty) {
        continue;
      }
      std::shared_ptr<CountBasedPassAnalysis> count_analysis =
          std::static_pointer_cast<CountBasedPassAnalysis>(analysis);
      report->transform_counts[pass->getPassName()] +=
          count_analysis->num_positive_transforms;
      if (count_analysis->num_positive_transforms != 0) {
        VLOG(1) << "Pass " << pass->getPassName() << " transformed "
                << count_analysis->num_positive_transforms;
      }

      while (count_analysis->fixedPointOptimizationNeeded()) {
        count_analysis = std::static_pointer_cast<CountBasedPassAnalysis>(
            RunPassTimed(pass, graph));
        report->transform_counts[pass->getPassName()] +=
            count_analysis->num_positive_transforms;
        if (count_analysis->num_positive_transforms != 0) {
          VLOG(1) << "Pass " << pass->getPassName() << " transformed "
                  << count_analysis->num_positive_transforms;
        }
        fixed_point_optimization_done = true;
      }
    }
  } while (fixed_point_optimization_done);

  return report;
}
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
