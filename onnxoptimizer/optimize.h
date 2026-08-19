// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/proto_utils.h"
#include "onnxoptimizer/pass_manager.h"
#include "onnxoptimizer/pass_registry.h"
#include "onnxoptimizer/passes/cse_util.h"
#include "onnxoptimizer/passes/tensor_content_hash.h"
#include "vector"

namespace ONNX_NAMESPACE {
namespace optimization {

struct Optimizer {
  static GlobalPassRegistry passes;

 public:
  Optimizer(const std::vector<std::string> &names, const bool fixed_point);
  ~Optimizer();

  // Optimize the ONNX C++ IR (Graph) in place, running the configured passes
  // directly on the graph. This avoids the ModelProto <-> Graph round-trip
  // entirely and is intended for C++ callers that already hold a Graph (e.g.
  // onnxsim's OptAndShape fixed point, which imports once and can keep
  // re-running passes on the same Graph across rounds where shape inference
  // made no change -- see onnxsim issue #633). Proto-level concerns such as
  // the ir_version upgrade and function copying are the caller's
  // responsibility, since those live on ModelProto rather than on Graph.
  //
  // If `report` is non-null it is filled with a map from pass name to the
  // total number of positive transforms that pass applied to the graph,
  // matching the ModelProto-based optimize() below.
  //
  // If `clear_tensor_digest_cache` is true (the default, and correct for
  // essentially every caller), the two tensor-hash caches consulted by
  // eliminate_duplicate_initializer and eliminate_common_subexpression --
  // TensorContentDigest's (tensor_content_hash.h, the typed-field path) and
  // CSETensorHash's raw_data-branch cache (cse_util.h's g_raw_hash_cache,
  // the common path for real exported models) -- are cleared before
  // running the passes, bounding their memory to the tensors this one
  // optimize() call touches. Pass false only if the caller itself manages
  // those caches' lifetime across several optimize() calls on the *same*
  // resident Graph (e.g. onnxsim's OptAndShape fixed point, which calls
  // this once per round but wants hashes computed in an earlier round to
  // stay cached in a later one) -- see ClearTensorContentDigestCache's
  // header comment for why that's safe to do explicitly (the same
  // reasoning applies to ClearRawHashCache).
  void optimize(Graph &graph,
                std::map<std::string, unsigned int> *report = nullptr,
                bool clear_tensor_digest_cache = true) {
    if (clear_tensor_digest_cache) {
      ClearTensorContentDigestCache();
      ClearRawHashCache();
    }
    auto analysis = this->pass_manager->run(graph);
    if (report != nullptr && analysis != nullptr) {
      *report = analysis->transform_counts;
    }
  }

  // If `report` is non-null it is filled with a map from pass name to the
  // total number of positive transforms that pass applied to the graph.
  ModelProto optimize(const ModelProto &_mp_in,
                      std::map<std::string, unsigned int> *report = nullptr) {
    const ModelProto *mp_in = &_mp_in;
    std::unique_ptr<ModelProto> copy_in;
    if (mp_in->ir_version() == 3) {
      // Upgrade ir_version to 4 so that initializer can be not in input
      copy_in = std::make_unique<ModelProto>(*mp_in);
      copy_in->set_ir_version(4);
      mp_in = copy_in.get();
    }
    std::shared_ptr<Graph> g(ImportModelProto(*mp_in));

    if (g.get() == nullptr) {
      std::cerr << "Warning: onnx optimizer is unable to parse input model. "
                << "(The IR version of the ONNX model may be too old.)"
                << std::endl;
      // If we can't parse the file, just return the input.
      return *mp_in;
    }

    ModelProto mp_out = PrepareOutput(*mp_in);
    this->optimize(*g, report);
    ExportModelProto(&mp_out, g);

    // Maybe we can optimize these functions, now just copy
    AddFunctionsToModel(*mp_in, mp_out);
    return mp_out;
  }

#ifdef ONNX_IR_PB_CONVERTER_HAS_CONSUMING_OVERLOADS
  // Consuming overload: same as above, but moves each initializer's raw
  // bytes out of `mp_in` on Import and out of the internal Graph on Export,
  // instead of copying them at each end of the ModelProto<->Graph round
  // trip. This roughly halves the memory traffic of one optimize() call for
  // weight-heavy models (see onnxsim issue #633), at the cost of leaving
  // `mp_in`'s initializer tensors with empty raw data afterward. Only call
  // this when `mp_in` is about to be discarded or overwritten by the caller
  // -- e.g. onnxsim's OptAndShape fixed point, which immediately
  // move-assigns this call's return value back over its input model on
  // every iteration.
  //
  // Only defined when compiled against an onnx fork that provides the
  // matching consuming ImportModelProto/ExportModelProto overloads (see
  // ONNX_IR_PB_CONVERTER_HAS_CONSUMING_OVERLOADS in ir_pb_converter.h) --
  // e.g. absent when this library is linked against onnxruntime's own
  // bundled, unpatched onnx copy instead.
  ModelProto optimize(ModelProto &mp_in,
                      std::map<std::string, unsigned int> *report = nullptr) {
    if (mp_in.ir_version() == 3) {
      // Rare legacy path; not worth threading the moving Import/Export
      // through, so fall back to the copying overload above.
      const ModelProto &const_mp_in = mp_in;
      return optimize(const_mp_in, report);
    }
    std::shared_ptr<Graph> g(ImportModelProto(mp_in));

    if (g.get() == nullptr) {
      std::cerr << "Warning: onnx optimizer is unable to parse input model. "
                << "(The IR version of the ONNX model may be too old.)"
                << std::endl;
      // If we can't parse the file, just return the input. ImportModelProto
      // fails before touching any tensor data (it only checks ir_version),
      // so mp_in is still intact here.
      return mp_in;
    }

    ModelProto mp_out = PrepareOutput(mp_in);
    this->optimize(*g, report);
    ExportModelProto(&mp_out, g, /*consume_tensor_data=*/true);

    // Maybe we can optimize these functions, now just copy
    AddFunctionsToModel(mp_in, mp_out);
    return mp_out;
  }
#endif  // ONNX_IR_PB_CONVERTER_HAS_CONSUMING_OVERLOADS

 private:
  std::shared_ptr<PassManager> pass_manager;

  void AddFunctionsToModel(const ModelProto &original_model,
                           ModelProto &output_model) {
    for (const auto &function_proto : original_model.functions()) {
      auto *p_f = output_model.add_functions();
      p_f->CopyFrom(function_proto);
    }
  }

  ModelProto AddInitializerToInput(const ModelProto &original_model) {
    ModelProto model = original_model;
    std::vector<std::string> input_names;
    for (const auto &x : model.graph().input()) {
      input_names.push_back(x.name());
    }
    for (const auto &x : model.graph().initializer()) {
      if (std::find(input_names.begin(), input_names.end(), x.name()) ==
          input_names.end()) {
        auto *value_info = model.mutable_graph()->add_input();
        value_info->set_name(x.name());
        TypeProto *type = value_info->mutable_type();
        auto *tensor = type->mutable_tensor_type();
        tensor->set_elem_type(x.data_type());
        auto *shape = tensor->mutable_shape();
        for (const auto &dim : x.dims()) {
          TensorShapeProto::Dimension *new_dim = shape->add_dim();
          new_dim->set_dim_value(dim);
        }
      }
    }
    return model;
  }
};

const std::vector<std::string> GetAvailablePasses();

const std::vector<std::string> GetFuseAndEliminationPass();

// Control whether the optimizer passes treat graph initializers as constant
// tensors. The default (true) is onnxoptimizer's historical behaviour, in which
// an initializer-backed value is a constant and value-baking passes
// (fuse_bn_into_conv, fuse_add_bias_into_conv, nop-reshape/expand on a constant
// shape, ...) may consume and fold it. When set to false, initializers are
// treated as non-constant, so those passes leave initializer-backed values --
// and the weights they represent -- untouched; Constant *nodes* are still
// treated as constants. The setting is thread-local and stays in effect until
// changed, so callers that flip it should restore it afterwards.
void SetInitializersAsConstants(bool value);
bool InitializersAsConstants();

ModelProto Optimize(const ModelProto &mp_in,
                    const std::vector<std::string> &names,
                    std::map<std::string, unsigned int> *report = nullptr);

ModelProto OptimizeFixed(const ModelProto &mp_in,
                         const std::vector<std::string> &names,
                         std::map<std::string, unsigned int> *report = nullptr);

// In-place counterparts that operate directly on the ONNX C++ IR (Graph),
// skipping the ModelProto <-> Graph conversion entirely. For C++ callers
// that already hold a Graph -- see Optimizer::optimize(Graph&, ...)'s doc
// comment. Unlike the consuming ModelProto overloads below, these do not
// depend on ONNX_IR_PB_CONVERTER_HAS_CONSUMING_OVERLOADS: they never touch
// ModelProto at all, so they work identically whether this library is
// linked against onnxsim's onnx fork or onnxruntime's bundled, unpatched
// onnx copy.
// `clear_tensor_digest_cache`: see Optimizer::optimize(Graph&, ...)'s doc
// comment above -- the default (true) is correct for essentially every
// caller.
void OptimizeGraph(Graph &graph, const std::vector<std::string> &names,
                   std::map<std::string, unsigned int> *report = nullptr,
                   bool clear_tensor_digest_cache = true);

void OptimizeGraphFixed(Graph &graph, const std::vector<std::string> &names,
                        std::map<std::string, unsigned int> *report = nullptr,
                        bool clear_tensor_digest_cache = true);

#ifdef ONNX_IR_PB_CONVERTER_HAS_CONSUMING_OVERLOADS
// Consuming overloads: see Optimizer::optimize(ModelProto&, ...)'s doc
// comment. Only call these when `mp_in` is about to be discarded or
// overwritten by the caller.
ModelProto Optimize(ModelProto &mp_in, const std::vector<std::string> &names,
                    std::map<std::string, unsigned int> *report = nullptr);

ModelProto OptimizeFixed(ModelProto &mp_in,
                         const std::vector<std::string> &names,
                         std::map<std::string, unsigned int> *report = nullptr);
#endif  // ONNX_IR_PB_CONVERTER_HAS_CONSUMING_OVERLOADS
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
