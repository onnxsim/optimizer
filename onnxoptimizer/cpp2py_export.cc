// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "onnxoptimizer/model_util.h"
#include "onnxoptimizer/optimize.h"

namespace ONNX_NAMESPACE {
namespace nb = nanobind;
using namespace nanobind::literals;

template <typename Proto>
bool ParseProtoFromPyBytes(Proto* proto, const nb::bytes& bytes) {
  // Get the buffer from Python bytes object
  char* buffer = nullptr;
  Py_ssize_t length = 0;
  PyBytes_AsStringAndSize(bytes.ptr(), &buffer, &length);

  return ParseProtoFromBytes(proto, buffer, length);
}

NB_MODULE(onnx_opt_cpp2py_export, onnx_opt_cpp2py_export) {
  onnx_opt_cpp2py_export.doc() = "ONNX Optimizer";

  onnx_opt_cpp2py_export.def(
      "optimize",
      [](const nb::bytes& bytes, const std::vector<std::string>& names) {
        ModelProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        // Explicitly const: pins overload resolution to the copying
        // Optimize(const ModelProto&, ...) even though `proto` happens to be
        // unused afterward here, so this call site's behavior can't change
        // silently if a consuming overload is ever added upstream of it (see
        // Optimizer::optimize(ModelProto&, ...) in optimize.h).
        auto const result = optimization::Optimize(
            static_cast<const ModelProto&>(proto), names);
        std::string out;
        result.SerializeToString(&out);
        return nb::bytes(out.data(), out.size());
      });

  onnx_opt_cpp2py_export.def(
      "optimize_fixedpoint",
      [](const nb::bytes& bytes, const std::vector<std::string>& names) {
        ModelProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        auto const result = optimization::OptimizeFixed(
            static_cast<const ModelProto&>(proto), names);
        std::string out;
        result.SerializeToString(&out);
        return nb::bytes(out.data(), out.size());
      });

  onnx_opt_cpp2py_export.def(
      "optimize_from_path", [](const std::string& import_model_path,
                               const std::string& export_model_path,
                               const std::vector<std::string>& names,
                               const std::string& export_data_file_name) {
        ModelProto proto{};
        optimization::loadModel(&proto, import_model_path, true);
        auto result = optimization::Optimize(
            static_cast<const ModelProto&>(proto), names);
        optimization::saveModel(&result, export_model_path, true,
                                export_data_file_name);
      });

  onnx_opt_cpp2py_export.def(
      "optimize_fixedpoint_from_path",
      [](const std::string& import_model_path,
         const std::string& export_model_path,
         const std::vector<std::string>& names,
         const std::string& export_data_file_name) {
        ModelProto proto{};
        optimization::loadModel(&proto, import_model_path, true);
        auto result = optimization::OptimizeFixed(
            static_cast<const ModelProto&>(proto), names);
        optimization::saveModel(&result, export_model_path, true,
                                export_data_file_name);
      });
  onnx_opt_cpp2py_export.def("get_available_passes",
                             &optimization::GetAvailablePasses);
  onnx_opt_cpp2py_export.def("get_fuse_and_elimination_passes",
                             &optimization::GetFuseAndEliminationPass);
  // Toggle whether the passes treat graph initializers as constant tensors
  // (default true). See SetInitializersAsConstants in optimize.h.
  onnx_opt_cpp2py_export.def("set_initializers_as_constants",
                             &optimization::SetInitializersAsConstants);
  onnx_opt_cpp2py_export.def("initializers_as_constants",
                             &optimization::InitializersAsConstants);
}
}  // namespace ONNX_NAMESPACE
