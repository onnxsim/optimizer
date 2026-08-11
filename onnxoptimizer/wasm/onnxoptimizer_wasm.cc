// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// A small C ABI over onnxoptimizer for the Emscripten build that backs the
// npm package (see js/README.md).
//
// It is deliberately narrower than onnxoptimizer/c_api: lists of passes cross
// the boundary as a single '\n'-joined string instead of a NULL-terminated
// array of pointers, so the JavaScript wrapper only ever has to allocate and
// free one buffer per call. Failures leave a human readable message behind in
// onnxopt_last_error() instead of only printing to stderr, because a WASM
// module's stderr is usually invisible to the caller.

#include <emscripten/emscripten.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"
#include "onnxoptimizer/optimize.h"

namespace {

std::string& LastError() {
  static std::string last_error;
  return last_error;
}

// Copies `str` into a NUL terminated malloc'd buffer, so JavaScript can read
// it with UTF8ToString and release it with onnxopt_free.
char* CopyToHeap(const std::string& str) {
  char* buf = static_cast<char*>(std::malloc(str.size() + 1));
  if (!buf) {
    return nullptr;
  }
  std::memcpy(buf, str.data(), str.size());
  buf[str.size()] = '\0';
  return buf;
}

char* JoinToHeap(const std::vector<std::string>& items) {
  std::string joined;
  for (const auto& item : items) {
    if (!joined.empty()) {
      joined += '\n';
    }
    joined += item;
  }
  return CopyToHeap(joined);
}

std::vector<std::string> Split(const char* joined) {
  std::vector<std::string> items;
  if (!joined) {
    return items;
  }
  const char* begin = joined;
  for (const char* it = joined;; ++it) {
    if (*it == '\n' || *it == '\0') {
      if (it != begin) {
        items.emplace_back(begin, it - begin);
      }
      if (*it == '\0') {
        break;
      }
      begin = it + 1;
    }
  }
  return items;
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE const char* onnxopt_version(void) {
  return ONNX_OPTIMIZER_VERSION_STRING;
}

// The message describing why the last onnxopt_optimize call failed. Valid
// until the next call into the module; owned by the module, do not free.
EMSCRIPTEN_KEEPALIVE const char* onnxopt_last_error(void) {
  return LastError().c_str();
}

EMSCRIPTEN_KEEPALIVE void onnxopt_free(void* ptr) {
  std::free(ptr);
}

/// '\n'-joined pass names. Caller frees with onnxopt_free.
EMSCRIPTEN_KEEPALIVE char* onnxopt_available_passes(void) {
  return JoinToHeap(ONNX_NAMESPACE::optimization::GetAvailablePasses());
}

/// '\n'-joined pass names. Caller frees with onnxopt_free.
EMSCRIPTEN_KEEPALIVE char* onnxopt_fuse_and_elimination_passes(void) {
  return JoinToHeap(
      ONNX_NAMESPACE::optimization::GetFuseAndEliminationPass());
}

/// Optimizes the serialized ModelProto in [mp_in, mp_in + mp_in_size).
///
/// `passes` is a '\n'-joined list of pass names; when it is NULL or empty the
/// fuse and elimination passes are used. On success 1 is returned and
/// *mp_out / *mp_out_size describe a malloc'd buffer the caller must release
/// with onnxopt_free. On failure 0 is returned and onnxopt_last_error()
/// describes the problem.
EMSCRIPTEN_KEEPALIVE int onnxopt_optimize(const void* mp_in,
                                          const size_t mp_in_size,
                                          const char* passes,
                                          const int fixed_point, void** mp_out,
                                          size_t* mp_out_size) {
  LastError().clear();
  if (!mp_out || !mp_out_size) {
    LastError() = "output arguments must not be null";
    return 0;
  }
  *mp_out = nullptr;
  *mp_out_size = 0;
  if (!mp_in || mp_in_size == 0) {
    LastError() = "the model is empty";
    return 0;
  }

  std::vector<std::string> names = Split(passes);
  if (names.empty()) {
    names = ONNX_NAMESPACE::optimization::GetFuseAndEliminationPass();
  }

  try {
    ONNX_NAMESPACE::ModelProto proto{};
    if (!ONNX_NAMESPACE::ParseProtoFromBytes(
            &proto, static_cast<const char*>(mp_in), mp_in_size)) {
      LastError() = "the model cannot be parsed as an ONNX ModelProto";
      return 0;
    }

    const ONNX_NAMESPACE::ModelProto result =
        fixed_point ? ONNX_NAMESPACE::optimization::OptimizeFixed(proto, names)
                    : ONNX_NAMESPACE::optimization::Optimize(proto, names);

    std::string serialized;
    if (!result.SerializeToString(&serialized)) {
      LastError() = "the optimized model cannot be serialized";
      return 0;
    }

    void* buf = std::malloc(serialized.size());
    if (!buf) {
      LastError() = "out of memory";
      return 0;
    }
    std::memcpy(buf, serialized.data(), serialized.size());
    *mp_out = buf;
    *mp_out_size = serialized.size();
    return 1;
  } catch (const std::exception& e) {
    LastError() = e.what();
    return 0;
  } catch (...) {
    LastError() = "unknown error";
    return 0;
  }
}

}  // extern "C"
