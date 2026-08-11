#!/usr/bin/env bash
# SPDX-FileCopyrightText: ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

# Builds the WebAssembly module of the npm package into js/dist.
#
# Requirements:
#   * an activated emsdk, so that emcmake is on PATH
#     (https://emscripten.org/docs/getting_started/downloads.html)
#   * a host protoc, because ONNX generates its C++ sources with a protoc built
#     for the host, not for wasm. Set PROTOC to point at one, otherwise the
#     protoc on PATH is used.
#
# Environment variables:
#   PROTOC          host protoc executable (default: the one on PATH)
#   BUILD_DIR       CMake build directory (default: <repo>/build-wasm)
#   BUILD_TYPE      CMake build type (default: Release)
#   LITE_PROTO      ON/OFF, build ONNX against protobuf-lite (default: ON).
#                   Lite protobuf drops the reflection and text-format code,
#                   which the optimizer does not use, and takes ~2 MB off the
#                   generated .wasm.
#   JOBS            parallel build jobs (default: number of online CPUs)

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
JS_DIR=$(dirname "$SCRIPT_DIR")
ROOT_DIR=$(dirname "$JS_DIR")

BUILD_DIR=${BUILD_DIR:-$ROOT_DIR/build-wasm}
BUILD_TYPE=${BUILD_TYPE:-Release}
LITE_PROTO=${LITE_PROTO:-ON}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}

if ! command -v emcmake > /dev/null; then
  echo "error: emcmake not found on PATH. Install and activate emsdk first:" >&2
  echo "       https://emscripten.org/docs/getting_started/downloads.html" >&2
  exit 1
fi

PROTOC=${PROTOC:-$(command -v protoc || true)}
if [ -z "$PROTOC" ]; then
  echo "error: no host protoc found. Install protobuf's compiler or set PROTOC." >&2
  exit 1
fi

if [ ! -f "$ROOT_DIR/third_party/onnx/CMakeLists.txt" ]; then
  echo "error: third_party/onnx is empty. Run:" >&2
  echo "       git submodule update --init --recursive" >&2
  exit 1
fi

set -x
emcmake cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DONNX_OPTIMIZER_BUILD_WASM=ON \
  -DONNX_CUSTOM_PROTOC_EXECUTABLE="$PROTOC" \
  -DONNX_USE_LITE_PROTO="$LITE_PROTO" \
  -DONNX_BUILD_PYTHON=OFF \
  -DONNX_BUILD_TESTS=OFF

cmake --build "$BUILD_DIR" --target onnxoptimizer_wasm -j "$JOBS"

mkdir -p "$JS_DIR/dist"
cp "$BUILD_DIR/onnxoptimizer.mjs" "$BUILD_DIR/onnxoptimizer.wasm" "$JS_DIR/dist/"
set +x

echo "built $JS_DIR/dist/onnxoptimizer.mjs and $JS_DIR/dist/onnxoptimizer.wasm"
