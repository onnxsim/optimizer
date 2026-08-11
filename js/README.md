<!--
SPDX-FileCopyrightText: ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# onnxoptimizer

[![npm version](https://img.shields.io/npm/v/onnxoptimizer.svg)](https://www.npmjs.com/package/onnxoptimizer)

[onnxoptimizer](https://github.com/onnxsim/optimizer) compiled to WebAssembly:
run ONNX graph optimization passes from Node.js or the browser, with no native
dependency and no ONNX Runtime.

The module is pure graph rewriting — the same passes the Python package runs.
It does not evaluate the model, so it needs no execution provider. Tools that
also want constant folding (onnx-simplifier, for instance) can pair it with
[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web).

## Install

```bash
npm install onnxoptimizer
```

The package is ESM only and needs Node.js 18 or newer.

## Usage

```js
import { readFile, writeFile } from 'node:fs/promises';
import { optimize } from 'onnxoptimizer';

const model = await readFile('model.onnx');
// The fuse and elimination passes, like the Python API's default.
const optimized = await optimize(model);
await writeFile('model.opt.onnx', optimized);
```

`optimize` takes the serialized `ModelProto` as a `Uint8Array`, an
`ArrayBuffer` or any typed array, and returns a new `Uint8Array`. The input is
left untouched.

Choose the passes yourself, and repeat them until the model stops changing:

```js
import { getAvailablePasses, optimize } from 'onnxoptimizer';

console.log(await getAvailablePasses());

const optimized = await optimize(model, {
  passes: ['eliminate_identity', 'fuse_bn_into_conv'],
  fixedPoint: true,
});
```

### Managing the module yourself

The functions above share one lazily instantiated WebAssembly module. When you
need more control — one module per worker, your own copy of the `.wasm`, or the
module's log output — instantiate it yourself. `createOnnxOptimizer` forwards
its argument to the Emscripten factory, so `wasmBinary`, `locateFile`, `print`
and `printErr` all work as usual:

```js
import { createOnnxOptimizer } from 'onnxoptimizer';

const optimizer = await createOnnxOptimizer({
  printErr: (message) => console.warn(`[onnxoptimizer] ${message}`),
});

console.log(optimizer.version);              // e.g. "0.4.2"
console.log(optimizer.getFuseAndEliminationPasses());

const optimized = optimizer.optimize(model); // synchronous, once loaded
```

### In the browser

`dist/onnxoptimizer.mjs` loads `dist/onnxoptimizer.wasm` from next to itself,
which every bundler that understands `import.meta.url` resolves on its own. If
yours does not, fetch the `.wasm` and hand it over:

```js
const wasmBinary = await (await fetch('/onnxoptimizer.wasm')).arrayBuffer();
const optimizer = await createOnnxOptimizer({ wasmBinary });
```

Optimizing a model blocks until it finishes, so run it in a Web Worker if the
models are large.

## Errors

A rejected model, an unknown pass name or a pass that throws all raise an
`Error` carrying the message from the optimizer:

```js
try {
  optimizer.optimize(model, { passes: ['no_such_pass'] });
} catch (error) {
  console.error(error.message); // unknown optimization pass no_such_pass
}
```

## Building from a checkout

The published package ships the prebuilt module in `dist/`. To build it:

```bash
git submodule update --init --recursive
# Requires an activated emsdk (emcmake on PATH) and a host protoc.
./js/scripts/build_wasm.sh
cd js && npm test
```

See [`js/PUBLISHING.md`](./PUBLISHING.md) for how releases reach npm.

## License

Apache-2.0, the same as onnxoptimizer itself.
