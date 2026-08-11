// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * JavaScript API of onnxoptimizer.
 *
 * The heavy lifting happens in dist/onnxoptimizer.wasm; this module owns the
 * bookkeeping around it -- copying models in and out of the WebAssembly heap,
 * freeing what the module allocates, and turning failures into exceptions.
 */

/** Size of a pointer and of size_t in the wasm32 module. */
const POINTER_SIZE = 4;

let factoryPromise;
let sharedOptimizerPromise;

async function loadFactory() {
  try {
    const generated = await import('../dist/onnxoptimizer.mjs');
    return generated.default;
  } catch (cause) {
    throw new Error(
      'the onnxoptimizer WebAssembly module is missing. When working from a ' +
        'checkout, build it with js/scripts/build_wasm.sh.',
      { cause },
    );
  }
}

function toBytes(model) {
  if (model instanceof Uint8Array) {
    return model;
  }
  if (model instanceof ArrayBuffer) {
    return new Uint8Array(model);
  }
  if (ArrayBuffer.isView(model)) {
    return new Uint8Array(model.buffer, model.byteOffset, model.byteLength);
  }
  throw new TypeError(
    'the model must be a Uint8Array, an ArrayBuffer or a typed array of the ' +
      'serialized ONNX ModelProto',
  );
}

function normalizePasses(passes) {
  if (passes === undefined || passes === null) {
    return [];
  }
  if (!Array.isArray(passes)) {
    throw new TypeError('passes must be an array of pass names');
  }
  return passes.map((pass) => {
    if (typeof pass !== 'string' || pass.length === 0) {
      throw new TypeError('every pass name must be a non-empty string');
    }
    // Pass names cross into the module as a '\n'-joined string.
    if (pass.includes('\n')) {
      throw new TypeError(`a pass name must not contain a newline: ${pass}`);
    }
    return pass;
  });
}

/** An initialized onnxoptimizer WebAssembly module. */
export class OnnxOptimizer {
  #module;

  /**
   * Wraps an already initialized Emscripten module. Use
   * {@link createOnnxOptimizer} instead of calling this directly.
   */
  constructor(module) {
    this.#module = module;
  }

  /** The onnxoptimizer version the module was built from. */
  get version() {
    return this.#module.UTF8ToString(this.#module._onnxopt_version());
  }

  /** Every pass onnxoptimizer knows about. */
  getAvailablePasses() {
    return this.#takeStringList(this.#module._onnxopt_available_passes());
  }

  /** The passes used when {@link optimize} is called without any. */
  getFuseAndEliminationPasses() {
    return this.#takeStringList(
      this.#module._onnxopt_fuse_and_elimination_passes(),
    );
  }

  /**
   * Optimizes a serialized ONNX ModelProto and returns the optimized one.
   *
   * @param model the serialized ModelProto
   * @param options.passes pass names; defaults to the fuse and elimination
   *   passes, like the Python API does
   * @param options.fixedPoint whether to repeat the passes until the model
   *   stops changing
   */
  optimize(model, options = {}) {
    const { passes, fixedPoint = false } = options;
    const bytes = toBytes(model);
    const names = normalizePasses(passes);
    if (bytes.byteLength === 0) {
      throw new TypeError('the model is empty');
    }

    const module = this.#module;
    let modelPointer = 0;
    let passesPointer = 0;
    let outPointer = 0;
    let outSizePointer = 0;
    let optimizedPointer = 0;
    try {
      modelPointer = this.#allocate(bytes.byteLength);
      module.HEAPU8.set(bytes, modelPointer);
      passesPointer = this.#allocateString(names.join('\n'));
      outPointer = this.#allocate(POINTER_SIZE);
      outSizePointer = this.#allocate(POINTER_SIZE);

      const ok = module._onnxopt_optimize(
        modelPointer,
        bytes.byteLength,
        passesPointer,
        fixedPoint ? 1 : 0,
        outPointer,
        outSizePointer,
      );
      // Read the heap views only after the call: growing the WebAssembly
      // memory replaces them.
      if (!ok) {
        throw new Error(
          module.UTF8ToString(module._onnxopt_last_error()) ||
            'the model cannot be optimized',
        );
      }
      optimizedPointer = module.HEAPU32[outPointer / POINTER_SIZE];
      const optimizedSize = module.HEAPU32[outSizePointer / POINTER_SIZE];
      // slice() copies out of the WebAssembly heap, so the result survives the
      // free below and any later memory growth.
      return module.HEAPU8.slice(
        optimizedPointer,
        optimizedPointer + optimizedSize,
      );
    } finally {
      if (optimizedPointer) {
        module._onnxopt_free(optimizedPointer);
      }
      for (const pointer of [
        modelPointer,
        passesPointer,
        outPointer,
        outSizePointer,
      ]) {
        if (pointer) {
          module._free(pointer);
        }
      }
    }
  }

  #allocate(size) {
    const pointer = this.#module._malloc(size);
    if (!pointer) {
      throw new Error(
        `the WebAssembly module cannot allocate ${size} bytes for the model`,
      );
    }
    return pointer;
  }

  #allocateString(text) {
    const size = this.#module.lengthBytesUTF8(text) + 1;
    const pointer = this.#allocate(size);
    this.#module.stringToUTF8(text, pointer, size);
    return pointer;
  }

  /** Reads a '\n'-joined list the module allocated, and frees it. */
  #takeStringList(pointer) {
    if (!pointer) {
      throw new Error('the WebAssembly module cannot allocate the pass list');
    }
    try {
      const joined = this.#module.UTF8ToString(pointer);
      return joined.length === 0 ? [] : joined.split('\n');
    } finally {
      this.#module._onnxopt_free(pointer);
    }
  }
}

/**
 * Instantiates the WebAssembly module.
 *
 * `moduleOptions` is passed straight to the Emscripten factory, so the usual
 * knobs apply -- `wasmBinary` to supply the .wasm bytes yourself, `locateFile`
 * to resolve them from somewhere else, `print` / `printErr` to capture the
 * module's output.
 */
export async function createOnnxOptimizer(moduleOptions = {}) {
  factoryPromise ??= loadFactory();
  const factory = await factoryPromise;
  return new OnnxOptimizer(await factory({ ...moduleOptions }));
}

function sharedOptimizer() {
  sharedOptimizerPromise ??= createOnnxOptimizer();
  return sharedOptimizerPromise;
}

/**
 * Optimizes a model with a lazily instantiated, process-wide module.
 *
 * Use {@link createOnnxOptimizer} instead when you need to configure the
 * module, or when you want one module per worker.
 */
export async function optimize(model, options) {
  return (await sharedOptimizer()).optimize(model, options);
}

/** {@link OnnxOptimizer#getAvailablePasses} on the shared module. */
export async function getAvailablePasses() {
  return (await sharedOptimizer()).getAvailablePasses();
}

/** {@link OnnxOptimizer#getFuseAndEliminationPasses} on the shared module. */
export async function getFuseAndEliminationPasses() {
  return (await sharedOptimizer()).getFuseAndEliminationPasses();
}

/** The onnxoptimizer version the shared module was built from. */
export async function getVersion() {
  return (await sharedOptimizer()).version;
}
