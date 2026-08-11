// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/** A serialized ONNX ModelProto. */
export type ModelBytes = Uint8Array | ArrayBuffer | ArrayBufferView;

export interface OptimizeOptions {
  /**
   * Pass names to run. Defaults to the fuse and elimination passes, like the
   * Python API does.
   */
  passes?: readonly string[];
  /** Repeat the passes until the model stops changing. Defaults to false. */
  fixedPoint?: boolean;
}

/**
 * Options forwarded to the Emscripten module factory. Only the keys the
 * package documents are listed; anything else Emscripten accepts is passed
 * through as well.
 */
export interface ModuleOptions {
  /** The contents of onnxoptimizer.wasm, when you load them yourself. */
  wasmBinary?: ArrayBuffer | ArrayBufferView;
  /** Resolves the module's data files, onnxoptimizer.wasm in particular. */
  locateFile?: (path: string, prefix: string) => string;
  /** Receives the module's stdout. Defaults to console.log. */
  print?: (message: string) => void;
  /** Receives the module's stderr. Defaults to console.error. */
  printErr?: (message: string) => void;
  [option: string]: unknown;
}

/** An initialized onnxoptimizer WebAssembly module. */
export declare class OnnxOptimizer {
  constructor(module: unknown);

  /** The onnxoptimizer version the module was built from. */
  readonly version: string;

  /** Every pass onnxoptimizer knows about. */
  getAvailablePasses(): string[];

  /** The passes used when {@link optimize} is called without any. */
  getFuseAndEliminationPasses(): string[];

  /** Optimizes a serialized ONNX ModelProto and returns the optimized one. */
  optimize(model: ModelBytes, options?: OptimizeOptions): Uint8Array;
}

/** Instantiates the WebAssembly module. */
export declare function createOnnxOptimizer(
  moduleOptions?: ModuleOptions,
): Promise<OnnxOptimizer>;

/** Optimizes a model with a lazily instantiated, process-wide module. */
export declare function optimize(
  model: ModelBytes,
  options?: OptimizeOptions,
): Promise<Uint8Array>;

/** {@link OnnxOptimizer.getAvailablePasses} on the shared module. */
export declare function getAvailablePasses(): Promise<string[]>;

/** {@link OnnxOptimizer.getFuseAndEliminationPasses} on the shared module. */
export declare function getFuseAndEliminationPasses(): Promise<string[]>;

/** The onnxoptimizer version the shared module was built from. */
export declare function getVersion(): Promise<string>;
