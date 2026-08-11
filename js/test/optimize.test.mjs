// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { after, before, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  createOnnxOptimizer,
  getAvailablePasses,
  getFuseAndEliminationPasses,
  getVersion,
  optimize,
} from '../src/index.mjs';
import { opTypes } from './proto.mjs';

const identityChain = readFileSync(
  fileURLToPath(new URL('./fixtures/identity_chain.onnx', import.meta.url)),
);
const versionNumber = readFileSync(
  fileURLToPath(new URL('../../VERSION_NUMBER', import.meta.url)),
  'utf8',
).trim();

describe('onnxoptimizer', () => {
  let optimizer;
  const stderr = [];

  before(async () => {
    optimizer = await createOnnxOptimizer({
      printErr: (message) => stderr.push(message),
    });
  });

  after(() => {
    // Surface what the module logged when something above went wrong.
    if (stderr.length > 0) {
      console.log(`module stderr:\n${stderr.join('\n')}`);
    }
  });

  it('reports the version it was built from', () => {
    assert.equal(optimizer.version, versionNumber);
  });

  it('lists the available passes', () => {
    const available = optimizer.getAvailablePasses();
    assert.ok(available.includes('eliminate_identity'), available.join(', '));
    assert.ok(available.includes('fuse_bn_into_conv'), available.join(', '));

    const defaults = optimizer.getFuseAndEliminationPasses();
    assert.ok(defaults.length > 0);
    assert.deepEqual(
      defaults.filter((pass) => !available.includes(pass)),
      [],
    );
  });

  it('runs the fuse and elimination passes by default', () => {
    assert.deepEqual(opTypes(identityChain), ['Identity', 'Relu', 'Identity']);

    const optimized = optimizer.optimize(identityChain);

    assert.ok(optimized instanceof Uint8Array);
    assert.deepEqual(opTypes(optimized), ['Relu']);
  });

  it('runs only the requested passes', () => {
    const optimized = optimizer.optimize(identityChain, {
      passes: ['eliminate_identity'],
    });
    assert.deepEqual(opTypes(optimized), ['Relu']);

    // A pass that does not apply to this model leaves it alone.
    const untouched = optimizer.optimize(identityChain, {
      passes: ['fuse_bn_into_conv'],
    });
    assert.deepEqual(opTypes(untouched), ['Identity', 'Relu', 'Identity']);
  });

  it('optimizes to a fixed point on request', () => {
    const optimized = optimizer.optimize(identityChain, {
      passes: ['eliminate_identity'],
      fixedPoint: true,
    });
    assert.deepEqual(opTypes(optimized), ['Relu']);
  });

  it('leaves an already optimized model unchanged', () => {
    const once = optimizer.optimize(identityChain);
    const twice = optimizer.optimize(once);
    assert.deepEqual(Array.from(twice), Array.from(once));
  });

  it('accepts any view of the model bytes', () => {
    const expected = Array.from(optimizer.optimize(identityChain));

    const arrayBuffer = identityChain.buffer.slice(
      identityChain.byteOffset,
      identityChain.byteOffset + identityChain.byteLength,
    );
    assert.deepEqual(Array.from(optimizer.optimize(arrayBuffer)), expected);
    assert.deepEqual(
      Array.from(optimizer.optimize(new Uint8Array(arrayBuffer))),
      expected,
    );
  });

  it('does not modify the model it is given', () => {
    const before = Array.from(identityChain);
    optimizer.optimize(identityChain);
    assert.deepEqual(Array.from(identityChain), before);
  });

  it('rejects an unknown pass', () => {
    assert.throws(
      () => optimizer.optimize(identityChain, { passes: ['no_such_pass'] }),
      /no_such_pass/,
    );
  });

  it('rejects bytes that are not a ModelProto', () => {
    assert.throws(
      () => optimizer.optimize(Uint8Array.from([0xff, 0xff, 0xff, 0xff])),
      /cannot be parsed/,
    );
  });

  it('keeps working after a failure', () => {
    assert.throws(() => optimizer.optimize(Uint8Array.from([0xff, 0xff])));
    assert.deepEqual(opTypes(optimizer.optimize(identityChain)), ['Relu']);
  });

  it('rejects an empty model', () => {
    assert.throws(() => optimizer.optimize(new Uint8Array(0)), TypeError);
  });

  it('rejects arguments of the wrong type', () => {
    assert.throws(() => optimizer.optimize('a model'), TypeError);
    assert.throws(
      () => optimizer.optimize(identityChain, { passes: 'eliminate_identity' }),
      TypeError,
    );
    assert.throws(
      () => optimizer.optimize(identityChain, { passes: [''] }),
      TypeError,
    );
  });

  it('exposes the same API through the shared module', async () => {
    assert.equal(await getVersion(), versionNumber);
    assert.ok((await getAvailablePasses()).includes('eliminate_identity'));
    assert.ok((await getFuseAndEliminationPasses()).length > 0);
    assert.deepEqual(opTypes(await optimize(identityChain)), ['Relu']);
  });
});
