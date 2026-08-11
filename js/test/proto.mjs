// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Just enough protobuf wire-format decoding for the tests to look inside an
 * optimized ModelProto without depending on an ONNX package.
 */

const WIRE_VARINT = 0;
const WIRE_FIXED64 = 1;
const WIRE_LENGTH_DELIMITED = 2;
const WIRE_FIXED32 = 5;

// ModelProto.graph, GraphProto.node and NodeProto.op_type field numbers.
const MODEL_GRAPH = 7;
const GRAPH_NODE = 1;
const NODE_OP_TYPE = 4;

function readVarint(bytes, offset) {
  let value = 0;
  let scale = 1;
  let index = offset;
  for (;;) {
    if (index >= bytes.length) {
      throw new Error('truncated varint');
    }
    const byte = bytes[index++];
    value += (byte & 0x7f) * scale;
    if ((byte & 0x80) === 0) {
      return [value, index];
    }
    scale *= 128;
  }
}

/** Yields the fields of the message in [start, end). */
export function* fields(bytes, start = 0, end = bytes.length) {
  let index = start;
  while (index < end) {
    let key;
    [key, index] = readVarint(bytes, index);
    const field = { number: Math.floor(key / 8), wire: key % 8 };
    switch (field.wire) {
      case WIRE_LENGTH_DELIMITED: {
        let length;
        [length, index] = readVarint(bytes, index);
        yield { ...field, start: index, end: index + length };
        index += length;
        break;
      }
      case WIRE_VARINT: {
        let value;
        [value, index] = readVarint(bytes, index);
        yield { ...field, value };
        break;
      }
      case WIRE_FIXED32:
        yield field;
        index += 4;
        break;
      case WIRE_FIXED64:
        yield field;
        index += 8;
        break;
      default:
        throw new Error(`unsupported wire type ${field.wire}`);
    }
  }
}

function findField(bytes, number, start, end) {
  for (const field of fields(bytes, start, end)) {
    if (field.number === number && field.wire === WIRE_LENGTH_DELIMITED) {
      return field;
    }
  }
  return undefined;
}

/** The op types of the nodes of a serialized ModelProto, in graph order. */
export function opTypes(model) {
  const bytes = model instanceof Uint8Array ? model : new Uint8Array(model);
  const graph = findField(bytes, MODEL_GRAPH, 0, bytes.length);
  if (!graph) {
    throw new Error('the model has no graph');
  }
  const decoder = new TextDecoder();
  const types = [];
  for (const field of fields(bytes, graph.start, graph.end)) {
    if (field.number !== GRAPH_NODE || field.wire !== WIRE_LENGTH_DELIMITED) {
      continue;
    }
    const opType = findField(bytes, NODE_OP_TYPE, field.start, field.end);
    types.push(opType ? decoder.decode(bytes.subarray(opType.start, opType.end)) : '');
  }
  return types;
}
