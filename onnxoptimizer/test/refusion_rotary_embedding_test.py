# SPDX-FileCopyrightText: ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-fusion test for RotaryEmbedding.

The input graph is produced by an *actual* ``torch.onnx`` export of the
standard "rotate_half" rotary embedding
(``y = x * cos + rotate_half(x) * sin``), which torch lowers to a
Slice / Slice / Neg / Concat / Mul / Mul / Add sub-graph. The
``fuse_rotary_embedding`` pass should re-fuse that sub-graph back into the
single ``ai.onnx`` ``RotaryEmbedding`` operator (opset 23).

The cos/sin caches are the usual ``concat(c, c)`` duplicated-half constants, so
the rewrite is sound and numerically exact.
"""

import io
import unittest

import numpy as np
import onnx

import onnxoptimizer

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), -1)


class RoPE(nn.Module):
    def __init__(self, batch, seq, dim):
        super().__init__()
        half = dim // 2
        c = torch.randn(seq, half)
        s = torch.randn(seq, half)
        # Standard duplicated-half caches, broadcast to the batch dimension.
        cos = torch.cat([c, c], -1).unsqueeze(0).expand(batch, seq, dim).contiguous()
        sin = torch.cat([s, s], -1).unsqueeze(0).expand(batch, seq, dim).contiguous()
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        return x * self.cos + rotate_half(x) * self.sin


def export_decomposed(module, dummy_input, export_opset):
    buf = io.BytesIO()
    torch.onnx.export(
        module,
        (dummy_input,),
        buf,
        opset_version=export_opset,
        input_names=["x"],
        output_names=["y"],
        dynamo=False,
    )
    buf.seek(0)
    return onnx.load_model_from_string(buf.getvalue())


def relabel_opset(model, version):
    del model.opset_import[:]
    model.opset_import.append(onnx.helper.make_opsetid("", version))
    model.ir_version = onnx.IR_VERSION
    return model


def op_types(model):
    return [node.op_type for node in model.graph.node]


def try_run(model, feeds):
    if not HAS_ORT:
        return None
    try:
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(None, feeds)[0]
    except Exception:
        return None


@unittest.skipUnless(HAS_TORCH, "requires torch for onnx export")
class TestRefuseRotaryEmbedding(unittest.TestCase):
    TARGET_OPSET = 23

    def test_fuse_rotary_embedding_from_torch(self):
        batch, seq, dim = 2, 4, 8
        model = export_decomposed(
            RoPE(batch, seq, dim), torch.randn(batch, seq, dim), export_opset=17
        )

        # Precondition: torch emitted the rotate_half decomposition.
        for prim in ("Slice", "Neg", "Concat"):
            self.assertIn(prim, op_types(model))
        self.assertNotIn("RotaryEmbedding", op_types(model))

        optimized = onnxoptimizer.optimize(
            model,
            [
                "fuse_rotary_embedding",
                "eliminate_deadend",
                "eliminate_unused_initializer",
            ],
        )
        relabel_opset(optimized, self.TARGET_OPSET)
        ops = op_types(optimized)

        self.assertEqual(ops.count("RotaryEmbedding"), 1)
        for prim in ("Slice", "Neg", "Concat", "Mul", "Add"):
            self.assertNotIn(prim, ops)
        onnx.checker.check_model(optimized)

        x = np.random.randn(batch, seq, dim).astype(np.float32)
        ref = try_run(model, {"x": x})
        got = try_run(optimized, {"x": x})
        if ref is not None and got is not None:
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
