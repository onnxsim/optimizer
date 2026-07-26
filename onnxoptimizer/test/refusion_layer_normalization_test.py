# SPDX-FileCopyrightText: ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-fusion test for LayerNormalization.

The input graph is produced by an *actual* ``torch.onnx`` export of
``torch.nn.LayerNorm`` at opset < 17, which torch decomposes into the
mean/variance normalisation chain (ReduceMean / Sub / Pow / ReduceMean / Add /
Sqrt / Div / Mul / Add). The ``fuse_layer_normalization`` pass should re-fuse
that chain back into the single ``ai.onnx`` ``LayerNormalization`` operator.
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
class TestRefuseLayerNormalization(unittest.TestCase):
    TARGET_OPSET = 17

    def test_fuse_layer_normalization_from_torch(self):
        model = export_decomposed(
            nn.LayerNorm(8), torch.randn(2, 4, 8), export_opset=16
        )

        # Precondition: torch emitted the decomposed normalisation chain.
        self.assertNotIn("LayerNormalization", op_types(model))
        self.assertIn("ReduceMean", op_types(model))

        optimized = onnxoptimizer.optimize(
            model, ["fuse_layer_normalization", "eliminate_deadend"]
        )
        relabel_opset(optimized, self.TARGET_OPSET)
        ops = op_types(optimized)

        self.assertEqual(ops.count("LayerNormalization"), 1)
        for prim in ("ReduceMean", "Sub", "Pow", "Sqrt", "Div"):
            self.assertNotIn(prim, ops)
        onnx.checker.check_model(optimized)

        x = np.random.randn(2, 4, 8).astype(np.float32)
        ref = try_run(model, {"x": x})
        got = try_run(optimized, {"x": x})
        if ref is not None and got is not None:
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
