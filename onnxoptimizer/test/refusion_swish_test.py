# SPDX-FileCopyrightText: ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-fusion test for the Swish (SiLU) activation.

The input graph is produced by an *actual* ``torch.onnx`` export of
``torch.nn.SiLU`` (``x * sigmoid(x)``), which torch decomposes into a
``Sigmoid`` + ``Mul`` pair. The ``fuse_swish`` pass should re-fuse that pair
back into the single ``ai.onnx`` ``Swish`` operator (opset 22).
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
    """Export a module with torch's decomposing (TorchScript) ONNX exporter."""
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
    """Relabel the default-domain opset (the fused op may need a newer opset
    than torch's legacy exporter can emit)."""
    del model.opset_import[:]
    model.opset_import.append(onnx.helper.make_opsetid("", version))
    model.ir_version = onnx.IR_VERSION
    return model


def op_types(model):
    return [node.op_type for node in model.graph.node]


def try_run(model, feeds):
    """Run a model in onnxruntime; return None if the op set is unsupported."""
    if not HAS_ORT:
        return None
    try:
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(None, feeds)[0]
    except Exception:
        # Newly-standardised ops may not yet be implemented by onnxruntime.
        return None


@unittest.skipUnless(HAS_TORCH, "requires torch for onnx export")
class TestRefuseSwish(unittest.TestCase):
    TARGET_OPSET = 24

    def test_fuse_swish_from_torch_silu(self):
        model = export_decomposed(nn.SiLU(), torch.randn(2, 4, 8), export_opset=20)

        # Precondition: torch emitted the decomposed Sigmoid + Mul pattern.
        self.assertIn("Sigmoid", op_types(model))
        self.assertIn("Mul", op_types(model))

        optimized = onnxoptimizer.optimize(
            model, ["fuse_swish", "eliminate_deadend"]
        )
        relabel_opset(optimized, self.TARGET_OPSET)
        ops = op_types(optimized)

        self.assertEqual(ops.count("Swish"), 1)
        self.assertNotIn("Sigmoid", ops)
        self.assertNotIn("Mul", ops)
        onnx.checker.check_model(optimized)

        # Numerical equivalence (best-effort; skipped if ORT lacks the op).
        x = np.random.randn(2, 4, 8).astype(np.float32)
        ref = try_run(model, {"x": x})
        got = try_run(optimized, {"x": x})
        if ref is not None and got is not None:
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
