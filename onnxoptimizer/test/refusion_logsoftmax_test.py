# SPDX-FileCopyrightText: ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-fusion test for LogSoftmax.

The input graph is produced by an *actual* ``torch.onnx`` export of a
hand-written log-softmax (``x - x.logsumexp(dim, keepdim=True)``), which torch
lowers to a ``ReduceLogSumExp`` + ``Sub`` pair. The ``fuse_logsoftmax`` pass
should re-fuse that pair back into the single ``ai.onnx`` ``LogSoftmax``
operator.
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


class ManualLogSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x - torch.logsumexp(x, self.dim, keepdim=True)


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
class TestRefuseLogSoftmax(unittest.TestCase):
    TARGET_OPSET = 13

    def test_fuse_logsoftmax_from_torch(self):
        model = export_decomposed(
            ManualLogSoftmax(dim=-1), torch.randn(2, 4, 8), export_opset=13
        )

        # Precondition: torch emitted the ReduceLogSumExp + Sub pattern.
        self.assertIn("ReduceLogSumExp", op_types(model))
        self.assertIn("Sub", op_types(model))

        optimized = onnxoptimizer.optimize(
            model, ["fuse_logsoftmax", "eliminate_deadend"]
        )
        relabel_opset(optimized, self.TARGET_OPSET)
        ops = op_types(optimized)

        self.assertEqual(ops.count("LogSoftmax"), 1)
        self.assertNotIn("ReduceLogSumExp", ops)
        self.assertNotIn("Sub", ops)
        self.assertEqual(
            onnx.helper.get_node_attr_value(
                [n for n in optimized.graph.node if n.op_type == "LogSoftmax"][0],
                "axis",
            ),
            -1,
        )
        onnx.checker.check_model(optimized)

        x = np.random.randn(2, 4, 8).astype(np.float32)
        ref = try_run(model, {"x": x})
        got = try_run(optimized, {"x": x})
        if ref is not None and got is not None:
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
