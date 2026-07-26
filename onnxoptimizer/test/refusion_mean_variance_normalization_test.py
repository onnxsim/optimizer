# SPDX-FileCopyrightText: ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-fusion test for MeanVarianceNormalization.

The input graph is produced by an *actual* ``torch.onnx`` export of a
hand-written ``(x - mean) / sqrt(var)`` (biased variance, no epsilon), which
torch lowers to a ReduceMean / Sub / Mul / ReduceMean / Sqrt / Div chain. The
``fuse_mean_variance_normalization`` pass should re-fuse that chain back into
the single ``ai.onnx`` ``MeanVarianceNormalization`` operator.
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


class MVN(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        m = x.mean(self.dim, keepdim=True)
        d = x - m
        v = (d * d).mean(self.dim, keepdim=True)
        return d / torch.sqrt(v)


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
class TestRefuseMeanVarianceNormalization(unittest.TestCase):
    TARGET_OPSET = 13

    def test_fuse_mvn_from_torch(self):
        model = export_decomposed(MVN(dim=-1), torch.randn(2, 4, 8), export_opset=13)

        # Precondition: torch emitted the decomposed mean/variance chain.
        self.assertEqual(op_types(model).count("ReduceMean"), 2)

        optimized = onnxoptimizer.optimize(
            model, ["fuse_mean_variance_normalization", "eliminate_deadend"]
        )
        relabel_opset(optimized, self.TARGET_OPSET)
        ops = op_types(optimized)

        self.assertEqual(ops.count("MeanVarianceNormalization"), 1)
        for prim in ("ReduceMean", "Sub", "Sqrt", "Div"):
            self.assertNotIn(prim, ops)
        onnx.checker.check_model(optimized)

        x = np.random.randn(2, 4, 8).astype(np.float32)
        ref = try_run(model, {"x": x})
        got = try_run(optimized, {"x": x})
        if ref is not None and got is not None:
            np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
