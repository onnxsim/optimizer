# SPDX-FileCopyrightText: ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regenerates the ONNX models the JavaScript tests run against.

    python3 js/test/fixtures/make_fixtures.py

The models are checked in so that `npm test` needs nothing but Node.js.
"""

import pathlib

import onnx
from onnx import TensorProto, helper

HERE = pathlib.Path(__file__).parent


def save(model: onnx.ModelProto, name: str) -> None:
    onnx.checker.check_model(model)
    path = HERE / name
    path.write_bytes(model.SerializeToString())
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def identity_chain() -> onnx.ModelProto:
    """x -> Identity -> Relu -> Identity -> y.

    The fuse and elimination passes (and eliminate_identity on its own) leave
    only the Relu behind.
    """
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["a"], name="first_identity"),
            helper.make_node("Relu", ["a"], ["b"], name="relu"),
            helper.make_node("Identity", ["b"], ["y"], name="second_identity"),
        ],
        "identity_chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    return helper.make_model(graph, producer_name="onnxoptimizer-js-tests")


if __name__ == "__main__":
    save(identity_chain(), "identity_chain.onnx")
