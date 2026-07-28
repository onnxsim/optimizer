# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

# ATTENTION: The code in this file is highly EXPERIMENTAL.
# Adventurous users should note that the APIs will probably change.

"""ONNX optimizer."""

from __future__ import annotations

__all__ = [
    "optimize",
    "get_available_passes",
    "get_fuse_and_elimination_passes",
    "main",
]


import os
import tempfile

import onnx

import onnxoptimizer.onnx_opt_cpp2py_export as _c
from onnxoptimizer.onnx_opt_cpp2py_export import (
    get_available_passes,
    get_fuse_and_elimination_passes,
)
from onnxoptimizer.onnxoptimizer_main import main

from .version import version as __version__  # noqa


def optimize(
    model: onnx.ModelProto,
    passes: list[str] | None = None,
    fixed_point: bool = False,
    initializers_as_constants: bool = True,
) -> onnx.ModelProto:
    """Apply the optimization on the serialized ModelProto.

    Arguments:
        model: ONNX model.
        passes: Optimization names.
        fixed_point: Whether to run the passes to a fixed point.
        initializers_as_constants: Whether the passes may treat graph
            initializers as constant tensors (the default, ``True``). When set
            to ``False`` initializers are treated as non-constant, so
            value-baking passes such as ``fuse_bn_into_conv`` leave
            initializer-backed weights untouched; ``Constant`` nodes are still
            treated as constants.

    Return:
        Optimized model.
    """

    if passes is None:
        passes = get_fuse_and_elimination_passes()
    if not isinstance(model, onnx.ModelProto):
        raise TypeError(f"Optimizer only accepts ModelProto, incorrect type: {type(model)}")
    # The C++ core reads this switch from thread-local state deep inside the
    # passes, so set it around the call and restore it afterwards to avoid
    # leaking the setting to unrelated callers on the same thread.
    previous = _c.initializers_as_constants()
    _c.set_initializers_as_constants(initializers_as_constants)
    try:
        return _optimize_impl(model, passes, fixed_point)
    finally:
        _c.set_initializers_as_constants(previous)


def _optimize_impl(
    model: onnx.ModelProto, passes: list[str], fixed_point: bool
) -> onnx.ModelProto:
    try:
        model_str = model.SerializeToString()
        if fixed_point:
            optimized_model_str = _c.optimize_fixedpoint(model_str, passes)
        else:
            optimized_model_str = _c.optimize(model_str, passes)

        return onnx.load_from_string(optimized_model_str)
    except ValueError:
        with (
            tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as file_src,
            tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as file_dest,
            tempfile.NamedTemporaryFile(delete=False) as data_file_src,
            tempfile.NamedTemporaryFile(delete=False) as data_file_dest,
        ):
            data_src_rel_filename = os.path.relpath(
                data_file_src.name, os.path.dirname(file_src.name)
            )
            data_dest_rel_filename = os.path.relpath(
                data_file_dest.name, os.path.dirname(file_dest.name)
            )
            try:
                onnx.save(
                    model,
                    file_src.name,
                    save_as_external_data=True,
                    location=data_src_rel_filename,
                    convert_attribute=True,
                )
                if fixed_point:
                    _c.optimize_fixedpoint_from_path(
                        file_src.name, file_dest.name, passes, data_dest_rel_filename
                    )
                else:
                    _c.optimize_from_path(
                        file_src.name, file_dest.name, passes, data_dest_rel_filename
                    )
                return onnx.load(file_dest.name, load_external_data=True)
            finally:
                os.remove(file_src.name)
                os.remove(file_dest.name)
                os.remove(data_file_src.name)
                os.remove(data_file_dest.name)
