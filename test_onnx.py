# Owner(s): ["module: onnx"]

"""Tests for onnx export that don't run the exported model."""

from __future__ import annotations

import contextlib
import io
import itertools
import unittest
import unittest.mock
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

import onnx
import onnx.numpy_helper
import pytorch_test_common

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.onnx import symbolic_helper, utils
from torch.onnx._internal import registration
from torch.testing._internal import common_quantization, common_utils, jit_utils


def export_to_onnx(
    model: Union[torch.nn.Module, torch.jit.ScriptFunction],
    input: Union[torch.Tensor, Tuple[torch.Tensor]],
    custom_ops: Optional[
        Iterable[Union[contextlib.AbstractContextManager, contextlib.ContextDecorator]]
    ] = None,
    mocks: Optional[Iterable] = None,
    operator_export_type: torch.onnx.OperatorExportTypes = torch.onnx.OperatorExportTypes.ONNX,
    opset_version: int = 17,
    **torch_onnx_export_kwargs,
) -> onnx.ModelProto:
    """Exports `model(input)` to ONNX and returns it.

    Custom operators and/or unittest patches can be used help reproducing specific behaviors.

    Args:
        model: model to export
        input: model input with same format as `torch.onnx.export(..,args,...)`
        custom_ops: list of custom operators to use during export
        mocks: list of mocks to use during export
        operator_export_type: export type as described by `torch.onnx.export(...operator_export_type,...)`
        opset_version: ONNX opset version as described by `torch.onnx.export(...opset_version,...)`
        torch_onnx_export_kwargs: extra torch.onnx.export kwargs arguments
    Returns:
        A valid ONNX model (`onnx.ModelProto`)
    """
    custom_ops = custom_ops or []
    mocks = mocks or []
    with contextlib.ExitStack() as stack:
        for ctx in itertools.chain(custom_ops, mocks):
            stack.enter_context(ctx)

        f = io.BytesIO()
        torch.onnx.export(
            model,
            input,
            f,
            operator_export_type=operator_export_type,
            opset_version=opset_version,
            **torch_onnx_export_kwargs,
        )

    # Validate ONNX graph before returning it
    onnx_model = onnx.load_from_string(f.getvalue())
    onnx.checker.check_model(onnx_model)
    return onnx_model


@common_utils.instantiate_parametrized_tests
class TestONNXExport(pytorch_test_common.ExportTestCase):
    def test_fuse_addmm(self):
        class AddmmModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x

        x = torch.ones(3, 3)
        f = io.BytesIO()
        torch.onnx.export(AddmmModel(), x, f, verbose=False)

    def test_onnx_transpose_incomplete_tensor_type(self):
        # Smoke test to get us into the state where we are attempting to export
        # a transpose op, where the input is a TensorType without size information.
        # This would previously not work, since we would
        # take the size of the input and use the length of its sizes as the
        # number of dimensions in the permutation.
        class Foo(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.contiguous().transpose(0, 1).sum()

        class TraceMe(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Foo()

            def forward(self, x):
                return self.foo(x)

        tm = TraceMe()
        tm = torch.jit.trace(tm, torch.rand(3, 4))
        f = io.BytesIO()
        torch.onnx.export(tm, (torch.rand(3, 4),), f)

    def test_export_tensoroption_to(self):
        def foo(x):
            return x[0].clone().detach().cpu() + x

        traced = torch.jit.trace(foo, (torch.rand([2])))

        torch.onnx.export_to_pretty_string(traced, (torch.rand([2]),))