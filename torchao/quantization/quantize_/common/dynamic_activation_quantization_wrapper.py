# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    TorchAOBaseTensor,
)

from .config import AOBaseTensorConfig
from .quantize_tensor import quantize_tensor

__all__ = [
    "DynamicActivationQuantizationWrapper",
]

aten = torch.ops.aten


class DynamicActivationQuantizationWrapper(TorchAOBaseTensor):
    """
    Applies activation quantization for linear operator, this is used to support
    dynamic quantization, user can pass in a `act_tensor_config`
    that is used to quantize the activation

    Args:
      `original_weight_tensor`: the weight tensor, if weight need to be quantized as well, we'd need
        to apply quantization to weight first, e.g. for int8 dynamic activation int8 weight quantization
        we will first apply int8 quantization to weight and then apply DynamicActivationQuantizationWrapper
        on top of it
      `act_tensor_config` (AOBaseTensorConfig): A tensor config used for dynamically quantize activations
         it will be applied to activation in lienar or bmm op right now
    """

    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        act_tensor_config: AOBaseTensorConfig,
    ):
        kwargs = {}
        dtype = original_weight_tensor.dtype
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        kwargs["device"] = original_weight_tensor.device
        shape = original_weight_tensor.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        original_weight_tensor: torch.Tensor,
        act_tensor_config: AOBaseTensorConfig,
    ):
        self.original_weight_tensor = original_weight_tensor
        self.act_tensor_config = act_tensor_config

    def __repr__(self):
        return f"{self.__class__.__name__}({self.act_tensor_config}, {self.original_weight_tensor})"

    def __tensor_flatten__(self):
        return ["original_weight_tensor"], [self.act_tensor_config]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        original_weight_tensor = tensor_data_dict["original_weight_tensor"]
        (act_tensor_config,) = tensor_attributes
        return cls(original_weight_tensor, act_tensor_config)

    @classmethod
    def from_float(
        cls,
        weight: torch.Tensor,
        act_tensor_config: AOBaseTensorConfig,
    ):
        return cls(weight, act_tensor_config)

    def _apply_fn_to_data(self, fn):
        return self.__class__(
            fn(self.original_weight_tensor),
            self.act_tensor_config,
        )

    def to(self, *args, **kwargs):
        kwargs = self._get_to_kwargs(*args, **kwargs)
        return self.__class__(
            self.original_weight_tensor.to(**kwargs),
            self.act_tensor_config,
        )


def _same_metadata(
    self: DynamicActivationQuantizationWrapper,
    src: DynamicActivationQuantizationWrapper,
):
    return (
        isinstance(self, DynamicActivationQuantizationWrapper)
        and isinstance(src, DynamicActivationQuantizationWrapper)
        and self.shape == src.shape
        and self.act_tensor_config == src.act_tensor_config
    )


implements = DynamicActivationQuantizationWrapper.implements


@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    if isinstance(weight_tensor, DynamicActivationQuantizationWrapper):
        if input_tensor.numel() == 0:
            return input_tensor
        original_weight_tensor = weight_tensor.original_weight_tensor
        c = weight_tensor.act_tensor_config
        quantized_input = quantize_tensor(input_tensor, c)
        return torch.nn.functional.linear(quantized_input, original_weight_tensor, bias)

    raise NotImplementedError(
        "DynamicActivationQuantizationWrapper: No specialized dispatch found for linear op"
    )


@implements(torch.bmm)
def _(func, types, args, kwargs):
    input_tensor, weight_tensor = (
        args[0],
        args[1],
    )
    if isinstance(weight_tensor, DynamicActivationQuantizationWrapper):
        original_weight_tensor = weight_tensor.original_weight_tensor
        c = weight_tensor.act_tensor_config
        quantized_input = quantize_tensor(input_tensor, c)
        return torch.bmm(quantized_input, original_weight_tensor)

    raise NotImplementedError(
        "DynamicActivationQuantizationWrapper: No specialized dispatch found for bmm op"
    )


@implements([aten.mm.default, aten.addmm.default])
def _(func, types, args, kwargs):
    if not args[0].is_floating_point():
        raise NotImplementedError(
            "DynamicActivationQuantizationWrapper: expecting a floating point input"
        )

    if func == aten.addmm.default:
        assert args[1].shape[-1] == args[2].shape[0], (
            f"need mat1 shape: {args[1].shape} final"
            f"dim to match mat2 shape: {args[2].shape} first dim "
        )
        input_tensor, weight_tensor, bias = (
            args[1],
            args[2],
            args[0],
        )
        if isinstance(weight_tensor, DynamicActivationQuantizationWrapper):
            transposed_weight_tensor = weight_tensor.t()
            if input_tensor.numel() == 0:
                return input_tensor
            original_weight_tensor = transposed_weight_tensor.original_weight_tensor
            c = transposed_weight_tensor.act_tensor_config
            quantized_input = quantize_tensor(input_tensor, c)
            return torch.nn.functional.linear(quantized_input, original_weight_tensor, bias)
    else:
        # aten.mm.default
        assert args[0].shape[-1] == args[1].shape[0], (
            f"need mat1 shape: {args[0].shape} final dim"
            f"to match mat2 shape: {args[1].shape} first dim"
        )
        input_tensor, weight_tensor = (
            args[0],
            args[1],
        )
        if isinstance(weight_tensor, DynamicActivationQuantizationWrapper):
            transposed_weight_tensor = weight_tensor.t()
            if input_tensor.numel() == 0:
                return input_tensor
            original_weight_tensor = transposed_weight_tensor.original_weight_tensor
            c = transposed_weight_tensor.act_tensor_config
            quantized_input = quantize_tensor(input_tensor, c)
            return torch.nn.functional.linear(quantized_input, original_weight_tensor, bias)

    raise NotImplementedError(
        "DynamicActivationQuantizationWrapper: No specialized dispatch found for linear op"
    )


@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(func)
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.clone)
    )


@implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0].to(*args[1:], **kwargs)._apply_fn_to_data(torch.clone),
    )


@implements(aten.copy_.default)
def _(func, types, args, kwargs):
    self = args[0]
    src = args[1]
    if _same_metadata(self, src):
        self_tensors = self.__tensor_flatten__()[0]
        for tensor_name in self_tensors:
            getattr(self, tensor_name).copy_(getattr(src, tensor_name))
        return
    raise ValueError(
        f"Not supported args for copy_ due to metadata mistach: {args[0], args[1]}"
    )


@implements(aten.t.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.t)
    )


@implements(aten.transpose.int)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        DynamicActivationQuantizationWrapper(
            func(args[0].original_weight_tensor, *args[1:]),
            args[0].act_tensor_config,
        ),
    )


@implements(aten.cat.default)
def _(func, types, args, kwargs):
    ls = args[0]
    act_tensor_config = ls[0].act_tensor_config
    # make sure all activation tensor config are the same
    for i in range(1, len(ls)):
        assert act_tensor_config == ls[i].act_tensor_config

    original_weights = [x.original_weight_tensor for x in ls]

    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        DynamicActivationQuantizationWrapper(
            func(original_weights, *args[1:]),
            act_tensor_config,
        ),
    )


@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        DynamicActivationQuantizationWrapper(
            func(args[0].original_weight_tensor, *args[1:]),
            args[0].act_tensor_config,
        ),
    )


@implements(aten.select.int)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        DynamicActivationQuantizationWrapper(
            func(args[0].original_weight_tensor, *args[1:]),
            args[0].act_tensor_config,
        ),
    )


@implements(aten.index.Tensor)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        DynamicActivationQuantizationWrapper(
            func(args[0].original_weight_tensor, *args[1:]),
            args[0].act_tensor_config,
        ),
    )


# this is needed for DTensor.from_local() and for flattening tensor
@implements(aten.view.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        DynamicActivationQuantizationWrapper(
            func(args[0].original_weight_tensor, *args[1:]),
            args[0].act_tensor_config,
        ),
    )


if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with DynamicActivationQuantizationWrapper weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([DynamicActivationQuantizationWrapper])
