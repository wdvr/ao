# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, Dict

import torch

from torchao.quantization.quantize_.common.config import AOBaseTensorConfig
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

__all__ = [
    "register_quantize_tensor_handler",
    "quantize_tensor",
]

_QUANTIZE_TENSOR_CONFIG_HANDLER: Dict[
    AOBaseTensorConfig, Callable[[torch.Tensor, AOBaseTensorConfig], torch.Tensor]
] = {}


def register_quantize_tensor_handler(config_type):
    """
    A decorator to register a quantize function to map from a tensor quantization
    configuration (child of `AOBaseTensorConfig`) to a function that quantize
    a `torch.Tensor` according to the specified configuration.

    Example::
       class Float8TensorConfig(AOBaseTensorConfig):
          ...


       @register_quantize_tensor_handler(Float8TensorConfig)
       def _(tensor: torch.Tensor, config: Float8TensorConfig):
           ...


       # when implementing linear for dynamic quantization
       def linear(...):
           # quantize_tensor quantized input_tensor to a quantized tensor
           quantized_input = quantize_tensor(input_tensor, weight.act_quant_config)
           ...


    """

    @functools.wraps(config_type)
    def decorator(func):
        _QUANTIZE_TENSOR_CONFIG_HANDLER[config_type] = func
        return func  # needed to make the functions usable externally

    if TORCH_VERSION_AT_LEAST_2_5:
        torch.serialization.add_safe_globals([config_type])

    return decorator


def quantize_tensor(tensor, config):
    """Applies the config to quantize tensor, to get a quantized Tensor.

    Example::
       class Float8TensorConfig(AOBaseTensorConfig):
          ...


       @register_quantize_tensor_handler(Float8TensorConfig)
       def _(tensor: torch.Tensor, config: Float8TensorConfig):
           ...


       # when implementing linear for dynamic quantization
       def linear(...):
           # quantize_tensor quantized input_tensor to a quantized tensor
           quantized_input = quantize_tensor(input_tensor, weight.act_quant_config)
           ...
    """
    handler = _QUANTIZE_TENSOR_CONFIG_HANDLER.get(type(config))
    return handler(tensor, config)
