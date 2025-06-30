# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import ClassVar

__all__ = [
    "AOBaseTensorConfig",
]


class AOBaseTensorConfig(abc.ABC):
    """This is used when we need to apply quantization for some Tensors,
    used in `to_tensor` API, example::

       class Float8TensorConfig(AOBaseTensorConfig):
          ...


       @register_to_tensor_handler(Float8TensorConfig)
       def _(tensor: torch.Tensor, config: Float8TensorConfig):
           ...


       # when implementing linear for dynamic quantization
       def linear(...):
           quantized_input = to_tensor(input_tensor, weight.act_quant_config)
           ...
    """

    # Base Version of a config
    VERSION: ClassVar[int] = 1
