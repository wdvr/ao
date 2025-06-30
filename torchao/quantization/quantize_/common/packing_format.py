# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class PackingFormat(str, Enum):
    """Packing format for Tensor subclasses in torchao, enum for how
    the values are packed and laid out in the stored Tensor.
    """

    """
    plain means the non-packed, row-major format
    """
    PLAIN = "plain"

    """
    preshuffled is referring to the preshuffled format used by fbgemm kernels
    """
    PRESHUFFLED = "preshuffled"
    _LEGACY = "_legacy"
