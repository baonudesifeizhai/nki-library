# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BLAS primitives for nkiprimitives."""

from .activation import Activation, activation
from .broadcast import Broadcast, broadcast
from .matmul import Matmul
from .quantize_mx import QuantizeMX
from .reciprocal import Reciprocal, reciprocal
from .tensor_scalar import TensorScalar, tensor_scalar
from .tensor_tensor import TensorTensor
from .transpose import Transpose, transpose

__all__ = [
    # Classes (for tiled operations)
    "Activation",
    "Broadcast",
    "Matmul",
    "QuantizeMX",
    "Reciprocal",
    "TensorScalar",
    "TensorTensor",
    "Transpose",
    # Compact functions (whole tensor, no tiling)
    "activation",
    "broadcast",
    "reciprocal",
    "tensor_scalar",
    "transpose",
]
