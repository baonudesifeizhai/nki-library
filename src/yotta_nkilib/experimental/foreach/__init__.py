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

"""NKI kernels for foreach operations."""

from .foreach_elementwise import (
    add_scalar_kernel,
    add_tensor_kernel,
    addcdiv_kernel,
    addcmul_kernel,
    div_scalar_kernel,
    div_tensor_kernel,
    lerp_kernel,
    mul_scalar_kernel,
    mul_tensor_kernel,
    sqrt_kernel,
    sub_scalar_kernel,
    sub_tensor_kernel,
)
from .foreach_norm import l1_norm_kernel, l2_norm_kernel, linf_norm_kernel

__all__ = [
    "add_scalar_kernel",
    "sub_scalar_kernel",
    "mul_scalar_kernel",
    "div_scalar_kernel",
    "add_tensor_kernel",
    "sub_tensor_kernel",
    "mul_tensor_kernel",
    "div_tensor_kernel",
    "addcdiv_kernel",
    "addcmul_kernel",
    "lerp_kernel",
    "sqrt_kernel",
    "l1_norm_kernel",
    "l2_norm_kernel",
    "linf_norm_kernel",
]
