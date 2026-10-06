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

"""PyTorch reference implementations for foreach norm kernels."""

import torch


def l2_norm_torch_ref(data: torch.Tensor, numel: int) -> torch.Tensor:
    """L2 norm reference: sqrt(sum(x^2))."""
    return torch.linalg.vector_norm(data.float(), ord=2).to(data.dtype).reshape(1, 1)


def l1_norm_torch_ref(data: torch.Tensor, numel: int) -> torch.Tensor:
    """L1 norm reference: sum(|x|)."""
    return torch.linalg.vector_norm(data.float(), ord=1).to(data.dtype).reshape(1, 1)


def linf_norm_torch_ref(data: torch.Tensor, numel: int) -> torch.Tensor:
    """Linf norm reference: max(|x|)."""
    return torch.linalg.vector_norm(data.float(), ord=float("inf")).to(data.dtype).reshape(1, 1)
