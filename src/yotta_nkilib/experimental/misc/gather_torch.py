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

"""PyTorch reference implementation for the gather kernel."""

import torch


def gather_torch_ref(input: torch.Tensor, dim: int, index: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of the gather kernel.

    Equivalent to ``input[index]`` for ``dim=0`` with a 2D ``input`` and a 1D ``index``.
    Called by ``torch_ref_wrapper``, which converts numpy arrays to torch tensors
    before invoking this function.

    Args:
        input (torch.Tensor): [N, D], Source tensor to gather rows from.
        dim (int): Dimension along which to gather. Must be 0 to match the kernel contract.
        index (torch.Tensor): [K], 1D integer tensor of row indices into ``input``.

    Returns:
        dict[str, torch.Tensor]: ``{"output_0": gathered}`` where ``gathered`` has shape
            [K, D] and ``gathered[i, :] = input[index[i], :]``.

    Notes:
        - ``index`` is cast to int64 to satisfy torch advanced-indexing requirements.
        - Only ``dim == 0`` is supported, matching the kernel's restriction.
    """
    if dim != 0:
        raise ValueError(f"gather currently only supports dim=0, got dim={dim}")
    return {"output_0": input[index.long()]}
