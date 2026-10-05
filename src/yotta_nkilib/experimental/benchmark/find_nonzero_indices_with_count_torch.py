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

"""PyTorch reference implementation for find_nonzero_indices_with_count kernel."""

import torch

from .find_nonzero_indices_with_count import PADDING_VALUE


def find_nonzero_indices_with_count_torch_ref(input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """PyTorch reference for find_nonzero_indices_with_count kernel.

    Finds nonzero indices in a 1D tensor and returns them with a count.

    Args:
        input_tensor (torch.Tensor): [1, T], Input tensor.

    Returns:
        dict with 'output' tensor of shape [1, T+1] int32.
        Format: [idx1, idx2, ..., -1, -1, ..., count]
    """
    T = input_tensor.shape[-1]
    output = torch.full((1, T + 1), PADDING_VALUE, dtype=torch.int32)

    nonzero_indices = torch.nonzero(input_tensor[0], as_tuple=False).squeeze(-1)
    count = nonzero_indices.shape[0]

    if count > 0:
        output[0, :count] = nonzero_indices.to(torch.int32)
    output[0, -1] = count

    return {"output": output}
