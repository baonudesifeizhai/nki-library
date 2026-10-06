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

"""PyTorch reference implementation for linear_scan kernel."""

import torch


def linear_scan_torch_ref(decay, data, initial=None):
    """
    PyTorch reference implementation of linear scan.

    Computes result[t] = decay[t] * result[t-1] + data[t] sequentially
    along the last dimension.

    Args:
        decay (torch.Tensor): Decay coefficients of shape (..., P, L).
        data (torch.Tensor): Additive input of shape (..., P, L).
        initial (torch.Tensor, optional): Initial state of shape (..., P, 1).

    Returns:
        dict: {"result": torch.Tensor, "final_state": torch.Tensor}
    """
    orig_shape = decay.shape
    if decay.dim() == 2:
        decay = decay.unsqueeze(0)
        data = data.unsqueeze(0)
        if initial is not None:
            initial = initial.unsqueeze(0)

    outer, P, L = decay.shape
    result = torch.zeros(outer, P, L, dtype=torch.float32)

    if initial is not None:
        state = initial[:, :, 0].float()
    else:
        state = torch.zeros((outer, P), dtype=torch.float32)

    for t in range(L):
        state = decay[:, :, t].float() * state + data[:, :, t].float()
        result[:, :, t] = state

    final_state = state.unsqueeze(-1)
    return {
        "result": result.reshape(orig_shape),
        "final_state": final_state.float(),
    }
