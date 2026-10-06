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

"""PyTorch reference implementation for selective_scan kernel."""

import torch


def selective_scan_torch_ref(x, dt, A, B, C, D=None, initial_state=None):
    """
    PyTorch reference implementation of selective scan (SSM).

    Implements fused discretization, recurrence, and output projection
    matching the Mamba-style selective scan kernel.

    Args:
        x (torch.Tensor): Input tensor of shape [batch, channels, L].
        dt (torch.Tensor): Time step tensor of shape [batch, channels, L].
        A (torch.Tensor): State transition matrix of shape [channels, state_size].
        B (torch.Tensor): Input projection of shape [batch, state_size, L].
        C (torch.Tensor): Output projection of shape [batch, state_size, L].
        D (torch.Tensor, optional): Skip connection weights of shape [channels].
        initial_state (torch.Tensor, optional): Initial state of shape
            [batch, channels, state_size].

    Returns:
        dict: {"y": torch.Tensor, "final_state": torch.Tensor}
    """
    batch_size, channels, L = x.shape
    _, state_size = A.shape

    x_f = x.float()
    dt_f = dt.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()

    y = torch.zeros((batch_size, channels, L), dtype=torch.float32)
    final_state = torch.zeros((batch_size, channels, state_size), dtype=torch.float32)

    for b in range(batch_size):
        for n in range(state_size):
            if initial_state is not None:
                state = initial_state[b, :, n].float().clone()
            else:
                state = torch.zeros(channels, dtype=torch.float32)

            for t in range(L):
                decay = torch.exp(dt_f[b, :, t] * A_f[:, n])
                inp = dt_f[b, :, t] * x_f[b, :, t] * B_f[b, n, t]
                state = decay * state + inp
                y[b, :, t] += C_f[b, n, t] * state

            final_state[b, :, n] = state

    if D is not None:
        D_f = D.float()
        y += D_f.unsqueeze(0).unsqueeze(-1) * x_f

    return {"y": y, "final_state": final_state}
