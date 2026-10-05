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

"""PyTorch reference implementation for cross entropy loss kernel."""

import torch
import torch.nn.functional as F


def cross_entropy_forward_torch_ref(
    logits_hbm: torch.Tensor,
    targets_hbm: torch.Tensor,
    positions_per_batch: int = 32,
    chunk_size: int = 32768,
    dtype=None,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of cross entropy forward pass.

    This is a reference implementation for testing the NKI cross_entropy_forward kernel.
    It implements the same mathematical operation using PyTorch operations.

    Args:
        logits_hbm: Input logits tensor of shape [num_positions, vocab_size]
        targets_hbm: Target indices tensor of shape [num_positions]
        positions_per_batch: Hardware param (ignored, included for signature matching)
        chunk_size: Hardware param (ignored, included for signature matching)
        dtype: Hardware param (ignored, included for signature matching)

    Returns:
        dict with keys:
            loss_hbm: Cross entropy loss per position [num_positions]
            lse_state_hbm: Log-sum-exp values per position [num_positions]
    """
    targets = targets_hbm.long() if targets_hbm.dtype != torch.long else targets_hbm
    loss = F.cross_entropy(logits_hbm, targets, reduction='none')
    # Compute log-sum-exp for backward pass
    lse = torch.logsumexp(logits_hbm, dim=1)
    return {"loss_hbm": loss, "lse_state_hbm": lse}


def cross_entropy_backward_torch_ref(
    logits_hbm: torch.Tensor,
    targets_hbm: torch.Tensor,
    lse_state_hbm: torch.Tensor = None,
    reduction: str = "mean",
    positions_per_batch: int = 32,
    chunk_size: int = 32768,
    dtype=None,
    inplace: bool = True,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of cross entropy backward pass.

    This is a reference implementation for testing the NKI cross_entropy_backward kernel.
    It uses PyTorch's autograd to compute the gradient of cross entropy loss with respect
    to logits, which is guaranteed to be correct.

    Args:
        logits_hbm: Input logits tensor of shape [num_positions, vocab_size]
        targets_hbm: Target indices tensor of shape [num_positions]
        lse_state_hbm: LSE state from forward pass (ignored, included for signature matching)
        reduction: 'mean' or 'sum'
        positions_per_batch: Hardware param (ignored, included for signature matching)
        chunk_size: Hardware param (ignored, included for signature matching)
        dtype: Hardware param (ignored, included for signature matching)
        inplace: Hardware param (ignored, included for signature matching)

    Returns:
        dict with keys:
            grad_logits_hbm: Gradient with respect to logits [num_positions, vocab_size]
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean' or 'sum'.")

    targets = targets_hbm.long() if targets_hbm.dtype != torch.long else targets_hbm
    logits_copy = logits_hbm.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(logits_copy, targets, reduction=reduction)
    loss.backward()
    return {"logits_hbm": logits_copy.grad}
