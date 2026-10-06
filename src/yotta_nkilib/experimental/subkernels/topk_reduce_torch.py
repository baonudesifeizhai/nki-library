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

"""PyTorch reference implementation for topk_reduce kernel."""

import torch


def topk_reduce_torch_ref(input: torch.Tensor, T: int, K: int, token_base_index: int = 1) -> torch.Tensor:
    """
    Compute MoE Top-K reduction across sparse all_to_all_v() collective output buffer.

    Gathers scattered rows by packed global token index and reduces along
    the K dimension.

    Token indices are 1-indexed (token 0 → index 1, token 1 → index 2, etc.),
    and padded rows must have index -1.

    Dimensions:
        TK_padded: n_src_ranks * T, padded input row count
        H: Hidden dimension size (must be divisible by LNC)
        T: Total number of input tokens (up to 128)
        K: Number of routed experts per token (up to 8)

    Args:
        input (torch.Tensor): [TK_padded, H + 2], bf16/fp16. Sparse input buffer containing T*K
            scattered outputs. Global token index is packed as int32 in the final 2x
            columns of each row (1-indexed, -1 for padding).
        T (int): Total number of input tokens.
        K (int): Number of routed experts per token.

    Returns:
        torch.Tensor: [T, H], bf16/fp16. Ordered and reduced output.
            out[t] = sum of all rows with index t+1.

    Pseudocode:
        global_token_indices = extract_int32_index(input[:, H:])
        for token_idx in range(T):
            matching_rows = find_rows_where(global_token_indices == token_idx + 1)
            output[token_idx] = sum(input[matching_rows, :H])
    """
    H = input.shape[1] - 2

    # Extract packed int32 global token indices from last 2 bf16 columns
    idx_bf16 = input[:, H:].to(torch.bfloat16).contiguous()
    global_token_indices = idx_bf16.view(torch.int32).squeeze(-1)

    # For each token, gather matching rows and sum
    out = torch.zeros(T, H, dtype=input.dtype)
    for t in range(T):
        mask = global_token_indices == token_base_index + t
        out[t] = input[mask, :H].sum(dim=0)

    return out
