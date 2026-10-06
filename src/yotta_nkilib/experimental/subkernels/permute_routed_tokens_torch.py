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

"""PyTorch reference implementation for permute_routed_tokens kernel."""

import torch

from .argsort_unstable_torch import argsort_unstable_torch_ref


def permute_routed_tokens_torch_ref(
    hidden_input: torch.Tensor,
    expert_index: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
) -> torch.Tensor:
    """
    Sort tokens by expert and pack hidden states, affinities, and token indices into a [T*K, n_output_cols] buffer.

    PyTorch reference implementation for permute_routed_tokens kernel.

    Dimensions:
        T: Number of tokens.
        H: Hidden dimension size.
        n_input_cols: Number of columns in hidden_input. When bf16, n_input_cols=H.
            When fp8, n_input_cols contains H and may contain additional columns for quantization scales.
        n_concat_cols: Number of columns for affinities (bf16) and token index (int32), viewed as hidden_input dtype.
        n_output_cols: n_input_cols + n_concat_cols
        K: Top-K experts per token.
        E: Total number of experts.

    Args:
        hidden_input (torch.Tensor): [T, n_input_cols] bf16 or fp8 tensor of hidden states.
            When hidden states are fp8, each row contains packed scales.
        expert_index (torch.Tensor): [T, K] int32 tensor of top-K expert indices per token.
        expert_affinities_masked (torch.Tensor): [T, E] bf16 tensor of expert affinities,
            with zeros for non-routed token/expert pairs.

    Returns:
        torch.Tensor: [T*K, n_output_cols] bf16 or fp8 tensor where each row is
            [hidden_state, affinity, token_index] sorted by expert index.
    """

    # When hidden_input is fp8, convert to uint8 internally before downcasting pre-concat.
    # Necessary because torch doesn't have complete support for slicing fp8 on CPU.
    original_hidden_input_dtype = hidden_input.dtype
    is_fp8 = original_hidden_input_dtype != torch.bfloat16
    if is_fp8:
        hidden_input = hidden_input.view(torch.uint8)

    T, K = expert_index.shape
    _, E = expert_affinities_masked.shape

    # Get argsort indices
    sorted_flat = argsort_unstable_torch_ref(
        expert_index.reshape(1, T * K), descending=False
    ).flatten()  # (T*K,) indices into flattened expert_index

    # Sort hidden_input, with reinterpret to bf16. Reinterpreting is a no-op when hidden_input is bf16;
    # when hidden_input is fp8, this allows us to concat in bf16 (supported by torch) before reinterpreting to fp8 post-concat.
    token_indices = sorted_flat // K
    hidden_sorted = hidden_input[token_indices]
    hidden_sorted = hidden_sorted.view(torch.bfloat16)

    # Sort affinities_sorted by T*K
    sorted_expert_indices = expert_index.flatten()[sorted_flat]
    sorted_expert_offsets = torch.arange(0, T * E, E, dtype=torch.int32).repeat_interleave(K)[sorted_flat]
    sorted_expert_indices_with_offset = sorted_expert_indices + sorted_expert_offsets

    affinities_sorted = expert_affinities_masked.flatten()[sorted_expert_indices_with_offset]

    # Concat in bf16 (torch has limited support for concat with 1B dtypes)
    affinities_sorted = affinities_sorted.reshape(T * K, 1)
    token_indices = token_indices.reshape(T * K, 1).view(torch.bfloat16)
    result = torch.cat([hidden_sorted, affinities_sorted, token_indices], dim=1)

    # Reinterpret concatenated result as fp8 when hidden_input is fp8
    if is_fp8:
        result = result.view(original_hidden_input_dtype)
    return result
