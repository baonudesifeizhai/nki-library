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

"""PyTorch reference implementation for argsort_unstable kernel."""

import torch

_ELEMS_PER_PASS = 8


def argsort_unstable_torch_ref(
    data: torch.Tensor,
    descending: bool = False,
    output_in_sbuf: bool = False,
) -> torch.Tensor:
    """Argsort unstable, matching ordering produced by argsort_unstable kernel.

    Args:
        data: (1, N) tensor
        descending: sort direction
        output_in_sbuf: unused, for signature compatibility with kernel

    Returns:
        (1, N) int32 tensor of argsort indices
    """
    data_f32 = data.flatten().float().clone()
    N = data_f32.shape[0]
    num_passes = N // _ELEMS_PER_PASS
    indices = torch.zeros(N, dtype=torch.int32)

    for pass_idx in range(num_passes):
        # max8: find 8 largest values in descending order
        top_vals, _ = torch.topk(data_f32, _ELEMS_PER_PASS, largest=True, sorted=True)

        # nc_match_replace8: hardware processes vals in reverse order (j=7 down to 0),
        # finding first occurrence of each val and writing index to dst_idx[j].
        pass_indices = torch.zeros(_ELEMS_PER_PASS, dtype=torch.int32)
        for val_idx in range(_ELEMS_PER_PASS - 1, -1, -1):
            val = top_vals[val_idx]
            matches = torch.where(data_f32 == val)[0]
            pos = matches[0].item()
            pass_indices[val_idx] = pos
            data_f32[pos] = float('-inf')

        # Write to output in order when sorting descending and in reverse when sorting ascending
        if descending:
            start = _ELEMS_PER_PASS * pass_idx
            indices[start : start + _ELEMS_PER_PASS] = pass_indices
        else:
            start = _ELEMS_PER_PASS * (num_passes - pass_idx) - 1
            for elem_idx in range(_ELEMS_PER_PASS):
                indices[start - elem_idx] = pass_indices[elem_idx]

    return indices.unsqueeze(0)
