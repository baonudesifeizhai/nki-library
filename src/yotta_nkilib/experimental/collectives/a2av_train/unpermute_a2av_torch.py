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
"""Torch reference for capacity-bounded unpermute_a2av."""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from ..collectives_torch import _to_numpy, _to_torch, get_pg
from .permute_a2av_torch import _validate_routed_row_permutation


def unpermute_a2av_torch_ref(
    output: np.ndarray,
    send_indices: np.ndarray,
    send_counts: np.ndarray,
    recv_counts: np.ndarray,
    replica_group: ReplicaGroup,
) -> dict:
    """Emulate packed reverse A2AV and restore unique routed rows."""
    N, EP = send_indices.shape
    _, H = output.shape
    dtype = output.dtype
    pg = get_pg(replica_group)
    _validate_routed_row_permutation(send_indices, send_counts)

    send_total = int(recv_counts.sum())
    send_flat = _to_torch(output[:send_total, :]).reshape(-1)
    input_sizes = [int(recv_counts[0, peer]) * H for peer in range(EP)]
    output_sizes = [int(send_counts[0, peer]) * H for peer in range(EP)]
    recv_flat = torch.zeros(sum(output_sizes), dtype=send_flat.dtype)
    dist.all_to_all_single(
        recv_flat,
        send_flat,
        output_split_sizes=output_sizes,
        input_split_sizes=input_sizes,
        group=pg,
    )

    result = np.zeros((N, H), dtype=dtype)
    offset = 0
    for peer in range(EP):
        count = int(send_counts[0, peer])
        if count:
            chunk = _to_numpy(recv_flat[offset : offset + count * H].reshape(count, H), dtype)
            indices = send_indices[:count, peer].astype(np.intp)
            result[indices, :] = chunk
        offset += count * H
    return {"result": result}
