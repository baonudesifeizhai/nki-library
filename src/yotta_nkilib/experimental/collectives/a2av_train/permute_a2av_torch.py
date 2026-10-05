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
"""Torch reference for capacity-bounded permute_a2av."""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from ..collectives_torch import _to_numpy, _to_torch, get_pg


def _validate_routed_row_permutation(
    send_indices: np.ndarray,
    send_counts: np.ndarray,
) -> None:
    """Validate that active routing entries assign every routed row once."""
    num_rows, ep_size = send_indices.shape
    active_indices = np.concatenate([send_indices[: int(send_counts[0, peer]), peer] for peer in range(ep_size)])
    expected = np.arange(num_rows, dtype=active_indices.dtype)
    if active_indices.shape != expected.shape or not np.array_equal(
        np.sort(active_indices),
        expected,
    ):
        raise ValueError(
            "active send_indices must be a permutation of routed rows "
            f"[0, {num_rows}), got {active_indices.shape[0]} active entries"
        )


def permute_a2av_torch_ref(
    hidden_states: np.ndarray,
    send_indices: np.ndarray,
    send_counts: np.ndarray,
    recv_capacity: int,
    replica_group: ReplicaGroup,
) -> dict:
    """Gather routed rows and emulate packed explicit-displacement A2AV."""
    N, H = hidden_states.shape
    EP = send_indices.shape[1]
    dtype = hidden_states.dtype
    pg = get_pg(replica_group)
    _validate_routed_row_permutation(send_indices, send_counts)

    send_count_chunks = [torch.tensor([int(send_counts[0, peer])], dtype=torch.int64) for peer in range(EP)]
    recv_count_chunks = [torch.zeros(1, dtype=torch.int64) for _ in range(EP)]
    dist.all_to_all(recv_count_chunks, send_count_chunks, group=pg)
    recv_counts = np.array([[int(chunk[0]) for chunk in recv_count_chunks]], dtype=np.uint32)
    recv_total = int(recv_counts.sum())
    if recv_total > recv_capacity:
        raise ValueError(f"receive total {recv_total} exceeds capacity {recv_capacity}")

    torch_dtype = _to_torch(hidden_states[:1]).dtype
    send_chunks = []
    send_sizes = []
    for peer in range(EP):
        count = int(send_counts[0, peer])
        send_sizes.append(count * H)
        if count:
            indices = send_indices[:count, peer].astype(np.intp)
            send_chunks.append(_to_torch(hidden_states[indices, :]).reshape(-1))
        else:
            send_chunks.append(torch.zeros(0, dtype=torch_dtype))
    send_flat = torch.cat(send_chunks)

    recv_sizes = [int(recv_counts[0, peer]) * H for peer in range(EP)]
    recv_flat = torch.zeros(sum(recv_sizes), dtype=send_flat.dtype)
    dist.all_to_all_single(
        recv_flat,
        send_flat,
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=pg,
    )

    recv_data = np.zeros((recv_capacity, H), dtype=dtype)
    if recv_total:
        recv_data[:recv_total, :] = _to_numpy(recv_flat.reshape(recv_total, H), dtype)

    send_displs = np.concatenate([[0], np.cumsum(send_counts[0, :-1])]).astype(np.uint32)
    recv_displs = np.concatenate([[0], np.cumsum(recv_counts[0, :-1])]).astype(np.uint32)
    metadata = np.stack(
        [
            send_counts[0].astype(np.uint32) * H,
            send_displs * H,
            recv_counts[0] * H,
            recv_displs * H,
        ]
    )
    return {"recv_data": recv_data, "metadata": metadata}
