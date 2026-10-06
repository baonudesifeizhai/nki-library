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

"""Torch reference implementations for collective operation kernels.

Each torch_ref has the exact same signature as its kernel. Cross-rank
communication uses get_rank()/get_pg() from distributed_adapter, which
handles both real torch.distributed and simulated environments.
"""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from .distributed_adapter import get_pg, get_rank  # noqa: F401 — re-exported for backward compat

# ==================== Torch References ====================
# Each has the exact same signature as its kernel.
# Uses get_rank()/get_pg() for cross-rank communication — works in both sim and real dist.
# Uses get_pg(replica_group)/get_rank() for cross-rank communication, like kernels use ncc.


def _to_torch(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor, handling bfloat16 and other non-native dtypes."""
    try:
        return torch.from_numpy(arr.copy())
    except TypeError:
        return torch.from_numpy(arr.astype(np.float32))


def _to_numpy(t: torch.Tensor, target_dtype: np.dtype) -> np.ndarray:
    """Convert torch tensor back to numpy with target dtype."""
    arr = t.numpy()
    if arr.dtype != target_dtype:
        arr = arr.astype(target_dtype)
    return arr


def all_reduce_hbm_torch_ref(input: np.ndarray, replica_group: ReplicaGroup) -> dict:
    """all_reduce sum across all ranks."""
    dtype = input.dtype
    t = _to_torch(input)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=get_pg(replica_group))
    return {"out": _to_numpy(t, dtype)}


def all_gather_hbm_torch_ref(input: np.ndarray, replica_group: ReplicaGroup, num_ranks: int) -> dict:
    """all_gather: concatenate all ranks' inputs along dim 0."""
    dtype = input.dtype
    t = _to_torch(input)
    output_list = [torch.zeros_like(t) for _ in range(num_ranks)]
    dist.all_gather(output_list, t, group=get_pg(replica_group))
    return {"out": _to_numpy(torch.cat(output_list, dim=0), dtype)}


def reduce_scatter_hbm_torch_ref(input: np.ndarray, replica_group: ReplicaGroup, num_ranks: int) -> dict:
    """reduce_scatter: sum all inputs, scatter chunks."""
    dtype = input.dtype
    t = _to_torch(input)
    chunks = list(t.chunk(num_ranks, dim=0))
    output = torch.zeros_like(chunks[0])
    dist.reduce_scatter(output, chunks, op=dist.ReduceOp.SUM, group=get_pg(replica_group))
    return {"out": _to_numpy(output, dtype)}


def all_to_all_hbm_torch_ref(input: np.ndarray, replica_group: ReplicaGroup) -> dict:
    """all_to_all: exchange chunks across ranks."""
    dtype = input.dtype
    pg = get_pg(replica_group)
    num_ranks = pg.size()
    t = _to_torch(input)
    chunks = list(t.chunk(num_ranks, dim=0))
    output_chunks = [torch.zeros_like(c) for c in chunks]
    dist.all_to_all(output_chunks, chunks, group=pg)
    return {"out": _to_numpy(torch.cat(output_chunks, dim=0), dtype)}


def rank_id_torch_ref(in_tensor: np.ndarray) -> dict:
    """rank_id select: output = in_tensor[rank_id]."""
    return {"out": in_tensor[get_rank()]}


def dma_copy_rank_id_torch_ref(in_tensor: np.ndarray, rank_id_lookup: np.ndarray) -> dict:
    """dma_copy rank_id: output = in_tensor[rank_id]."""
    return {"out": in_tensor[get_rank()]}
