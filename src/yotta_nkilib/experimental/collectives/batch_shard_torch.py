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
"""Torch reference for attn_q_batch_shard kernel."""

from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from .batch_shard import AttnQBatchShardLayout
from .collectives_torch import _to_numpy, _to_torch, get_pg, get_rank


def attn_q_batch_shard_torch_ref(
    input: np.ndarray,
    iota_workers: np.ndarray,
    gathered_buf: np.ndarray,
    gqa_group_size: int,
    replica_group: ReplicaGroup,
    layout: AttnQBatchShardLayout = AttnQBatchShardLayout.NBSd,
    rank_id_in: Optional[np.ndarray] = None,
) -> dict:
    """batch_shard: all_gather heads within gqa_group, then slice batch for this rank."""
    dtype = input.dtype
    pg = get_pg(replica_group)
    rank_id = get_rank()
    is_nbsd = layout == AttnQBatchShardLayout.NBSd

    t = _to_torch(input)
    gathered_list = [torch.zeros_like(t) for _ in range(gqa_group_size)]
    dist.all_gather(gathered_list, t, group=pg)
    gathered = torch.cat(gathered_list, dim=0)

    # B is at dim 1 for both layouts after all_gather on dim 0
    rank_in_group = rank_id % gqa_group_size
    batch = gathered.shape[1]
    batch_per_rank = batch // gqa_group_size
    b0 = rank_in_group * batch_per_rank
    b1 = b0 + batch_per_rank

    if is_nbsd:
        result = gathered[:, b0:b1, :, :]
    else:
        sliced = gathered[:, b0:b1, :, :]
        d = input.shape[0]
        result = sliced.reshape(gqa_group_size, d, batch_per_rank, sliced.shape[-1])

    return {"q_out": _to_numpy(result, dtype)}
