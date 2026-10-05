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

"""PyTorch reference for ring_attention_unpermute kernel (striped -> contiguous)."""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from ..collectives.distributed_adapter import get_pg, get_rank


def ring_attention_unpermute_torch_ref(
    x: np.ndarray,
    replica_groups: tuple = None,
    num_workers: int = 1,
) -> dict:
    """PyTorch reference for striped-to-contiguous unpermute. Same signature as the kernel.

    Each rank holds tokens at striped positions [rank, rank+cp, rank+2*cp, ...].
    After unpermute, each rank holds a contiguous chunk of the global sequence:
    rank 0 gets [0, seqlen/cp), rank 1 gets [seqlen/cp, 2*seqlen/cp), etc.

    Implementation: all_gather all ranks' data, reconstruct global order by
    interleaving, then slice this rank's contiguous chunk.

    Args:
        x: [bs, seqlen_per_rank, d] — this rank's striped tokens
        replica_groups: replica group specification for collective communication
        num_workers: number of CP ranks in the ring

    Returns:
        dict with "out_x": [bs, seqlen_per_rank, d] — this rank's contiguous chunk
    """
    rank_id = get_rank()

    if isinstance(replica_groups, ReplicaGroup):
        groups = [list(g) for g in replica_groups._value]
    else:
        groups = [list(g) for g in replica_groups]

    my_group = next(g for g in groups if rank_id in g)
    my_worker_idx = my_group.index(rank_id)
    nw = len(my_group)

    rg = replica_groups if isinstance(replica_groups, ReplicaGroup) else ReplicaGroup(groups)
    try:
        pg = get_pg(rg)
    except KeyError:
        total_ranks = sum(len(g) for g in groups)
        default_rg = ReplicaGroup([list(range(total_ranks))])
        pg = get_pg(default_rg)

    use_subgroup_filter = pg.size() > nw

    # Gather all ranks' striped data
    x_flat = x.reshape(-1, x.shape[-1]).astype(np.float32)  # [bs * spr, d]
    t = torch.from_numpy(x_flat)
    gathered = [torch.zeros_like(t) for _ in range(pg.size())]
    dist.all_gather(gathered, t, group=pg)
    if use_subgroup_filter:
        gathered = [gathered[r] for r in my_group]

    bs = x.shape[0]
    spr = x.shape[1]
    d = x.shape[2]
    seqlen = spr * nw

    # Reconstruct global tensor by interleaving striped positions
    x_workers = [g.numpy().reshape(bs, spr, d) for g in gathered]
    x_global = np.empty((bs, seqlen, d), dtype=np.float32)
    for w in range(nw):
        x_global[:, w::nw, :] = x_workers[w]

    # Slice this rank's contiguous chunk
    out_x = x_global[:, my_worker_idx * spr : (my_worker_idx + 1) * spr, :]

    return {"out_x": out_x.astype(x.dtype)}
