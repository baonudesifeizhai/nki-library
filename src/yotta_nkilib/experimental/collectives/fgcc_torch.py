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
"""Torch reference for allgather_compute_matmul (FGCC) kernel."""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from .collectives_torch import _to_numpy, _to_torch, get_pg


def allgather_compute_matmul_torch_ref(
    lhs: np.ndarray, rhs: np.ndarray, tp_degree: int, num_groups: int, force_hbm_cc: bool = False
) -> dict:
    """fgcc: all_gather lhs rows, then matmul with local rhs shard."""
    dtype = lhs.dtype
    t_lhs = _to_torch(lhs)
    rg = ReplicaGroup([list(range(tp_degree))])
    gathered = [torch.zeros_like(t_lhs) for _ in range(tp_degree)]
    dist.all_gather(gathered, t_lhs, group=get_pg(rg))
    lhs_full = torch.cat(gathered, dim=0).to(torch.float32)
    rhs_f32 = _to_torch(rhs).to(torch.float32)
    result = lhs_full @ rhs_f32
    return {"result": _to_numpy(result, dtype)}
