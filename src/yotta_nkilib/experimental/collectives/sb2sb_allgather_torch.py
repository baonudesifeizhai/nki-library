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
"""Torch references for SBUF-to-SBUF all-gather kernels."""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from .collectives_torch import _to_numpy, _to_torch, get_pg


def allgather_sb2sb_torch_ref(inp: np.ndarray, replica_groups: ReplicaGroup, tp_degree: int) -> dict:
    """sb2sb all_gather: gather along k dimension (axis=1)."""
    dtype = inp.dtype
    t = _to_torch(inp)
    gathered = [torch.zeros_like(t) for _ in range(tp_degree)]
    dist.all_gather(gathered, t, group=get_pg(replica_groups))
    return {"out": _to_numpy(torch.cat(gathered, dim=1), dtype)}


def allgather_sb2sb_tiled_torch_ref(inp: np.ndarray, replica_groups: ReplicaGroup, tp_degree: int) -> dict:
    """sb2sb tiled all_gather: gather along k dimension (axis=1)."""
    dtype = inp.dtype
    t = _to_torch(inp)
    gathered = [torch.zeros_like(t) for _ in range(tp_degree)]
    dist.all_gather(gathered, t, group=get_pg(replica_groups))
    return {"result": _to_numpy(torch.cat(gathered, dim=1), dtype)}
