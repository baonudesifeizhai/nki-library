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
"""Torch reference for fine_grained_allgather kernel."""

import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup

from .collectives_torch import _to_numpy, _to_torch, get_pg


def fine_grained_allgather_torch_ref(
    lhs: np.ndarray, tp_degree: int, num_groups: int, force_hbm_cc: bool = False
) -> dict:
    """fg_allgather: all_gather rows from all ranks."""
    dtype = lhs.dtype
    t = _to_torch(lhs)
    rg = ReplicaGroup([list(range(tp_degree))])
    gathered = [torch.zeros_like(t) for _ in range(tp_degree)]
    dist.all_gather(gathered, t, group=get_pg(rg))
    return {"result": _to_numpy(torch.cat(gathered, dim=0), dtype)}
