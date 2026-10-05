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

"""PyTorch reference implementation for build_all_to_all_v_metadata kernel."""

import torch

from ...core.utils.kernel_assert import kernel_assert


def build_all_to_all_v_metadata_torch_ref(
    expert_index: torch.Tensor,
    replica_group_size: int,
    E: int,
    recv_counts_known: bool = False,
    has_rdispls: bool = False,
) -> torch.Tensor:
    """Compute send_counts and send_displs metadata for all_to_all_v.

    Args:
        expert_index (torch.Tensor): [T, K] int32 tensor of expert indices per token.
        replica_group_size (int): Number of destination ranks.
        E (int): Total number of experts.
        recv_counts_known (bool): Not currently supported.
        has_rdispls (bool): Not currently supported.

    Returns:
        torch.Tensor: [n_rows, replica_group_size] int32 tensor where n_rows is 4 if has_rdispls else 3.
            Row 0: send counts, Row 1: send displacements, Row 2: recv counts (zeros), Row 3 (optional): recv displacements (zeros).
    """

    # Validation
    kernel_assert(
        not recv_counts_known, f"Torch ref does not yet support recv_counts_known=True, got {recv_counts_known=}"
    )
    kernel_assert(not has_rdispls, f"Torch ref does not yet support has_rdispls=True, got {has_rdispls=}")
    kernel_assert(
        E % replica_group_size == 0, f"Expected E divisible by replica_group_size, got {E=}, {replica_group_size=}"
    )

    # FIXME: change to torch.uint32 when nkilib testing infra supports it
    per_expert_counts = torch.bincount(expert_index.flatten().int(), minlength=E)
    send_counts = per_expert_counts.reshape(replica_group_size, -1).sum(dim=1).to(torch.int32)
    send_displs = torch.zeros(replica_group_size, dtype=torch.int32)
    send_displs[1:] = torch.cumsum(send_counts, 0)[:-1]
    recv_counts = torch.zeros(replica_group_size, dtype=torch.int32)
    rows = [send_counts, send_displs, recv_counts]
    return torch.stack(rows, dim=0)
