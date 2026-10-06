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

"""Shared utilities for foreach kernels."""

import nki.language as nl

from ...core.utils.kernel_helpers import div_ceil, get_program_sharding_info


def get_spmd_tiling_info(numel: int, tile_size: int):
    """
    Return SPMD partitioning and tiling info for the current core.

    Evenly partitions numel across SPMD cores, then computes aligned
    tile counts and tail remainder for the current core's slice.

    Args:
        numel (int): Total number of elements to partition.
        tile_size (int): Number of elements per tile (P_MAX * f_size).

    Returns:
        tuple: (core_base, core_n_aligned, num_tiles, tail, prog_id)
            - core_base: Starting element index for this core
            - core_n_aligned: Number of elements aligned to P_MAX
            - num_tiles: Number of full tiles to process
            - tail: Remaining elements after alignment (0 to P_MAX-1)
            - prog_id: Current program ID
    """
    P_MAX = nl.tile_size.pmax
    _, num_progs, prog_id = get_program_sharding_info()
    per_prog = div_ceil(numel, num_progs)
    core_base = prog_id * per_prog
    core_n = min(per_prog, numel - core_base)
    core_n_aligned = (core_n // P_MAX) * P_MAX
    num_tiles = div_ceil(core_n_aligned, tile_size)
    tail = core_n - core_n_aligned
    return core_base, core_n_aligned, num_tiles, tail, prog_id
