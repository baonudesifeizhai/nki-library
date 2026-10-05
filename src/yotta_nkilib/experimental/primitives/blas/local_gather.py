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

"""
NKI Primitives LocalGather Module.

Gathers elements from a source buffer based on indices using nisa.local_gather.
Primary use case: gathering expert affinities based on expert indices in MoE.
"""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from ..tile_stream import TileStream

# Hardware constants
P_MAX = 128
PARTITIONS_PER_GPSIMD_CORE = 16


class LocalGather(nl.NKIObject):
    """Gather elements from source buffer based on indices.

    Uses nisa.local_gather to collect elements from src based on index values.
    Handles index preparation including transpose for proper alignment.

    Primary use case: Gather expert affinities for MoE based on expert indices.

    Args:
        dst: Destination TileStream for gathered values.
             Shape should be (P_MAX, PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE)
             or equivalent flattened shape.
        src: Source TileStream containing data to gather from.
             Shape should be (P_MAX, E) where E is the gather dimension.
        index: TileStream containing indices for gathering.
               Shape should be (T, K) where T <= P_MAX and K <= 16.

    Example:
        # Gather expert affinities based on expert indices
        # affinities: [P_MAX, E] - affinity for each expert
        # expert_idx: [T, K] - top-K expert indices per token
        # gathered: [P_MAX, 16, 16] - gathered affinities

        affinities = TileStream((P_MAX, E), None, nl.bfloat16)
        expert_idx = TileStream((T, K), None, nl.int32)
        gathered = TileStream((P_MAX, 16, 16), None, nl.bfloat16)

        blas.LocalGather(dst=gathered, src=affinities, index=expert_idx).execute()
    """

    def __init__(
        self,
        dst: TileStream,
        src: TileStream,
        index: TileStream,
    ) -> None:
        self._dst = dst
        self._src = src
        self._index = index
        self._name = f"LocalGather(dst={dst.get_name()}, src={src.get_name()}, index={index.get_name()})"

        # Get dimensions
        src_shape = self._src.get_logical_shape()
        index_shape = self._index.get_logical_shape()

        kernel_assert(
            len(src_shape) == 2,
            f"LocalGather '{self._name}' expects src to be 2D (P, E), got shape {src_shape}",
        )
        kernel_assert(
            len(index_shape) == 2,
            f"LocalGather '{self._name}' expects index to be 2D (T, K), got shape {index_shape}",
        )

        self._T = index_shape[0]
        self._K = index_shape[1]
        self._E = src_shape[1]

        kernel_assert(
            self._K <= PARTITIONS_PER_GPSIMD_CORE,
            f"LocalGather '{self._name}' K={self._K} exceeds max {PARTITIONS_PER_GPSIMD_CORE}",
        )
        kernel_assert(
            self._E > 1,
            f"LocalGather '{self._name}' E={self._E} must be > 1 (local_gather requires src_buffer_size > 1)",
        )
        kernel_assert(
            self._T <= P_MAX,
            f"LocalGather '{self._name}' T={self._T} exceeds max {P_MAX}",
        )

    def execute(self):
        """Execute the local gather operation."""
        T, K, E = self._T, self._K, self._E

        # Get raw containers
        src_container = self._src.get_container()
        index_container = self._index.get_container()
        dst_container = self._dst.get_container()

        # Reshape containers to 2D for easier manipulation
        src_2d = src_container.reshape((P_MAX, E))
        index_2d = index_container.reshape((T, K))

        # Convert indices to uint16 for local_gather
        expert_idx_u16 = nl.ndarray(
            (P_MAX, PARTITIONS_PER_GPSIMD_CORE), dtype=nl.uint16, buffer=nl.sbuf, name="expert_idx_u16"
        )
        nisa.memset(dst=expert_idx_u16, value=0.0)
        nisa.tensor_copy(dst=expert_idx_u16[0:T, 0:K], src=index_2d[0:T, 0:K])

        # Prepare index values - need to transpose for local_gather alignment
        index_values = nl.ndarray(
            (P_MAX, PARTITIONS_PER_GPSIMD_CORE), dtype=nl.uint16, buffer=nl.sbuf, name="index_values"
        )
        nisa.memset(dst=index_values, value=0.0)

        if T <= PARTITIONS_PER_GPSIMD_CORE:
            # Optimized path for small token counts (T <= 16)
            expert_indices_trans = nl.ndarray(
                (PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE),
                dtype=nl.uint16,
                buffer=nl.sbuf,
                name="expert_indices_trans",
            )
            nisa.nc_transpose(
                dst=expert_indices_trans,
                data=expert_idx_u16[0:PARTITIONS_PER_GPSIMD_CORE, 0:PARTITIONS_PER_GPSIMD_CORE],
                engine=nisa.vector_engine,
            )
            nisa.tensor_copy(
                dst=index_values[0:PARTITIONS_PER_GPSIMD_CORE, 0:PARTITIONS_PER_GPSIMD_CORE],
                src=expert_indices_trans,
            )
        else:
            # Path for larger token counts (T > 16)
            active_channels = div_ceil(T, PARTITIONS_PER_GPSIMD_CORE)
            for channel_idx in range(active_channels):
                nisa.dma_transpose(
                    dst=index_values.ap(
                        pattern=[
                            [PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE],
                            [1, 1],
                            [1, 1],
                            [1, PARTITIONS_PER_GPSIMD_CORE],
                        ],
                        offset=channel_idx * PARTITIONS_PER_GPSIMD_CORE * PARTITIONS_PER_GPSIMD_CORE,
                    ),
                    src=expert_idx_u16.ap(
                        pattern=[
                            [PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE],
                            [1, 1],
                            [1, 1],
                            [1, PARTITIONS_PER_GPSIMD_CORE],
                        ],
                        offset=channel_idx * PARTITIONS_PER_GPSIMD_CORE * PARTITIONS_PER_GPSIMD_CORE,
                    ),
                )

        # Perform local gather
        ga_sb_fdim = PARTITIONS_PER_GPSIMD_CORE * PARTITIONS_PER_GPSIMD_CORE

        # Reshape dst to expected shape for local_gather
        gathered = dst_container.reshape((P_MAX, PARTITIONS_PER_GPSIMD_CORE, PARTITIONS_PER_GPSIMD_CORE))
        nisa.memset(dst=gathered, value=0.0)

        nisa.local_gather(
            dst=gathered.ap([[ga_sb_fdim, P_MAX], [1, ga_sb_fdim]]),
            src_buffer=src_2d,
            index=index_values[:, :],
            num_elem_per_idx=1,
            num_valid_indices=ga_sb_fdim,
        )

        self._dst.reset_cur_tile()

    def get_name(self) -> str:
        return self._name
