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
NKI Primitives Gather Module.

Gather operation using local_gather on GpSimd Engine.
dst[i] = src[index[i]] - gather elements from src based on indices.
"""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.logging import get_logger
from ..tile_stream import TileStream

# Hardware constants
PARTITIONS_PER_GPSIMD_CORE = 16  # GpSimd operates on 16 partitions


class Gather(nl.NKIObject):
    """Gather elements from src based on indices.

    Semantics: dst[p, f] = src[p, index[p, f]] for each partition p

    This wraps nisa.local_gather which operates on GpSimd Engine.

    Constraints:
        - Index must be convertible to uint16
        - Partition dim must be divisible by 16 (GPSIMD constraint)
        - src and dst must have same dtype (no casting in gather)
        - index free dim determines dst free dim

    Args:
        dst: Output TileStream, shape compatible with gathered result
        src: Source TileStream to gather from, shape [P, E] where E is num elements
        index: Index TileStream, shape [P, K] where K is num indices per partition

    Example:
        # Gather expert affinities based on expert indices
        # affinities[T, E] -> gathered[T, K] using expert_index[T, K]
        Gather(dst=gathered, src=affinities, index=expert_index).execute()
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
        self._name = f"Gather(src={src.get_name()}, index={index.get_name()}, dst={dst.get_name()})"
        self._logger = get_logger(self._name)

        # Validate shapes
        src_shape = self._src.get_logical_shape()
        index_shape = self._index.get_logical_shape()
        dst_shape = self._dst.get_logical_shape()

        kernel_assert(
            len(src_shape) == 2,
            f"blas.Gather '{self._name}' requires 2D src, got shape {src_shape}",
        )
        kernel_assert(
            len(index_shape) == 2,
            f"blas.Gather '{self._name}' requires 2D index, got shape {index_shape}",
        )
        kernel_assert(
            len(dst_shape) == 2,
            f"blas.Gather '{self._name}' requires 2D dst, got shape {dst_shape}",
        )

        src_P, src_E = src_shape
        index_P, index_K = index_shape
        dst_P, dst_K = dst_shape

        # Partition dims must match
        kernel_assert(
            src_P == index_P == dst_P,
            f"blas.Gather '{self._name}' requires matching partition dims, got src_P={src_P}, index_P={index_P}, dst_P={dst_P}",
        )

        # Partition dim must be divisible by 16 (GPSIMD constraint)
        kernel_assert(
            src_P % PARTITIONS_PER_GPSIMD_CORE == 0 or src_P <= PARTITIONS_PER_GPSIMD_CORE,
            f"blas.Gather '{self._name}' requires partition dim divisible by 16, got {src_P}",
        )

        # dst free dim must match index free dim
        kernel_assert(
            dst_K == index_K,
            f"blas.Gather '{self._name}' requires dst_K == index_K, got dst_K={dst_K}, index_K={index_K}",
        )

        # src must have > 1 element per partition (local_gather constraint)
        kernel_assert(
            src_E > 1,
            f"blas.Gather '{self._name}' requires src to have > 1 element per partition, got {src_E}",
        )

        # dtypes must match (no casting in gather)
        kernel_assert(
            self._src.get_dtype() == self._dst.get_dtype(),
            f"blas.Gather '{self._name}' requires src and dst to have same dtype, got src={self._src.get_dtype()}, dst={self._dst.get_dtype()}",
        )

        self._P = src_P
        self._E = src_E
        self._K = index_K

    def execute(self):
        """Execute the gather operation."""
        # Get containers
        src_container = self._src.get_container()
        index_container = self._index.get_container()
        dst_container = self._dst.get_container()

        # Convert index to uint16 if needed (local_gather requires uint16)
        if index_container.dtype != nl.uint16:
            index_u16 = nl.ndarray(index_container.shape, dtype=nl.uint16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=index_u16, src=index_container)
            index_container = index_u16

        # Determine number of valid indices
        num_valid_indices = self._P * self._K

        # Perform local_gather
        # local_gather semantics: dst[p, f] = src_buffer[p, index[p, f]]
        nisa.local_gather(
            dst=dst_container,
            src_buffer=src_container,
            index=index_container,
            num_elem_per_idx=1,
            num_valid_indices=num_valid_indices,
        )

    def get_name(self) -> str:
        return self._name
