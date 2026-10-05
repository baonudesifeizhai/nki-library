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
"""Capacity-bounded all-to-all-v dispatch for MoE training."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup
from nki.isa.constants import oob_mode

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from .a2av_train_utils import (
    _A2AV_METADATA_NUM_ROWS,
    _LNC2_CORE_IDS,
    _exclusive_cumsum_u32,
    _mask_row_indices,
    _masked_packed_row_indices,
    _start_count_exchange,
    _validate_a2av_indices_counts,
    _validate_metadata_extent,
    _validate_trn3_a2av_group_size,
    _write_packed_a2av_v_metadata,
)

TILE_SIZE = 128


@nki.jit
def permute_a2av(
    hidden_states: nl.NkiTensor,
    send_indices: nl.NkiTensor,
    send_counts: nl.NkiTensor,
    recv_capacity: int,
    replica_group: ReplicaGroup,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Gather and exchange routed rows into one packed capacity buffer.

    Summary:
        Packs ``N`` local routed rows by destination rank, overlaps an
        ``EP``-element count exchange with local indirect DMA, and receives
        source-major rows into a static ``[C, H]`` buffer using explicit A2AV
        receive displacements. This experimental kernel is intended for Trn3
        LNC=2, ``EP >= 8`` in complete four-rank device groups, and row totals
        below ``2**24`` so the float32 prefix scan remains exact.

    ``send_counts`` must sum to ``N``. The kernel exchanges those counts as
    ``[EP, 1]``, computes source-ordered receive displacements, and uses explicit
    A2AV receive displacements. The caller must guarantee that the total rows
    received by every rank fit ``recv_capacity``. A2AV validates destination
    size against metadata at execution, but the kernel cannot resize the static
    output or recover from an undersized capacity. The kernel requires an
    LNC=2 launch and a sequential replica-group rank list that contains all
    four ranks from each of at least two participating devices.

    Dimensions:
        N:  number of local routed rows.
        H:  hidden dimension.
        EP: number of expert-parallel ranks.
        C:  static receive capacity (``recv_capacity``).

    Args:
        hidden_states (nl.NkiTensor): [N, H]@HBM routed input rows.
        send_indices (nl.NkiTensor): [N, EP]@HBM, int32. Source-row indices
            packed into each destination column. Only slots below the matching
            ``send_counts`` entry are read.
        send_counts (nl.NkiTensor): [1, EP]@HBM, int32/uint32 row counts.
        recv_capacity (int): Static number of rows in the packed receive output.
            For top-k routing with unique expert IDs and ``L`` local experts
            per destination rank, a dropless bound is
            ``global_token_count * min(top_k, L)``. This reduces to
            ``global_token_count`` when ``L == 1``. Duplicate expert IDs
            within one token invalidate this bound.
        replica_group (ReplicaGroup): EP replica group.

    Returns:
        recv_data (nl.NkiTensor): [C, H]@HBM. Source-major packed rows followed
            by deterministic zero padding.
        metadata (nl.NkiTensor): [4, EP]@HBM, uint32. Send counts/displacements
            and receive counts/displacements, all measured in elements.

    Notes:
        The API routes opaque rows and does not depend on a framework, model,
        or local-expert count. Consumers that require expert-major output must
        provide their own per-expert regrouping metadata.
        Every count and displacement multiplied by ``H`` must fit uint32
        because A2AV metadata is measured in elements.

    Pseudocode:
        Start all-to-all exchange of send_counts.
        Prefix-sum send_counts and pack hidden_states by destination rank.
        Prefix-sum the exchanged receive counts.
        Build explicit send and receive A2AV metadata.
        Zero the static receive capacity.
        Run A2AV directly into source-major packed receive intervals.
        Return the packed rows and metadata.
    """
    N, H = hidden_states.shape
    EP = send_indices.shape[1]
    dtype = hidden_states.dtype
    NUM_TILES = div_ceil(N, TILE_SIZE)

    _validate_a2av_indices_counts(send_indices, send_counts, N, EP)
    _validate_trn3_a2av_group_size(EP)
    kernel_assert(
        recv_capacity >= 1,
        f"recv_capacity must be >= 1, got {recv_capacity}",
    )
    _validate_metadata_extent(N, H, "hidden_states")
    _validate_metadata_extent(recv_capacity, H, "recv_data")

    recv_counts_exchange_hbm = _start_count_exchange(send_counts, EP, replica_group)
    send_counts_sb, send_displs_sb, send_displs_hbm = _exclusive_cumsum_u32(
        send_counts,
        EP,
        hbm_name="packed_send_displs",
    )
    nisa.core_barrier(send_displs_hbm, _LNC2_CORE_IDS)

    send_hbm = nl.ndarray((N, H), dtype=dtype, buffer=nl.shared_hbm, name="packed_dispatch_send")
    send_counts_hbm_1d = send_counts.reshape((EP,))
    send_displs_hbm_1d = send_displs_hbm.reshape((EP,))

    for ep_dst in nl.affine_range(EP):
        for tile_idx in nl.affine_range(NUM_TILES):
            tile_start = tile_idx * TILE_SIZE
            packed_rows, valid_slots = _masked_packed_row_indices(
                send_counts_hbm_1d,
                send_displs_hbm_1d,
                ep_dst,
                tile_start,
                TILE_SIZE,
                N,
            )

            gather_indices = nl.ndarray((TILE_SIZE, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.memset(gather_indices, N)
            tile_end = min(tile_start + TILE_SIZE, N)
            nisa.dma_copy(
                dst=gather_indices[0 : tile_end - tile_start, :],
                src=send_indices[tile_start:tile_end, ep_dst : ep_dst + 1],
            )
            _mask_row_indices(gather_indices, valid_slots, N)

            tile_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=tile_sb,
                src=hidden_states.ap(
                    pattern=[[N, TILE_SIZE], [1, H]],
                    offset=0,
                    vector_offset=gather_indices,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )
            nisa.dma_copy(
                dst=send_hbm.ap(
                    pattern=[[N, TILE_SIZE], [1, H]],
                    offset=0,
                    vector_offset=packed_rows,
                    indirect_dim=0,
                ),
                src=tile_sb,
                oob_mode=oob_mode.skip,
            )
        nisa.core_barrier(send_hbm, _LNC2_CORE_IDS)

    # The receive-count DMA is the first packing dependency and requires no extra cross-core barrier.
    recv_counts_hbm = recv_counts_exchange_hbm.reshape((1, EP))
    recv_counts_sb, recv_displs_sb, _ = _exclusive_cumsum_u32(
        recv_counts_hbm,
        EP,
        hbm_name="packed_recv_displs",
    )
    nisa.core_barrier(recv_counts_exchange_hbm, _LNC2_CORE_IDS)

    metadata = nl.ndarray(
        (_A2AV_METADATA_NUM_ROWS, EP),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
        name="packed_dispatch_meta",
    )
    _write_packed_a2av_v_metadata(
        metadata,
        send_counts_sb,
        send_displs_sb,
        recv_counts_sb,
        recv_displs_sb,
        H,
        EP,
    )
    nisa.core_barrier(metadata, _LNC2_CORE_IDS)

    recv_hbm = nl.ndarray(
        (recv_capacity, H),
        dtype=dtype,
        buffer=nl.shared_hbm,
        name="packed_dispatch_recv",
    )
    zero_tile = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(zero_tile, 0)
    for tile_idx in nl.affine_range(div_ceil(recv_capacity, TILE_SIZE)):
        tile_start = tile_idx * TILE_SIZE
        tile_end = min(tile_start + TILE_SIZE, recv_capacity)
        nisa.dma_copy(
            dst=recv_hbm[tile_start:tile_end, :],
            src=zero_tile[0 : tile_end - tile_start, :],
        )
    nisa.core_barrier(recv_hbm, _LNC2_CORE_IDS)

    nisa.core_barrier(send_hbm, _LNC2_CORE_IDS)
    nisa.core_barrier(metadata, _LNC2_CORE_IDS)
    ncc.all_to_all_v(
        srcs=[send_hbm],
        dsts=[recv_hbm],
        replica_group=replica_group,
        metadata_tensor=metadata,
        recv_counts_known=True,
        has_rdispls=True,
    )
    nisa.core_barrier(recv_hbm, _LNC2_CORE_IDS)

    recv_data = nl.ndarray((recv_capacity, H), dtype=dtype, buffer=nl.shared_hbm)
    metadata_out = nl.ndarray((_A2AV_METADATA_NUM_ROWS, EP), dtype=nl.uint32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=recv_data, src=recv_hbm)
    nisa.dma_copy(dst=metadata_out, src=metadata)
    nisa.core_barrier(recv_data, _LNC2_CORE_IDS)
    nisa.core_barrier(metadata_out, _LNC2_CORE_IDS)
    return recv_data, metadata_out
