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
"""Capacity-bounded all-to-all-v combine and unpermute for MoE training."""

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
    _LNC2_PEER_PROGRAM_ID,
    _SENDRECV_PIPE_ID,
    _exclusive_cumsum_u32,
    _mask_row_indices,
    _masked_packed_row_indices,
    _validate_a2av_indices_counts,
    _validate_metadata_extent,
    _validate_trn3_a2av_group_size,
    _write_packed_a2av_v_metadata,
)

TILE_SIZE = 128


@nki.jit
def unpermute_a2av(
    output: nl.NkiTensor,
    send_indices: nl.NkiTensor,
    send_counts: nl.NkiTensor,
    recv_counts: nl.NkiTensor,
    replica_group: ReplicaGroup,
) -> nl.NkiTensor:
    """Return packed expert output to origin ranks and restore routed rows.

    Summary:
        Reverses packed dispatch for Trn3 LNC=2 by sending only active rows
        from ``[C, H]``, receiving exactly ``N`` routed rows with explicit
        displacements, and restoring routed-row order. Supports ``EP >= 8`` in
        complete four-rank device groups and count totals below ``2**24`` so
        float32 prefix scans remain exact.

    This reverses :func:`permute_a2av`. Dispatch receive counts and
    displacements become combine send metadata; dispatch send counts and
    displacements become explicit combine receive metadata. Active entries in
    ``send_indices`` must collectively be a permutation of the routed-row IDs:
    indirect DMA stores are not atomic duplicate-safe scatter-add operations.
    The dispatch routing table satisfies this invariant. The kernel requires
    an LNC=2 launch and a sequential
    replica-group rank list that contains all four ranks from each of at least
    two participating devices.

    Dimensions:
        N:  Number of original local routed rows.
        H:  Hidden dimension.
        EP: Number of expert-parallel ranks.
        C:  Static packed expert-output capacity.

    Args:
        output (nl.NkiTensor): [C, H]@HBM packed expert output.
        send_indices (nl.NkiTensor): [N, EP]@HBM, int32 dispatch routing table.
        send_counts (nl.NkiTensor): [1, EP]@HBM original dispatch send counts.
        recv_counts (nl.NkiTensor): [1, EP]@HBM original dispatch receive counts.
        replica_group (ReplicaGroup): EP replica group.

    Returns:
        result (nl.NkiTensor): [N, H]@HBM in original routed-row order.

    Notes:
        Active routing entries must assign every routed-row index exactly once.
        The row identity can represent any framework or any number of local
        experts; expert-major layout construction is outside this rank-level
        collective.
        Every count and displacement multiplied by ``H`` must fit uint32
        because A2AV metadata is measured in elements.

    Pseudocode:
        Prefix-sum dispatch send and receive counts.
        Reverse dispatch metadata for combine A2AV.
        Send only the active prefix of packed expert output.
        Receive source-major returned rows into a packed N-row buffer.
        Scatter each peer interval to its unique original routed-row indices.
        Add the private LNC partial results and return routed-row order.
    """
    C, H = output.shape
    N, EP = send_indices.shape
    dtype = output.dtype
    NUM_TILES = div_ceil(N, TILE_SIZE)

    _validate_a2av_indices_counts(send_indices, send_counts, N, EP)
    _validate_trn3_a2av_group_size(EP)
    kernel_assert(
        tuple(recv_counts.shape) == (1, EP),
        f"recv_counts must be (1, EP={EP}), got {tuple(recv_counts.shape)}",
    )
    kernel_assert(
        recv_counts.dtype in (nl.int32, nl.uint32),
        f"recv_counts must be int32/uint32, got {recv_counts.dtype}",
    )
    kernel_assert(C >= 1, f"output capacity must be >= 1, got C={C}")
    _validate_metadata_extent(C, H, "output")
    _validate_metadata_extent(N, H, "result")

    send_counts_sb, send_displs_sb, send_displs_hbm = _exclusive_cumsum_u32(
        send_counts,
        EP,
        hbm_name="packed_combine_recv_displs",
    )
    nisa.core_barrier(send_displs_hbm, _LNC2_CORE_IDS)
    recv_counts_sb, recv_displs_sb, _ = _exclusive_cumsum_u32(
        recv_counts,
        EP,
        hbm_name="packed_combine_send_displs",
    )

    metadata = nl.ndarray(
        (_A2AV_METADATA_NUM_ROWS, EP),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
        name="packed_combine_meta",
    )
    _write_packed_a2av_v_metadata(
        metadata,
        recv_counts_sb,
        recv_displs_sb,
        send_counts_sb,
        send_displs_sb,
        H,
        EP,
    )
    nisa.core_barrier(metadata, _LNC2_CORE_IDS)

    output_hbm = nl.ndarray((C, H), dtype=dtype, buffer=nl.shared_hbm, name="packed_combine_send")
    got_hbm = nl.ndarray((C, H), dtype=dtype, buffer=nl.shared_hbm, name="packed_combine_recv")
    nisa.dma_copy(dst=output_hbm, src=output)
    nisa.core_barrier(output_hbm, _LNC2_CORE_IDS)
    nisa.core_barrier(metadata, _LNC2_CORE_IDS)
    ncc.all_to_all_v(
        srcs=[output_hbm],
        dsts=[got_hbm],
        replica_group=replica_group,
        metadata_tensor=metadata,
        recv_counts_known=True,
        has_rdispls=True,
    )
    nisa.core_barrier(got_hbm, _LNC2_CORE_IDS)

    n_prgs = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    prg_id = nl.program_id(axis=0) if nl.program_ndim() != 0 else 0
    ep_per_shard_0 = div_ceil(EP, n_prgs)
    ep_per_shard_1 = EP // n_prgs
    ep_per_shard = ep_per_shard_0 if prg_id == 0 else ep_per_shard_1
    peer_start = 0 if prg_id == 0 else ep_per_shard_0

    partial = nl.ndarray((N, H), dtype=dtype, buffer=nl.private_hbm, name="packed_combine_partial")
    zero_tile = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(zero_tile, 0)
    for tile_idx in nl.affine_range(NUM_TILES):
        tile_start = tile_idx * TILE_SIZE
        tile_end = min(tile_start + TILE_SIZE, N)
        nisa.dma_copy(
            dst=partial[tile_start:tile_end, :],
            src=zero_tile[0 : tile_end - tile_start, :],
        )
    nisa.core_barrier(got_hbm, _LNC2_CORE_IDS)

    send_counts_hbm_1d = send_counts.reshape((EP,))
    send_displs_hbm_1d = send_displs_hbm.reshape((EP,))
    for peer_offset in nl.sequential_range(ep_per_shard_0):
        peer_idx = min(peer_start + peer_offset, EP - 1)
        for tile_idx in nl.sequential_range(NUM_TILES):
            tile_start = tile_idx * TILE_SIZE
            packed_rows, valid_slots = _masked_packed_row_indices(
                send_counts_hbm_1d,
                send_displs_hbm_1d,
                peer_idx,
                tile_start,
                TILE_SIZE,
                N,
            )
            if peer_offset >= ep_per_shard:
                nisa.memset(valid_slots, 0)
                nisa.memset(packed_rows, N)

            scatter_indices = nl.ndarray((TILE_SIZE, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.memset(scatter_indices, N)
            tile_end = min(tile_start + TILE_SIZE, N)
            if peer_offset < ep_per_shard:
                nisa.dma_copy(
                    dst=scatter_indices[0 : tile_end - tile_start, :],
                    src=send_indices[tile_start:tile_end, peer_idx : peer_idx + 1],
                )
            _mask_row_indices(scatter_indices, valid_slots, N)

            data_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=data_sb,
                src=got_hbm.ap(
                    pattern=[[C, TILE_SIZE], [1, H]],
                    offset=0,
                    vector_offset=packed_rows,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )
            current_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=current_sb,
                src=partial.ap(
                    pattern=[[N, TILE_SIZE], [1, H]],
                    offset=0,
                    vector_offset=scatter_indices,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )
            nisa.tensor_tensor(dst=current_sb, data1=current_sb, data2=data_sb, op=nl.add)
            nisa.dma_copy(
                dst=partial.ap(
                    pattern=[[N, TILE_SIZE], [1, H]],
                    offset=0,
                    vector_offset=scatter_indices,
                    indirect_dim=0,
                ),
                src=current_sb,
                oob_mode=oob_mode.skip,
            )
        nisa.core_barrier(got_hbm, _LNC2_CORE_IDS)

    nisa.core_barrier(got_hbm, _LNC2_CORE_IDS)
    result = nl.ndarray((N, H), dtype=dtype, buffer=nl.shared_hbm, name="packed_combine_result")
    for tile_idx in nl.affine_range(NUM_TILES):
        tile_start = tile_idx * TILE_SIZE
        tile_end = min(tile_start + TILE_SIZE, N)
        tile_size = tile_end - tile_start
        local_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=local_sb[0:tile_size, :], src=partial[tile_start:tile_end, :])
        if n_prgs > 1:
            peer_sb = nl.ndarray((TILE_SIZE, H), dtype=dtype, buffer=nl.sbuf)
            nisa.sendrecv(
                src=local_sb,
                dst=peer_sb,
                send_to_rank=_LNC2_PEER_PROGRAM_ID - prg_id,
                recv_from_rank=_LNC2_PEER_PROGRAM_ID - prg_id,
                pipe_id=_SENDRECV_PIPE_ID,
            )
            nisa.tensor_tensor(dst=local_sb, data1=local_sb, data2=peer_sb, op=nl.add)
        if prg_id == 0:
            nisa.dma_copy(dst=result[tile_start:tile_end, :], src=local_sb[0:tile_size, :])

    nisa.core_barrier(result, _LNC2_CORE_IDS)
    return result
