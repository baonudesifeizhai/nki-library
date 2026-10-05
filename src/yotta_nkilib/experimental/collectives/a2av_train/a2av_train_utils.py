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
"""Shared helpers for the a2av_train dispatch / combine kernels.

This module exposes module-private helpers used by ``permute_a2av`` and
``unpermute_a2av`` to build the inputs ``ncc.all_to_all_v`` expects:

- :func:`_exclusive_cumsum_u32` computes the per-destination *scatter
  displacement* table (exclusive cumulative sum in tokens) and materialises
  it in both SBUF and HBM so the packer loop can look it up via indirect DMA.
- :func:`_write_a2av_v_metadata` writes the ``(4, EP)`` uint32 metadata
  tensor consumed by the collective (counts, send displacements, then two zero
  rows). The runtime writes receive counts to row 2 when
  ``recv_counts_known=False``. With ``has_rdispls=False``, row 3 remains zero
  and A2AV uses implicit equally spaced receive slots of
  ``dst.numel() / EP`` elements.
- :func:`_start_count_exchange` exchanges one count per peer before dispatch
  packing so the compiler can overlap metadata communication with local DMA.
- :func:`_masked_packed_row_indices` and :func:`_mask_row_indices` map inactive
  lanes to a one-past-end sentinel for indirect DMA with ``oob_mode.skip``.
- :func:`_write_packed_a2av_v_metadata` writes explicit send and receive counts
  and displacements for capacity-bounded packed A2AV.
- :func:`_validate_a2av_indices_counts` asserts the shared shape/dtype
  contract on ``send_indices`` and ``counts`` (plus ``T >= 1``, ``EP >= 2``).
  Called at entry from both ``permute_a2av`` and ``unpermute_a2av``.

All helpers are plain Python functions (not decorated with
``@nki.jit``) so they inline into the caller's tracing context.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ....core.utils.kernel_assert import kernel_assert

_UINT32_MAX = (1 << 32) - 1
_TRN3_LNC2_RANKS_PER_DEVICE = 4
_TRN3_A2AV_MIN_DEVICES = 2
_A2AV_METADATA_NUM_ROWS = 4
_A2AV_METADATA_ROW_HEIGHT = 1
_A2AV_SEND_COUNTS_ROW = 0
_A2AV_SEND_DISPLS_ROW = 1
_A2AV_RECV_COUNTS_ROW = 2
_A2AV_RECV_DISPLS_ROW = 3
_LNC2_CORE_IDS = (0, 1)
_LNC2_PEER_PROGRAM_ID = 1
_SENDRECV_PIPE_ID = 0


def _validate_trn3_a2av_group_size(EP: int) -> None:
    """Assert the rank-count portion of the Trn3 LNC=2 A2AV contract.

    Trn3 A2AV requires more than one participating device and complete
    four-rank membership for every LNC=2 device in a replica-group rank list.
    The caller must separately ensure that each rank list is sequential and
    aligned to device boundaries.

    Args:
        EP (int): Number of ranks in each A2AV replica-group rank list.

    Returns:
        None.
    """
    minimum_ranks = _TRN3_LNC2_RANKS_PER_DEVICE * _TRN3_A2AV_MIN_DEVICES
    kernel_assert(
        EP >= minimum_ranks,
        (
            "Trn3 LNC=2 A2AV requires more than one device, with "
            f"{_TRN3_LNC2_RANKS_PER_DEVICE} ranks per device; got EP={EP}, "
            f"minimum={minimum_ranks}"
        ),
    )
    kernel_assert(
        EP % _TRN3_LNC2_RANKS_PER_DEVICE == 0,
        (
            "Trn3 LNC=2 A2AV requires complete device membership, so EP must "
            f"be divisible by {_TRN3_LNC2_RANKS_PER_DEVICE}; got EP={EP}"
        ),
    )


def _validate_metadata_extent(
    row_capacity: int,
    hidden_size: int,
    tensor_name: str,
) -> None:
    """Assert that a row-major tensor extent fits A2AV uint32 metadata.

    Args:
        row_capacity (int): Static number of rows addressable by metadata.
        hidden_size (int): Elements per row.
        tensor_name (str): User-facing tensor name for the assertion message.

    Returns:
        None.
    """
    kernel_assert(
        row_capacity * hidden_size <= _UINT32_MAX,
        (
            f"{tensor_name} extent must fit uint32 A2AV element metadata, got "
            f"rows={row_capacity}, hidden_size={hidden_size}, "
            f"elements={row_capacity * hidden_size}, max={_UINT32_MAX}"
        ),
    )


def _exclusive_cumsum_u32(
    counts_hbm: nl.NkiTensor,
    EP: int,
    hbm_name: str = "a2av_v_sdispls_hbm",
) -> tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """Exclusive cumulative sum of ``counts_hbm`` in tokens.

    Produces the scatter-displacement table
    ``[0, counts[0], counts[0] + counts[1], ..., sum(counts[:EP - 1])]``,
    which is the per-destination row offset into the packed send buffer
    that ``ncc.all_to_all_v`` expects.

    Args:
        counts_hbm (nl.NkiTensor): [1, EP] int32/uint32 HBM tensor.
            Per-destination counts in tokens.
        EP (int): collective world size (static).

    Returns:
        counts_sb (nl.NkiTensor): [1, EP] uint32 SBUF. Raw counts (in tokens)
            loaded from ``counts_hbm``; returned so callers can reuse
            without a second DMA.
        sdispls_sb (nl.NkiTensor): [1, EP] uint32 SBUF. Exclusive cumulative
            sum (scatter displacements) in tokens.
        sdispls_hbm (nl.NkiTensor): [1, EP] uint32 shared_hbm. Same data as
            ``sdispls_sb``, materialised in HBM for indirect-DMA lookups
            from the packer loops.

    Notes:
        ``tensor_tensor_scan`` requires float32 operands, so counts are
        cast through float32 and back to uint32. This is safe for the
        count ranges used in practice (sums up to hundreds of thousands
        of tokens fit comfortably in fp32 mantissa).
    """
    counts_sb = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=counts_sb, src=counts_hbm)

    counts_f = nl.ndarray((1, EP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(counts_f, counts_sb)

    sdispls_f = nl.ndarray((1, EP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(sdispls_f, 0.0)
    init_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(init_sb, 0.0)
    # ones_sb acts as a multiplicative identity for the scan:
    #   acc = acc * 1 + counts[i]   → exclusive cumulative sum.
    ones_sb = nl.ndarray((1, EP - 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_sb, 1.0)
    nisa.tensor_tensor_scan(
        initial=init_sb,
        data0=ones_sb,
        op0=nl.multiply,
        data1=counts_f[0, : EP - 1],
        op1=nl.add,
        dst=sdispls_f[0, 1:],
    )

    sdispls_sb = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(sdispls_sb, sdispls_f)
    sdispls_hbm = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.shared_hbm, name=hbm_name)
    nisa.dma_copy(dst=sdispls_hbm, src=sdispls_sb)

    return counts_sb, sdispls_sb, sdispls_hbm


def _start_count_exchange(
    send_counts_hbm: nl.NkiTensor,
    EP: int,
    replica_group: ReplicaGroup,
) -> nl.NkiTensor:
    """Exchange one row count per peer before independent payload packing."""
    counts_send_hbm = nl.ndarray(
        (EP, 1),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
        name="packed_count_send",
    )
    counts_recv_hbm = nl.ndarray(
        (EP, 1),
        dtype=nl.uint32,
        buffer=nl.shared_hbm,
        name="packed_count_recv",
    )
    nisa.dma_copy(dst=counts_send_hbm, src=send_counts_hbm.reshape((EP, 1)))
    nisa.core_barrier(counts_send_hbm, (0, 1))
    ncc.all_to_all(
        srcs=[counts_send_hbm],
        dsts=[counts_recv_hbm],
        replica_group=replica_group,
        collective_dim=0,
    )
    nisa.core_barrier(counts_recv_hbm, (0, 1))
    return counts_recv_hbm


def _masked_packed_row_indices(
    counts_hbm_1d: nl.NkiTensor,
    displs_hbm_1d: nl.NkiTensor,
    peer_idx: int,
    tile_start: int,
    tile_size: int,
    row_limit: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Build packed row indices and a 0/1 validity mask for one peer/tile.

    Slots outside either the runtime peer count or the static row limit are
    mapped to the one-past-end ``row_limit`` sentinel.
    """
    slot_rows = nl.ndarray((tile_size, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(dst=slot_rows, pattern=[[0, 1]], offset=tile_start, channel_multiplier=1)

    count_tile = nl.ndarray((tile_size, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=count_tile,
        src=counts_hbm_1d.ap(pattern=[[0, tile_size], [1, 1]], offset=peer_idx),
    )

    valid_count = nl.ndarray((tile_size, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=valid_count, data1=slot_rows, data2=count_tile, op=nl.less)
    valid_range = nl.ndarray((tile_size, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=valid_range, data=slot_rows, op0=nl.less, operand0=row_limit)
    valid = nl.ndarray((tile_size, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=valid, data1=valid_count, data2=valid_range, op=nl.bitwise_and)

    displ_tile = nl.ndarray((tile_size, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=displ_tile,
        src=displs_hbm_1d.ap(pattern=[[0, tile_size], [1, 1]], offset=peer_idx),
    )
    packed_rows = nl.ndarray((tile_size, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=packed_rows, data1=slot_rows, data2=displ_tile, op=nl.add)
    _mask_row_indices(packed_rows, valid, row_limit)
    return packed_rows, valid


def _mask_row_indices(
    row_indices: nl.NkiTensor,
    valid: nl.NkiTensor,
    row_limit: int,
) -> None:
    """Map invalid indirect-DMA lanes to the one-past-end row sentinel."""
    nisa.tensor_tensor(
        dst=row_indices,
        data1=row_indices,
        data2=valid,
        op=nl.multiply,
    )
    invalid = nl.ndarray(row_indices.shape, dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=invalid,
        data=valid,
        op0=nl.subtract,
        operand0=1,
        reverse0=True,
    )
    nisa.tensor_scalar(
        dst=invalid,
        data=invalid,
        op0=nl.multiply,
        operand0=row_limit,
    )
    nisa.tensor_tensor(
        dst=row_indices,
        data1=row_indices,
        data2=invalid,
        op=nl.add,
    )


def _write_a2av_v_metadata(
    metadata_hbm: nl.NkiTensor,
    counts_sb: nl.NkiTensor,
    sdispls_sb: nl.NkiTensor,
    H: int,
    EP: int,
) -> None:
    """Write the four rows of the ``(4, EP)`` uint32 metadata tensor that
    ``ncc.all_to_all_v`` consumes.

        row 0: counts   * H    (send-side counts, in elements)
        row 1: sdispls  * H    (packed send displacements, in elements)
        row 2: zeros           (runtime fills recv_counts when
                                recv_counts_known=False)
        row 3: zeros           (implicit fixed-stride receive slots are used
                                when has_rdispls=False)

    Args:
        metadata_hbm (nl.NkiTensor): [4, EP] uint32 shared_hbm tensor,
            pre-allocated by the caller so the caller controls the
            tensor ``name``.
        counts_sb (nl.NkiTensor): [1, EP] uint32 SBUF. Counts in tokens.
        sdispls_sb (nl.NkiTensor): [1, EP] uint32 SBUF. Exclusive cumulative
            sum of ``counts_sb`` in tokens.
        H (int): hidden size (elements per token).
        EP (int): collective world size.
    """
    counts_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(data=counts_sb, op0=nl.multiply, operand0=H, dst=counts_el)

    sdispls_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(data=sdispls_sb, op0=nl.multiply, operand0=H, dst=sdispls_el)

    zeros_sb = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.memset(zeros_sb, 0)

    nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_SEND_COUNTS_ROW : _A2AV_SEND_COUNTS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=counts_el,
    )
    nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_SEND_DISPLS_ROW : _A2AV_SEND_DISPLS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=sdispls_el,
    )
    nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_RECV_COUNTS_ROW : _A2AV_RECV_COUNTS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=zeros_sb,
    )
    nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_RECV_DISPLS_ROW : _A2AV_RECV_DISPLS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=zeros_sb,
    )


def _write_packed_a2av_v_metadata(
    metadata_hbm: nl.NkiTensor,
    send_counts_sb: nl.NkiTensor,
    send_displs_sb: nl.NkiTensor,
    recv_counts_sb: nl.NkiTensor,
    recv_displs_sb: nl.NkiTensor,
    H: int,
    EP: int,
) -> None:
    """Write explicit packed A2AV metadata in element units.

    Args:
        metadata_hbm (nl.NkiTensor): [4, EP] uint32 shared-HBM destination.
        send_counts_sb (nl.NkiTensor): [1, EP] uint32 send counts in rows.
        send_displs_sb (nl.NkiTensor): [1, EP] uint32 send displacements in rows.
        recv_counts_sb (nl.NkiTensor): [1, EP] uint32 receive counts in rows.
        recv_displs_sb (nl.NkiTensor): [1, EP] uint32 receive displacements in
            rows.
        H (int): Elements per routed row.
        EP (int): Collective world size.

    Returns:
        None.
    """
    send_counts_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=send_counts_el,
        data=send_counts_sb,
        op0=nl.multiply,
        operand0=H,
    )
    send_displs_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=send_displs_el,
        data=send_displs_sb,
        op0=nl.multiply,
        operand0=H,
    )
    recv_counts_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=recv_counts_el,
        data=recv_counts_sb,
        op0=nl.multiply,
        operand0=H,
    )
    recv_displs_el = nl.ndarray((1, EP), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=recv_displs_el,
        data=recv_displs_sb,
        op0=nl.multiply,
        operand0=H,
    )
    nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_SEND_COUNTS_ROW : _A2AV_SEND_COUNTS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=send_counts_el,
    )
    nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_SEND_DISPLS_ROW : _A2AV_SEND_DISPLS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=send_displs_el,
    )
    nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_RECV_COUNTS_ROW : _A2AV_RECV_COUNTS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=recv_counts_el,
    )
    nisa.dma_copy(
        dst=metadata_hbm[
            _A2AV_RECV_DISPLS_ROW : _A2AV_RECV_DISPLS_ROW + _A2AV_METADATA_ROW_HEIGHT,
            :,
        ],
        src=recv_displs_el,
    )


def _validate_a2av_indices_counts(
    send_indices: nl.NkiTensor,
    counts: nl.NkiTensor,
    T: int,
    EP: int,
) -> None:
    """Assert the shape / dtype contract shared by ``permute_a2av`` and
    ``unpermute_a2av``.

    Checks:
      - ``send_indices`` is ``[T, EP]`` int32.
      - ``counts`` is ``[1, EP]`` int32/uint32.
      - ``EP`` and ``T`` are positive (``EP >= 2`` because the scan helper
        needs at least two slots).

    Args:
        send_indices (nl.NkiTensor): [T, EP] int32 HBM tensor. MoE routing
            indices shared across dispatch and combine.
        counts (nl.NkiTensor): [1, EP] int32/uint32 HBM tensor. ``send_counts``
            for dispatch, ``recv_counts`` for combine.
        T (int): number of local tokens (compile-time).
        EP (int): collective world size (compile-time).
    """
    kernel_assert(
        T >= 1,
        f"T must be >= 1, got T={T}",
    )
    kernel_assert(
        EP >= 2,
        f"EP must be >= 2 (exclusive-cumsum scan requires at least two slots), got EP={EP}",
    )
    kernel_assert(
        tuple(send_indices.shape) == (T, EP),
        f"send_indices must be (T={T}, EP={EP}), got {tuple(send_indices.shape)}",
    )
    kernel_assert(
        send_indices.dtype == nl.int32,
        f"send_indices must be int32, got {send_indices.dtype}",
    )
    kernel_assert(
        tuple(counts.shape) == (1, EP),
        f"counts must be (1, EP={EP}), got {tuple(counts.shape)}",
    )
    kernel_assert(
        counts.dtype in (nl.int32, nl.uint32),
        f"counts must be int32/uint32, got {counts.dtype}",
    )
