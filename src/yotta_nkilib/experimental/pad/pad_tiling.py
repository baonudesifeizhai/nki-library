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

"""Tiling strategy for the generic pad kernel.

Determines how to partition the input tensor into tiles that:
1. Are contiguous in memory (for efficient DMA).
2. Meet the minimum DMA transfer size (≥ 4 KiB per partition).
3. Fit in SBUF together with the padded output tile.

The strategy also decides, per dimension, whether padding is applied in SBUF
during tile processing or deferred to a post-loop DMA pass.
"""

from typing import List, Tuple

import nki.language as nl

from ...core.utils.allocator import sizeinbytes
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from ...core.utils.logging import get_logger
from .pad_params import PadParams

_logger = get_logger("pad_kernel")

_MIN_DMA_BYTES = 4096


def _find_tile_dim(sizes: List[int], elem_size: int) -> Tuple[int, int]:
    """Find the dimension to tile on and the tile size.

    Scans inner-to-outer, accumulating the contiguous byte size. The first
    dimension whose cumulative size reaches ``_MIN_DMA_BYTES`` is chosen,
    with a tile size that produces chunks of at least ``_MIN_DMA_BYTES``.

    Returns:
        ``(dim, tile_size)``.
    """
    cumulative_size = elem_size
    for dim in range(len(sizes) - 1, -1, -1):
        cumulative_size *= sizes[dim]
        n_chunks = cumulative_size // _MIN_DMA_BYTES
        if n_chunks == 0:
            continue
        return dim, div_ceil(sizes[dim], n_chunks)

    return 0, sizes[0]


def _choose_sbuf_padding(
    tile_sizes: Tuple[int, int, int],
    n_tiles: Tuple[int, int, int],
    params: PadParams,
    mode: str,
    elem: int,
) -> Tuple[bool, bool, bool]:
    """Decide per-dim whether padding is applied in SBUF or deferred to DMA.

    Evaluates inner-to-outer (W → H → D), greedily assigning SBUF padding
    to inner dims first (they benefit most from avoiding per-element DMA).
    """
    sbuf_bytes = nl.tile_size.total_available_sbuf_size
    src_bytes = tile_sizes[0] * tile_sizes[1] * tile_sizes[2] * elem

    sbuf_pad = [False, False, False]
    padded = list(tile_sizes)

    for dim in (2, 1, 0):  # W, H, D
        # Reflect and circular map padding indices to arbitrary positions in
        # the source dimension (e.g., circular wraps to the opposite end).
        # When a dim is multi-tiled, the mapped position may fall outside the
        # current tile, so SBUF padding is only correct if the full dimension
        # is in one tile. Replicate and constant have no such dependency
        # (replicate uses the tile edge, constant uses a scalar).
        can_use_sbuf = (n_tiles[dim] == 1) or mode in ("replicate", "constant")
        if not can_use_sbuf:
            continue

        extra = params.total(dim) if n_tiles[dim] == 1 else max(params.before(dim), params.after(dim))

        trial = list(padded)
        trial[dim] += extra
        if src_bytes + trial[0] * trial[1] * trial[2] * elem <= sbuf_bytes:
            padded[dim] = trial[dim]
            sbuf_pad[dim] = True

    return (sbuf_pad[0], sbuf_pad[1], sbuf_pad[2])


def _validate_sbuf_fit(
    tile_sizes: Tuple[int, int, int],
    n_tiles: Tuple[int, int, int],
    sbuf_pad: Tuple[bool, bool, bool],
    params: PadParams,
    elem: int,
) -> None:
    """Assert that the worst-case tile (src + padded) fits in SBUF."""
    sbuf_bytes = nl.tile_size.total_available_sbuf_size
    src_bytes = tile_sizes[0] * tile_sizes[1] * tile_sizes[2] * elem

    padded = list(tile_sizes)
    for dim in range(3):
        if sbuf_pad[dim]:
            padded[dim] += params.total(dim) if n_tiles[dim] == 1 else max(params.before(dim), params.after(dim))

    padded_bytes = padded[0] * padded[1] * padded[2] * elem
    kernel_assert(
        src_bytes + padded_bytes <= sbuf_bytes,
        f"pad: src tile ({tile_sizes[0]}×{tile_sizes[1]}×{tile_sizes[2]}) + padded tile "
        f"({padded[0]}×{padded[1]}×{padded[2]}) = {src_bytes + padded_bytes}B "
        f"exceeds SBUF capacity of {sbuf_bytes}B",
    )


class PadTilingStrategy(nl.NKIObject):
    """Encapsulates all tiling decisions for the pad kernel."""

    def __init__(
        self,
        tile_sizes: Tuple[int, int, int],  # (d_tile, h_tile, w_tile) — elements per tile per dim
        n_tiles: Tuple[int, int, int],  # (n_d, n_h, n_w) — number of tiles per dim
        src_sizes: Tuple[int, int, int],  # (D, H, W) — spatial dimensions (after sharding)
        sbuf_pad: Tuple[bool, bool, bool],  # whether padding is applied in SBUF per dim
        params: PadParams,  # padding params for this shard
    ):
        self.tile_sizes = tile_sizes
        self.n_tiles = n_tiles
        self.src_sizes = src_sizes
        self.sbuf_pad = sbuf_pad
        self.params = params

    def needs_deferred_padding(self, dim: int) -> bool:
        """Whether *dim* requires a post-loop DMA pass to fill padding."""
        return not self.sbuf_pad[dim] and self.params.total(dim) > 0

    def tile_params(self, parent_params: PadParams, dim: int, tile_idx: int) -> PadParams:
        """Create PadParams for a specific tile along *dim*."""
        tile_size = self.tile_sizes[dim]
        src_size = self.src_sizes[dim]
        tile_start = tile_idx * tile_size
        tile_count = min(tile_size, src_size - tile_start)

        if not self.sbuf_pad[dim]:
            pad_before, pad_after = 0, 0
        else:
            pad_before = parent_params.before(dim) if tile_idx == 0 else 0
            pad_after = parent_params.after(dim) if tile_start + tile_count == src_size else 0

        new_pad = list(parent_params.pad)
        new_pad[dim] = (pad_before, pad_after)
        return PadParams(pad=tuple(new_pad), mode=parent_params.mode)


def compute_tiling_strategy(D: int, H: int, W: int, params: PadParams, mode: str, dtype) -> PadTilingStrategy:
    """Compute the tiling strategy for a pad operation.

    Partitions the spatial dimensions into tiles that are:
    - Contiguous in memory, so each DMA transfer moves a dense block.
    - Large enough (≥ 4 KiB per partition) to saturate DMA bandwidth.
    - Small enough to fit in SBUF (source tile + padded output tile).

    The tiled dimension is chosen inner-to-outer: the innermost dimension
    whose cumulative byte size reaches the DMA threshold is tiled. All
    dimensions outer to it are iterated one-at-a-time (tile=1) to preserve
    contiguity. Inner dimensions are kept at full size.
    """
    elem = sizeinbytes(dtype)
    tile_dim, tile_size = _find_tile_dim([D, H, W], elem)

    # Outer dims (before tile_dim) get tile=1 to keep DMA transfers contiguous.
    # The tiled dim gets the computed tile_size. Inner dims keep their full size.
    d_tile = 1 if tile_dim > 0 else tile_size
    h_tile = 1 if tile_dim > 1 else (tile_size if tile_dim == 1 else H)
    w_tile = tile_size if tile_dim == 2 else W
    n_d, n_h, n_w = div_ceil(D, d_tile), div_ceil(H, h_tile), div_ceil(W, w_tile)

    tile_sizes = (d_tile, h_tile, w_tile)
    n_tiles = (n_d, n_h, n_w)

    sbuf_pad = _choose_sbuf_padding(tile_sizes, n_tiles, params, mode, elem)
    _validate_sbuf_fit(tile_sizes, n_tiles, sbuf_pad, params, elem)

    _logger.info(f"Tiling: src=({D},{H},{W}), tile_sizes={tile_sizes}, n_tiles={n_tiles}, sbuf_pad={sbuf_pad}")

    return PadTilingStrategy(
        tile_sizes=tile_sizes,
        n_tiles=n_tiles,
        src_sizes=(D, H, W),
        sbuf_pad=sbuf_pad,
        params=params,
    )
