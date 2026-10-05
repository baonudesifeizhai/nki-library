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
Output projection utility functions for experimental kernels.
"""

from typing import Tuple

from nki.language import affine_range

from ...core.utils.allocator import BufferManager, sizeinbytes
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil

# Maximum free dimension tile size for Matmul operations
F_MAX = 512


def budget_weight_blocks(
    h_sharded: int,
    num_bxs_tiles: int,
    w_dtype,
    io_dtype,
    n_size: int,
    use_double_row: bool,
    out_in_sb: bool,
    sbm: BufferManager,
) -> Tuple[int, int, int]:
    """
    Calculate optimal h_block_size, number of weight buffer slots, and output interleave degree.

    This function determines:
    1. h_block_size: The optimal block size for the H dimension. Starts with 2048 (if divisible)
       and reduces if needed to ensure at least 2 weight buffers fit for double-buffering.
    2. num_w_h_blocks: How many weight buffer slots to allocate (for circular buffering).
    3. out_sb_interleave_degree: Interleave degree for output multi-buffering.

    The h_block_size is always a multiple of F_MAX (512) to avoid partial tile issues in Matmul.

    Args:
        h_sharded: Per-core hidden dimension (h_size // num_prgs).
        num_bxs_tiles: Number of bxs tiles in the tile loop.
        w_dtype: Weight data type.
        io_dtype: Data type of attention input and kernel output.
        n_size: Number of heads after packing.
        use_double_row: Whether double row optimization is enabled.
        out_in_sb: Whether output stays in SBUF instead of being written to HBM.
        sbm: Buffer manager.

    Returns:
        (h_block_size, num_w_h_blocks, out_sb_interleave_degree): Optimal h_block size,
        number of weight buffer slots, and interleave degree for output.
    """
    if sbm.is_auto_alloc():
        h_block_size = 2048 if h_sharded % 2048 == 0 else h_sharded
        num_h_blocks = h_sharded // h_block_size
        return h_block_size, num_h_blocks, 1

    # Calculate sizes
    w_dtype_size = sizeinbytes(w_dtype)
    io_dtype_size = sizeinbytes(io_dtype)
    out_sb_bytes = h_sharded * io_dtype_size
    free_space = sbm.get_free_space()

    n_heads_per_block = n_size // 2 if use_double_row else n_size
    w_elts_per_head_factor = 2 if use_double_row else 1  # multiplied by h_block_size later

    MAX_OUT_SB_INTERLEAVE = 4
    MIN_WEIGHT_BUFFERS = 2  # Target at least 2 for double-buffering

    max_interleave = 1 if out_in_sb else min(num_bxs_tiles, MAX_OUT_SB_INTERLEAVE)

    # Build candidate h_block_sizes: must be multiple of F_MAX
    # If h_sharded is not evenly divisible, the last block will be a partial (remainder) block
    if h_sharded % 2048 == 0:
        candidate_h_blocks = [2048]
    else:
        # Try multiples of F_MAX from largest to smallest
        candidate_h_blocks = []
        # Start from h_sharded rounded up to F_MAX (for single-block case) down to F_MAX
        h_block = div_ceil(h_sharded, F_MAX) * F_MAX
        while h_block >= F_MAX:
            candidate_h_blocks.append(h_block)
            h_block -= F_MAX

    best_h_block = F_MAX
    best_w_blocks = 0
    best_interleave = 1

    for h_block_size in candidate_h_blocks:
        if h_block_size > h_sharded:
            # Single block case: h_block_size covers entire h_sharded (with padding)
            h_block_size = h_sharded  # Use actual h_sharded for single block

        # Use ceiling division to account for remainder block
        num_h_blocks_per_prg = div_ceil(h_sharded, h_block_size)
        if num_h_blocks_per_prg == 0:
            continue

        # Calculate remainder size for partial last block
        remainder = h_sharded % h_block_size
        last_block_size = remainder if remainder > 0 else h_block_size

        w_elts_per_head = w_elts_per_head_factor * h_block_size
        bytes_per_h_block = n_heads_per_block * w_elts_per_head * w_dtype_size
        bytes_per_last_block = n_heads_per_block * w_elts_per_head_factor * last_block_size * w_dtype_size

        for interleave in affine_range(max_interleave, 0, -1):
            remaining = free_space - interleave * out_sb_bytes
            if remaining <= 0:
                continue

            # Calculate how many buffers fit, accounting for smaller last block if preloading all
            if num_h_blocks_per_prg == 1:
                # Single block case: only need space for the (possibly partial) block
                num_w = 1 if remaining >= bytes_per_last_block else 0
            else:
                # Multi-block: full-size buffers for circular buffering
                num_w = min(num_h_blocks_per_prg, remaining // bytes_per_h_block)

            if num_w < MIN_WEIGHT_BUFFERS:
                continue  # Need at least 2 for double-buffering

            # Prefer: larger h_block_size first, then more weight blocks, then larger interleave
            if (
                h_block_size > best_h_block
                or (h_block_size == best_h_block and num_w > best_w_blocks)
                or (h_block_size == best_h_block and num_w == best_w_blocks and interleave > best_interleave)
            ):
                best_h_block = h_block_size
                best_w_blocks = num_w
                best_interleave = interleave

    # Fallback: if no config with 2+ buffers found, try to fit at least 1
    if best_w_blocks == 0:
        for h_block_size in candidate_h_blocks:
            if h_block_size > h_sharded:
                continue
            num_h_blocks_per_prg = h_sharded // h_block_size
            if num_h_blocks_per_prg == 0:
                continue
            w_elts_per_head = w_elts_per_head_factor * h_block_size
            bytes_per_h_block = n_heads_per_block * w_elts_per_head * w_dtype_size
            remaining = free_space - out_sb_bytes  # interleave=1
            if remaining > 0 and bytes_per_h_block <= remaining:
                best_h_block = h_block_size
                best_w_blocks = min(num_h_blocks_per_prg, remaining // bytes_per_h_block)
                best_interleave = 1
                break

    kernel_assert(
        best_w_blocks > 0,
        f"Not enough SBUF space for output projection weights. h_sharded={h_sharded}, n_size={n_size}, free={free_space}",
    )

    return best_h_block, best_w_blocks, best_interleave
