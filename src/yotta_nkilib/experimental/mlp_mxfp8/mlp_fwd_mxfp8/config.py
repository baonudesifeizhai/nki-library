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

"""Configuration dataclasses and auto-tuning logic for MXFP8 MLP forward pass matmuls."""

from dataclasses import dataclass
from typing import Dict, Tuple

from nki.language import NKIObject

from ....core.utils.kernel_helpers import div_ceil
from ..common_utils import get_tile_sizes

# Base tile sizes
TILE_M = 128  # M tile size (stationary dimension)
TILE_K = 128  # K tile size after quantization (Q_TILE_K)
TILE_N = 512  # N tile size (moving dimension)
L_TILE_K = 512  # DGT load tile K size (must be 512 for quantize_mx)


@dataclass
class BlockConfig(NKIObject):
    """Blocking factors for matmul tuning.

    TILES_IN_BLOCK_M: Number of M tiles (128 each) to process together
    TILES_IN_BLOCK_N: Number of N tiles (512 each) to process together
    TILES_IN_BLOCK_K: Number of K tiles (512 DGT load -> 128 matmul) to accumulate in PSUM
    """

    TILES_IN_BLOCK_M: int = 8
    TILES_IN_BLOCK_N: int = 1
    TILES_IN_BLOCK_K: int = 8


@dataclass
class MatmulConfig(NKIObject):
    """Per-operation configuration for forward pass matmuls."""

    gate_up: BlockConfig = None
    down: BlockConfig = None

    def __post_init__(self):
        if self.gate_up == None:
            self.gate_up = BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=8)
        if self.down == None:
            self.down = BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8)


DEFAULT_MATMUL_CONFIG = MatmulConfig()


def _largest_divisor(n: int, candidates: list) -> int:
    """Return the largest value from candidates that divides n.

    Assumes candidates are sorted in decreasing order so the first
    match is the largest divisor.
    """
    for c in candidates:
        if n >= c and n % c == 0:
            return c
    return 1


def get_autotuned_config(seq_len: int, hidden_size: int, intermediate_size: int, lnc: int = 2) -> MatmulConfig:
    """Auto-select the best matmul configuration based on input shapes.

    K tiles are L_TILE_K=512 elements for DGT.

    Strategy: maximize K-blocking (fewer tensor_tensor adds) and N-blocking
    (better weight reuse), while keeping M-blocking at 8 for DGT efficiency.

    Gate/up: x[S,H] @ W_gate_up[2I,H].T
      M = S/TILE_M per core, K = H/L_TILE_K, N = I/TILE_N
    Down: intermediate[S,I] @ W_down[H,I].T
      M = S/TILE_M per core, K = I/L_TILE_K, N = H/TILE_N
    """
    s_per_core = seq_len // lnc if lnc > 1 else seq_len

    # Compute dynamic tile sizes for each phase
    gu_tiles = get_tile_sizes(hidden_size, s_per_core, intermediate_size)
    down_tiles_d = get_tile_sizes(intermediate_size, s_per_core, hidden_size)

    num_s_tiles = div_ceil(s_per_core, gu_tiles['tile_m'])

    # K dimension tile counts
    num_h_k_tiles = div_ceil(hidden_size, gu_tiles['l_tile_k'])
    num_i_k_tiles = div_ceil(intermediate_size, down_tiles_d['l_tile_k'])  # down K tiles

    # N dimension tile counts
    num_i_n_tiles = div_ceil(intermediate_size, gu_tiles['tile_n'])  # gate/up N tiles
    num_h_n_tiles = div_ceil(hidden_size, down_tiles_d['tile_n'])  # down N tiles

    # M blocking: 8 is optimal for DGT block loads
    m_blocking = _largest_divisor(num_s_tiles, [16, 8, 4, 2, 1])

    # K blocking: start at 1 for mxfp8 (DGT+quantize doubles SBUF pressure vs bf16).
    # Higher values can be explored but may cause SBUF spills.
    gate_up_k = 1
    down_k = 1

    # N blocking: maximize to reduce weight re-loads and tensor_tensor adds
    # Cover all N-tiles in one block when possible
    gate_up_n = _largest_divisor(num_i_n_tiles, [num_i_n_tiles, 8, 6, 4, 3, 2, 1])
    down_n = _largest_divisor(num_h_n_tiles, [num_h_n_tiles, 8, 6, 4, 3, 2, 1])

    return MatmulConfig(
        gate_up=BlockConfig(TILES_IN_BLOCK_M=m_blocking, TILES_IN_BLOCK_N=gate_up_n, TILES_IN_BLOCK_K=gate_up_k),
        down=BlockConfig(TILES_IN_BLOCK_M=m_blocking, TILES_IN_BLOCK_N=down_n, TILES_IN_BLOCK_K=down_k),
    )


"""
Shape-specific tuned configs for Qwen3 8B.

Key tuning levers:
- TILES_IN_BLOCK_K controls PSUM accumulation depth (fewer tensor_tensor adds)
- TILES_IN_BLOCK_N controls weight reuse across M-tiles
Higher N-blocking = weights loaded once per K-tile, reused across all M-tiles.

Gate/up: K = H/512 tiles, N = I/512 tiles
Down:    K = I/512 tiles, N = H/512 tiles
"""
SHAPE_TUNED_CONFIGS: Dict[Tuple[int, int, int], MatmulConfig] = {
    # TP1: seq=4096, H=4096, I=12288 — Gate/up: M=8, N=2, K=8; Down: M=8, N=4, K=8
    (4096, 4096, 12288): MatmulConfig(
        gate_up=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=8),
        down=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=8),
    ),
    # TP2: seq=4096, H=4096, I=6144 — Gate/up: M=8, N=2, K=8; Down: M=8, N=4, K=4
    (4096, 4096, 6144): MatmulConfig(
        gate_up=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=8),
        down=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=4, TILES_IN_BLOCK_K=4),
    ),
    # TP4: seq=4096, H=4096, I=3072 — Gate/up: M=8, N=2, K=8; Down: M=8, N=8, K=2
    (4096, 4096, 3072): MatmulConfig(
        gate_up=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=8),
        down=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=8, TILES_IN_BLOCK_K=2),
    ),
    # TP8: seq=4096, H=4096, I=1536 — Gate/up: M=8, N=1, K=8; Down: M=8, N=8, K=1
    (4096, 4096, 1536): MatmulConfig(
        gate_up=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=1, TILES_IN_BLOCK_K=8),
        down=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=8, TILES_IN_BLOCK_K=1),
    ),
    # TODO: TUNE THE CONFIG. TEMPORARY TO TEST FUNCTIONALITY. NOT PERFORMANT
    (4096, 5120, 3200): MatmulConfig(
        gate_up=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=1, TILES_IN_BLOCK_K=8),
        down=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=8, TILES_IN_BLOCK_K=1),
    ),
    # TODO: TUNE THE CONFIG. TEMPORARY TO TEST FUNCTIONALITY. NOT PERFORMANT
    (4096, 5120, 6400): MatmulConfig(
        gate_up=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=1, TILES_IN_BLOCK_K=8),
        down=BlockConfig(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=8, TILES_IN_BLOCK_K=1),
    ),
}


def get_config_for_shape(seq_len: int, hidden_size: int, intermediate_size: int, lnc: int = 2) -> MatmulConfig:
    """Get the best config for a specific shape."""
    key = (seq_len, hidden_size, intermediate_size)
    if key in SHAPE_TUNED_CONFIGS:
        return SHAPE_TUNED_CONFIGS[key]
    return get_autotuned_config(seq_len=seq_len, hidden_size=hidden_size, intermediate_size=intermediate_size, lnc=lnc)
