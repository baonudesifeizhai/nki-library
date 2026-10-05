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

"""Recompute methods for MLP backward pass activation checkpointing.

When the forward pass does not save certain intermediates, the backward pass
must recompute them. Each function here is a self-contained recompute step
that can be plugged into the backward kernel as needed.

Dependency chain:
    gate_pre = hidden_states @ W_gate.T
    gate_act = SiLU(gate_pre)              (requires gate_pre)
    up       = hidden_states @ W_up.T
    intermediate = gate_act * up            (requires gate_act + up)
"""

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_helpers import div_ceil
from ...matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from ...matmul_mxfp8.matmul_mxfp8_generic_api import generic_matmul_mxfp8_api
from ...mxfp_utils.mxfp8_utils.common_dataclasses import TensorDescriptor
from ...mxfp_utils.mxfp8_utils.common_utils import get_active_sbm
from ...mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import INTERLEAVE_FACTOR
from ..common_utils import (
    NUM_LNC2_CORES,
    _allocate_spill_buffer,
    _build_matmul_params,
    _compute_load_tile_shape,
    apply_activation_clamp,
    get_tile_sizes,
)
from .config import ClampLimits


def recompute_gate_up_projection(
    hidden_td: TensorDescriptor,
    gate_up_td: TensorDescriptor,
    gate_pre_td: TensorDescriptor,
    up_td: TensorDescriptor,
    s_base_offset: int,
    dtype: type,
    fp8_x4_dtype: type,
    gate_config: MatmulMxfp8KernelConfig = None,
    up_config: MatmulMxfp8KernelConfig = None,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
    clamp_limits: ClampLimits = None,
) -> None:
    """Recompute gate_pre and up projections using generic_matmul_mxfp8_api.

    Computes:
        gate_pre = input_hidden_states @ W_gate.T    [S, I]
        up       = input_hidden_states @ W_up.T      [S, I]

    Uses paired generic_matmul_mxfp8_api calls with shared-stationary (lhs_sbuf_td)
    pattern: gate call loads hidden, up call reuses it from SBUF.

    Dimensions derived from tensor descriptors:
        hidden_td.logical_shape = (H, S)  — K=H, F=S
        gate_up_td.logical_shape = (H, 2I)  — K=H, F=2I
        S_local from hidden_td.sharded_logical_shape[1]

    Args:
        hidden_td (TensorDescriptor): [S, H], input hidden states (is_f_by_k=True).
        gate_up_td (TensorDescriptor): [2I, H], fused gate+up weight matrix (is_f_by_k=True).
        gate_pre_td (TensorDescriptor): [S, I], output buffer for gate pre-activation.
        up_td (TensorDescriptor): [S, I], output buffer for up projection.
        s_base_offset (int): Row offset for LNC sharding.
        dtype: Compute dtype (nl.bfloat16).
        fp8_x4_dtype: MXFP8 quantized dtype.
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_N (int): Number of N tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles to accumulate in PSUM.
        spill_reload (bool): Whether to spill quantized operands to HBM for reload.
        use_scale_packing (bool): Whether to pack scales for MXFP8 quantization.
        run_with_lnc2 (bool): Whether running with LNC2 (uses private_hbm for spill buffers).

    Returns:
        None. Results are written to gate_pre_td.data and up_td.data.
    """
    sbm = get_active_sbm()

    # Dynamic tile sizes: K=H, M=S_local, N=I (same mapping as fwd gate/up)
    H = hidden_td.logical_shape[0]
    I = gate_up_td.sharded_logical_shape[1]
    S_local = hidden_td.sharded_logical_shape[1]

    # Use gate_config for shared M/K loop structure (validated compatible with up_config)
    tile_m = gate_config.tile_m
    tile_n = gate_config.tile_n
    l_tile_k = gate_config.tile_k
    TILES_IN_BLOCK_M = gate_config.TILES_IN_BLOCK_M
    TILES_IN_BLOCK_N = gate_config.TILES_IN_BLOCK_N
    TILES_IN_BLOCK_K = gate_config.TILES_IN_BLOCK_K

    matmul_tile_k_physical = l_tile_k // INTERLEAVE_FACTOR
    rc_tiles = {
        'tile_m': tile_m,
        'tile_n': tile_n,
        'l_tile_k': l_tile_k,
        'matmul_tile_k_physical': matmul_tile_k_physical,
        'lhs_matmul_tile_physical': gate_config.lhs_matmul_tile_shape_physical,
        'rhs_matmul_tile_physical': gate_config.rhs_matmul_tile_shape_physical,
        'lhs_load_tile': gate_config.lhs_load_tile_shape,
        'rhs_load_tile': gate_config.rhs_load_tile_shape,
        'lhs_quantize_tile': gate_config.lhs_quantize_tile_shape,
        'rhs_quantize_tile': gate_config.rhs_quantize_tile_shape,
    }

    NUM_S_TILES_LOCAL = div_ceil(S_local, tile_m)

    BLOCK_N_GU = TILES_IN_BLOCK_N * tile_n

    NUM_K_TILES = div_ceil(H, l_tile_k)
    NUM_I_TILES = div_ceil(I, tile_n)
    NUM_M_BLOCKS = div_ceil(NUM_S_TILES_LOCAL, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_I_TILES, TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)

    # Compute load tile shapes from TD state
    lhs_load_tile_shape = _compute_load_tile_shape(hidden_td, rc_tiles, tile_m)
    rhs_load_tile_shape = _compute_load_tile_shape(gate_up_td, rc_tiles, tile_n)

    # Convert offsets to physical for pre-swizzled inputs
    s_base_physical = (
        s_base_offset * INTERLEAVE_FACTOR if (hidden_td.is_swizzled and not hidden_td.is_quantized) else s_base_offset
    )
    rhs_n_offset_up = I * INTERLEAVE_FACTOR if (gate_up_td.is_swizzled and not gate_up_td.is_quantized) else I

    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=rc_tiles,
    )

    # Spill/reload buffers (skip for pre-quantized inputs)
    hiddenq_td = None
    gate_wq_td = None
    up_wq_td = None
    if spill_reload:
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm

        if not hidden_td.is_quantized:
            hiddenq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )
        if not gate_up_td.is_quantized:
            gate_wq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )
            up_wq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )

    sbuf_step_p = TILES_IN_BLOCK_M * BLOCK_N_GU

    for m_block_idx in nl.sequential_range(NUM_M_BLOCKS):
        m_block_start = m_block_idx * TILES_IN_BLOCK_M

        for n_block_idx in range(NUM_N_BLOCKS):
            n_block_start = n_block_idx * TILES_IN_BLOCK_N

            gate_sbuf = sbm.alloc_stack(shape=(tile_m, TILES_IN_BLOCK_M * BLOCK_N_GU), dtype=nl.float32, buffer=nl.sbuf)
            up_sbuf = sbm.alloc_stack(shape=(tile_m, TILES_IN_BLOCK_M * BLOCK_N_GU), dtype=nl.float32, buffer=nl.sbuf)
            gate_output_td = TensorDescriptor(data=gate_sbuf)
            up_output_td = TensorDescriptor(data=up_sbuf)

            for k_block_idx in nl.sequential_range(NUM_K_BLOCKS):
                hidden_sbuf_td = TensorDescriptor(is_quantized=True)

                # Gate matmul — loads hidden, fills hidden_sbuf_td
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=hidden_td,
                    rhs_hbm_td=gate_up_td,
                    bd=bd,
                    output_td=gate_output_td,
                    block_idx_m=(m_block_idx, m_block_idx + 1),
                    block_idx_n=(n_block_idx, n_block_idx + 1),
                    block_idx_k=(k_block_idx, k_block_idx + 1),
                    lhs_sbuf_td=hidden_sbuf_td,
                    lhs_m_offset=s_base_physical,
                    rhs_n_offset=0,
                    TILES_IN_LOAD_M=gate_config.TILES_IN_LOAD_M,
                    TILES_IN_LOAD_N=gate_config.TILES_IN_LOAD_N,
                    lhs_matmul_tile_shape_physical=rc_tiles['lhs_matmul_tile_physical'],
                    rhs_matmul_tile_shape_physical=rc_tiles['rhs_matmul_tile_physical'],
                    lhs_load_tile_shape=lhs_load_tile_shape or rc_tiles['lhs_load_tile'],
                    rhs_load_tile_shape=rhs_load_tile_shape or rc_tiles['rhs_load_tile'],
                    lhs_quantize_tile_shape=rc_tiles['lhs_quantize_tile'],
                    rhs_quantize_tile_shape=rc_tiles['rhs_quantize_tile'],
                    spill_reload=spill_reload,
                    lhsq_td=hiddenq_td,
                    rhsq_td=gate_wq_td,
                    use_scale_packing=use_scale_packing,
                    initialize_accumulator=(k_block_idx == 0),
                )

                # Up matmul — reuses hidden from hidden_sbuf_td
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=hidden_td,
                    rhs_hbm_td=gate_up_td,
                    bd=bd,
                    output_td=up_output_td,
                    block_idx_m=(m_block_idx, m_block_idx + 1),
                    block_idx_n=(n_block_idx, n_block_idx + 1),
                    block_idx_k=(k_block_idx, k_block_idx + 1),
                    lhs_sbuf_td=hidden_sbuf_td,
                    lhs_m_offset=s_base_physical,
                    rhs_n_offset=rhs_n_offset_up,
                    TILES_IN_LOAD_M=up_config.TILES_IN_LOAD_M,
                    TILES_IN_LOAD_N=up_config.TILES_IN_LOAD_N,
                    lhs_matmul_tile_shape_physical=rc_tiles['lhs_matmul_tile_physical'],
                    rhs_matmul_tile_shape_physical=rc_tiles['rhs_matmul_tile_physical'],
                    lhs_load_tile_shape=lhs_load_tile_shape or rc_tiles['lhs_load_tile'],
                    rhs_load_tile_shape=rhs_load_tile_shape or rc_tiles['rhs_load_tile'],
                    lhs_quantize_tile_shape=rc_tiles['lhs_quantize_tile'],
                    rhs_quantize_tile_shape=rc_tiles['rhs_quantize_tile'],
                    spill_reload=spill_reload,
                    lhsq_td=hiddenq_td,
                    rhsq_td=up_wq_td,
                    use_scale_packing=use_scale_packing,
                    initialize_accumulator=(k_block_idx == 0),
                )

            # Store gate_pre and up to HBM with boundary clamping
            for tile_m_idx in range(TILES_IN_BLOCK_M):
                m_off = m_block_start * tile_m + tile_m_idx * tile_m
                if m_off >= S_local:
                    continue
                global_s = s_base_offset + m_off
                actual_m = min(tile_m, S_local - m_off)
                for tile_n_idx in range(TILES_IN_BLOCK_N):
                    i_off = (n_block_start + tile_n_idx) * tile_n
                    if i_off >= I:
                        continue
                    actual_n = min(tile_n, I - i_off)
                    sbuf_offset = tile_m_idx * BLOCK_N_GU + tile_n_idx * tile_n

                    gate_tile = gate_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)
                    up_tile = up_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)

                    if clamp_limits is not None:
                        gate_tile_buf = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=gate_tile_buf, src=gate_tile)
                        apply_activation_clamp(
                            gate_tile_buf,
                            clamp_limits.non_linear_clamp_upper_limit,
                            clamp_limits.non_linear_clamp_lower_limit,
                        )
                        up_tile_buf = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=up_tile_buf, src=up_tile)
                        apply_activation_clamp(
                            up_tile_buf, clamp_limits.linear_clamp_upper_limit, clamp_limits.linear_clamp_lower_limit
                        )
                        nisa.dma_copy(
                            dst=gate_pre_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n],
                            src=gate_tile_buf,
                        )
                        nisa.dma_copy(
                            dst=up_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n],
                            src=up_tile_buf,
                        )
                    else:
                        nisa.dma_copy(
                            dst=gate_pre_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n],
                            src=gate_tile,
                        )
                        nisa.dma_copy(
                            dst=up_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n],
                            src=up_tile,
                        )


def recompute_gate_act(
    gate_pre_td: TensorDescriptor,
    gate_act_td: TensorDescriptor,
    s_base_offset: int,
    dtype: type,
    run_with_lnc2: bool = True,
) -> None:
    """Recompute gate_act = SiLU(gate_pre) from checkpointed or recomputed gate_pre.

    Tile-by-tile element-wise SiLU activation. Used when gate_act was not
    checkpointed but gate_pre is available (either checkpointed or recomputed).

    Dimensions derived from tensor descriptors:
        gate_pre_td.logical_shape = (I, S)  — unswizzled BF16 [S, I]

    Args:
        gate_pre_td (TensorDescriptor): [S, I], gate pre-activation (input).
        gate_act_td (TensorDescriptor): [S, I], output buffer for SiLU(gate_pre).
        s_base_offset (int): Row offset for LNC sharding.
        dtype: Compute dtype (nl.bfloat16).
        run_with_lnc2 (bool): Whether running with LNC2.

    Returns:
        None. Results are written to gate_act_td.data.

    Pseudocode:
        for each s_tile in S tiles:
            for each i_tile in I tiles:
                gate_pre_tile = load(gate_pre_td)
                silu_out = SiLU(gate_pre_tile)
                store silu_out to gate_act_td
    """
    sbm = get_active_sbm()

    I = gate_pre_td.logical_shape[0]
    S = gate_pre_td.logical_shape[1]
    S_local = S // NUM_LNC2_CORES if run_with_lnc2 else S

    tiles = get_tile_sizes(I, S_local, I)  # use I as both K and N for element-wise tile sizing
    tile_m = tiles['tile_m']
    tile_n = tiles['tile_n']

    NUM_S_TILES_LOCAL = div_ceil(S_local, tile_m)
    NUM_I_TILES = div_ceil(I, tile_n)

    for s_tile in nl.sequential_range(NUM_S_TILES_LOCAL):
        global_s = s_base_offset + s_tile * tile_m
        actual_m = min(tile_m, S_local - s_tile * tile_m)
        for i_tile in nl.affine_range(NUM_I_TILES):
            i_off = i_tile * tile_n
            actual_n = min(tile_n, I - i_off)

            gate_pre_tile = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=gate_pre_tile, src=gate_pre_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n]
            )

            silu_out = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
            nisa.activation(dst=silu_out, op=nl.silu, data=gate_pre_tile)

            nisa.dma_copy(dst=gate_act_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n], src=silu_out)


def recompute_intermediate(
    gate_act_td: TensorDescriptor,
    up_td: TensorDescriptor,
    intermediate_td: TensorDescriptor,
    s_base_offset: int,
    dtype: type,
    run_with_lnc2: bool = True,
) -> None:
    """Recompute intermediate = gate_act * up from checkpointed or recomputed gate_act and up.

    Tile-by-tile element-wise multiply. Used when intermediate (the gated activation)
    was not checkpointed but gate_act and up are available.

    Dimensions derived from tensor descriptors:
        gate_act_td.logical_shape = (I, S)  — unswizzled BF16 [S, I]

    Args:
        gate_act_td (TensorDescriptor): [S, I], SiLU(gate_pre) (input).
        up_td (TensorDescriptor): [S, I], up projection (input).
        intermediate_td (TensorDescriptor): [S, I], output buffer for gate_act * up.
        s_base_offset (int): Row offset for LNC sharding.
        dtype: Compute dtype (nl.bfloat16).
        run_with_lnc2 (bool): Whether running with LNC2.

    Returns:
        None. Results are written to intermediate_td.data.

    Pseudocode:
        for each s_tile in S tiles:
            for each i_tile in I tiles:
                gate_act_tile = load(gate_act_td)
                up_tile = load(up_td)
                result = gate_act_tile * up_tile
                store result to intermediate_td
    """
    sbm = get_active_sbm()

    I = gate_act_td.logical_shape[0]
    S = gate_act_td.logical_shape[1]
    S_local = S // NUM_LNC2_CORES if run_with_lnc2 else S

    tiles = get_tile_sizes(I, S_local, I)  # use I as both K and N for element-wise tile sizing
    tile_m = tiles['tile_m']
    tile_n = tiles['tile_n']

    NUM_S_TILES_LOCAL = div_ceil(S_local, tile_m)
    NUM_I_TILES = div_ceil(I, tile_n)

    for s_tile in nl.sequential_range(NUM_S_TILES_LOCAL):
        global_s = s_base_offset + s_tile * tile_m
        actual_m = min(tile_m, S_local - s_tile * tile_m)
        for i_tile in nl.affine_range(NUM_I_TILES):
            i_off = i_tile * tile_n
            actual_n = min(tile_n, I - i_off)

            gate_act_tile = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=gate_act_tile, src=gate_act_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n]
            )

            up_tile = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=up_tile, src=up_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n])

            result = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=result, data1=gate_act_tile, data2=up_tile, op=nl.multiply)

            nisa.dma_copy(
                dst=intermediate_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n], src=result
            )
