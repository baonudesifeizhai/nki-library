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
#
# Added by Yotta Labs: bwmm_shard_on_block from bwmm_shard_on_block.py with the
# outer block loop split into a static part and a dynamic tail (see below).

"""``bwmm_shard_on_block`` with a dynamic tail: the prefill MoE kernel sizes its
block loop for the worst-case routing, about twice the blocks a real batch
fills (T=2048, one EP rank of Qwen3-30B-A3B: 191 static blocks, ~104 used), and
each spare block still costs its full PE work. Here the first
``n_static_outer`` outer iterations stay static (fully pipelined); each later
one runs under a zero-or-one trip-count ``fori_loop`` keyed by
``outer_active``, so spare iterations cost nothing.

Only the main function is repeated here (body unchanged apart from the loop
split); the helpers come from bwmm_shard_on_block.py.
"""

from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl

from .bwmm_shard_on_block import (
    BLOCK_PARALLEL_FACTOR,
    GUP_LOAD_COALESCE_FACTOR,
    GUP_PROJ_DIM,
    _CHUNKED_I_TP_THRESHOLD,
    _I_TP_CHUNK_SIZE,
    DimensionSizes,
    _clip_gate_up_projections,
    bwmm_load_old_block,
    bwmm_output_initialization,
    compute_block_output,
    compute_same_weights_block_parallel_hbm,
    load_and_broadcast_down_bias,
    load_and_transpose_gup_bias,
    load_down_proj_weight,
    load_gate_up_proj_weights,
    reduce_outputs,
    shard_strat2blk_idx,
    shard_strat2new_blk_idx_offset,
)
from .moe_cte_utils import (
    DVE_CHANNELS_PER_BANK,
    N_PSUM_BANKS,
    PSUM_SIZE,
    TILE_SIZE,
    BlockShardStrategy,
    Configs,
    InputTensors,
    SkipMode,
    calculate_expert_affinities,
    compute_intermediate_states,
    div_ceil,
    load_block_expert,
)
from ...utils import common_types
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_program_sharding_info


@nki.jit
def bwmm_shard_on_block_dyn_tail(
    hidden_states: nl.NkiTensor,
    expert_affinities_masked: nl.NkiTensor,
    gate_up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    block_size: int,
    token_position_to_id: nl.NkiTensor,
    block_to_expert: nl.NkiTensor,
    gate_and_up_proj_bias: Optional[nl.NkiTensor] = None,
    down_proj_bias: Optional[nl.NkiTensor] = None,
    gate_up_proj_scale: Optional[nl.NkiTensor] = None,
    down_proj_scale: Optional[nl.NkiTensor] = None,
    down_activations: Optional[nl.NkiTensor] = None,
    activation_function: common_types.ActFnType = common_types.ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype: Any = nl.bfloat16,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: common_types.ExpertAffinityScaleMode = common_types.ExpertAffinityScaleMode.POST_SCALE,
    n_block_per_iter: int = 1,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.PING_PONG,
    outer_active: Optional[nl.NkiTensor] = None,
    n_static_outer: Optional[int] = None,
):
    """
    Blockwise matrix multiplication kernel for context-encoding MoE layers.

    This kernel implements blockwise matrix multiplication for mixture-of-experts (MoE) layers, processing tokens
    through expert-specific gate, up, and down projections. The computation combines static optimization benefits
    with dynamic early-exit capabilities by using a hybrid loop structure. Optimized for block-level sharding
    with PING_PONG strategy and supports FP8 quantization, multiple expert affinity scaling modes, and TopK > 1
    accumulation patterns. Optimized for block sizes 128-512 tokens, 8-64 experts, and sequence lengths up to 32K
    tokens. Best performance when I_TP >= 512 and batch size * sequence length <= 4096.

    Dimensions:
        T: Total number of input tokens
        H: Hidden dimension size
        B: Block size (tokens per block)
        E: Number of experts
        N: Total number of blocks (T / B)
        I_TP: Intermediate size divided by tensor parallelism degree

    Args:
        hidden_states (nl.NkiTensor): [T, H], Input token embeddings in HBM
        expert_affinities_masked (nl.NkiTensor): [(T+1)*E, 1], Expert routing weights for token assignments in HBM
        gate_up_proj_weight (nl.NkiTensor): [E, H, 2, I_TP], Combined gate and up projection weights in HBM
        down_proj_weight (nl.NkiTensor): [E, I_TP, H], Down projection weights in HBM
        block_size (int): Number of tokens processed per block
        token_position_to_id (nl.NkiTensor): [N*B], Mapping from block positions to token IDs in HBM
        block_to_expert (nl.NkiTensor): [N, 1], Expert assignment for each block in HBM
        gate_and_up_proj_bias (nl.NkiTensor, optional): [E, 2, I_TP], Bias terms for gate/up projections in HBM
        down_proj_bias (nl.NkiTensor, optional): [E, 1, H], Bias terms for down projection in HBM
        gate_up_proj_scale (nl.NkiTensor, optional): [E, 1, 2*I_TP], Dequantization scales for gate/up weights in HBM
        down_proj_scale (nl.NkiTensor, optional): [E, 1, H], Dequantization scales for down weights in HBM
        down_activations (nl.NkiTensor, optional): [N, B, H], Storage for intermediate activations in HBM
        activation_function (ActFnType): Activation function type (SiLU, GELU, etc.)
        skip_dma (SkipMode): DMA skip configuration for memory optimization
        compute_dtype (nki.dtype): Data type for internal computations (default: bfloat16)
        is_tensor_update_accumulating (bool): Enable accumulation for TopK > 1 scenarios
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Expert affinity application mode
        n_block_per_iter (int): Number of blocks processed per iteration
        gate_clamp_upper_limit (float, optional): Upper clamp limit for gate projections
        gate_clamp_lower_limit (float, optional): Lower clamp limit for gate projections
        up_clamp_upper_limit (float, optional): Upper clamp limit for up projections
        up_clamp_lower_limit (float, optional): Lower clamp limit for up projections
        block_sharding_strategy (BlockShardStrategy): Block distribution strategy across cores

    Returns:
        output (nl.NkiTensor): Expert-processed token representations in HBM. Shape depends on accumulation mode:
            - Single expert (is_tensor_update_accumulating=False): [T, H]
            - Multiple experts (is_tensor_update_accumulating=True): [T, 2, H] for cross-core accumulation

    Notes:
        - Currently only supports PING_PONG block sharding strategy
        - Static loop processes N-E blocks with compile-time optimizations
        - Dynamic loop handles remaining blocks with early-exit capability
        - Supports FP8 quantization with dequantization scales
        - Expert affinity scaling modes: PRE_SCALE, POST_SCALE, PRE_SCALE_DELAYED
        - Multi-shard execution requires num_shards == 2 for accumulation

    Pseudocode:
        # Initialize output tensor
        output = zeros(T, H)

        # Process blocks in parallel across shards
        for block_idx in shard_blocks:
            # Load expert weights for current block
            expert_id = block_to_expert[block_idx]
            gup_weights = load_weights(gate_up_proj_weight[expert_id])
            down_weights = load_weights(down_proj_weight[expert_id])

            # Load block tokens
            token_ids = token_position_to_id[block_idx * B : (block_idx + 1) * B]
            hidden = hidden_states[token_ids]  # [B, H]

            # Gate and Up projections
            gate_proj = hidden @ gup_weights[:, 0, :]  # [B, I_TP]
            up_proj = hidden @ gup_weights[:, 1, :]    # [B, I_TP]

            # Apply activation and element-wise multiply
            intermediate = activation_fn(gate_proj) * up_proj  # [B, I_TP]

            # Down projection
            block_output = intermediate @ down_weights  # [B, H]

            # Scale by expert affinity and accumulate
            affinities = expert_affinities_masked[token_ids, expert_id]
            output[token_ids] += block_output * affinities

        return output
    """
    kernel_assert(
        block_sharding_strategy == BlockShardStrategy.PING_PONG, "Currently only support PING-PONG sharding strategy"
    )
    kernel_assert(
        block_sharding_strategy == BlockShardStrategy.PING_PONG, "Currently only support PING-PONG sharding strategy"
    )
    kernel_assert(
        expert_affinities_scaling_mode != common_types.ExpertAffinityScaleMode.PRE_SCALE_DELAYED,
        "Currently ExpertAffinityScaleMode PRE_SCALE_DELAYED is not support ",
    )
    # Infer configurations from the input shapes
    T, H = hidden_states.shape
    B = block_size
    E, I_TP, _ = down_proj_weight.shape
    N = token_position_to_id.shape[0] // B
    NUM_TILES = B // TILE_SIZE
    shard_strat = block_sharding_strategy

    weights_dtype = compute_dtype
    _, num_shards, shard_id = get_program_sharding_info()
    dims = DimensionSizes(T=T, H=H, B=B, E=E, N=N, I_TP=I_TP)

    cfg = Configs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=False,
        linear_bias=(gate_and_up_proj_bias is not None and down_proj_bias is not None),
        activation_function=activation_function,
        is_quant=gate_up_proj_scale is not None and down_proj_scale is not None,
        fuse_gate_and_up_load=True,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
    )

    inps = InputTensors(
        hidden_states=hidden_states,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        down_proj_weight=down_proj_weight,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
    )

    NUM_STATIC_BLOCKS = N
    if is_tensor_update_accumulating:
        output = nl.ndarray((dims.T, 2, dims.H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
        bwmm_output_initialization(output, shard_id=shard_id)
    else:
        output = nl.ndarray((dims.T, dims.H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # Placeholder for FP8
    gup_scale = None
    down_scale = None
    n_blocks_per_shard = div_ceil(NUM_STATIC_BLOCKS, num_shards)
    # Actual number of valid blocks for this shard under interleaved distribution
    n_blocks_this_shard = div_ceil(max(NUM_STATIC_BLOCKS - shard_id, 0), num_shards)
    n_shard_tile_count = div_ceil(n_blocks_per_shard, BLOCK_PARALLEL_FACTOR)
    all_block_expert_broadcasted_per_shard = nl.ndarray((1, n_blocks_per_shard), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=all_block_expert_broadcasted_per_shard, value=E)
    nisa.dma_copy(
        dst=all_block_expert_broadcasted_per_shard[0:1, 0:n_blocks_this_shard],
        src=block_to_expert.reshape((block_to_expert.shape[0], 1)).ap(
            pattern=[[1, 1], [2, n_blocks_this_shard]], offset=shard_id
        ),
    )
    all_block_expert_real = nl.ndarray((1, n_blocks_per_shard), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=all_block_expert_real, src=all_block_expert_broadcasted_per_shard)

    if skip_dma.skip_weight:
        # Convert multi-dimensional ndarray to list of ndarrays
        gup_weights_load_dst_lst = []
        for h_tile_idx in range(dims.h_tile_count):
            inner_lst = []
            for h_subtile_idx in range(dims.h_subtile_count_gup):
                tmp = nl.ndarray(
                    (TILE_SIZE, GUP_LOAD_COALESCE_FACTOR, GUP_PROJ_DIM, I_TP), dtype=weights_dtype, buffer=nl.sbuf
                )
                nisa.memset(dst=tmp, value=0)
                inner_lst.append(tmp)
            gup_weights_load_dst_lst.append(inner_lst)

        down_weights_load_dst_lst = []
        for n_i in range(dims.gup_tile_count):
            down_weights_load_dst_lst.append(nl.ndarray((TILE_SIZE, H), dtype=weights_dtype, buffer=nl.sbuf))

        is_weight_same_as_prev_hbm = compute_same_weights_block_parallel_hbm(
            N, block_to_expert=block_to_expert, num_shards=num_shards, shard_id=shard_id, shard_strat=shard_strat
        )

        on_false = nl.ndarray((1, n_blocks_per_shard), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=on_false, value=E)
        need_skip = nl.ndarray((1, n_blocks_per_shard), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=need_skip,
            src=is_weight_same_as_prev_hbm.reshape((1, n_blocks_per_shard)).ap(
                pattern=[[1, 1], [1, n_blocks_per_shard]]
            ),
        )
        nisa.tensor_copy_predicated(
            dst=all_block_expert_broadcasted_per_shard,
            src=on_false,
            predicate=need_skip,
        )
    else:
        gup_weights_load_dst_lst = None
        down_weights_load_dst_lst = None

    block_hidden_states_lst = []
    for _ in range(BLOCK_PARALLEL_FACTOR):
        inner_lst = []
        for _ in range(NUM_TILES):
            tmp = nl.ndarray((TILE_SIZE, H), dtype=compute_dtype, buffer=nl.sbuf)
            nisa.memset(dst=tmp, value=0.0)
            inner_lst.append(tmp)
        block_hidden_states_lst.append(inner_lst)

    token_indices_lst = []
    for _ in range(BLOCK_PARALLEL_FACTOR):
        token_indices_lst.append(nl.ndarray((TILE_SIZE, NUM_TILES), dtype=nl.int32, buffer=nl.sbuf))

    # Outer iterations from n_static_outer on run only when outer_active says
    # one of their blocks holds tokens ([num_shards, n_shard_tile_count] int32,
    # 1 = run): a zero trip-count loop skips a spare iteration's PE work. The
    # flags must agree across shards; an iteration taken on one shard only
    # never completes.
    if n_static_outer is None or outer_active is None:
        n_static_outer = n_shard_tile_count
    n_static_outer = min(n_static_outer, n_shard_tile_count)
    if n_static_outer < n_shard_tile_count:
        outer_active_sb = nl.ndarray((1, n_shard_tile_count), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=outer_active_sb, src=outer_active[shard_id : shard_id + 1, 0:n_shard_tile_count])

    for outer_block_iter in range(n_shard_tile_count):
        if outer_block_iter < n_static_outer:
            _outer_iteration(
                outer_block_iter,
                B=B,
                E=E,
                H=H,
                I_TP=I_TP,
                N=N,
                NUM_TILES=NUM_TILES,
                activation_function=activation_function,
                all_block_expert_broadcasted_per_shard=all_block_expert_broadcasted_per_shard,
                all_block_expert_real=all_block_expert_real,
                block_hidden_states_lst=block_hidden_states_lst,
                block_to_expert=block_to_expert,
                cfg=cfg,
                compute_dtype=compute_dtype,
                dims=dims,
                down_proj_weight=down_proj_weight,
                down_scale=down_scale,
                down_weights_load_dst_lst=down_weights_load_dst_lst,
                expert_affinities_masked=expert_affinities_masked,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                gate_up_proj_weight=gate_up_proj_weight,
                gup_scale=gup_scale,
                gup_weights_load_dst_lst=gup_weights_load_dst_lst,
                hidden_states=hidden_states,
                inps=inps,
                is_tensor_update_accumulating=is_tensor_update_accumulating,
                n_blocks_per_shard=n_blocks_per_shard,
                output=output,
                shard_id=shard_id,
                shard_strat=shard_strat,
                skip_dma=skip_dma,
                token_indices_lst=token_indices_lst,
                token_position_to_id=token_position_to_id,
                weights_dtype=weights_dtype,
            )
        else:
            active_reg = nisa.register_alloc()
            nisa.register_load(active_reg, outer_active_sb[0:1, outer_block_iter : outer_block_iter + 1])

            def _conditional_iteration(_):
                _outer_iteration(
                    outer_block_iter,
                    B=B,
                    E=E,
                    H=H,
                    I_TP=I_TP,
                    N=N,
                    NUM_TILES=NUM_TILES,
                    activation_function=activation_function,
                    all_block_expert_broadcasted_per_shard=all_block_expert_broadcasted_per_shard,
                    all_block_expert_real=all_block_expert_real,
                    block_hidden_states_lst=block_hidden_states_lst,
                    block_to_expert=block_to_expert,
                    cfg=cfg,
                    compute_dtype=compute_dtype,
                    dims=dims,
                    down_proj_weight=down_proj_weight,
                    down_scale=down_scale,
                    down_weights_load_dst_lst=down_weights_load_dst_lst,
                    expert_affinities_masked=expert_affinities_masked,
                    expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                    gate_up_proj_weight=gate_up_proj_weight,
                    gup_scale=gup_scale,
                    gup_weights_load_dst_lst=gup_weights_load_dst_lst,
                    hidden_states=hidden_states,
                    inps=inps,
                    is_tensor_update_accumulating=is_tensor_update_accumulating,
                    n_blocks_per_shard=n_blocks_per_shard,
                    output=output,
                    shard_id=shard_id,
                    shard_strat=shard_strat,
                    skip_dma=skip_dma,
                    token_indices_lst=token_indices_lst,
                    token_position_to_id=token_position_to_id,
                    weights_dtype=weights_dtype,
                )

            nl.fori_loop(0, active_reg, _conditional_iteration)
    # END OF STATIC LOOP

    # final accumulation
    if is_tensor_update_accumulating and num_shards > 1:
        kernel_assert(num_shards == 2, "only support reducing data from 2 shards")
        reduce_tile_size = 128
        if skip_dma.skip_token:
            reduce_tiles = div_ceil(T, 128)
        else:
            reduce_tiles = div_ceil(T - 1, 128)

        nc0_tiles = reduce_tiles // num_shards
        nc1_tiles = reduce_tiles - nc0_tiles
        zeros_dummy = nl.ndarray((reduce_tile_size, 1, H), dtype=output.dtype, buffer=nl.sbuf)
        nisa.memset(dst=zeros_dummy, value=0.0)
        if num_shards == 1:
            nisa.core_barrier(output, (0))
        elif num_shards == 2:
            nisa.core_barrier(output, (0, 1))

        if shard_id == 0:
            reduce_outputs(output, zeros_dummy, nc0_tiles, reduce_tile_size, 0, H)

        if shard_id == 1:
            reduce_outputs(output, zeros_dummy, nc1_tiles, reduce_tile_size, nc0_tiles, H)

    return output


def _outer_iteration(
    outer_block_iter,
    B,
    E,
    H,
    I_TP,
    N,
    NUM_TILES,
    activation_function,
    all_block_expert_broadcasted_per_shard,
    all_block_expert_real,
    block_hidden_states_lst,
    block_to_expert,
    cfg,
    compute_dtype,
    dims,
    down_proj_weight,
    down_scale,
    down_weights_load_dst_lst,
    expert_affinities_masked,
    expert_affinities_scaling_mode,
    gate_up_proj_weight,
    gup_scale,
    gup_weights_load_dst_lst,
    hidden_states,
    inps,
    is_tensor_update_accumulating,
    n_blocks_per_shard,
    output,
    shard_id,
    shard_strat,
    skip_dma,
    token_indices_lst,
    token_position_to_id,
    weights_dtype,
):
    """One outer iteration of the static block loop (BLOCK_PARALLEL_FACTOR blocks
    per shard): the loop body of nkilib's bwmm_shard_on_block, unchanged."""
    block_psum_tiles = div_ceil(B, PSUM_SIZE)
    free_size = min(PSUM_SIZE, B)
    block_hidden_states_T_lst = []
    for k_ in range(BLOCK_PARALLEL_FACTOR):
        outer_lst = []
        for h_tile_idx in range(dims.h_tile_count):
            inner_lst = []
            for h_subtile_idx in range(dims.h_subtile_count):
                tmp = nl.ndarray((TILE_SIZE, block_psum_tiles, free_size), dtype=compute_dtype, buffer=nl.sbuf)
                nisa.memset(value=0, dst=tmp)
                inner_lst.append(tmp)

            outer_lst.append(inner_lst)
        block_hidden_states_T_lst.append(outer_lst)
    # parallel load and transpose input
    for inner_block_iter in range(BLOCK_PARALLEL_FACTOR):
        linear_idx = outer_block_iter * BLOCK_PARALLEL_FACTOR + inner_block_iter
        block_idx = 2 * linear_idx + shard_id

        if block_idx < N:
            shared_block_idx = shard_strat2blk_idx(shard_strat, outer_block_iter, inner_block_iter)
            local_block_idx = shared_block_idx + shard_strat2new_blk_idx_offset(
                shard_id, shard_strat, n_blocks_per_shard
            )

            offset = local_block_idx * B
            nisa.dma_copy(
                dst=token_indices_lst[inner_block_iter].ap(pattern=[[NUM_TILES, TILE_SIZE], [1, NUM_TILES]]),
                src=token_position_to_id.reshape((token_position_to_id.shape[0], 1)).ap(
                    pattern=[[1, TILE_SIZE], [TILE_SIZE, NUM_TILES]], offset=offset
                ),
            )

            if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.PRE_SCALE:
                v_expert = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                block_expert = load_block_expert(block_to_expert, local_block_idx)
                shuffle_mask = [0] * DVE_CHANNELS_PER_BANK
                for channel_idx in range(4):
                    nisa.nc_stream_shuffle(
                        dst=v_expert[
                            DVE_CHANNELS_PER_BANK * channel_idx : DVE_CHANNELS_PER_BANK * (channel_idx + 1), 0:1
                        ],
                        src=block_expert.ap(pattern=[[1, 1], [1, 1]], offset=0),
                        shuffle_mask=shuffle_mask,
                    )

                expert_affinity_f32_lst = []
                for _ in range(NUM_TILES):
                    expert_affinity_f32_lst.append(nl.ndarray((TILE_SIZE, 1), dtype=nl.float32, buffer=nl.sbuf))

                for token_tile_idx in range(NUM_TILES):
                    addr = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_scalar(
                        dst=addr,
                        data=token_indices_lst[inner_block_iter][0:TILE_SIZE, token_tile_idx],
                        op0=nl.multiply,
                        operand0=E,
                    )
                    addr_fin = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=addr_fin, data1=addr, data2=v_expert, op=nl.add)

                    if skip_dma.skip_token:
                        nisa.tensor_scalar(dst=addr_fin, data=addr_fin, op0=nl.minimum, operand0=-1)

                    expert_affinity_dtype = nl.ndarray((TILE_SIZE, 1), dtype=compute_dtype, buffer=nl.sbuf)
                    if skip_dma.skip_token:
                        nisa.memset(value=0, dst=expert_affinity_dtype)

                    # nl.load with indirect indexing -> nisa.dma_copy with .ap()
                    num_cols = expert_affinities_masked.shape[1]
                    addr_fin_reshaped = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=addr_fin_reshaped, src=addr_fin[0:TILE_SIZE, 0:1])

                    nisa.dma_copy(
                        dst=expert_affinity_dtype[0:TILE_SIZE, 0:1],
                        src=expert_affinities_masked.ap(
                            pattern=[[num_cols, TILE_SIZE], [1, 1]],
                            offset=0,
                            vector_offset=addr_fin_reshaped,
                            indirect_dim=0,
                        ),
                        oob_mode=nisa.oob_mode.skip if skip_dma.skip_token else nisa.oob_mode.error,
                    )

                    # Cast to float32
                    nisa.tensor_copy(
                        dst=expert_affinity_f32_lst[token_tile_idx][0:TILE_SIZE, 0:1],
                        src=expert_affinity_dtype[0:TILE_SIZE, 0:1],
                    )

            for token_tile_idx in range(NUM_TILES):
                block_token_mapping = token_indices_lst[inner_block_iter].ap(
                    pattern=[[NUM_TILES, TILE_SIZE], [1, 1]],
                    offset=token_tile_idx,
                )
                nisa.dma_copy(
                    dst=block_hidden_states_lst[inner_block_iter][token_tile_idx][0:TILE_SIZE, nl.ds(0, H)],
                    src=hidden_states.ap(
                        pattern=[[H, TILE_SIZE], [1, H]],
                        offset=0,
                        vector_offset=block_token_mapping,
                        indirect_dim=0,
                    ),
                    oob_mode=nisa.oob_mode.skip if skip_dma.skip_token else nisa.oob_mode.error,
                )

                if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.PRE_SCALE:
                    nisa.tensor_scalar(
                        dst=block_hidden_states_lst[inner_block_iter][token_tile_idx][0:TILE_SIZE, nl.ds(0, H)],
                        data=block_hidden_states_lst[inner_block_iter][token_tile_idx][0:TILE_SIZE, nl.ds(0, H)],
                        op0=nl.multiply,
                        operand0=expert_affinity_f32_lst[token_tile_idx][0:TILE_SIZE, 0],
                        engine=nisa.vector_engine,
                    )

            block_free_tiles = min(PSUM_SIZE // TILE_SIZE, B // TILE_SIZE)

            for token_tile_idx in range(block_psum_tiles):
                # ═══════════════════════════════════════════════════════════════════════
                # DEFINE 8 PSUM BANK BUFFERS: 2 sets × (N_PSUM_BANKS // 2) h_subtiles
                # ═══════════════════════════════════════════════════════════════════════
                num_subtiles_per_set = N_PSUM_BANKS // 2
                tmp_psum_set_0 = []
                tmp_psum_set_1 = []
                for _ in range(num_subtiles_per_set):
                    tmp_psum_set_0.append(nl.ndarray((TILE_SIZE, free_size), dtype=compute_dtype, buffer=nl.psum))
                    tmp_psum_set_1.append(nl.ndarray((TILE_SIZE, free_size), dtype=compute_dtype, buffer=nl.psum))

                # ═══════════════════════════════════════════════════════════════════════
                # PROCESS H_TILES IN PAIRS (fill 8 banks, then copy 8 banks)
                # ═══════════════════════════════════════════════════════════════════════
                num_pairs = (dims.h_tile_count + 1) // 2

                for h_tile_pair in range(num_pairs):
                    h_tile_0 = h_tile_pair * 2  # Even h_tile → Set 0
                    h_tile_1 = h_tile_pair * 2 + 1  # Odd h_tile → Set 1

                    # ───────────────────────────────────────────────────────────────────
                    # PHASE 1: FILL ALL 8 PSUM BANKS (all transposes first)
                    # ───────────────────────────────────────────────────────────────────

                    # Fill Set 0 with h_tile_0
                    if h_tile_0 < dims.h_tile_count:
                        for batch_tile_idx in range(block_free_tiles):
                            input_tile_idx = block_free_tiles * token_tile_idx + batch_tile_idx
                            for h_subtile_idx in range(num_subtiles_per_set):
                                nisa.nc_transpose(
                                    dst=tmp_psum_set_0[h_subtile_idx][
                                        0:TILE_SIZE, batch_tile_idx * TILE_SIZE : (batch_tile_idx + 1) * TILE_SIZE
                                    ],
                                    data=block_hidden_states_lst[inner_block_iter][input_tile_idx][
                                        0:TILE_SIZE,
                                        nl.ds(TILE_SIZE * h_subtile_idx + PSUM_SIZE * h_tile_0, TILE_SIZE),
                                    ],
                                )

                    # Fill Set 1 with h_tile_1
                    if h_tile_1 < dims.h_tile_count:
                        for batch_tile_idx in range(block_free_tiles):
                            input_tile_idx = block_free_tiles * token_tile_idx + batch_tile_idx
                            for h_subtile_idx in range(num_subtiles_per_set):
                                nisa.nc_transpose(
                                    dst=tmp_psum_set_1[h_subtile_idx][
                                        0:TILE_SIZE, batch_tile_idx * TILE_SIZE : (batch_tile_idx + 1) * TILE_SIZE
                                    ],
                                    data=block_hidden_states_lst[inner_block_iter][input_tile_idx][
                                        0:TILE_SIZE,
                                        nl.ds(TILE_SIZE * h_subtile_idx + PSUM_SIZE * h_tile_1, TILE_SIZE),
                                    ],
                                )

                    # ───────────────────────────────────────────────────────────────────
                    # PHASE 2: COPY ALL 8 PSUM BANKS (close results copied together!)
                    # ───────────────────────────────────────────────────────────────────

                    # Copy Set 0 (h_tile_0, all h_subtiles together)
                    if h_tile_0 < dims.h_tile_count:
                        for h_subtile_idx in range(num_subtiles_per_set):
                            nisa.tensor_copy(
                                dst=block_hidden_states_T_lst[inner_block_iter][h_tile_0][h_subtile_idx][
                                    0:TILE_SIZE, token_tile_idx, nl.ds(0, free_size)
                                ],
                                src=tmp_psum_set_0[h_subtile_idx],
                                engine=nisa.scalar_engine,
                            )

                    # Copy Set 1 (h_tile_1, all h_subtiles together)
                    if h_tile_1 < dims.h_tile_count:
                        for h_subtile_idx in range(num_subtiles_per_set):
                            nisa.tensor_copy(
                                dst=block_hidden_states_T_lst[inner_block_iter][h_tile_1][h_subtile_idx][
                                    0:TILE_SIZE, token_tile_idx, nl.ds(0, free_size)
                                ],
                                src=tmp_psum_set_1[h_subtile_idx],
                                engine=nisa.scalar_engine,
                            )

    # sequential load weights and compute
    for inner_block_iter in range(BLOCK_PARALLEL_FACTOR):
        linear_idx = outer_block_iter * BLOCK_PARALLEL_FACTOR + inner_block_iter
        block_idx = 2 * linear_idx + shard_id
        if block_idx < N:
            shared_block_idx = shard_strat2blk_idx(shard_strat, outer_block_iter, inner_block_iter)
            local_block_idx = shared_block_idx + shard_strat2new_blk_idx_offset(
                shard_id, shard_strat, n_blocks_per_shard
            )
            block_expert = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=block_expert, src=all_block_expert_broadcasted_per_shard[0:1, linear_idx : linear_idx + 1]
            )
            real_expert = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=real_expert, src=all_block_expert_real[0:1, linear_idx : linear_idx + 1])

            # load bias
            if cfg.linear_bias:
                gate_up_bias_T = load_and_transpose_gup_bias(inps, dims, cfg, real_expert, skip_dma)

            if block_idx != shard_id:
                block_old = (
                    bwmm_load_old_block(
                        output,
                        token_indices_lst[inner_block_iter],
                        NUM_TILES,
                        compute_dtype,
                        skip_dma,
                        shard_id=shard_id,
                    )
                    if is_tensor_update_accumulating
                    else None
                )
            else:
                block_old = []
                for alloc_idx in range(NUM_TILES):
                    tmp = nl.ndarray((TILE_SIZE, H), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.memset(tmp, value=0.0)
                    block_old.append(tmp)

            free_size = block_hidden_states_T_lst[0][0][0].shape[-1]

            # Use original tiling for small I_TP, chunked for large I_TP
            USE_CHUNKED_I_TP = I_TP > _CHUNKED_I_TP_THRESHOLD

            if not USE_CHUNKED_I_TP:
                gup_weights = load_gate_up_proj_weights(
                    gate_up_proj_weight, block_expert, weights_dtype, skip_dma, load_dst=gup_weights_load_dst_lst
                )

                dp_weights = load_down_proj_weight(
                    down_proj_weight, block_expert, weights_dtype, skip_dma, load_dst=down_weights_load_dst_lst
                )

                gate_and_up_proj_states_lst_psum = []
                gate_and_up_proj_states_lst_sbuf = []
                for _ in range(GUP_PROJ_DIM):
                    n_psum_lst = []
                    n_sbuf_lst = []
                    for _ in range(dims.n_psum_tile_count):
                        gup_psum_lst = []
                        gup_sbuf_lst = []
                        for _ in range(dims.gup_tile_count):
                            gup_psum_lst.append(
                                nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.psum)
                            )
                            gup_sbuf_lst.append(
                                nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.sbuf)
                            )
                        n_psum_lst.append(gup_psum_lst)
                        n_sbuf_lst.append(gup_sbuf_lst)
                    gate_and_up_proj_states_lst_psum.append(n_psum_lst)
                    gate_and_up_proj_states_lst_sbuf.append(n_sbuf_lst)

                for h_tile_idx in range(dims.h_tile_count):
                    for h_subtile_idx in range(dims.h_subtile_count_gup):
                        for op_idx in range(GUP_LOAD_COALESCE_FACTOR):
                            is_last_accumulation = (
                                h_tile_idx == dims.h_tile_count - 1
                                and h_subtile_idx == dims.h_subtile_count_gup - 1
                                and op_idx == 1
                            )

                            if not is_last_accumulation:
                                for i_tile_idx in range(dims.gup_tile_count):
                                    num_valid_k = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)
                                    for projection_idx in range(GUP_PROJ_DIM):
                                        for batch_tile_idx in range(dims.n_psum_tile_count):
                                            nisa.nc_matmul(
                                                dst=gate_and_up_proj_states_lst_psum[projection_idx][
                                                    batch_tile_idx
                                                ][i_tile_idx][0:num_valid_k, nl.ds(0, free_size)],
                                                stationary=gup_weights[h_tile_idx][h_subtile_idx][
                                                    nl.ds(0, TILE_SIZE),
                                                    op_idx,
                                                    projection_idx,
                                                    nl.ds(TILE_SIZE * i_tile_idx, num_valid_k),
                                                ],
                                                moving=block_hidden_states_T_lst[inner_block_iter][h_tile_idx][
                                                    h_subtile_idx * GUP_LOAD_COALESCE_FACTOR + op_idx
                                                ][0:TILE_SIZE, batch_tile_idx, nl.ds(0, free_size)],
                                            )
                            else:
                                for i_tile_idx in range(dims.gup_tile_count + 1):
                                    if i_tile_idx > 0:
                                        prev_tile = i_tile_idx - 1
                                        num_valid_k = min(TILE_SIZE, I_TP - TILE_SIZE * prev_tile)
                                        for projection_idx in range(GUP_PROJ_DIM):
                                            for batch_tile_idx in range(dims.n_psum_tile_count):
                                                nisa.tensor_copy(
                                                    dst=gate_and_up_proj_states_lst_sbuf[projection_idx][
                                                        batch_tile_idx
                                                    ][prev_tile][0:num_valid_k, nl.ds(0, free_size)],
                                                    src=gate_and_up_proj_states_lst_psum[projection_idx][
                                                        batch_tile_idx
                                                    ][prev_tile][0:num_valid_k, nl.ds(0, free_size)],
                                                    engine=nisa.scalar_engine,
                                                )
                                                if cfg.linear_bias:
                                                    nisa.tensor_tensor(
                                                        dst=gate_and_up_proj_states_lst_sbuf[projection_idx][
                                                            batch_tile_idx
                                                        ][prev_tile][0:TILE_SIZE, nl.ds(0, free_size)],
                                                        data1=gate_and_up_proj_states_lst_sbuf[projection_idx][
                                                            batch_tile_idx
                                                        ][prev_tile][0:TILE_SIZE, nl.ds(0, free_size)],
                                                        data2=gate_up_bias_T.ap(
                                                            pattern=[
                                                                [2 * dims.gup_tile_count, TILE_SIZE],
                                                                [0, free_size],
                                                            ],
                                                            offset=prev_tile * 2 + projection_idx,
                                                        ),
                                                        op=nl.add,
                                                    )

                                    if i_tile_idx < dims.gup_tile_count:
                                        num_valid_k = min(TILE_SIZE, I_TP - TILE_SIZE * i_tile_idx)
                                        for projection_idx in range(GUP_PROJ_DIM):
                                            for batch_tile_idx in range(dims.n_psum_tile_count):
                                                nisa.nc_matmul(
                                                    dst=gate_and_up_proj_states_lst_psum[projection_idx][
                                                        batch_tile_idx
                                                    ][i_tile_idx][0:num_valid_k, nl.ds(0, free_size)],
                                                    stationary=gup_weights[h_tile_idx][h_subtile_idx][
                                                        nl.ds(0, TILE_SIZE),
                                                        op_idx,
                                                        projection_idx,
                                                        nl.ds(TILE_SIZE * i_tile_idx, num_valid_k),
                                                    ],
                                                    moving=block_hidden_states_T_lst[inner_block_iter][h_tile_idx][
                                                        h_subtile_idx * 2 + op_idx
                                                    ][0:TILE_SIZE, batch_tile_idx, nl.ds(0, free_size)],
                                                )

                _clip_gate_up_projections(gate_and_up_proj_states_lst_sbuf, dims, cfg, free_size)

                # TODO: PRE_SCALE_DELAYED support is still under development
                expert_affinity_T_broadcasted = None

                intermediate_states = compute_intermediate_states(
                    gate_and_up_proj_states_lst_sbuf,
                    B,
                    I_TP,
                    compute_dtype,
                    activation_function=activation_function,
                    expert_affinity_T_broadcasted=expert_affinity_T_broadcasted,
                    gup_scale=gup_scale,
                )

                expert_affinity_f32 = (
                    calculate_expert_affinities(
                        expert_affinities_masked,
                        token_indices_lst[inner_block_iter],
                        real_expert,
                        E,
                        NUM_TILES,
                        compute_dtype,
                        skip_dma,
                    )
                    if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.POST_SCALE
                    else None
                )
                down_activations = None

                if cfg.linear_bias:
                    down_bias_broadcasted = load_and_broadcast_down_bias(inps, dims, cfg, real_expert, skip_dma)
                else:
                    down_bias_broadcasted = None

                block_new_lst = compute_block_output(
                    intermediate_states,
                    dp_weights,
                    expert_affinity_f32,
                    block_old,
                    down_activations,
                    local_block_idx,
                    H,
                    I_TP,
                    NUM_TILES,
                    output_dtype=output.dtype,
                    down_bias_broadcasted=down_bias_broadcasted,
                    is_tensor_update_accumulating=is_tensor_update_accumulating,
                    down_scale=down_scale,
                )

            else:
                # ═══════════════════════════════════════════════════════════
                # CHUNKED PATH: load weights per I_TP chunk, fused down_proj
                # ═══════════════════════════════════════════════════════════
                I_TP_CHUNK = min(_I_TP_CHUNK_SIZE, I_TP)
                I_TP_CHUNK_TILES = div_ceil(I_TP_CHUNK, TILE_SIZE)
                N_I_CHUNKS = div_ceil(I_TP, I_TP_CHUNK)

                gate_and_up_proj_states_lst_sbuf = []
                for _ in range(GUP_PROJ_DIM):
                    n_sbuf_lst = []
                    for _ in range(dims.n_psum_tile_count):
                        gup_sbuf_lst = []
                        for _ in range(dims.gup_tile_count):
                            gup_sbuf_lst.append(
                                nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.sbuf)
                            )
                        n_sbuf_lst.append(gup_sbuf_lst)
                    gate_and_up_proj_states_lst_sbuf.append(n_sbuf_lst)

                for projection_idx in range(GUP_PROJ_DIM):
                    for i_chunk_idx in range(N_I_CHUNKS):
                        i_chunk_start = i_chunk_idx * I_TP_CHUNK
                        i_chunk_end = min(i_chunk_start + I_TP_CHUNK, I_TP)
                        chunk_i_tp = i_chunk_end - i_chunk_start
                        chunk_n_tiles = div_ceil(chunk_i_tp, TILE_SIZE)

                        # Load weights for this I_TP chunk: [h_outer][h_inner] each (TILE_SIZE, chunk_i_tp)
                        chunk_weights = []
                        for h_tile_idx in range(dims.h_tile_count):
                            h_inner_lst = []
                            for h_subtile_idx in range(dims.h_subtile_count_gup):
                                for op_idx in range(GUP_LOAD_COALESCE_FACTOR):
                                    h_inner_lst.append(
                                        nl.ndarray(
                                            (TILE_SIZE, chunk_i_tp),
                                            dtype=gate_up_proj_weight.dtype,
                                            buffer=nl.sbuf,
                                        )
                                    )
                            chunk_weights.append(h_inner_lst)

                        _, H_w, _, I_TP_w = gate_up_proj_weight.shape
                        for h_tile_idx in range(dims.h_tile_count):
                            for h_subtile_idx in range(dims.h_subtile_count_gup):
                                for op_idx in range(GUP_LOAD_COALESCE_FACTOR):
                                    h_offset = (
                                        PSUM_SIZE * h_tile_idx
                                        + GUP_LOAD_COALESCE_FACTOR * TILE_SIZE * h_subtile_idx
                                        + TILE_SIZE * op_idx
                                    )
                                    num_h = min(TILE_SIZE, H - h_offset)
                                    w_idx = h_subtile_idx * GUP_LOAD_COALESCE_FACTOR + op_idx

                                    if num_h < TILE_SIZE:
                                        nisa.memset(dst=chunk_weights[h_tile_idx][w_idx], value=0.0)

                                    # gate_up_proj_weight: [E, H, 2, I_TP]
                                    # Access: [expert, h_offset:h_offset+num_h, projection_idx, i_chunk_start:i_chunk_end]
                                    offset = h_offset * (2 * I_TP_w) + projection_idx * I_TP_w + i_chunk_start

                                    nisa.dma_copy(
                                        dst=chunk_weights[h_tile_idx][w_idx][0:num_h, 0:chunk_i_tp],
                                        src=gate_up_proj_weight.ap(
                                            pattern=[
                                                [2 * I_TP_w, num_h],
                                                [1, chunk_i_tp],
                                            ],
                                            offset=offset,
                                            scalar_offset=block_expert,
                                        ),
                                        oob_mode=nisa.oob_mode.skip
                                        if skip_dma.skip_weight
                                        else nisa.oob_mode.error,
                                    )

                        # Allocate psum for this chunk's i_tiles
                        chunk_psum = []
                        for _ in range(dims.n_psum_tile_count):
                            tile_lst = []
                            for _ in range(chunk_n_tiles):
                                tile_lst.append(
                                    nl.ndarray((TILE_SIZE, free_size), dtype=nl.float32, buffer=nl.psum)
                                )
                            chunk_psum.append(tile_lst)

                        # i_tile outer, H inner — psum accumulates across H
                        for local_i_idx in range(chunk_n_tiles):
                            i_start = TILE_SIZE * local_i_idx
                            num_i = min(TILE_SIZE, chunk_i_tp - i_start)

                            for h_tile_idx in range(dims.h_tile_count):
                                for h_subtile_idx in range(dims.h_subtile_count_gup):
                                    for op_idx in range(GUP_LOAD_COALESCE_FACTOR):
                                        w_idx = h_subtile_idx * GUP_LOAD_COALESCE_FACTOR + op_idx
                                        for batch_tile_idx in range(dims.n_psum_tile_count):
                                            nisa.nc_matmul(
                                                dst=chunk_psum[batch_tile_idx][local_i_idx][
                                                    0:num_i, nl.ds(0, free_size)
                                                ],
                                                stationary=chunk_weights[h_tile_idx][w_idx][
                                                    nl.ds(0, TILE_SIZE),
                                                    nl.ds(i_start, num_i),
                                                ],
                                                moving=block_hidden_states_T_lst[inner_block_iter][h_tile_idx][
                                                    h_subtile_idx * GUP_LOAD_COALESCE_FACTOR + op_idx
                                                ][0:TILE_SIZE, batch_tile_idx, nl.ds(0, free_size)],
                                            )

                        # Copy psum → sbuf (+ optional bias)
                        for local_i_idx in range(chunk_n_tiles):
                            global_i_idx = i_chunk_idx * I_TP_CHUNK_TILES + local_i_idx
                            num_i = min(TILE_SIZE, chunk_i_tp - TILE_SIZE * local_i_idx)
                            for batch_tile_idx in range(dims.n_psum_tile_count):
                                if cfg.linear_bias:
                                    nisa.tensor_tensor(
                                        dst=gate_and_up_proj_states_lst_sbuf[projection_idx][batch_tile_idx][
                                            global_i_idx
                                        ][0:num_i, nl.ds(0, free_size)],
                                        data1=chunk_psum[batch_tile_idx][local_i_idx][0:num_i, nl.ds(0, free_size)],
                                        data2=gate_up_bias_T.ap(
                                            pattern=[
                                                [2 * dims.gup_tile_count, num_i],
                                                [0, free_size],
                                            ],
                                            offset=global_i_idx * 2 + projection_idx,
                                        ),
                                        op=nl.add,
                                    )
                                else:
                                    nisa.tensor_copy(
                                        dst=gate_and_up_proj_states_lst_sbuf[projection_idx][batch_tile_idx][
                                            global_i_idx
                                        ][0:num_i, nl.ds(0, free_size)],
                                        src=chunk_psum[batch_tile_idx][local_i_idx][0:num_i, nl.ds(0, free_size)],
                                    )

                _clip_gate_up_projections(gate_and_up_proj_states_lst_sbuf, dims, cfg, free_size)

                # TODO: PRE_SCALE_DELAYED support is still under development
                expert_affinity_T_broadcasted = None

                # ───────────────────────────────────────────────────────────────
                # Fused per-chunk: activation + down_proj
                # ───────────────────────────────────────────────────────────────
                expert_affinity_f32 = (
                    calculate_expert_affinities(
                        expert_affinities_masked,
                        token_indices_lst[inner_block_iter],
                        real_expert,
                        E,
                        NUM_TILES,
                        compute_dtype,
                        skip_dma,
                    )
                    if expert_affinities_scaling_mode == common_types.ExpertAffinityScaleMode.POST_SCALE
                    else None
                )
                down_activations = None

                if cfg.linear_bias:
                    down_bias_broadcasted = load_and_broadcast_down_bias(inps, dims, cfg, real_expert, skip_dma)
                else:
                    down_bias_broadcasted = None

                _, I_TP_w_dp, H_dp = down_proj_weight.shape

                # Save original block_old; start chunk accumulation from zeros
                original_block_old = block_old
                block_old_zero = []
                for alloc_idx in range(NUM_TILES):
                    tmp = nl.ndarray((TILE_SIZE, H), dtype=compute_dtype, buffer=nl.sbuf)
                    nisa.memset(tmp, value=0.0)
                    block_old_zero.append(tmp)
                block_old = block_old_zero

                for i_chunk_idx in range(N_I_CHUNKS):
                    i_chunk_start = i_chunk_idx * I_TP_CHUNK
                    i_chunk_end = min(i_chunk_start + I_TP_CHUNK, I_TP)
                    chunk_i_tp = i_chunk_end - i_chunk_start
                    chunk_n_tiles = div_ceil(chunk_i_tp, TILE_SIZE)

                    # Build chunk-sized gate_and_up_proj view
                    chunk_gup_sbuf = []
                    for proj_idx in range(GUP_PROJ_DIM):
                        n_lst = []
                        for batch_tile_idx in range(dims.n_psum_tile_count):
                            i_lst = []
                            for local_i in range(chunk_n_tiles):
                                global_i = i_chunk_idx * I_TP_CHUNK_TILES + local_i
                                i_lst.append(gate_and_up_proj_states_lst_sbuf[proj_idx][batch_tile_idx][global_i])
                            n_lst.append(i_lst)
                        chunk_gup_sbuf.append(n_lst)

                    chunk_intermediate = compute_intermediate_states(
                        chunk_gup_sbuf,
                        B,
                        chunk_i_tp,
                        compute_dtype,
                        activation_function=activation_function,
                        expert_affinity_T_broadcasted=expert_affinity_T_broadcasted,
                        gup_scale=gup_scale,
                    )

                    # Load down_proj weights for this chunk
                    chunk_dp_weights = []
                    for local_i in range(chunk_n_tiles):
                        chunk_dp_weights.append(
                            nl.ndarray((TILE_SIZE, H), dtype=down_proj_weight.dtype, buffer=nl.sbuf)
                        )
                    for local_i in range(chunk_n_tiles):
                        global_i_start = i_chunk_start + TILE_SIZE * local_i
                        num_i = min(TILE_SIZE, I_TP - global_i_start)
                        if num_i < TILE_SIZE:
                            nisa.memset(dst=chunk_dp_weights[local_i], value=0.0)
                        offset_dp = global_i_start * H_dp
                        nisa.dma_copy(
                            dst=chunk_dp_weights[local_i][0:num_i, 0:H],
                            src=down_proj_weight.ap(
                                pattern=[[H_dp, num_i], [1, H_dp]],
                                offset=offset_dp,
                                scalar_offset=block_expert,
                            ),
                            oob_mode=nisa.oob_mode.skip if skip_dma.skip_weight else nisa.oob_mode.error,
                        )

                    # Down projection for this chunk — plain accumulation (no affinity/bias yet)
                    block_new_lst = compute_block_output(
                        chunk_intermediate,
                        chunk_dp_weights,
                        None,
                        block_old,
                        down_activations,
                        local_block_idx,
                        H,
                        chunk_i_tp,
                        NUM_TILES,
                        output_dtype=output.dtype,
                        down_bias_broadcasted=None,
                        is_tensor_update_accumulating=True,
                        down_scale=down_scale,
                    )
                    # Use accumulated result as block_old for next chunk
                    block_old = block_new_lst

                # After all chunks: block_new_lst = sum(chunk_down_proj)
                # Apply bias, then affinity scaling, then add original block_old
                if down_bias_broadcasted is not None:
                    for token_tile_idx in range(NUM_TILES):
                        nisa.tensor_tensor(
                            dst=block_new_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                            data1=block_new_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                            data2=down_bias_broadcasted[0:TILE_SIZE, 0:H],
                            op=nl.add,
                        )

                if expert_affinity_f32 is not None:
                    if is_tensor_update_accumulating and original_block_old is not None:
                        # final = down_result * affinity + original_block_old
                        for token_tile_idx in range(NUM_TILES):
                            nisa.scalar_tensor_tensor(
                                dst=block_new_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                                data=block_new_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                                op0=nl.multiply,
                                operand0=expert_affinity_f32[token_tile_idx][0:TILE_SIZE, 0],
                                op1=nl.add,
                                operand1=original_block_old[token_tile_idx][0:TILE_SIZE, 0:H],
                            )
                    else:
                        for token_tile_idx in range(NUM_TILES):
                            nisa.tensor_scalar(
                                dst=block_new_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                                data=block_new_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                                op0=nl.multiply,
                                operand0=expert_affinity_f32[token_tile_idx][0:TILE_SIZE, 0],
                            )
                else:
                    # No affinity: just add original block_old if accumulating
                    if is_tensor_update_accumulating and original_block_old is not None:
                        for token_tile_idx in range(NUM_TILES):
                            nisa.tensor_tensor(
                                dst=block_new_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                                data1=block_new_lst[token_tile_idx][0:TILE_SIZE, 0:H],
                                data2=original_block_old[token_tile_idx][0:TILE_SIZE, 0:H],
                                op=nl.add,
                            )

            # Placeholder for TopK > 1 case, need to add full implementation
            if is_tensor_update_accumulating:
                for token_tile_idx in range(NUM_TILES):
                    token_idx = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=token_idx,
                        src=token_indices_lst[inner_block_iter][0:TILE_SIZE, token_tile_idx : token_tile_idx + 1],
                        engine=nisa.scalar_engine,
                    )

                    nisa.dma_copy(
                        dst=output.ap(
                            pattern=[[2 * H, TILE_SIZE], [1, H]],
                            offset=shard_id * H,
                            vector_offset=token_idx,
                            indirect_dim=0,
                        ),
                        src=block_new_lst[token_tile_idx].ap(pattern=[[H, TILE_SIZE], [1, H]], offset=0),
                        oob_mode=nisa.oob_mode.skip if skip_dma.skip_token else nisa.oob_mode.error,
                    )
            else:
                for token_tile_idx in range(NUM_TILES):
                    token_idx = nl.ndarray((TILE_SIZE, 1), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=token_idx,
                        src=token_indices_lst[inner_block_iter][0:TILE_SIZE, token_tile_idx : token_tile_idx + 1],
                    )

                    nisa.dma_copy(
                        dst=output.ap(
                            pattern=[[H, TILE_SIZE], [1, H]], offset=0, vector_offset=token_idx, indirect_dim=0
                        ),
                        src=block_new_lst[token_tile_idx].ap(pattern=[[H, TILE_SIZE], [1, H]], offset=0),
                        oob_mode=nisa.oob_mode.skip if skip_dma.skip_token else nisa.oob_mode.error,
                    )
