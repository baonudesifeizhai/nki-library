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

"""RoPE kernel for HuggingFace tensor layout [batch, heads, seq, head_dim]."""

from typing import Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info

NUM_COALESCE_TILES = 8


@nki.jit
def rope_hf(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    q_out: nl.NkiTensor,
    k_out: nl.NkiTensor,
    cos: Optional[nl.NkiTensor] = None,
    sin: Optional[nl.NkiTensor] = None,
    rope_cache: Optional[nl.NkiTensor] = None,
    backward: bool = False,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Apply Rotary Position Embedding (RoPE) to query and key tensors using HuggingFace layout.

    This kernel uses tensor layout [batch, heads, seq, head_dim], common in HuggingFace
    models (e.g., Llama). Optimized for seq_len divisible by 128 * num_lnc_shards.

    Dimensions:
        batch_size: Batch size
        q_heads: Number of query attention heads
        k_heads: Number of key attention heads
        seq_len: Sequence length (must be divisible by 128 * num_lnc_shards)
        head_dim: Head dimension size

    Args:
        q (nl.NkiTensor): [batch_size, q_heads, seq_len, head_dim] @ HBM, Query tensor
        k (nl.NkiTensor): [batch_size, k_heads, seq_len, head_dim] @ HBM, Key tensor
        q_out (nl.NkiTensor): [batch_size, q_heads, seq_len, head_dim] @ HBM, Output query tensor
        k_out (nl.NkiTensor): [batch_size, k_heads, seq_len, head_dim] @ HBM, Output key tensor
        cos (Optional[nl.NkiTensor]): [optional(batch_size), seq_len, head_dim] @ HBM, Cosine embeddings.
            Required if rope_cache is None.
        sin (Optional[nl.NkiTensor]): [optional(batch_size), seq_len, head_dim] @ HBM, Sine embeddings.
            Required if rope_cache is None.
        rope_cache (Optional[nl.NkiTensor]): [optional(batch_size), seq_len, head_dim*2] @ HBM,
            Packed cos/sin tensor. First half contains cos, second half contains sin.
        backward (bool): If True, compute backward pass (gradient w.r.t. inputs). Default False.

    Returns:
        Tuple[q_out, k_out]: Query and key tensors with rotary embeddings applied.

    Notes:
        - Either (cos, sin) or rope_cache must be provided, not both
        - seq_len must be divisible by (128 * num_lnc_shards)
        - Use `rope` kernel for layout [head_dim, batch, heads, seq_len]

    Pseudocode:
        for batch_id in range(batch_size):
            for seq_tile_idx in range(num_seq_tiles):
                # Load cos/sin tiles for this sequence chunk
                cos_tile = load(cos[seq_start:seq_end])
                sin_tile = load(sin[seq_start:seq_end])

                # Apply RoPE to all Q heads
                for head_id in range(q_heads):
                    q_tile = load(q[batch_id, head_id, seq_start:seq_end])
                    q_out[batch_id, head_id, seq_start:seq_end] = rotate(q_tile, cos_tile, sin_tile)

                # Apply RoPE to all K heads
                for head_id in range(k_heads):
                    k_tile = load(k[batch_id, head_id, seq_start:seq_end])
                    k_out[batch_id, head_id, seq_start:seq_end] = rotate(k_tile, cos_tile, sin_tile)
    """
    _, num_shards, shard_id = get_verified_program_sharding_info()
    _validate_apply_rope_inputs(q, k, cos, sin, rope_cache, num_shards)

    batch_size, _, seq_len, head_dim = q.shape

    # how much of total seq_len dim per shard (half for lnc 2)
    seq_per_shard = seq_len // num_shards
    seq_shard_offset = shard_id * seq_per_shard
    seq_tile_size = min(nl.tile_size.pmax, seq_per_shard)

    # how many tiles on shard dim
    num_seq_tiles_per_shard = div_ceil(seq_per_shard, seq_tile_size)
    for batch_id in range(batch_size):
        for seq_tile_idx in range(0, num_seq_tiles_per_shard, NUM_COALESCE_TILES):
            # Number of tiles to process at a time
            num_tiles = min(NUM_COALESCE_TILES, num_seq_tiles_per_shard - seq_tile_idx)
            seq_start = seq_shard_offset + seq_tile_idx * seq_tile_size

            cos_tile = nl.ndarray((seq_tile_size, num_tiles, head_dim), dtype=q.dtype, buffer=nl.sbuf)
            sin_tile = nl.ndarray((seq_tile_size, num_tiles, head_dim), dtype=q.dtype, buffer=nl.sbuf)

            seq_end = seq_start + num_tiles * seq_tile_size

            if rope_cache != None:
                rope_tensor = rope_cache[batch_id, :, :] if rope_cache.ndim == 3 else rope_cache
                rope_tensor = rope_tensor[seq_start:seq_end, :]
                rope_tensor = rope_tensor.reshape_dim(0, (num_tiles, seq_tile_size)).permute((1, 0, 2))
                cos_ap_src = rope_tensor[:, :, :head_dim]
                sin_ap_src = rope_tensor[:, :, head_dim:]
            else:
                cos_tensor = cos[batch_id, seq_start:seq_end, :] if len(cos.shape) == 3 else cos[seq_start:seq_end, :]
                sin_tensor = sin[batch_id, seq_start:seq_end, :] if len(sin.shape) == 3 else sin[seq_start:seq_end, :]
                cos_ap_src = cos_tensor.reshape_dim(0, (num_tiles, seq_tile_size)).permute((1, 0, 2))
                sin_ap_src = sin_tensor.reshape_dim(0, (num_tiles, seq_tile_size)).permute((1, 0, 2))

            nisa.dma_copy(cos_tile, cos_ap_src)
            nisa.dma_copy(sin_tile, sin_ap_src)

            _apply_rope_all_heads(q, q_out, cos_tile, sin_tile, batch_id, seq_start, backward)
            _apply_rope_all_heads(k, k_out, cos_tile, sin_tile, batch_id, seq_start, backward)

    return q_out, k_out


def _validate_apply_rope_inputs(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    cos: Optional[nl.NkiTensor],
    sin: Optional[nl.NkiTensor],
    rope_cache: Optional[nl.NkiTensor],
    num_shards: int,
) -> None:
    """
    Validate inputs for apply_rope operation.

    Args:
        q (nl.NkiTensor): [batch_size, q_heads, seq_len, head_dim], Query tensor
        k (nl.NkiTensor): [batch_size, k_heads, seq_len, head_dim], Key tensor
        cos (Optional[nl.NkiTensor]): [optional(batch_size), seq_len, head_dim], Cosine embeddings
        sin (Optional[nl.NkiTensor]): [optional(batch_size), seq_len, head_dim], Sine embeddings
        rope_cache (Optional[nl.NkiTensor]): [optional(batch_size), seq_len, head_dim*2], Packed cos/sin
        num_shards (int): Number of LNC shards

    Returns:
        None

    Notes:
        - Either (cos, sin) or rope_cache must be provided, not both
        - seq_len must be divisible by (128 * num_shards)
    """
    kernel_assert(len(q.shape) == 4, f"apply_rope only supports 4D tensors, got input shape {q.shape}")
    kernel_assert(q.shape[-1] == k.shape[-1], f"Head dim mismatch: q={q.shape[-1]}, k={k.shape[-1]}")

    head_dim = q.shape[-1]

    if rope_cache != None:
        # Packed rope_cache format: contains both cos and sin
        kernel_assert(
            rope_cache.shape[-1] == head_dim * 2,
            f"Packed rope_cache expects head_dim*2={head_dim * 2}, got {rope_cache.shape[-1]}",
        )
        kernel_assert(cos == None and sin == None, "Expected either rope_cache or separate cos/sin tensors, got both")
        rope_tensor = rope_cache
    else:
        # Separate cos/sin tensors
        kernel_assert(cos != None and sin != None, "Expected either rope_cache or separate cos/sin tensors, got none")
        kernel_assert(
            cos.shape == sin.shape, f"Shape of cos Tensor: {cos.shape} doesn't match shape of sin Tensor: {sin.shape}"
        )
        kernel_assert(cos.shape[-1] == head_dim, f"Head dim mismatch: q={head_dim}, cos={cos.shape[-1]}")
        rope_tensor = cos

    # Validate batch dimension (optional for rope cache)
    kernel_assert(
        len(rope_tensor.shape) == 2 or (len(rope_tensor.shape) == 3 and rope_tensor.shape[0] == q.shape[0]),
        f"rope_cache or cos tensor batch size {rope_tensor.shape} doesn't match q batch size {q.shape}",
    )
    kernel_assert(
        rope_tensor.shape[-2] >= q.shape[2],
        f"rope_cache or cos tensor seq len {rope_tensor.shape[-2]} smaller than q seq len {q.shape[2]}",
    )
    kernel_assert(
        q.shape[2] % (nl.tile_size.pmax * num_shards) == 0,
        f"seq len must be multiple of 128 * LNC={num_shards}, got {q.shape[2]}",
    )


def _apply_rope_all_heads(
    x: nl.NkiTensor,
    x_out: nl.NkiTensor,
    cos_tile: nl.NkiTensor,
    sin_tile: nl.NkiTensor,
    batch_id: int,
    seq_start: int,
    backward: bool = False,
) -> None:
    """
    Apply rotary embedding to a tensor for all attention heads using preloaded cos/sin tiles.

    Args:
        x (nl.NkiTensor): [batch_size, num_heads, seq_len, head_dim] @ HBM, Input tensor
        x_out (nl.NkiTensor): [batch_size, num_heads, seq_len, head_dim] @ HBM, Output tensor
        cos_tile (nl.NkiTensor): [seq_tile_size, num_tiles, head_dim] @ SBUF, Cosine embedding tile
        sin_tile (nl.NkiTensor): [seq_tile_size, num_tiles, head_dim] @ SBUF, Sine embedding tile
        batch_id (int): Current batch index
        seq_start (int): Starting sequence position
        backward (bool): If True, compute backward pass rotation. Default False.

    Returns:
        None

    Notes:
        - Processes all heads sequentially for the given batch and sequence tile
        - Writes results directly to x_out in HBM
    """
    seq_tile_size, num_tiles, head_dim = cos_tile.shape
    num_heads = x.shape[1]

    # Process all heads sequentially
    for head_id in range(num_heads):
        x_tile = nl.ndarray((seq_tile_size, num_tiles, head_dim), dtype=x.dtype, buffer=nl.sbuf)
        # Select batch+head, slice seq range, reshape into tiles, permute to (seq_tile, num_tiles, head_dim)
        x_ap_src = (
            x[batch_id, head_id, seq_start : seq_start + num_tiles * seq_tile_size, :]
            .reshape_dim(0, (num_tiles, seq_tile_size))
            .permute((1, 0, 2))
        )
        nisa.dma_copy(x_tile, x_ap_src)

        x_rotated = _apply_rope_single(x_tile, cos_tile, sin_tile, backward)
        x_out_view = (
            x_out[batch_id, head_id, seq_start : seq_start + num_tiles * seq_tile_size, :]
            .reshape_dim(0, (num_tiles, seq_tile_size))
            .permute((1, 0, 2))
        )
        nisa.dma_copy(x_out_view, x_rotated)


def _apply_rope_single(
    x_tile: nl.NkiTensor,
    cos_tile: nl.NkiTensor,
    sin_tile: nl.NkiTensor,
    backward: bool = False,
) -> nl.NkiTensor:
    """
    Apply rotary embedding to a single tensor tile in SBUF.

    Args:
        x_tile (nl.NkiTensor): [seq_tile_size, num_tiles, head_dim] @ SBUF, Input tensor tile
        cos_tile (nl.NkiTensor): [seq_tile_size, num_tiles, head_dim] @ SBUF, Cosine embedding tile
        sin_tile (nl.NkiTensor): [seq_tile_size, num_tiles, head_dim] @ SBUF, Sine embedding tile
        backward (bool): If True, compute backward pass rotation. Default False.

    Returns:
        nl.NkiTensor: [seq_tile_size, num_tiles, head_dim] @ SBUF, Rotated tensor tile

    Notes:
        - Forward: y = [x1, x2] * [cos1, cos2] + [-x2, x1] * [sin1, sin2]
        - Backward reverses the rotation direction for gradient computation
    """
    seq_size, num_tiles, head_dim = x_tile.shape
    result = nl.ndarray((seq_size, num_tiles, head_dim), dtype=x_tile.dtype, buffer=nl.sbuf)

    # Create temporary tensors for intermediate calculations
    temp1 = nl.ndarray((seq_size, num_tiles, head_dim // 2), dtype=x_tile.dtype, buffer=nl.sbuf)
    temp2 = nl.ndarray((seq_size, num_tiles, head_dim // 2), dtype=x_tile.dtype, buffer=nl.sbuf)

    if not backward:
        # Forward: y = [x1, x2] * [cos1, cos2] + [-x2, x1] * [sin1, sin2]
        # result = x_tile * cos_tile
        nisa.tensor_tensor(dst=result, data1=x_tile, data2=cos_tile, op=nl.multiply)

        # temp1 = x_tile[..., head_dim//2:] * sin_tile[..., :head_dim//2]
        nisa.tensor_tensor(
            dst=temp1, data1=x_tile[:, :, head_dim // 2 :], data2=sin_tile[:, :, : head_dim // 2], op=nl.multiply
        )

        # result[..., :head_dim//2] -= temp1
        nisa.tensor_tensor(
            dst=result[:, :, : head_dim // 2], data1=result[:, :, : head_dim // 2], data2=temp1, op=nl.subtract
        )

        # temp2 = x_tile[..., :head_dim//2] * sin_tile[..., head_dim//2:]
        nisa.tensor_tensor(
            dst=temp2, data1=x_tile[:, :, : head_dim // 2], data2=sin_tile[:, :, head_dim // 2 :], op=nl.multiply
        )

        # result[..., head_dim//2:] += temp2
        nisa.tensor_tensor(
            dst=result[:, :, head_dim // 2 :], data1=result[:, :, head_dim // 2 :], data2=temp2, op=nl.add
        )
    else:
        # Backward: dy = [dx1, dx2] * [cos1, cos2] + [dx2, -dx1] * [sin2, sin1]
        # result = x_tile * cos_tile
        nisa.tensor_tensor(dst=result, data1=x_tile, data2=cos_tile, op=nl.multiply)

        # temp1 = x_tile[..., head_dim//2:] * sin_tile[..., head_dim//2:]
        nisa.tensor_tensor(
            dst=temp1, data1=x_tile[:, :, head_dim // 2 :], data2=sin_tile[:, :, head_dim // 2 :], op=nl.multiply
        )

        # result[..., :head_dim//2] += temp1
        nisa.tensor_tensor(
            dst=result[:, :, : head_dim // 2], data1=result[:, :, : head_dim // 2], data2=temp1, op=nl.add
        )

        # temp2 = x_tile[..., :head_dim//2] * sin_tile[..., :head_dim//2]
        nisa.tensor_tensor(
            dst=temp2, data1=x_tile[:, :, : head_dim // 2], data2=sin_tile[:, :, : head_dim // 2], op=nl.multiply
        )

        # result[..., head_dim//2:] -= temp2
        nisa.tensor_tensor(
            dst=result[:, :, head_dim // 2 :], data1=result[:, :, head_dim // 2 :], data2=temp2, op=nl.subtract
        )

    return result
