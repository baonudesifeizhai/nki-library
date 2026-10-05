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

"""PyTorch reference implementation for qkv_cte_mla kernel."""

from typing import Dict

import nki.language as nl
import numpy as np
import torch

from ...core.subkernels.rmsnorm_torch import rms_norm_torch_ref
from ...core.utils.mx_torch_common import (
    mx_matmul,
    quantize_to_mx,
    unpack_float8_e4m3fn_x4,
)

# MX quantization constants
_Q_WIDTH = 4  # MX quantization group width (number of elements packed together)
_PMAX = 128  # Hardware partition dimension size (nl.tile_size.pmax resolves to -1 on host)
_DS_SCALE_BLOCK = 128  # DeepSeek block-128 scale block size


def _broadcast_compact_scales(compact_scale, in_dim, out_dim):
    """Broadcast a DeepSeek compact block-128 scale to MX matmul layout.

    Args:
        compact_scale: numpy array [in_dim // 128, ceil(out_dim / 128)] uint8.
        in_dim: K dimension of the matmul.
        out_dim: N dimension of the matmul.

    Returns:
        numpy array [in_dim // 32, out_dim] uint8 with each compact scale
        replicated across the 4 MX-scale rows it covers (along K) and across
        128 columns of N (or fewer for the trailing partial block).
    """
    if isinstance(compact_scale, torch.Tensor):
        compact_np = compact_scale.cpu().numpy()
    else:
        compact_np = compact_scale
    full = np.repeat(compact_np, 4, axis=0)
    full = np.repeat(full, _DS_SCALE_BLOCK, axis=1)
    return full[: in_dim // 32, :out_dim].astype(np.uint8)


def qkv_mla_mx_torch_ref(
    x_hbm: torch.Tensor,
    wqkv_a_hbm,  # MX x4 packed numpy
    wqkv_a_scale_hbm,  # numpy [P//8, fused_dim]
    wq_b_hbm,  # MX x4 packed numpy
    wq_b_scale_hbm,  # numpy [P//8, q_out_dim]
    wkv_b_hbm,  # MX x4 packed numpy
    wkv_b_scale_hbm,  # numpy [P//8, kv_b_out_dim]
    q_norm_gamma_hbm: torch.Tensor,
    kv_norm_gamma_hbm: torch.Tensor,
    cos_cache_hbm: torch.Tensor,
    sin_cache_hbm: torch.Tensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
    norm_eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    PyTorch reference implementation for MLA QKV CTE kernel.

    Implements Multi-head Latent Attention (MLA) QKV projection with:
        1. Unswizzle input (MX layout)
        2. Fused QKV_A matmul: [B,S,H] @ W_qkv_a -> [qk_lora_rank + kv_lora_rank + qk_rope_head_dim]
        3. Split into Q compressed (qr) and KV_A output (kv + k_pe)
        4. RMSNorm on qr, then Q_B matmul -> [n_heads, qk_head_dim] -> split q_nope, q_pe
        5. RMSNorm on kv, then KV_B matmul -> [n_heads, qk_nope_head_dim + v_head_dim] -> split k_nope, v
        6. RoPE on q_pe (per head) and k_pe (shared across heads)
        7. Concatenate: q = [q_nope, q_pe_rope], k = [k_nope, k_pe_rope_broadcast]

    Args:
        x_hbm: Input hidden states [B, S, H] (MX-swizzled).
        wqkv_a_hbm: Fused QKV_A weights, MX x4 packed numpy [H//4, fused_dim*4].
        wqkv_a_scale_hbm: QKV_A weight scales numpy [P//8, fused_dim].
        wq_b_hbm: Q_B weights, MX x4 packed numpy [qk_lora_rank//4, q_out_dim*4].
        wq_b_scale_hbm: Q_B weight scales numpy [P//8, q_out_dim].
        wkv_b_hbm: KV_B weights, MX x4 packed numpy [kv_lora_rank//4, kv_b_out_dim*4].
        wkv_b_scale_hbm: KV_B weight scales numpy [P//8, kv_b_out_dim].
        q_norm_gamma_hbm: Q RMSNorm gamma [1, qk_lora_rank].
        kv_norm_gamma_hbm: KV RMSNorm gamma [1, kv_lora_rank].
        cos_cache_hbm: RoPE cosine cache [B, S, qk_rope_head_dim].
        sin_cache_hbm: RoPE sine cache [B, S, qk_rope_head_dim].
        n_heads: Number of attention heads.
        qk_nope_head_dim: Non-positional head dimension for Q/K.
        qk_rope_head_dim: Positional (RoPE) head dimension for Q/K.
        v_head_dim: Value head dimension.
        kv_lora_rank: KV LoRA rank.
        qk_lora_rank: QK LoRA rank.
        norm_eps: RMSNorm epsilon. Default: 1e-6.

    Returns:
        Dict with keys "q", "k", "v" as torch.bfloat16 tensors:
            - "q": [B, S, n_heads, qk_nope_head_dim + qk_rope_head_dim]
            - "k": [B, S, n_heads, qk_nope_head_dim + qk_rope_head_dim]
            - "v": [B, S, n_heads, v_head_dim]
    """
    x = x_hbm.to(torch.float32)
    B, S, H = x.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_a_out_dim = kv_lora_rank + qk_rope_head_dim
    fused_out_dim = qk_lora_rank + kv_a_out_dim

    cos_cache = cos_cache_hbm.to(torch.float32)
    sin_cache = sin_cache_hbm.to(torch.float32)
    q_norm_gamma = q_norm_gamma_hbm.to(torch.float32)
    kv_norm_gamma = kv_norm_gamma_hbm.to(torch.float32)

    # Unswizzle MX input layout
    x = x.reshape(B * S, _Q_WIDTH, H // (_PMAX * _Q_WIDTH), _PMAX).permute(0, 2, 3, 1).reshape(B, S, H)

    def _mx_matmul_torch(inp_bf16, weights_packed_np, weights_scale_np, in_dim, out_dim):
        """MX block-quantize input, then mx_matmul with packed weights."""
        b, s, _ = inp_bf16.shape
        # Reshape to [P, F] layout for MX quantization
        hidden_np = inp_bf16.reshape(b * s, in_dim).T.numpy()  # [in_dim, b*s]
        hidden_np = (
            hidden_np.reshape(in_dim // _Q_WIDTH, _Q_WIDTH, b * s)
            .transpose(0, 2, 1)
            .reshape(in_dim // _Q_WIDTH, _Q_WIDTH * b * s)
            .astype(np.float32)
        )
        hidden_mx, hidden_scale = quantize_to_mx(hidden_np, nl.float8_e4m3fn_x4)

        hidden_mx_torch = unpack_float8_e4m3fn_x4(hidden_mx)
        weights_unpacked = unpack_float8_e4m3fn_x4(weights_packed_np)

        hidden_scale_torch = torch.from_numpy(hidden_scale.astype(np.float64))
        if isinstance(weights_scale_np, torch.Tensor):
            w_scale_torch = weights_scale_np.to(torch.float64)
        else:
            w_scale_torch = torch.from_numpy(weights_scale_np.astype(np.float64))

        result = mx_matmul(
            stationary=hidden_mx_torch,
            moving=weights_unpacked,
            stationary_scale=hidden_scale_torch,
            moving_scale=w_scale_torch,
        )
        return result.reshape(b, s, out_dim)

    # Broadcast DeepSeek compact block-128 scales to MX matmul layout for ref.
    wqkv_a_scale_full = _broadcast_compact_scales(wqkv_a_scale_hbm, H, fused_out_dim)
    wq_b_scale_full = _broadcast_compact_scales(wq_b_scale_hbm, qk_lora_rank, n_heads * qk_head_dim)
    kv_b_out_dim = n_heads * (qk_nope_head_dim + v_head_dim)
    wkv_b_scale_full = _broadcast_compact_scales(wkv_b_scale_hbm, kv_lora_rank, kv_b_out_dim)

    # Step 1: Fused QKV_A matmul
    fused_out = _mx_matmul_torch(x, wqkv_a_hbm, wqkv_a_scale_full, H, fused_out_dim)

    # Un-swizzle the matmul output: wqkv_a columns were pre-swizzled from
    # [I//512, 4, 128] to [4, I//512, 128] order. Reverse this for qr and kv
    # portions before splitting. k_pe portion was not swizzled.
    def _unswizzle_cols(t, dim):
        num_512_tiles = dim // (_PMAX * _Q_WIDTH)
        # Inverse of swizzle [grp, pos, sub] -> [sub, grp, pos]
        # Un-swizzle: [sub, grp, pos] -> [grp, pos, sub]
        idx = np.arange(dim).reshape(_Q_WIDTH, num_512_tiles, _PMAX).transpose(1, 2, 0).reshape(dim)
        return t[..., idx]

    qr_swizzled = fused_out[..., :qk_lora_rank]
    kv_raw_swizzled = fused_out[..., qk_lora_rank:]
    kv_swizzled = kv_raw_swizzled[..., :kv_lora_rank]
    k_pe = kv_raw_swizzled[..., kv_lora_rank:]  # not swizzled

    qr = _unswizzle_cols(qr_swizzled, qk_lora_rank)
    kv = _unswizzle_cols(kv_swizzled, kv_lora_rank)

    # Step 2: Q path - RMSNorm + Q_B matmul
    qr = rms_norm_torch_ref(qr, q_norm_gamma, eps=norm_eps)
    q = _mx_matmul_torch(qr, wq_b_hbm, wq_b_scale_full, qk_lora_rank, n_heads * qk_head_dim)
    q = q.reshape(B, S, n_heads, qk_head_dim)
    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    # Step 3: KV path - split, RMSNorm, KV_B matmul
    kv = rms_norm_torch_ref(kv, kv_norm_gamma, eps=norm_eps)
    kv_out = _mx_matmul_torch(kv, wkv_b_hbm, wkv_b_scale_full, kv_lora_rank, kv_b_out_dim)
    kv_out = kv_out.reshape(B, S, n_heads, qk_nope_head_dim + v_head_dim)
    k_nope, v = kv_out.split([qk_nope_head_dim, v_head_dim], dim=-1)

    # Step 4: RoPE (first-second-half interleave) on q_pe and k_pe
    half = qk_rope_head_dim // 2
    cos = cos_cache[:, :, :half].unsqueeze(2)  # [B, S, 1, half]
    sin = sin_cache[:, :, :half].unsqueeze(2)

    # Q RoPE: per-head [B, S, n_heads, qk_rope_head_dim]
    q_pe_first = q_pe[:, :, :, :half]
    q_pe_second = q_pe[:, :, :, half:]
    q_pe_rope = torch.cat(
        [
            q_pe_first * cos - q_pe_second * sin,
            q_pe_first * sin + q_pe_second * cos,
        ],
        dim=-1,
    )

    # K RoPE: shared single head [B, S, qk_rope_head_dim] -> [B, S, 1, qk_rope_head_dim]
    k_pe = k_pe.unsqueeze(2)  # [B, S, 1, qk_rope_head_dim]
    k_pe_first = k_pe[:, :, :, :half]
    k_pe_second = k_pe[:, :, :, half:]
    k_pe_rope = torch.cat(
        [
            k_pe_first * cos - k_pe_second * sin,
            k_pe_first * sin + k_pe_second * cos,
        ],
        dim=-1,
    )

    # Step 5: Concatenate
    q = torch.cat([q_nope, q_pe_rope], dim=-1)
    k = torch.cat([k_nope, k_pe_rope.expand(B, S, n_heads, qk_rope_head_dim)], dim=-1)

    return {
        "q": q.to(torch.bfloat16),
        "k": k.to(torch.bfloat16),
        "v": v.to(torch.bfloat16),
    }


def qkv_mla_mx_deepseek_v4_torch_ref(
    x_hbm: torch.Tensor,
    wqkv_hbm,
    wqkv_scale_hbm,
    wq_b_hbm,
    wq_b_scale_hbm,
    q_norm_gamma_hbm: torch.Tensor,
    kv_norm_gamma_hbm: torch.Tensor,
    cos_cache_hbm: torch.Tensor,
    sin_cache_hbm: torch.Tensor,
    n_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
    norm_eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """PyTorch reference for v4 MLA QKV.

    Pipeline:
        Fused: x -> wqkv -> split [qr, kv]
        Q: qr -> q_norm(gamma) -> wq_b -> unflatten -> rsqrt_norm -> RoPE(q[..., -rd:])
        KV: kv -> kv_norm(gamma) -> RoPE(kv[..., -rd:])
    """
    x = x_hbm.to(torch.float32)
    B, S, H = x.shape
    kv_dim = kv_lora_rank + qk_rope_head_dim
    q_out_dim = n_heads * head_dim
    qk_nope_head_dim = head_dim - qk_rope_head_dim

    cos_cache = cos_cache_hbm.to(torch.float32)
    sin_cache = sin_cache_hbm.to(torch.float32)
    q_norm_gamma = q_norm_gamma_hbm.to(torch.float32)
    kv_norm_gamma = kv_norm_gamma_hbm.to(torch.float32)

    # Unswizzle MX input layout
    x = x.reshape(B * S, _Q_WIDTH, H // (_PMAX * _Q_WIDTH), _PMAX).permute(0, 2, 3, 1).reshape(B, S, H)

    def _mx_matmul_torch(inp_bf16, weights_packed_np, weights_scale_np, in_dim, out_dim):
        b, s, _ = inp_bf16.shape
        hidden_np = inp_bf16.reshape(b * s, in_dim).T.numpy()
        hidden_np = (
            hidden_np.reshape(in_dim // _Q_WIDTH, _Q_WIDTH, b * s)
            .transpose(0, 2, 1)
            .reshape(in_dim // _Q_WIDTH, _Q_WIDTH * b * s)
            .astype(np.float32)
        )
        hidden_mx, hidden_scale = quantize_to_mx(hidden_np, nl.float8_e4m3fn_x4)
        hidden_mx_torch = unpack_float8_e4m3fn_x4(hidden_mx)
        weights_unpacked = unpack_float8_e4m3fn_x4(weights_packed_np)
        hidden_scale_torch = torch.from_numpy(hidden_scale.astype(np.float64))
        if isinstance(weights_scale_np, torch.Tensor):
            w_scale_torch = weights_scale_np.to(torch.float64)
        else:
            w_scale_torch = torch.from_numpy(weights_scale_np.astype(np.float64))
        result = mx_matmul(
            stationary=hidden_mx_torch,
            moving=weights_unpacked,
            stationary_scale=hidden_scale_torch,
            moving_scale=w_scale_torch,
        )
        return result.reshape(b, s, out_dim)

    # Fused first projection: x -> wqkv -> split [qr, kv]
    kv_dim = kv_lora_rank + qk_rope_head_dim
    fused_out_dim = qk_lora_rank + kv_dim
    wqkv_scale_full = _broadcast_compact_scales(wqkv_scale_hbm, H, fused_out_dim)
    wq_b_scale_full = _broadcast_compact_scales(wq_b_scale_hbm, qk_lora_rank, q_out_dim)
    fused_out = _mx_matmul_torch(x, wqkv_hbm, wqkv_scale_full, H, fused_out_dim)

    def _unswizzle_cols(t, dim):
        num_512_tiles = dim // (_PMAX * _Q_WIDTH)
        idx = np.arange(dim).reshape(_Q_WIDTH, num_512_tiles, _PMAX).transpose(1, 2, 0).reshape(dim)
        return t[..., idx]

    # Un-swizzle qr columns (pre-swizzled in weights)
    qr_swizzled = fused_out[..., :qk_lora_rank]
    kv = fused_out[..., qk_lora_rank:]  # kv columns not swizzled

    qr = _unswizzle_cols(qr_swizzled, qk_lora_rank)

    # RMSNorm with gamma
    qr = rms_norm_torch_ref(qr, q_norm_gamma, eps=norm_eps)

    # Second Q matmul
    q = _mx_matmul_torch(qr, wq_b_hbm, wq_b_scale_full, qk_lora_rank, q_out_dim)
    q = q.reshape(B, S, n_heads, head_dim)

    # Post-wq_b rsqrt norm per head
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + norm_eps)

    # RoPE on q[..., -rd:]
    half = qk_rope_head_dim // 2
    cos = cos_cache[:, :, :half].unsqueeze(2)
    sin = sin_cache[:, :, :half].unsqueeze(2)

    q_nope = q[..., :qk_nope_head_dim]
    q_pe = q[..., qk_nope_head_dim:]
    q_pe_first = q_pe[..., :half]
    q_pe_second = q_pe[..., half:]
    q_pe_rope = torch.cat(
        [
            q_pe_first * cos - q_pe_second * sin,
            q_pe_first * sin + q_pe_second * cos,
        ],
        dim=-1,
    )
    q = torch.cat([q_nope, q_pe_rope], dim=-1)

    # KV path: kv_norm(gamma) -> RoPE
    kv = rms_norm_torch_ref(kv, kv_norm_gamma, eps=norm_eps)

    # RoPE on kv[..., -rd:]
    kv_nope = kv[..., :kv_lora_rank]
    kv_pe = kv[..., kv_lora_rank:]
    cos_1d = cos_cache[:, :, :half]
    sin_1d = sin_cache[:, :, :half]
    kv_pe_first = kv_pe[..., :half]
    kv_pe_second = kv_pe[..., half:]
    kv_pe_rope = torch.cat(
        [
            kv_pe_first * cos_1d - kv_pe_second * sin_1d,
            kv_pe_first * sin_1d + kv_pe_second * cos_1d,
        ],
        dim=-1,
    )
    kv = torch.cat([kv_nope, kv_pe_rope], dim=-1)

    return {
        "q": q.to(torch.bfloat16),
        "kv": kv.to(torch.bfloat16),
    }
