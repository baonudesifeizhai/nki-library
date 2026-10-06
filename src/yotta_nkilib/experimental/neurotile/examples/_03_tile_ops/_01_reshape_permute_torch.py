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
"""Torch references for 01_reshape_permute.py."""

import torch


def norm_reshape_transpose_torch_ref(src_hbm):
    # src: [4, 1024]  ->  out: [128, 32]; out[h0, p] = src.flat[p * 128 + h0]
    H0, BxS_H1 = 128, 32
    flat = src_hbm.reshape(-1)
    out = torch.empty((H0, BxS_H1), dtype=src_hbm.dtype)
    for h0 in range(H0):
        for p in range(BxS_H1):
            out[h0, p] = flat[p * 128 + h0]
    return out


def reshape_permute_load_torch_ref(src_hbm):
    # src: [4, 1024]  ->  out: [128, 4, 8]
    flat = src_hbm.reshape(-1)
    out = torch.empty((128, 4, 8), dtype=src_hbm.dtype)
    for h0 in range(128):
        for bxs in range(4):
            for h1 in range(8):
                out[h0, bxs, h1] = flat[h0 + bxs * 1024 + h1 * 128]
    return out


def bsh_to_h0_bs_h1_torch_ref(src_hbm):
    # src: [B, S, H]  ->  out flat: [H0, B*S, H1] reshaped back to kernel's output shape.
    B, S, H = src_hbm.shape
    H0, H1 = 8, 16
    return src_hbm.reshape(B, S, H0, H1).permute(2, 0, 1, 3).reshape(H0, B * S, H1)


def attention_qk_layout_torch_ref(q_hbm):
    # q: [B, S, H]  ->  out: [head_dim, B*S] (kernel selects head index 0)
    B, S, H = q_hbm.shape
    num_heads, head_dim = 8, 16
    q_perm = q_hbm.reshape(B, S, num_heads, head_dim).permute(3, 0, 1, 2).reshape(head_dim, B * S, num_heads)
    return q_perm[:, :, 0]


def tile_reshape_elementwise_torch_ref(src):
    return src * 2.0


def block_reshape_elementwise_torch_ref(src):
    return src * 2.0


def block_reshape_chunked_torch_ref(src):
    return src + 1.0
