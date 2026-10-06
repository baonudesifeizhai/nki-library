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
"""Torch references for 03_kv_cache.py."""

import torch


def kv_cache_load_torch_ref(kv_cache, batch_id, seq_offset_tensor):
    _, _, D = kv_cache.shape
    seq = int(seq_offset_tensor[0, 0].item())
    return kv_cache[batch_id, seq, :].reshape(1, D)


def kv_cache_load_raw_sbuf_index_torch_ref(kv_cache, batch_id, seq_offset_tensor):
    _, _, D = kv_cache.shape
    seq = int(seq_offset_tensor[0, 0].item())
    return kv_cache[batch_id, seq, :].reshape(1, D)


def kv_cache_multi_pos_torch_ref(kv_cache, batch_indices, seq_positions):
    K = batch_indices.shape[0]
    _, _, D = kv_cache.shape
    out = torch.zeros(K, D, dtype=kv_cache.dtype)
    for k in range(K):
        out[k, :] = kv_cache[batch_indices[k, 0], seq_positions[k, 0], :]
    return out
