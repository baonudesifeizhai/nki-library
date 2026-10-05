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
"""Torch references for 01_tiles.py."""

import torch

_TILE_P = 128
_TILE_F = 128


def basic_tile_copy_torch_ref(src):
    return src


def direct_indexing_torch_ref(src):
    return src[_TILE_P : 2 * _TILE_P, _TILE_F : 2 * _TILE_F]


def scale_torch_ref(src, factor=2.0):
    return src * factor


def load_tile_row_torch_ref(src):
    return src * 2.0


def load_tile_column_torch_ref(src):
    return src * 3.0


def load_subgrid_torch_ref(src):
    out = src.clone()
    out[_TILE_P : 3 * _TILE_P, _TILE_F : 3 * _TILE_F] *= 4.0
    return out


def load_negative_index_torch_ref(src):
    return src[-_TILE_P:, :] * 2.0


def load_strided_rows_torch_ref(src):
    even = torch.cat([src[0:_TILE_P, :], src[2 * _TILE_P : 3 * _TILE_P, :]], dim=0)
    return even * 2.0


def batched_tile_iteration_torch_ref(src):
    return src * 2.0


def higher_rank_tile_torch_ref(src):
    out = src.clone()
    out[:, ::2, :] *= 2.0
    out[:, 1::2, :] *= 3.0
    return out


def partition_tile_torch_ref(src):
    return src * 2.0


def column_tile_torch_ref(src):
    return src * 2.0
