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
"""Torch references for 04_subtile_indexing.py."""


def static_view_select_torch_ref(data, select_idx):
    return data[select_idx]


def chained_view_select_torch_ref(data, slab_idx, row_idx):
    # data: [D0, P, F] -> select data[slab_idx, row_idx, :] reshaped to [1, F]
    F = data.shape[-1]
    return data[slab_idx, row_idx, :].reshape(1, F)


def loop_view_select_torch_ref(data):
    return data.flip(0)


def tile_row_extract_torch_ref(data, row_idx):
    _, F = data.shape
    return data[row_idx, :].reshape(1, F)


def tile_element_extract_torch_ref(data, row_idx, col_idx):
    return data[row_idx, col_idx].reshape(1, 1)


def tile_subblock_extract_torch_ref(data, row_start, row_count, col_start, col_count):
    return data[row_start : row_start + row_count, col_start : col_start + col_count]


def dynamic_select_row_extract_torch_ref(weights, expert_id_tensor, row_idx):
    _, _, F = weights.shape
    eid = int(expert_id_tensor[0, 0].item())
    return weights[eid, row_idx, :].reshape(1, F)
