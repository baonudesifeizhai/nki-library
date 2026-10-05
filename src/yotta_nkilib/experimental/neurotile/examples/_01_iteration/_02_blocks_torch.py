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
"""Torch references for 02_blocks.py."""

_TILE_P = 128
_TILE_F = 128


def block_level_scale_torch_ref(src):
    return src * 2.0


def load_block_row_torch_ref(src):
    return src * 2.0


def load_block_subgrid_torch_ref(src):
    # Block-rows 1..2 cover tile-rows [2..6) -> elem rows [256..768).
    out = src.clone()
    out[2 * _TILE_P : 6 * _TILE_P, :] *= 4.0
    return out


def promote_tile_to_block_view_torch_ref(src):
    return src * 2.0


def descend_block_to_tile_view_torch_ref(src):
    return src * 2.0


def retile_block_view_torch_ref(src):
    return src * 2.0
