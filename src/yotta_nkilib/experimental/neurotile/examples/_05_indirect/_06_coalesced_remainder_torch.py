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
"""Torch references for 06_coalesced_remainder.py."""

import torch


def row_hoist_f_remainder_torch_ref(src):
    # Iterates all rows; trailing F-tile (cols 384..) uses oob skip → cols 384..F-1
    # in src are written, cols F..511 in the tile are skipped (and not in dst).
    return src.clone()


def col_hoist_p_remainder_torch_ref(src):
    # Iterates all (i, j) tiles; trailing P-tile (rows 256..) uses oob skip →
    # rows 256..P-1 in src are written, rows P..383 in the tile are skipped.
    return src.clone()


def range_slice_remainder_torch_ref(src):
    # Only tiles[1:3, :] are iterated → elem rows [128:384, :].
    # OOB rows [384:P) and cols beyond F in trailing tile stay at HBM zero-init.
    out = torch.zeros_like(src)
    out[128 : min(384, src.shape[0]), :] = src[128 : min(384, src.shape[0]), :]
    return out


def store_oob_mode_torch_ref(src):
    # Single tile-row (P=128) with F-remainder; row.is_remainder fires.
    # All in-bounds elements get written.
    return src.clone()


def multi_range_interior_torch_ref(src):
    # Only tiles[0:2, 0:3] iterated → elem rows [0:256], cols [0:384].
    out = torch.zeros_like(src)
    out[:256, :384] = src[:256, :384]
    return out
