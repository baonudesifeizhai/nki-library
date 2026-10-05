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
"""Torch references for 01_gather_scatter.py."""

import torch


def gather_torch_ref(data, indices):
    return data[indices.flatten().long(), :]


def scatter_torch_ref(source, indices, out_rows):
    K, D = source.shape
    out = torch.zeros(out_rows, D, dtype=source.dtype)
    for i in range(K):
        out[indices[i, 0], :] = source[i, :]
    return out


def scalar_gather_dim0_torch_ref(data, row_idx_tensor):
    N, D = data.shape
    row = int(row_idx_tensor[0, 0].item())
    return data[row, :].reshape(1, D)


def scalar_gather_dim1_torch_ref(data, col_idx_tensor):
    N, D = data.shape
    col = int(col_idx_tensor[0, 0].item())
    return data[:, col].reshape(N, 1)
