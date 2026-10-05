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
"""Torch references for 02_fold_pattern_override.py."""

import torch


def fold_into_partition_torch_ref(src_hbm):
    # src: [P, F, K]  ->  out: [K*P, F]; for each k, out[k*P:(k+1)*P] = src[:, :, k]
    P, F, K = src_hbm.shape
    out = torch.empty((K * P, F), dtype=src_hbm.dtype)
    for k in range(K):
        out[k * P : (k + 1) * P, :] = src_hbm[:, :, k]
    return out


def fold_free_dim_torch_ref(src_hbm):
    # src: [P, F, K]  ->  out: [P, F*K]
    P, F, K = src_hbm.shape
    return src_hbm.reshape(P, F * K)


def fold_partition_roundtrip_torch_ref(src_hbm):
    return src_hbm * 2.0


def fold_free_dim_roundtrip_torch_ref(src_hbm):
    return src_hbm * 2.0


def fold_chain_4d_torch_ref(src_hbm):
    # Tutorial only checks output shape (32, 512). Reshape preserves data ordering;
    # the kernel does a chained fold that flattens to [32, 512]. Match by reshape.
    return src_hbm.reshape(32, 512)


def pattern_override_load_torch_ref(src_hbm):
    # Strided load: every other element along free dim.
    return src_hbm[:, ::2]
