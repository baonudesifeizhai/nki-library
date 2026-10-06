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
"""Torch reference for the neurotile rmsnorm_tkg_nt example kernel.

Computes RMSNorm and reshapes to the kernel's [H0, BxS, H1] output layout.
"""

import torch


def rmsnorm_tkg_torch_ref(hidden, gamma, eps, H_actual):
    # hidden: [B, S_tkg, H]; gamma: [1, H]
    B, S_tkg, H = hidden.shape
    H_denom = H_actual if H_actual is not None else H
    H0 = 128
    H1 = H // H0
    BxS = B * S_tkg

    x = hidden.to(torch.float32)
    g = gamma.to(torch.float32).reshape(1, H)
    rms = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) / H_denom + eps)
    normalized = x * rms * g
    # Reshape: [B, S_tkg, H] -> [BxS, H0, H1] -> [H0, BxS, H1]
    return normalized.reshape(BxS, H0, H1).permute(1, 0, 2).to(hidden.dtype)
