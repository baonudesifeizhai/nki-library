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
"""Torch references for 01_mixed_precision.py."""

import torch


def mixed_precision_matmul_fp32_output_torch_ref(A_hbm, B_hbm):
    # Kernel does bf16 input matmul with fp32 accumulator.
    # Reference must round inputs to bf16 to match hardware precision.
    A_bf16 = A_hbm.T.to(torch.bfloat16).to(torch.float32)
    B_bf16 = B_hbm.to(torch.bfloat16).to(torch.float32)
    return A_bf16 @ B_bf16


def mixed_precision_matmul_bf16_output_torch_ref(A_hbm, B_hbm):
    return A_hbm.T.to(torch.bfloat16) @ B_hbm.to(torch.bfloat16)
