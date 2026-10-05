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

"""PyTorch reference implementation for depthwise Conv1D kernel testing."""

import torch
import torch.nn.functional as F


def depthwise_conv1d_implicit_gemm_torch_ref(
    img_ref: torch.Tensor,
    filter_ref: torch.Tensor,
    padding: tuple = ((0, 0), (0, 0)),
    stride: tuple = (1, 1),
    rhs_dilation: tuple = (1, 1),
    lhs_dilation: tuple = (1, 1),
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    in_perm: tuple = None,
    kern_perm: tuple = None,
    out_perm: tuple = None,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of depthwise Conv1D using conv2d with groups=C.

    Args:
        img_ref (torch.Tensor): [N, C, 1, W] input tensor
        filter_ref (torch.Tensor): [C, 1, 1, S] depthwise kernel weights
        padding (tuple): ((H_pad_l, H_pad_r), (W_pad_l, W_pad_r))
        stride (tuple): (stride_h, stride_w)
        rhs_dilation (tuple): RHS dilation (unused in reference)
        lhs_dilation (tuple): LHS dilation (unused in reference)
        feature_group_count (int): Number of feature groups (C for depthwise)
        batch_group_count (int): Number of batch groups

    Returns:
        dict[str, torch.Tensor]: {"output": tensor of shape [N, C, 1, Q]}
    """
    C = img_ref.shape[1]
    padding_pytorch = (0, padding[1][0])
    output = F.conv2d(img_ref, filter_ref, bias=None, stride=stride, padding=padding_pytorch, groups=C)
    return {"output": output}
