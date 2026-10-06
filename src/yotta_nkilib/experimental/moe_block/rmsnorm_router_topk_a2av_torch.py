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

"""PyTorch reference for rmsnorm_router_topk."""

from typing import Optional

import torch

from ...core.router_topk.router_topk import XSBLayout_tp102__0
from ...core.router_topk.router_topk_torch import router_topk_torch_ref
from ...core.subkernels.rmsnorm_torch import rms_norm_torch_ref
from ...core.utils.common_types import RouterActFnType


def rmsnorm_router_topk_a2av_torch_ref(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    router_weights: torch.Tensor,
    router_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    top_k: int = 1,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
) -> dict:
    """Torch reference for rmsnorm_router_topk.

    Returns dict with keys: norm_output, expert_index, expert_affinities.
    """
    B, S, H = hidden_states.shape
    T = B * S

    # RMSNorm
    norm_output = rms_norm_torch_ref(hidden_states, gamma, eps=eps).reshape(T, H)

    # Router TopK
    router_outputs = router_topk_torch_ref(
        x=norm_output,
        w=router_weights,
        w_bias=router_bias,
        router_logits=None,
        expert_affinities=None,
        expert_index=None,
        act_fn=router_act_fn,
        k=top_k,
        x_hbm_layout=1,  # [T, H] layout
        x_sb_layout=XSBLayout_tp102__0,
        router_pre_norm=False,
        norm_topk_prob=False,
    )

    return {
        "norm_output": norm_output,
        "expert_index": router_outputs["expert_index"],
        "expert_affinities": router_outputs["expert_affinities"],
    }
