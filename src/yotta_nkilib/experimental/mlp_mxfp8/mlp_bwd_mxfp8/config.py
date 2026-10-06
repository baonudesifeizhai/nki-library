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

"""Configuration dataclasses and auto-tuning logic for MXFP8 MLP backward pass matmuls."""

from dataclasses import dataclass
from typing import Optional

import nki.language as nl

from ...matmul_mxfp8.matmul_mxfp8_config import (
    MatmulMxfp8KernelConfig,
    resolve_matmul_config_with_validation,
)
from ...moe.bwd.moe_bwd_parameters import ClampLimits  # noqa: F401
from ...mxfp_utils.mxfp8_utils.common_dataclasses import TensorDescriptor

# ===========================================================================
# Per-matmul config using MatmulMxfp8KernelConfig
# ===========================================================================


@dataclass
class MlpBwdMatmulConfig(nl.NKIObject):
    """Per-matmul configuration for MLP backward pass.

    Each of the 6 individual matmuls can be independently tuned.
    Any field left as None will be auto-resolved by the kernel via
    resolve_matmul_config_with_validation().

    Paired matmul constraints:
      - recompute_gate and recompute_up must have matching tile_m, tile_k,
        TILES_IN_BLOCK_M, TILES_IN_BLOCK_K (shared LHS SBUF).
    """

    recompute_gate: Optional[MatmulMxfp8KernelConfig] = None
    recompute_up: Optional[MatmulMxfp8KernelConfig] = None
    phase1_down_proj: Optional[MatmulMxfp8KernelConfig] = None
    phase2_hidden_grad: Optional[MatmulMxfp8KernelConfig] = None
    phase3_wgrad: Optional[MatmulMxfp8KernelConfig] = None
    phase4_wgrad: Optional[MatmulMxfp8KernelConfig] = None

    def __str__(self) -> str:
        lines = ["MlpBwdMatmulConfig:"]
        for name in [
            "recompute_gate",
            "recompute_up",
            "phase1_down_proj",
            "phase2_hidden_grad",
            "phase3_wgrad",
            "phase4_wgrad",
        ]:
            cfg = getattr(self, name)
            if cfg is None:
                lines.append(f"  {name}: None")
            else:
                lines.append(
                    f"  {name}: M={cfg.M}, K={cfg.K}, N={cfg.N}, "
                    f"tile_m={cfg.tile_m}, tile_k={cfg.tile_k}, tile_n={cfg.tile_n}, "
                    f"TILES_IN_BLOCK_M={cfg.TILES_IN_BLOCK_M}, TILES_IN_BLOCK_N={cfg.TILES_IN_BLOCK_N}, "
                    f"TILES_IN_BLOCK_K={cfg.TILES_IN_BLOCK_K}"
                )
        return "\n".join(lines)

    def resolve_backward_phases(
        self,
        run_with_lnc2: bool,
        spill_reload: bool,
        use_scale_packing: bool,
        output_grad_td: TensorDescriptor,
        down_weight_T_td: TensorDescriptor,
        d_gate_up_td: TensorDescriptor,
        gate_up_weight_T_td: TensorDescriptor,
        scratch_td: TensorDescriptor,
        hidden_states_T_td: TensorDescriptor,
        output_grad_T_td: TensorDescriptor,
        intermediate_T_td: TensorDescriptor,
    ) -> None:
        """Resolve matmul configs for all 4 backward phases in place."""
        self.phase1_down_proj = resolve_matmul_config_with_validation(
            self.phase1_down_proj,
            output_grad_td,
            down_weight_T_td,
            run_with_lnc2,
            spill_reload,
            use_scale_packing,
            False,
        )
        self.phase2_hidden_grad = resolve_matmul_config_with_validation(
            self.phase2_hidden_grad,
            d_gate_up_td,
            gate_up_weight_T_td,
            run_with_lnc2,
            spill_reload,
            use_scale_packing,
            False,
        )
        self.phase3_wgrad = resolve_matmul_config_with_validation(
            self.phase3_wgrad,
            scratch_td,
            hidden_states_T_td,
            run_with_lnc2,
            spill_reload,
            use_scale_packing,
            False,
        )
        self.phase4_wgrad = resolve_matmul_config_with_validation(
            self.phase4_wgrad,
            output_grad_T_td,
            intermediate_T_td,
            run_with_lnc2,
            spill_reload,
            use_scale_packing,
            False,
        )

    def resolve_recompute_phases(
        self,
        hidden_states_td: TensorDescriptor,
        gate_up_weights_td: TensorDescriptor,
        run_with_lnc2: bool,
        spill_reload: bool,
        use_scale_packing: bool,
    ) -> None:
        """Resolve matmul configs for recompute gate/up phases in place."""
        self.recompute_gate = resolve_matmul_config_with_validation(
            self.recompute_gate,
            hidden_states_td,
            gate_up_weights_td,
            run_with_lnc2,
            spill_reload,
            use_scale_packing,
            False,
        )
        self.recompute_up = resolve_matmul_config_with_validation(
            self.recompute_up,
            hidden_states_td,
            gate_up_weights_td,
            run_with_lnc2,
            spill_reload,
            use_scale_packing,
            False,
        )
