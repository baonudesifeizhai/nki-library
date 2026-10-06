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
"""Runner for the single-rank mlp_cte kernel.

Usage:
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python run_mlp.py --config compare_2k
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python run_mlp.py --config compare_2k --verify
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import nki.isa as nisa
import numpy as np
import torch
from mlp_cte_nt import MLPConfig, mlp_cte

CONFIGS = {
    "small": {"M": 1024, "K": 512, "I": 512, "H": 512, "bxs": 4, "atol": 0.0625},
    "small_bxs2": {"M": 2048, "K": 512, "I": 512, "H": 512, "bxs": 2, "atol": 0.0625},
    "compare_2k": {"M": 8192, "K": 2048, "I": 2048, "H": 2048, "bxs": 2, "atol": 0.05},
    "compare_4k_i2560": {"M": 8192, "K": 4096, "I": 2560, "H": 4096, "bxs": 2, "atol": 0.1},
    "llama3_8b_tp4": {"M": 8192, "K": 4096, "I": 3584, "H": 4096, "bxs": 2, "atol": 0.15},
    "llama3_8b_tp8": {"M": 8192, "K": 4096, "I": 1792, "H": 4096, "bxs": 2, "atol": 0.15},
}


def mlp_reference_bf16(x, gate, up, down):
    gate_out = torch.matmul(x, gate)
    up_out = torch.matmul(x, up)
    swiglu = torch.nn.functional.silu(gate_out) * up_out
    return torch.matmul(swiglu, down).float().numpy()


def main():
    parser = argparse.ArgumentParser(description="Run neurotile mlp_cte kernel")
    parser.add_argument("--config", default="compare_2k", choices=list(CONFIGS.keys()))
    parser.add_argument("--bxs", type=int, default=None, help="Override BXS subtile count")
    parser.add_argument("--dge-mode", default=None, choices=["none", "hwdge", "swdge"], help="Override dge_mode")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    M, K, I, H = cfg["M"], cfg["K"], cfg["I"], cfg["H"]
    bxs = args.bxs if args.bxs is not None else cfg.get("bxs", 2)
    atol = cfg.get("atol", 0.1)
    dge_mode = None
    if args.dge_mode == "none":
        dge_mode = nisa.dge_mode.none
    elif args.dge_mode == "hwdge":
        dge_mode = nisa.dge_mode.hwdge
    elif args.dge_mode == "swdge":
        dge_mode = nisa.dge_mode.swdge
    mlp_config = MLPConfig(K=K, I=I, H=H, bxs_subtile_count=bxs, dge_mode=dge_mode)

    print(f"mlp_cte: M={M} K={K} I={I} H={H} BXS={bxs}", flush=True)

    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    torch.manual_seed(42)

    x_2d = (torch.randn(M, K) * 0.1).to(torch.bfloat16)
    x = x_2d.reshape(1, M, K)
    gate_t = (torch.randn(K, I) * 0.1).to(torch.bfloat16)
    up_t = (torch.randn(K, I) * 0.1).to(torch.bfloat16)
    down_t = (torch.randn(I, H) * 0.1).to(torch.bfloat16)

    x_dev = x.to(device)
    gate_dev = gate_t.to(device)
    up_dev = up_t.to(device)
    down_dev = down_t.to(device)

    result = mlp_cte[2](x_dev, gate_dev, up_dev, down_dev, mlp_config)
    result_np = result.cpu().to(torch.float32).numpy().reshape(-1, H)

    print(f"Result shape: {result_np.shape}, range: [{result_np.min():.4f}, {result_np.max():.4f}]", flush=True)

    if args.verify:
        expected = mlp_reference_bf16(x_2d, gate_t, up_t, down_t)
        max_err = float(np.abs(result_np - expected).max())
        if max_err < atol:
            print(f"PASSED (max_err={max_err:.6f})", flush=True)
        else:
            print(f"FAILED (max_err={max_err:.6f})", flush=True)
            sys.exit(1)
    else:
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
