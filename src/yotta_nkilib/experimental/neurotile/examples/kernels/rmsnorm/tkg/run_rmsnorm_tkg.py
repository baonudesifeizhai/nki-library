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
#!/usr/bin/env python3
"""
Compile and run rmsnorm_tkg kernel on NeuronCore for profiling.

Uses llama3_70b config dimensions (S_tkg=1, H=8192, lnc=2) with configurable batch size.

Usage:
    python examples/kernels/rmsnorm/tkg/run_rmsnorm_tkg.py --batch 16
"""

import argparse
import os
import sys

import numpy as np
import torch

# Add this directory to path for sibling imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from rmsnorm_tkg_nt import rmsnorm_tkg_kernel


def to_device(np_array):
    import torch_xla.core.xla_model as xm

    """Convert numpy array to XLA tensor on NeuronCore."""
    if np_array.dtype not in (np.float32, np.float64):
        t = torch.from_numpy(np.ascontiguousarray(np_array.astype(np.float32)))
        t = t.to(torch.bfloat16)
    else:
        t = torch.from_numpy(np.ascontiguousarray(np_array))
    return t.to(xm.xla_device())


def main():
    import torch_xla.core.xla_model as xm

    parser = argparse.ArgumentParser(description="Run rmsnorm_tkg kernel for profiling")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    # llama3_70b config dimensions with configurable batch
    batch = args.batch
    S_tkg = 1
    H = 8192
    lnc = 2
    eps = 1e-3
    H_actual = None  # no padding

    print("=" * 60)
    print("RMSNorm TKG Kernel -- llama3_70b config")
    print("=" * 60)
    print(f"  batch={batch}, S_tkg={S_tkg}, H={H}, lnc={lnc}, eps={eps}")
    print()

    np.random.seed(42)
    X = np.random.randn(batch, S_tkg, H).astype(np.float32) * 0.1
    gamma = np.random.uniform(0.5, 1.5, (1, H)).astype(np.float32)

    print(f"  X:     {X.shape} {X.dtype}")
    print(f"  gamma: {gamma.shape} {gamma.dtype}")
    print()

    X_dev = to_device(X)
    gamma_dev = to_device(gamma)

    print("Compiling and running kernel...")
    result = rmsnorm_tkg_kernel[lnc](X_dev, gamma_dev, eps, H_actual)
    xm.mark_step()
    xm.wait_device_ops()

    print("Kernel execution complete.")
    result_np = result.cpu().numpy()
    print(f"  Output: {result_np.shape} {result_np.dtype}")


if __name__ == "__main__":
    main()
