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
"""Dispatch mechanism for swapping kernel calls with torch ref implementations.

Usage:
    # Normal execution: kernel runs on hardware
    result = moe_tkg(...)

    # Debug mode: set env var, same code runs torch ref on CPU
    # $ NKILIB_USE_TORCH_REF=1 python my_model.py

    # Per-kernel override:
    # $ NKILIB_USE_TORCH_REF_moe_tkg=1 python my_model.py
"""

import importlib
import logging
import os
from typing import Callable, Optional

_logger = logging.getLogger(__name__)

_USE_TORCH_REF = os.environ.get("NKILIB_USE_TORCH_REF", "").lower() in ("1", "true", "yes", "on")


def dispatch(
    kernel_func: Callable, torch_ref_module: Optional[str] = None, torch_ref_name: Optional[str] = None
) -> Callable:
    """Return kernel_func normally, or its torch_ref if NKILIB_USE_TORCH_REF=1.

    Discovery order:
      1. Try explicit override (torch_ref_module/torch_ref_name) if provided
      2. Fall back to convention-based discovery

    Convention:
      kernel module "...rmsnorm_quant" -> torch_ref module "...rmsnorm_quant_torch"
      kernel func "rmsnorm_quant_kernel" -> torch_ref func "rmsnorm_quant_torch_ref"
      kernel func "moe_tkg" -> torch_ref func "moe_tkg_torch_ref"

    Args:
        kernel_func: The NKI kernel function (returned as-is when dispatch is inactive)
        torch_ref_module: Optional explicit module path for the torch_ref (override)
        torch_ref_name: Optional explicit function name for the torch_ref (override)

    Returns:
        kernel_func when dispatch is inactive, or torch_ref_wrapper(torch_ref) when active.
    """
    # Per-kernel override: NKILIB_USE_TORCH_REF_<name>=1
    kernel_name = kernel_func.__name__
    base = kernel_name.replace("_kernel", "") if kernel_name.endswith("_kernel") else kernel_name
    per_kernel_env = f"NKILIB_USE_TORCH_REF_{base}"
    use_ref = os.environ.get(per_kernel_env, "").lower() in ("1", "true", "yes", "on")
    if not use_ref:
        use_ref = _USE_TORCH_REF

    if not use_ref:
        return kernel_func

    # Lazy import to avoid circular deps and unnecessary torch import at module load
    from .torch_ref_wrapper import torch_ref_wrapper

    # 1. Try explicit override first
    if torch_ref_module is not None or torch_ref_name is not None:
        try:
            module = importlib.import_module(torch_ref_module)
            ref = getattr(module, torch_ref_name)
            _logger.info(f"NKILIB_USE_TORCH_REF: {kernel_name} -> {torch_ref_name} (explicit)")
            return torch_ref_wrapper(ref)
        except (ImportError, AttributeError):
            pass  # Fall through to convention

    # 2. Convention-based discovery
    ref_module_name = kernel_func.__module__ + "_torch"
    ref_name = base + "_torch_ref"

    module = importlib.import_module(ref_module_name)
    ref = getattr(module, ref_name)
    _logger.info(f"NKILIB_USE_TORCH_REF: {kernel_name} -> {ref_name} (convention)")
    return torch_ref_wrapper(ref)
