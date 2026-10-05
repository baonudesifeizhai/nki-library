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

"""PyTorch reference implementation for the RMSNorm-Quant kernel."""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import torch

from ..utils.common_types import DtypeMode, NormType, QuantizationType
from ..utils.kernel_helpers import get_max_positive_value_for_dtype

# FP8 clipping constants for quantization references.
# Legacy FP8 (float8_e4m3, TRN2 / gen3) max positive = 240.
# OCP-compliant FP8 (float8_e4m3fn, TRN3 / gen4) max positive = 448.
# The kernel picks the right dtype via nisa.get_nc_version(); the test must
# pass the matching dtype here so goldens clip to the same range.
FP8_E4M3_CLIP_VALUE = get_max_positive_value_for_dtype(nl.float8_e4m3)
FP8_E4M3FN_CLIP_VALUE = get_max_positive_value_for_dtype(nl.float8_e4m3fn)


def rmsnorm_quant_torch_ref(
    hidden: torch.Tensor,
    ln_w: torch.Tensor,
    kargs,
    input_dequant_scale: torch.Tensor = None,
    pre_norm_gamma: torch.Tensor = None,
    residual: torch.Tensor = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> dict[str, np.ndarray]:
    """Torch reference for rmsnorm_quant_kernel.

    Implements optional pre-normalization, residual addition, RMS normalization,
    and FP8 quantization (row or static) to serve as ground truth for correctness testing.

    Args:
        hidden (torch.Tensor): Input tensor of shape [B, S, H] to normalize and quantize.
        ln_w (torch.Tensor): RMSNorm weight (gamma) tensor of shape [H].
        kargs (RmsNormQuantKernelArgs): Kernel configuration containing norm_type,
            quantization_type, eps, and lower_bound.
        input_dequant_scale (torch.Tensor, optional): Dequantization scale for static
            quantization, shape [B, S, 1]. Required when quantization_type is STATIC.
        pre_norm_gamma (torch.Tensor, optional): Pre-normalization gamma weights of shape [H].
            When provided, applies RMSNorm to hidden before the main computation.
        residual (torch.Tensor, optional): Residual tensor of shape [B, S, H].
            When provided, added to hidden after optional pre-normalization.
        dtype_mode (DtypeMode): FP8 E4M3 dtype selection for quantized output.
            - ``DtypeMode.NON_OCP`` (default): ``nl.float8_e4m3`` (max=240).
            - ``DtypeMode.OCP``: ``nl.float8_e4m3fn`` (max=448).
            Pass ``nl.float8_e4m3fn`` (OCP, max=448) when the kernel runs on TRN3/gen4 so
            the reference clips to the same range as the kernel output. Tests typically
            branch on ``platform_target.is_trn3()`` to pick this.

    Returns:
        dict with:
          - "norm_quant": [B, S, H] fp8e4m3 quantized output (dtype = quant_dtype)
          - "dequant_scale": [B, S, 1] fp32 dequant scale (ROW quant only, None for STATIC)
          - "residual_out": [B, S, H] fp32 residual output (only when residual is provided)
    """
    quant_dtype = nl.float8_e4m3fn if dtype_mode == DtypeMode.OCP else nl.float8_e4m3
    FP8_RANGE = FP8_E4M3FN_CLIP_VALUE if quant_dtype == nl.float8_e4m3fn else FP8_E4M3_CLIP_VALUE

    inp = hidden.numpy().astype(np.float32)
    gamma = ln_w.numpy().astype(np.float32)
    if input_dequant_scale != None:
        in_dq_scale = input_dequant_scale.numpy().astype(np.float32)
    else:
        in_dq_scale = None

    quant_only = kargs.norm_type == NormType.NO_NORM
    quant_type = kargs.quantization_type
    eps = kargs.eps
    lower_bound = kargs.lower_bound

    # Optional pre-norm: RMSNorm(hidden, pre_norm_gamma)
    if pre_norm_gamma != None:
        png = pre_norm_gamma.numpy().astype(np.float32)
        rms = np.sqrt(np.mean(np.square(inp), axis=-1, keepdims=True) + eps)
        inv_rms = 1.0 / rms
        inp = inp * inv_rms
        inp *= png

    # Optional residual add
    residual_out = None
    if residual != None:
        res = residual.numpy().astype(np.float32)
        inp = inp + res
        residual_out = inp.copy()

    # RMSNorm
    if quant_only:
        norm = inp
    else:
        rms = np.sqrt(np.mean(np.square(inp), axis=-1, keepdims=True) + eps)
        inv_rms = 1.0 / rms
        norm = inp * inv_rms
        norm *= gamma

    # Quantization
    if quant_type == QuantizationType.ROW:
        norm_abs_max = np.abs(norm).max(axis=-1, keepdims=True)
        if lower_bound > 0:
            norm_abs_max = np.clip(norm_abs_max, a_min=None, a_max=lower_bound)
            norm = np.clip(norm, a_min=-lower_bound, a_max=lower_bound)
        dequant_scale = norm_abs_max / FP8_RANGE
        quant_scale = np.reciprocal(dequant_scale)
        norm_quant = norm * quant_scale
        dequant_scale = dt.static_cast(dequant_scale, np.float32)
    elif quant_type == QuantizationType.STATIC:
        quant_scale = np.reciprocal(in_dq_scale[0, 0])
        norm = norm * quant_scale
        norm_quant = np.clip(norm, a_min=-FP8_RANGE, a_max=FP8_RANGE)
        dequant_scale = None

    norm_quant = dt.static_cast(norm_quant, quant_dtype)

    return {"norm_quant": norm_quant, "dequant_scale": dequant_scale, "residual_out": residual_out}
