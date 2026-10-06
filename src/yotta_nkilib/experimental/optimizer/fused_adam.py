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

"""
This module implements a fused Adam/AdamW/AMSGrad optimizer kernel with SPMD tiling and Scalar Engine fusion.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Free-dim tile size. Needs ~5 SBUF tiles per iteration (p, g, ea, eas, t0).
_F_SIZE_BF16 = 4096
_F_SIZE_F32 = 2048


def _adam_tile_compute(
    param,
    grad,
    exp_avg,
    exp_avg_sq,
    max_exp_avg_sq,
    step_size,
    inv_bc2_sqrt,
    weight_decay,
    beta1,
    beta2,
    eps,
    one_minus_beta1,
    one_minus_beta2,
    decoupled_wd,
    amsgrad,
):
    """
    Pure Adam compute for one tile.

    All inputs/outputs are SBUF tensors. Performs the core Adam/AdamW/AMSGrad
    update computation including moment updates and parameter updates.

    Args:
        param (nl.ndarray): Parameter tensor in SBUF
        grad (nl.ndarray): Gradient tensor in SBUF
        exp_avg (nl.ndarray): First moment estimate in SBUF
        exp_avg_sq (nl.ndarray): Second moment estimate in SBUF
        max_exp_avg_sq (nl.ndarray): Max second moment (AMSGrad only) in SBUF
        step_size (nl.ndarray): Bias-corrected learning rate in SBUF
        inv_bc2_sqrt (nl.ndarray): Inverse bias correction sqrt in SBUF
        weight_decay (nl.ndarray): Weight decay factor in SBUF
        beta1 (float): First moment decay factor
        beta2 (float): Second moment decay factor
        eps (float): Epsilon for numerical stability
        one_minus_beta1 (float): 1 - beta1
        one_minus_beta2 (float): 1 - beta2
        decoupled_wd (bool): If True, use AdamW decoupled weight decay
        amsgrad (bool): If True, use AMSGrad variant

    Returns:
        tuple: (param, exp_avg, exp_avg_sq, max_exp_avg_sq) updated tensors
    """
    # Adam L2 reg: grad = grad + weight_decay * param
    if not decoupled_wd:
        nisa.scalar_tensor_tensor(
            dst=grad, data=param, op0=nl.multiply, operand0=weight_decay, op1=nl.add, operand1=grad
        )

    # First moment: exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    temp0 = nl.ndarray(exp_avg.shape, dtype=exp_avg.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=temp0, data=exp_avg, op0=nl.multiply, operand0=beta1, engine=nisa.scalar_engine)
    nisa.scalar_tensor_tensor(
        dst=exp_avg, data=grad, op0=nl.multiply, operand0=one_minus_beta1, op1=nl.add, operand1=temp0
    )

    # Second moment: exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
    nisa.tensor_scalar(dst=temp0, data=exp_avg_sq, op0=nl.multiply, operand0=beta2, engine=nisa.scalar_engine)
    nisa.tensor_tensor(dst=grad, data1=grad, data2=grad, op=nl.multiply)  # grad = grad^2
    nisa.scalar_tensor_tensor(
        dst=exp_avg_sq, data=grad, op0=nl.multiply, operand0=one_minus_beta2, op1=nl.add, operand1=temp0
    )

    # AMSGrad: denom_src = max(max_exp_avg_sq, exp_avg_sq)
    if amsgrad:
        nisa.tensor_tensor(dst=max_exp_avg_sq, data1=max_exp_avg_sq, data2=exp_avg_sq, op=nl.maximum)
        denom_src = max_exp_avg_sq
    else:
        denom_src = exp_avg_sq

    # denom = sqrt(denom_src) * inv_bc2_sqrt + eps; update = exp_avg / denom * step_size
    nisa.activation(dst=temp0, op=nl.sqrt, data=denom_src)
    nisa.tensor_scalar(
        dst=temp0,
        data=temp0,
        op0=nl.multiply,
        operand0=inv_bc2_sqrt,
        op1=nl.add,
        operand1=eps,
        engine=nisa.scalar_engine,
    )
    nisa.activation(dst=temp0, data=temp0, op=nl.reciprocal)
    nisa.tensor_tensor(dst=grad, data1=exp_avg, data2=temp0, op=nl.multiply)
    nisa.tensor_scalar(dst=grad, data=grad, op0=nl.multiply, operand0=step_size, engine=nisa.scalar_engine)

    # param_new: AdamW decoupled decay param = param * weight_decay - update; Adam: param = param - update
    if decoupled_wd:
        nisa.scalar_tensor_tensor(
            dst=param, data=param, op0=nl.multiply, operand0=weight_decay, op1=nl.subtract, operand1=grad
        )
    else:
        nisa.tensor_tensor(dst=param, data1=param, data2=grad, op=nl.subtract)

    return param, exp_avg, exp_avg_sq, max_exp_avg_sq


def _adam_tile_step(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    max_exp_avg_sq_ptr,
    param_out,
    exp_avg_out,
    exp_avg_sq_out,
    max_exp_avg_sq_out,
    shape,
    access_pattern,
    offset,
    step_size,
    inv_bc2_sqrt,
    weight_decay,
    beta1,
    beta2,
    eps,
    one_minus_beta1,
    one_minus_beta2,
    decoupled_wd,
    amsgrad,
):
    """
    Load inputs, call _adam_tile_compute, store outputs.

    Shape/access_pattern/offset differ between main and tail tiles.

    Args:
        param_ptr (nl.ndarray): Parameter tensor pointer on HBM
        grad_ptr (nl.ndarray): Gradient tensor pointer on HBM
        exp_avg_ptr (nl.ndarray): First moment pointer on HBM
        exp_avg_sq_ptr (nl.ndarray): Second moment pointer on HBM
        max_exp_avg_sq_ptr (nl.ndarray): Max second moment pointer on HBM
        param_out (nl.ndarray): Output parameter tensor on HBM
        exp_avg_out (nl.ndarray): Output first moment on HBM
        exp_avg_sq_out (nl.ndarray): Output second moment on HBM
        max_exp_avg_sq_out (nl.ndarray): Output max second moment on HBM
        shape (tuple): Shape of the tile
        access_pattern (list): Access pattern for DMA
        offset (int): Offset for DMA
        step_size (nl.ndarray): Bias-corrected learning rate
        inv_bc2_sqrt (nl.ndarray): Inverse bias correction sqrt
        weight_decay (nl.ndarray): Weight decay factor
        beta1 (float): First moment decay factor
        beta2 (float): Second moment decay factor
        eps (float): Epsilon for numerical stability
        one_minus_beta1 (float): 1 - beta1
        one_minus_beta2 (float): 1 - beta2
        decoupled_wd (bool): If True, use AdamW decoupled weight decay
        amsgrad (bool): If True, use AMSGrad variant
    """
    param = nl.ndarray(shape, dtype=param_ptr.dtype, buffer=nl.sbuf)
    grad = nl.ndarray(shape, dtype=grad_ptr.dtype, buffer=nl.sbuf)
    exp_avg = nl.ndarray(shape, dtype=exp_avg_ptr.dtype, buffer=nl.sbuf)
    exp_avg_sq = nl.ndarray(shape, dtype=exp_avg_sq_ptr.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=param, src=param_ptr.ap(access_pattern, offset))
    nisa.dma_copy(dst=grad, src=grad_ptr.ap(access_pattern, offset))
    nisa.dma_copy(dst=exp_avg, src=exp_avg_ptr.ap(access_pattern, offset))
    nisa.dma_copy(dst=exp_avg_sq, src=exp_avg_sq_ptr.ap(access_pattern, offset))

    if amsgrad:
        max_exp_avg_sq = nl.ndarray(shape, dtype=max_exp_avg_sq_ptr.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=max_exp_avg_sq, src=max_exp_avg_sq_ptr.ap(access_pattern, offset))
    else:
        max_exp_avg_sq = None

    param, exp_avg, exp_avg_sq, max_exp_avg_sq = _adam_tile_compute(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        step_size,
        inv_bc2_sqrt,
        weight_decay,
        beta1,
        beta2,
        eps,
        one_minus_beta1,
        one_minus_beta2,
        decoupled_wd,
        amsgrad,
    )

    nisa.dma_copy(dst=param_out.ap(access_pattern, offset), src=param)
    nisa.dma_copy(dst=exp_avg_out.ap(access_pattern, offset), src=exp_avg)
    nisa.dma_copy(dst=exp_avg_sq_out.ap(access_pattern, offset), src=exp_avg_sq)
    if amsgrad:
        nisa.dma_copy(dst=max_exp_avg_sq_out.ap(access_pattern, offset), src=max_exp_avg_sq)


def _adam_kernel_impl(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    max_exp_avg_sq_ptr,
    step_size_ptr,
    inv_bc2_sqrt_ptr,
    wd_factor_ptr,
    numel,
    beta1,
    beta2,
    eps,
    decoupled_wd,
    amsgrad,
):
    """
    Internal implementation for Adam/AdamW/AMSGrad optimizer kernel with SPMD tiling and Scalar Engine fusion.

    This kernel implements the Adam optimizer family (Adam, AdamW, AMSGrad) with efficient
    SPMD parallelization across 2 cores and fused scalar operations. It uses flat memory
    access patterns via .ap() for optimal performance.

    TODO: Specify intended usage range (e.g., parameter count, batch size)

    Dimensions:
        N: Number of elements in the parameter tensor (numel)
        P_MAX: Partition dimension size (128)

    Args:
        param_ptr (nl.ndarray): [N], Parameter tensor on HBM
        grad_ptr (nl.ndarray): [N], Gradient tensor on HBM
        exp_avg_ptr (nl.ndarray): [N], First moment estimate tensor on HBM
        exp_avg_sq_ptr (nl.ndarray): [N], Second moment estimate tensor on HBM
        max_exp_avg_sq_ptr (nl.ndarray): [N], Max second moment tensor on HBM (AMSGrad only)
        step_size_ptr (nl.ndarray): [P_MAX, 1], Bias-corrected step size on HBM
        inv_bc2_sqrt_ptr (nl.ndarray): [P_MAX, 1], Inverse bias correction sqrt on HBM
        wd_factor_ptr (nl.ndarray): [P_MAX, 1], Weight decay factor on HBM
        numel (int): Number of elements in the parameter tensor
        beta1 (float): First moment decay factor (typically 0.9)
        beta2 (float): Second moment decay factor (typically 0.999)
        eps (float): Epsilon for numerical stability (typically 1e-8)
        decoupled_wd (bool): If True, use AdamW decoupled weight decay; else Adam L2 reg
        amsgrad (bool): If True, use AMSGrad variant with max second moment

    Returns:
        param_out (nl.ndarray): [N], Updated parameter tensor on HBM
        exp_avg_out (nl.ndarray): [N], Updated first moment tensor on HBM
        exp_avg_sq_out (nl.ndarray): [N], Updated second moment tensor on HBM
        max_exp_avg_sq_out (nl.ndarray): [N], Updated max second moment tensor on HBM (if amsgrad=True)

    Notes:
        - SPMD parallelization splits work across num_programs cores
        - Scalar tensors (step_size, inv_bc2_sqrt, wd_factor) have shape [P_MAX, 1]
        - For AdamW (decoupled_wd=True): param = param * wd_factor - update
        - For Adam (decoupled_wd=False): grad = grad + wd_factor * param, then param = param - update
        - AMSGrad uses max(max_exp_avg_sq, exp_avg_sq) as denominator

    Pseudocode:
        # Variable naming conventions:
        # param = parameter tensor
        # grad = gradient tensor
        # exp_avg = first moment estimate (m_t = E[grad])
        # exp_avg_sq = second moment estimate (v_t = E[grad^2])
        # max_exp_avg_sq = running max of v_t (AMSGrad only, v_hat_t)
        # step_size = lr / (1 - beta1^t), bias-corrected learning rate
        # inv_bc2_sqrt = 1 / sqrt(1 - beta2^t), bias correction for v_t
        # wd_factor = weight decay factor: (1 - lr*lambda) for AdamW, lambda for Adam

        one_minus_beta1 = 1.0 - beta1
        one_minus_beta2 = 1.0 - beta2

        # SPMD split work across cores
        per_core = (numel + num_cores - 1) // num_cores
        core_base = core_id * per_core
        core_n = min(per_core, numel - core_base)

        # Process tiles
        for tile_idx in range(num_tiles):
            # Load tile from HBM to SBUF
            param_tile = load(param_ptr[tile_offset:tile_offset+tile_size])
            grad_tile = load(grad_ptr[tile_offset:tile_offset+tile_size])
            exp_avg_tile = load(exp_avg_ptr[tile_offset:tile_offset+tile_size])
            exp_avg_sq_tile = load(exp_avg_sq_ptr[tile_offset:tile_offset+tile_size])

            # Adam L2 regularization (if not decoupled)
            if not decoupled_wd:
                grad_tile = grad_tile + wd_factor * param_tile

            # Update first moment
            exp_avg_tile = beta1 * exp_avg_tile + (1 - beta1) * grad_tile

            # Update second moment
            exp_avg_sq_tile = beta2 * exp_avg_sq_tile + (1 - beta2) * grad_tile^2

            # Compute denominator
            if amsgrad:
                max_exp_avg_sq_tile = max(max_exp_avg_sq_tile, exp_avg_sq_tile)
                denom = sqrt(max_exp_avg_sq_tile) * inv_bc2_sqrt + eps
            else:
                denom = sqrt(exp_avg_sq_tile) * inv_bc2_sqrt + eps

            # Compute update
            update = exp_avg_tile / denom * step_size

            # Update parameter
            if decoupled_wd:
                param_tile = param_tile * wd_factor - update
            else:
                param_tile = param_tile - update

            # Store tile back to HBM
            store(param_out[tile_offset:tile_offset+tile_size], param_tile)
            store(exp_avg_out[tile_offset:tile_offset+tile_size], exp_avg_tile)
            store(exp_avg_sq_out[tile_offset:tile_offset+tile_size], exp_avg_sq_tile)

        # Handle tail (remainder elements)
        if tail > 0:
            # Process tail with adjusted tile size
            ...
    """
    P_MAX = nl.tile_size.pmax
    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2

    param_out = nl.ndarray(param_ptr.shape, dtype=param_ptr.dtype, buffer=nl.shared_hbm)
    exp_avg_out = nl.ndarray(param_ptr.shape, dtype=param_ptr.dtype, buffer=nl.shared_hbm)
    exp_avg_sq_out = nl.ndarray(param_ptr.shape, dtype=param_ptr.dtype, buffer=nl.shared_hbm)
    max_exp_avg_sq_out = nl.ndarray(param_ptr.shape, dtype=param_ptr.dtype, buffer=nl.shared_hbm)

    # Load scalar tensors (shape [P_MAX, 1], same value across all partitions)
    step_size = nl.ndarray(step_size_ptr.shape, dtype=step_size_ptr.dtype, buffer=nl.sbuf)
    inv_bc2_sqrt = nl.ndarray(inv_bc2_sqrt_ptr.shape, dtype=inv_bc2_sqrt_ptr.dtype, buffer=nl.sbuf)
    weight_decay = nl.ndarray(wd_factor_ptr.shape, dtype=wd_factor_ptr.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=step_size, src=step_size_ptr)
    nisa.dma_copy(dst=inv_bc2_sqrt, src=inv_bc2_sqrt_ptr)
    nisa.dma_copy(dst=weight_decay, src=wd_factor_ptr)

    f_size = _F_SIZE_F32 if param_ptr.dtype == nl.float32 else _F_SIZE_BF16
    tile_size = P_MAX * f_size

    # SPMD split
    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    per_prog = (numel + num_progs - 1) // num_progs
    core_base = prog_id * per_prog
    core_n = min(per_prog, numel - core_base)
    core_n_aligned = (core_n // P_MAX) * P_MAX
    tail = core_n - core_n_aligned
    num_tiles = (core_n_aligned + tile_size - 1) // tile_size

    for tile_idx in nl.affine_range(num_tiles):
        offset = core_base + tile_idx * tile_size
        masked_f = min(f_size, (core_n_aligned - tile_idx * tile_size) // P_MAX)
        _adam_tile_step(
            param_ptr,
            grad_ptr,
            exp_avg_ptr,
            exp_avg_sq_ptr,
            max_exp_avg_sq_ptr,
            param_out,
            exp_avg_out,
            exp_avg_sq_out,
            max_exp_avg_sq_out,
            (P_MAX, masked_f),
            [[masked_f, P_MAX], [1, masked_f]],
            offset,
            step_size,
            inv_bc2_sqrt,
            weight_decay,
            beta1,
            beta2,
            eps,
            one_minus_beta1,
            one_minus_beta2,
            decoupled_wd,
            amsgrad,
        )

    if tail > 0:
        tail_offset = core_base + core_n_aligned
        _adam_tile_step(
            param_ptr,
            grad_ptr,
            exp_avg_ptr,
            exp_avg_sq_ptr,
            max_exp_avg_sq_ptr,
            param_out,
            exp_avg_out,
            exp_avg_sq_out,
            max_exp_avg_sq_out,
            (tail, 1),
            [[1, tail], [1, 1]],
            tail_offset,
            step_size[0:tail, 0:1],
            inv_bc2_sqrt[0:tail, 0:1],
            weight_decay[0:tail, 0:1],
            beta1,
            beta2,
            eps,
            one_minus_beta1,
            one_minus_beta2,
            decoupled_wd,
            amsgrad,
        )

    if amsgrad:
        return param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out
    return param_out, exp_avg_out, exp_avg_sq_out


@nki.jit
def adam_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    max_exp_avg_sq_ptr,
    step_size_ptr,
    inv_bc2_sqrt_ptr,
    wd_factor_ptr,
    numel,
    beta1,
    beta2,
    eps,
    amsgrad=False,
):
    """
    Adam optimizer kernel with L2 regularization.

    This kernel implements the Adam optimizer with L2 weight regularization.
    For decoupled weight decay (AdamW), use adamw_kernel instead.

    TODO: Specify intended usage range (e.g., parameter count, batch size)

    Dimensions:
        N: Number of elements in the parameter tensor (numel)
        P_MAX: Partition dimension size (128)

    Args:
        param_ptr (nl.ndarray): [N], Parameter tensor on HBM
        grad_ptr (nl.ndarray): [N], Gradient tensor on HBM
        exp_avg_ptr (nl.ndarray): [N], First moment estimate tensor on HBM
        exp_avg_sq_ptr (nl.ndarray): [N], Second moment estimate tensor on HBM
        max_exp_avg_sq_ptr (nl.ndarray): [N], Max second moment tensor on HBM (AMSGrad only)
        step_size_ptr (nl.ndarray): [P_MAX, 1], Bias-corrected step size on HBM
        inv_bc2_sqrt_ptr (nl.ndarray): [P_MAX, 1], Inverse bias correction sqrt on HBM
        wd_factor_ptr (nl.ndarray): [P_MAX, 1], Weight decay coefficient (lambda) on HBM
        numel (int): Number of elements in the parameter tensor
        beta1 (float): First moment decay factor (typically 0.9)
        beta2 (float): Second moment decay factor (typically 0.999)
        eps (float): Epsilon for numerical stability (typically 1e-8)
        amsgrad (bool): If True, use AMSGrad variant (default: False)

    Returns:
        param_out (nl.ndarray): [N], Updated parameter tensor on HBM
        exp_avg_out (nl.ndarray): [N], Updated first moment tensor on HBM
        exp_avg_sq_out (nl.ndarray): [N], Updated second moment tensor on HBM
        max_exp_avg_sq_out (nl.ndarray): [N], Updated max second moment (if amsgrad=True)

    Notes:
        - Uses L2 regularization: grad = grad + lambda * param
        - For decoupled weight decay (AdamW), use adamw_kernel instead

    Pseudocode:
        # Adam with L2 regularization
        grad = grad + wd_factor * param
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
        denom = sqrt(exp_avg_sq) * inv_bc2_sqrt + eps
        update = exp_avg / denom * step_size
        param = param - update
    """
    return _adam_kernel_impl(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        max_exp_avg_sq_ptr,
        step_size_ptr,
        inv_bc2_sqrt_ptr,
        wd_factor_ptr,
        numel,
        beta1,
        beta2,
        eps,
        decoupled_wd=False,
        amsgrad=amsgrad,
    )


@nki.jit
def adamw_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    max_exp_avg_sq_ptr,
    step_size_ptr,
    inv_bc2_sqrt_ptr,
    wd_factor_ptr,
    numel,
    beta1,
    beta2,
    eps,
    amsgrad=False,
):
    """
    AdamW optimizer kernel with decoupled weight decay.

    This kernel implements the AdamW optimizer with decoupled weight decay.
    For L2 regularization (Adam), use adam_kernel instead.

    TODO: Specify intended usage range (e.g., parameter count, batch size)

    Dimensions:
        N: Number of elements in the parameter tensor (numel)
        P_MAX: Partition dimension size (128)

    Args:
        param_ptr (nl.ndarray): [N], Parameter tensor on HBM
        grad_ptr (nl.ndarray): [N], Gradient tensor on HBM
        exp_avg_ptr (nl.ndarray): [N], First moment estimate tensor on HBM
        exp_avg_sq_ptr (nl.ndarray): [N], Second moment estimate tensor on HBM
        max_exp_avg_sq_ptr (nl.ndarray): [N], Max second moment tensor on HBM (AMSGrad only)
        step_size_ptr (nl.ndarray): [P_MAX, 1], Bias-corrected step size on HBM
        inv_bc2_sqrt_ptr (nl.ndarray): [P_MAX, 1], Inverse bias correction sqrt on HBM
        wd_factor_ptr (nl.ndarray): [P_MAX, 1], Weight decay factor (1 - lr*lambda) on HBM
        numel (int): Number of elements in the parameter tensor
        beta1 (float): First moment decay factor (typically 0.9)
        beta2 (float): Second moment decay factor (typically 0.999)
        eps (float): Epsilon for numerical stability (typically 1e-8)
        amsgrad (bool): If True, use AMSGrad variant (default: False)

    Returns:
        param_out (nl.ndarray): [N], Updated parameter tensor on HBM
        exp_avg_out (nl.ndarray): [N], Updated first moment tensor on HBM
        exp_avg_sq_out (nl.ndarray): [N], Updated second moment tensor on HBM
        max_exp_avg_sq_out (nl.ndarray): [N], Updated max second moment (if amsgrad=True)

    Notes:
        - Uses decoupled weight decay: param = param * wd_factor - update
        - For L2 regularization (Adam), use adam_kernel instead

    Pseudocode:
        # AdamW with decoupled weight decay
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
        denom = sqrt(exp_avg_sq) * inv_bc2_sqrt + eps
        update = exp_avg / denom * step_size
        param = param * wd_factor - update
    """
    return _adam_kernel_impl(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        max_exp_avg_sq_ptr,
        step_size_ptr,
        inv_bc2_sqrt_ptr,
        wd_factor_ptr,
        numel,
        beta1,
        beta2,
        eps,
        decoupled_wd=True,
        amsgrad=amsgrad,
    )
