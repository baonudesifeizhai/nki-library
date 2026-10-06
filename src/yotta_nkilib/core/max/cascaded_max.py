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

"""This kernel implements cascaded max reduction to find the global maximum value and its index across a tensor using multi-stage parallel reduction optimized for NeuronCore architecture."""

import math
from dataclasses import dataclass
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from .cascaded_max_utils import predicated_folded_load, reduce


@nki.jit
def cascaded_max(input_tensor: nl.NkiTensor) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Find global maximum value and its index across input tensor using cascaded reduction.

    This kernel implements a multi-stage parallel reduction algorithm optimized for
    NeuronCore architecture to efficiently find the maximum value and its corresponding
    index across large tensors.

    Dimensions:
        B: Batch size
        S: Sequence length
        V: Vocabulary size (dimension to reduce over)

    Args:
        input_tensor (nl.NkiTensor): [B, S, V] or [BxS, V], Input tensor in HBM

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]: A tuple containing:
            - max_values: [B, S, 1], Maximum values with original shape preserved
            - max_indices: [B, S, 1], Global indices of maximum values

    Notes:
        - Supports multi-dimensional input (automatically flattens batch dimensions)
        - Uses LNC sharding for parallel execution across multiple cores
        - Only supports LNC2 for BxS > 1, defaults to LNC1 for BxS == 1
        - TODO: Specify intended usage range (e.g., vocabulary size, batch size)

    Pseudocode:

        # Create configuration
        config = CascadedMaxConfig(input_tensor.shape, input_tensor.dtype)

        # Reshape input to 2D
        input_2d = input_tensor.reshape([BxS, V])

        # Perform cascaded max reduction
        max_vals, max_idxs = cascaded_max_core(input_2d, config)

        # Write results to HBM with proper sharding
        output_max_values[program_slice] = max_vals[local_slice]
        output_max_indices[program_slice] = max_idxs[local_slice]

        # Reshape to original shape
        return max_values.reshape(original_shape), max_indices.reshape(original_shape)
    """
    config = CascadedMaxConfig(input_tensor.shape, inp_dtype=input_tensor.dtype)
    val, indx = cascaded_max_core(input_tensor.reshape(config.inp_shape), config)

    BxS_size = config.per_lnc_BxS
    BxS_global = config.BxS
    n_programs, program_id = config.n_prgs, config.prg_id

    output_shape_t = (1, BxS_global)
    max_values = nl.ndarray(output_shape_t, dtype=val.dtype, buffer=nl.shared_hbm, name="max_values")
    max_indices = nl.ndarray(output_shape_t, dtype=config.index_dtype, buffer=nl.shared_hbm, name="max_indices")
    unpadded_bxs_size = min(BxS_size, BxS_global - BxS_size * program_id)
    offset = program_id * BxS_size
    nisa.dma_copy(dst=max_values[:, offset : offset + unpadded_bxs_size], src=val[:, 0:unpadded_bxs_size])
    nisa.dma_copy(dst=max_indices[:, offset : offset + unpadded_bxs_size], src=indx[:, 0:unpadded_bxs_size])

    return max_values.reshape(config.out_shape), max_indices.reshape(config.out_shape)


@dataclass
class CascadedMaxConfig(nl.NKIObject):
    """
    Configuration for cascaded max reduction algorithm.

    This class encapsulates all parameters needed for the cascaded max reduction,
    including input shape validation, LNC sharding configuration, and cascading
    algorithm constants.

    Args:
        inp_shape (Tuple): Shape of input tensor (can be multi-dimensional)
        inp_dtype: Data type of input tensor (required)

    Attributes:
        inp_dtype: Input tensor data type
        index_dtype: Data type for indices (nl.uint32)
        BxS (int): Combined batch and sequence dimensions
        vocab_size (int): Size of vocabulary dimension (last dimension)
        out_shape (list): Output shape with last dimension set to 1
        inp_shape (list): Reshaped input as [BxS, vocab_size]
        n_prgs (int): Number of logical cores
        prg_id (int): Current program ID
        per_lnc_BxS (int): Batch size per logical core
        per_lnc_BxS_unpadded (int): Unpadded batch size for current core
        n_stages (int): Number of cascading stages
        stage_free_size (int): Free dimension size per stage
        padded_vocab_size (int): Padded vocabulary size
    """

    def __init__(self, inp_shape: Tuple, inp_dtype=None) -> None:
        kernel_assert(inp_dtype is not None, "inp_dtype must be set")
        self.inp_dtype = inp_dtype
        self.index_dtype = nl.uint32

        self.BxS = reduce('mul', inp_shape[:-1], 1)
        self.vocab_size = inp_shape[-1]
        self.out_shape = []
        for dim_size in inp_shape[:-1]:
            self.out_shape.append(dim_size)
        self.out_shape.append(1)
        self.inp_shape = [self.BxS, self.vocab_size]

        """
        Handle LNC sharding configuration.

        Eventually hope to remove LNC info from config once NKI support allows
        for helper functions to be LNC agnostic.
        """
        shard_info = get_verified_program_sharding_info("max", (0, 1), 2)
        self.n_prgs = shard_info[1]
        self.prg_id = shard_info[2]
        if self.BxS == 1:
            self.prg_id = 0
            self.n_prgs = 1
        self.per_lnc_BxS = div_ceil(self.BxS, self.n_prgs)
        self.per_lnc_BxS_unpadded = min(self.per_lnc_BxS, self.BxS - self.per_lnc_BxS * self.prg_id)
        kernel_assert(
            self.BxS_dim_valid(),
            f"cascaded_max expects BxS per worker {self.per_lnc_BxS} <= max_pdim {nl.tile_size.pmax}",
        )
        kernel_assert(self.inp_shape_valid(), f"cascaded_max expects input to be 2D, got {len(inp_shape)}")

        const_info = self._calculate_cascading_constants()
        self.n_stages = const_info[0]
        self.stage_free_size = const_info[1]
        self.padded_vocab_size = const_info[2]

    def _calculate_cascading_constants(self) -> Tuple[int, int, int]:
        """
        Calculate cascading algorithm constants.

        Returns:
            Tuple[int, int, int]: (n_stages, chunk_size, padded_vocab_size)
        """
        P_MAX = nl.tile_size.pmax
        max_n_stages = math.floor(P_MAX // self.per_lnc_BxS)
        n_stages = min(max_n_stages, self.vocab_size // P_MAX)

        chunk_size = math.ceil(self.vocab_size / n_stages)
        padded_vocab_size = chunk_size * n_stages

        return n_stages, chunk_size, padded_vocab_size

    def inp_shape_valid(self) -> bool:
        return len(self.inp_shape) == 2

    def BxS_dim_valid(self) -> bool:
        return self.per_lnc_BxS <= nl.tile_size.pmax


def cascaded_max_core(
    inp: nl.NkiTensor,
    config: CascadedMaxConfig,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Compute global maximum value and its index from input tensor using cascaded reduction.

    This function implements a multi-stage parallel reduction to find the maximum value
    across a tensor and return both the max value and its original index. The computation
    is distributed across multiple NeuronCores using a folded loading strategy followed
    by hierarchical max reduction.

    Dimensions:
        BxS: Combined batch and sequence dimensions
        V: Vocabulary size (free dimension)
        n_stages: Number of reduction stages

    Args:
        inp (nl.NkiTensor): [BxS, V], Input tensor in HBM to find maximum from
        config (CascadedMaxConfig): Configuration object containing algorithm parameters

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]: A tuple containing:
            - max_values: [BxS_size * n_programs, 1], Maximum values found by each program
            - max_indices: [BxS_size * n_programs, 1], Global indices of maximum values

    Notes:
        - Input is loaded with folding across n_stages partitions
        - Uses tensor_reduce to find local max in each partition (up to 8 candidates)
        - Converts local indices to global indices using stage offsets
        - Performs grouped_reduce_max to find global maximum across all partitions

    Pseudocode:
        # Load and fold input data
        values = predicated_folded_load(inp, n_stages)

        # Find local maxima per partition
        local_max = tensor_reduce(values, op=maximum)

        # Find indices of local maxima
        local_indices = nc_find_index8(values, local_max)

        # Convert to global indices
        global_indices = local_indices + stage_offsets

        # Find global maximum across all partitions
        final_max, final_index = grouped_reduce_max(local_max, global_indices)

        return final_max, final_index
    """
    n_stages = config.n_stages
    BxS_size = config.per_lnc_BxS
    n_programs, program_id = config.n_prgs, config.prg_id
    stage_free_size = config.stage_free_size

    total_partition_dim = n_stages * BxS_size

    i_p = nl.ds(0, total_partition_dim)
    i_f = nl.ds(0, stage_free_size)

    values = predicated_folded_load(data_hbm=inp, fold_factor=n_stages, n_programs=n_programs, program_id=program_id)
    kernel_assert(
        values.shape == (total_partition_dim, stage_free_size),
        f"shape mismatch expected {(total_partition_dim, stage_free_size)} but got {values.shape}",
    )

    value = nl.ndarray((total_partition_dim, 1), dtype=inp.dtype)
    ind_buf = nl.ndarray((total_partition_dim, 8), dtype=config.index_dtype)

    nisa.tensor_reduce(op=nl.maximum, data=values[i_p, i_f], dst=value[...], axis=1)

    nisa.nc_find_index8(data=values[:, :], vals=value.broadcast(1, 8), dst=ind_buf[...])

    ind_offset = _repeat(n_stages, stage_free_size, BxS_size)

    """
    The index path MUST run in fp32, never the logit dtype.

    A vocab index can reach vocab_size-1 (e.g. 16383 per shard / 262143 global).
    bf16 has an 8-bit mantissa and only represents integers exactly up to 256;
    near a 16384-wide shard its spacing is 128, so a true argmax at the top of a
    shard (e.g. 16383) rounds UP to 16384 == shard width, one past the last valid
    index -> out-of-range token. fp32 (23-bit mantissa) is exact through 2^24.
    """
    idx_dtype = nl.float32

    i_identity_nx1_p, i_identity_nx1_x = nl.ds(0, total_partition_dim), nl.ds(0, total_partition_dim)
    # Separate identities so the value transpose keeps the logit dtype while the
    # index transpose runs in fp32.
    identity_val = nl.ndarray((total_partition_dim, total_partition_dim), dtype=value.dtype, buffer=nl.sbuf)
    identity_idx = nl.ndarray((total_partition_dim, total_partition_dim), dtype=idx_dtype, buffer=nl.sbuf)
    nisa.memset(dst=identity_val, value=1.0)
    nisa.memset(dst=identity_idx, value=1.0)
    pattern = [[1, total_partition_dim]]
    nisa.affine_select(
        dst=identity_val[...],
        pattern=pattern,
        channel_multiplier=-1,
        cmp_op=nl.equal,
        on_true_tile=identity_val[i_identity_nx1_p, i_identity_nx1_x],
        on_false_value=0.0,
    )
    nisa.affine_select(
        dst=identity_idx[...],
        pattern=pattern,
        channel_multiplier=-1,
        cmp_op=nl.equal,
        on_true_tile=identity_idx[i_identity_nx1_p, i_identity_nx1_x],
        on_false_value=0.0,
    )
    ind_t = nl.ndarray((1, total_partition_dim), dtype=idx_dtype, buffer=nl.psum)
    ind_buf_float = nl.ndarray((total_partition_dim, 1), dtype=idx_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(src=ind_buf[:, 0:1], dst=ind_buf_float)
    nisa.nc_matmul(dst=ind_t[...], stationary=ind_buf_float, moving=identity_idx, is_transpose=True)

    ind_shifted = nl.ndarray((1, total_partition_dim), dtype=idx_dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ind_shifted, data1=ind_t, op=nl.add, data2=ind_offset)

    value_psum = nl.ndarray((1, total_partition_dim), dtype=value.dtype, buffer=nl.psum)
    nisa.nc_matmul(dst=value_psum, stationary=value[:, 0:1], moving=identity_val, is_transpose=True)
    final_max, global_index_f = _grouped_reduce_max(value_psum, ind_shifted, fold_factor=BxS_size)

    """
    Value-cast the fp32 index to the integer index dtype before the HBM output.

    The main entry dma_copy's this into a uint32 buffer; a dma_copy across
    mismatched dtypes bit-reinterprets rather than value-casts, so a float index
    slot would surface as a garbage signed int. tensor_copy performs a true numeric
    cast; fp32 holds every vocab index exactly, so this is lossless.
    """
    global_index = nl.ndarray(global_index_f.shape, dtype=config.index_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(src=global_index_f, dst=global_index)
    return final_max, global_index


def _grouped_reduce_max(
    input_tensor: nl.NkiTensor,
    index: nl.NkiTensor,
    fold_factor: int,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Compute grouped maximum values and their corresponding indices from input tensor.

    This function divides the input tensor into groups (folds) and finds the maximum
    value within each group along with the index of that maximum value.

    Args:
        input_tensor (nl.NkiTensor): [b, n], Input tensor containing values to find maximums from
        index (nl.NkiTensor): [b, n], Index tensor with indices corresponding to each element
        fold_factor (int): Number of groups to divide each batch into (must evenly divide n)

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]: A tuple containing:
            - reduced_max: [b, fold_factor], Maximum value from each group
            - final_index: [b, fold_factor], Index of maximum value in each group

    Notes:
        - Reshapes input into [b, fold_factor, elts_per_fold] where elts_per_fold = n // fold_factor
        - For each group, finds maximum value and identifies which element(s) equal that maximum
        - Uses mask to extract corresponding index from index tensor
        - All intermediate computations performed in SBUF for efficiency
    """
    b, n = input_tensor.shape
    elts_per_fold = n // fold_factor
    reshaped_shape = (b, fold_factor, elts_per_fold)
    mask = nl.ndarray((reshaped_shape), dtype=nl.uint8, buffer=nl.sbuf)
    reduced_max = nl.ndarray((b, fold_factor), dtype=input_tensor.dtype, buffer=nl.sbuf)
    masked_index = nl.ndarray((b, n), dtype=index.dtype, buffer=nl.sbuf)
    final_index = nl.ndarray((b, fold_factor), dtype=index.dtype, buffer=nl.sbuf)

    nisa.tensor_reduce(dst=reduced_max, op=nl.maximum, data=input_tensor.reshape(reshaped_shape), axis=2)
    nisa.tensor_tensor(
        dst=mask,
        data1=input_tensor.reshape(reshaped_shape),
        op=nl.equal,
        data2=reduced_max.reshape((b, fold_factor, 1)).broadcast(2, elts_per_fold),
    )
    nisa.tensor_tensor(dst=masked_index, data1=mask.reshape((b, n)), data2=index, op=nl.multiply)
    nisa.tensor_reduce(dst=final_index, op=nl.maximum, data=masked_index.reshape(reshaped_shape), axis=2)
    return reduced_max, final_index


def _repeat(n_stages: int, stage_size: int, repeat_count: int) -> nl.NkiTensor:
    """
    Generate offset pattern for index calculation.

    Args:
        n_stages (int): Number of stages
        stage_size (int): Stage size
        repeat_count (int): Repeat count

    Returns:
        nl.NkiTensor: [1, repeat_count * n_stages], Offset tensor with iota pattern

    Notes:
        - Uses iota instruction to generate sequential offsets
        - Pattern creates stage-based offset structure
    """
    offset_r = nl.ndarray((1, repeat_count * n_stages), dtype=nl.uint32, buffer=nl.sbuf)
    pattern = [[0, 1], [0, repeat_count], [stage_size, n_stages]]
    nisa.iota(pattern=pattern, dst=offset_r)
    return offset_r
