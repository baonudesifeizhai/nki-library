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

"""Unstable argsort of input data using Vector Engine."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert

_ELEMS_PER_PASS = 8


@nki.jit
def argsort_unstable(data, descending=False, output_in_sbuf=False):
    """
    Perform unstable argsort on 1D input buffer. Elements with equal values
    may appear in any order relative to their original positions.

    For example:

        data = [5, 2, 5, 3]

        Pass 0 (ascending mode):
          max8 vals = [5, 5, 3, 2, ...]
          nc_match_replace8 matches: pos 0, pos 2, pos 3, pos 1
          reversed output indices: [1, 3, 2, 0]

        Result: indices = [1, 3, 2, 0]  (values in order: 2, 3, 5, 5)
        Indices [2, 0] corresponding to values [5, 5] are not in original order.

    Dimensions:
        N: Number of elements to sort. Must be a multiple of 8.

    Args:
        data (nl.ndarray): [1, N] int32/float32 tensor in HBM or SBUF.
        descending (bool): When True, return indices for descending order. Defaults to ascending.
        output_in_sbuf (bool): When True, return SBUF output. Defaults to HBM output.

    Returns:
        indices (nl.ndarray): [1, N] uint32 tensor in HBM or SBUF containing the argsort indices.
    """

    # Extract shapes, validate
    input_1d = len(data.shape) == 1
    if input_1d:
        data = data.reshape((1, data.shape[0]))
    N = data.shape[1]
    kernel_assert(data.shape[0] == 1, f"Expected data.shape=[1, N], got {data.shape}")
    kernel_assert(N >= _ELEMS_PER_PASS, f"Expected N >= {_ELEMS_PER_PASS}, got {N}")
    kernel_assert(N % _ELEMS_PER_PASS == 0, f"Expected N divisible by {_ELEMS_PER_PASS}, got {N}")
    num_passes = N // _ELEMS_PER_PASS

    # Load data if not already in SBUF
    if data.buffer != nl.sbuf:
        data_sb = nl.ndarray(data.shape, data.dtype, buffer=nl.sbuf)
        nisa.dma_copy(data_sb, data)
    else:
        data_sb = data

    # Argsort data_sb using N/8 max8 + nc_match_replace8 passes
    argsort_indices_sb = nl.ndarray((1, N), dtype=nl.uint32, buffer=nl.sbuf)
    for pass_idx in nl.sequential_range(num_passes):
        val_buf = nl.ndarray((1, _ELEMS_PER_PASS), dtype=nl.float32)
        nisa.max8(dst=val_buf, src=data_sb)

        # dst_idx AP is reversed for ascending sort
        idx_pattern = [[N, 1], [1 if descending else -1, _ELEMS_PER_PASS]]
        idx_offset = _ELEMS_PER_PASS * pass_idx if descending else _ELEMS_PER_PASS * (num_passes - pass_idx) - 1
        nisa.nc_match_replace8(
            dst=data_sb,
            data=data_sb,
            vals=val_buf,
            imm=float('-inf'),
            dst_idx=argsort_indices_sb.ap(
                pattern=idx_pattern,
                offset=idx_offset,
            ),
        )

    # Optionally store output to HBM
    if output_in_sbuf:
        return argsort_indices_sb.reshape((N,)) if input_1d else argsort_indices_sb
    else:
        out_shape = (N,) if input_1d else (1, N)
        argsort_indices_hbm = nl.ndarray(out_shape, dtype=nl.uint32, buffer=nl.shared_hbm, name="argsort_indices_hbm")
        nisa.dma_copy(argsort_indices_hbm.reshape((1, N)) if input_1d else argsort_indices_hbm, argsort_indices_sb)
        return argsort_indices_hbm
