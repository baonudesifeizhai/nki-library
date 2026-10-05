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
Mixed-precision matmul: load(dtype=) casts on DMA so SBUF holds bfloat16
inputs, nc_matmul accumulates in float32 PSUM, store with the right
output dtype. Two patterns: float32 output, bfloat16 output.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt

TILE_M, TILE_K, TILE_N = 128, 128, 512


# ============================================================================
# Kernels
# ============================================================================


@nki.jit
def mixed_precision_matmul_fp32_output(A_hbm, B_hbm):
    """Load fp32 source as bfloat16 (DMA cast), accumulate in fp32 PSUM,
    store fp32 output."""
    # A_hbm: [M, K] fp32,  B_hbm: [K, N] fp32  ->  C: [M, N] fp32
    M, K = A_hbm.shape
    _, N = B_hbm.shape
    C = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    A_tiles = nt.tiles(A_hbm, tile_size=(TILE_K, TILE_M))
    B_tiles = nt.tiles(B_hbm, tile_size=(TILE_K, TILE_N))
    C_tiles = nt.tiles(C, tile_size=(TILE_M, TILE_N))
    K_TILES = A_tiles.shape[0]

    for m in range(C_tiles.shape[0]):
        # load(dtype=nl.bfloat16) casts on DMA: source is fp32, slot is bf16.
        lhs_tiles = A_tiles[:, m].load(dtype=nl.bfloat16)
        for n in range(C_tiles.shape[1]):
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(acc, 0.0)
            for k in range(K_TILES):
                lhs_tile = lhs_tiles[k]  # tile: [128, 128] bf16
                rhs_tile = B_tiles[k, n].load(dtype=nl.bfloat16)
                nisa.nc_matmul(acc, lhs_tile.data, rhs_tile.data)
            sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(sbuf, acc)
            C_tiles[m, n].store(sbuf)

    return C


@nki.jit
def mixed_precision_matmul_bf16_output(A_hbm, B_hbm):
    """Same as fp32-output but the output is bf16 (cast on store)."""
    # A_hbm: [M, K] fp32,  B_hbm: [K, N] fp32  ->  C: [M, N] bf16
    M, K = A_hbm.shape
    _, N = B_hbm.shape
    C = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    A_tiles = nt.tiles(A_hbm, tile_size=(TILE_K, TILE_M))
    B_tiles = nt.tiles(B_hbm, tile_size=(TILE_K, TILE_N))
    C_tiles = nt.tiles(C, tile_size=(TILE_M, TILE_N))
    K_TILES = A_tiles.shape[0]

    for m in range(C_tiles.shape[0]):
        lhs_tiles = A_tiles[:, m].load(dtype=nl.bfloat16)
        for n in range(C_tiles.shape[1]):
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(acc, 0.0)
            for k in range(K_TILES):
                lhs_tile = lhs_tiles[k]
                rhs_tile = B_tiles[k, n].load(dtype=nl.bfloat16)
                nisa.nc_matmul(acc, lhs_tile.data, rhs_tile.data)
            # Cast PSUM fp32 -> SBUF bf16 then store.
            sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(sbuf, acc)
            C_tiles[m, n].store(sbuf)

    return C


# ============================================================================
# Helpers
# ============================================================================


def to_device(t):
    import torch_xla.core.xla_model as xm

    return t.to(xm.xla_device())


def to_cpu(t):
    return t.cpu() if isinstance(t, torch.Tensor) else t


def _setup(M, K, N):
    torch.manual_seed(42)
    A = torch.rand(M, K, dtype=torch.float32)
    B = torch.rand(K, N, dtype=torch.float32)
    return A.T.contiguous(), B


# ============================================================================
# Tests
# ============================================================================


def test_mixed_precision_fp32_output():
    M, K, N = 256, 256, 512
    AT, B = _setup(M, K, N)
    result = mixed_precision_matmul_fp32_output(to_device(AT), to_device(B))
    # Reference matmul in bf16 (matches hardware precision).
    A_bf16 = AT.T.to(torch.bfloat16).to(torch.float32)
    B_bf16 = B.to(torch.bfloat16).to(torch.float32)
    expected = A_bf16 @ B_bf16
    assert torch.allclose(to_cpu(result), expected, rtol=1e-4, atol=1e-4)
    print("mixed_precision_matmul_fp32_output: PASSED")


def test_mixed_precision_bf16_output():
    M, K, N = 256, 256, 512
    AT, B = _setup(M, K, N)
    result = mixed_precision_matmul_bf16_output(to_device(AT), to_device(B))
    expected = AT.T.to(torch.bfloat16) @ B.to(torch.bfloat16)
    # bf16 matmul has ~0.5 ULP rounding per accumulation; relax tolerance.
    assert torch.allclose(
        to_cpu(result).to(torch.float32),
        expected.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    print("mixed_precision_matmul_bf16_output: PASSED")


def main():
    test_mixed_precision_fp32_output()
    test_mixed_precision_bf16_output()


if __name__ == "__main__":
    main()
