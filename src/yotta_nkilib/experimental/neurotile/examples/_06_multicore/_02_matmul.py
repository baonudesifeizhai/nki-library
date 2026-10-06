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
Multi-core matmul with block sharding on the M dimension. Shard every
view that touches M (output and lhsT); rhs stays unsharded since each
core needs the full K x N for its reduction.
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
def matmul_single_core(lhsT, rhs):
    """Single-core matmul baseline."""
    # lhsT: [K, M],  rhs: [K, N]  ->  C: [M, N]
    K, M = lhsT.shape
    _, N = rhs.shape
    C = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    lhsT_tiles = nt.tiles(lhsT, tile_size=(TILE_K, TILE_M))
    rhs_tiles = nt.tiles(rhs, tile_size=(TILE_K, TILE_N))
    C_tiles = nt.tiles(C, tile_size=(TILE_M, TILE_N))

    for m in range(C_tiles.shape[0]):
        for n in range(C_tiles.shape[1]):
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(acc, 0.0)
            for k in nl.affine_range(lhsT_tiles.shape[0]):
                lhs_tile = lhsT_tiles[k, m].load()
                rhs_tile = rhs_tiles[k, n].load()
                nisa.nc_matmul(acc, lhs_tile.data, rhs_tile.data)
            sbuf = nl.ndarray((TILE_M, TILE_N), dtype=acc.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(sbuf, acc)
            C_tiles[m, n].store(sbuf)

    return C


@nki.jit
def matmul_block_sharded(lhsT, rhs):
    """Block-sharded matmul -- each core owns a contiguous half of M."""
    # lhsT: [K, M],  rhs: [K, N]  ->  C: [M, N]   (M tiles split across cores)
    K, M = lhsT.shape
    _, N = rhs.shape
    C = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    r = nt.block_range(
        rank=nl.program_id(0),
        num_shards=nl.num_programs(0),
        total=nt.ceiling_div(M, TILE_M),
    )

    # lhsT has M on dim 1; output has M on dim 0. Both shard on the M axis.
    lhsT_tiles = nt.tiles(lhsT, tile_size=(TILE_K, TILE_M))[:, r]
    rhs_tiles = nt.tiles(rhs, tile_size=(TILE_K, TILE_N))  # not sharded -- full K x N
    C_tiles = nt.tiles(C, tile_size=(TILE_M, TILE_N))[r, :]

    for m in range(C_tiles.shape[0]):
        for n in range(C_tiles.shape[1]):
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(acc, 0.0)
            for k in nl.affine_range(lhsT_tiles.shape[0]):
                lhs_tile = lhsT_tiles[k, m].load()
                rhs_tile = rhs_tiles[k, n].load()
                nisa.nc_matmul(acc, lhs_tile.data, rhs_tile.data)
            sbuf = nl.ndarray((TILE_M, TILE_N), dtype=acc.dtype, buffer=nl.sbuf)
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
    A = torch.rand(M, K, dtype=torch.bfloat16)
    B = torch.rand(K, N, dtype=torch.bfloat16)
    return A.T.contiguous(), B, A @ B


# ============================================================================
# Tests
# ============================================================================


def test_single_core():
    M, K, N = 256, 256, 512
    AT, B, expected = _setup(M, K, N)
    result = matmul_single_core(to_device(AT), to_device(B))
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=1e-3, atol=1e-3)
    print("matmul_single_core: PASSED")


def test_block_sharded_lnc2():
    M, K, N = 256, 256, 512
    AT, B, expected = _setup(M, K, N)
    result = matmul_block_sharded[2](to_device(AT), to_device(B))
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=1e-3, atol=1e-3)
    print("matmul_block_sharded (LNC=2): PASSED")


def test_block_sharded_lnc2_larger():
    M, K, N = 512, 512, 1024
    AT, B, expected = _setup(M, K, N)
    result = matmul_block_sharded[2](to_device(AT), to_device(B))
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=1e-3, atol=1e-3)
    print("matmul_block_sharded LNC=2 (larger): PASSED")


def main():
    test_single_core()
    test_block_sharded_lnc2()
    test_block_sharded_lnc2_larger()


if __name__ == "__main__":
    main()
