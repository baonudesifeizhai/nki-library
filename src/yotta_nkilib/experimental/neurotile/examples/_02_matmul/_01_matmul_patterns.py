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
Matmul optimization patterns: baseline, hoisting LHS / RHS / both, and
.stream() double-buffering.
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
def matmul_baseline(lhsT_hbm, rhs_hbm):
    """Baseline matmul: both operands re-loaded inside the K loop.
    Highest HBM traffic; serves as the reference for hoisting savings."""
    # lhsT_hbm: [K, M],  rhs_hbm: [K, N]  ->  C: [M, N]
    K, M = lhsT_hbm.shape
    _, N = rhs_hbm.shape
    C = nl.ndarray((M, N), dtype=lhsT_hbm.dtype, buffer=nl.shared_hbm)

    lhsT_tiles = nt.tiles(lhsT_hbm, tile_size=(TILE_K, TILE_M))
    rhs_tiles = nt.tiles(rhs_hbm, tile_size=(TILE_K, TILE_N))
    C_tiles = nt.tiles(C, tile_size=(TILE_M, TILE_N))

    for m in range(C_tiles.shape[0]):
        for n in range(C_tiles.shape[1]):
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(acc, 0.0)
            for k in nl.affine_range(lhsT_tiles.shape[0]):
                lhs_tile = lhsT_tiles[k, m].load()  # tile: [128, 128]
                rhs_tile = rhs_tiles[k, n].load()  # tile: [128, 512]
                nisa.nc_matmul(acc, lhs_tile.data, rhs_tile.data)
            sbuf = nl.ndarray((TILE_M, TILE_N), dtype=acc.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(sbuf, acc)
            C_tiles[m, n].store(sbuf)

    return C


@nki.jit
def matmul_hoist_lhs(lhsT_hbm, rhs_hbm):
    """Hoist LHS column once per M-row; RHS streams from HBM each K step.
    Best when N is large (many RHS columns reuse the LHS column)."""
    # lhsT_hbm: [K, M],  rhs_hbm: [K, N]  ->  C: [M, N]
    K, M = lhsT_hbm.shape
    _, N = rhs_hbm.shape
    C = nl.ndarray((M, N), dtype=lhsT_hbm.dtype, buffer=nl.shared_hbm)

    lhsT_tiles = nt.tiles(lhsT_hbm, tile_size=(TILE_K, TILE_M))
    rhs_tiles = nt.tiles(rhs_hbm, tile_size=(TILE_K, TILE_N))
    C_tiles = nt.tiles(C, tile_size=(TILE_M, TILE_N))
    K_TILES = lhsT_tiles.shape[0]

    for m in range(C_tiles.shape[0]):
        # Hoist: preload all LHS K-tiles for this M row.
        lhs_tiles = lhsT_tiles[:, m].load()  # SBUF view: K tiles
        for n in range(C_tiles.shape[1]):
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(acc, 0.0)
            for k in range(K_TILES):
                lhs_tile = lhs_tiles[k]  # tile: [128, 128]
                rhs_tile = rhs_tiles[k, n].load()  # tile: [128, 512]
                nisa.nc_matmul(acc, lhs_tile.data, rhs_tile.data)
            sbuf = nl.ndarray((TILE_M, TILE_N), dtype=acc.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(sbuf, acc)
            C_tiles[m, n].store(sbuf)

    return C


@nki.jit
def matmul_hoist_rhs(lhsT_hbm, rhs_hbm):
    """Hoist RHS column once per N-column via dim=1 iteration; LHS streams.
    Best when M is large (many LHS rows reuse the RHS column)."""
    # lhsT_hbm: [K, M],  rhs_hbm: [K, N]  ->  C: [M, N]
    K, M = lhsT_hbm.shape
    _, N = rhs_hbm.shape
    C = nl.ndarray((M, N), dtype=lhsT_hbm.dtype, buffer=nl.shared_hbm)

    lhsT_tiles = nt.tiles(lhsT_hbm, tile_size=(TILE_K, TILE_M))
    rhs_tiles = nt.tiles(rhs_hbm, tile_size=(TILE_K, TILE_N))
    C_tiles = nt.tiles(C, tile_size=(TILE_M, TILE_N))
    K_TILES = lhsT_tiles.shape[0]

    # Column-first iteration: walk dim 1 of C_tiles.
    for n in range(C_tiles.shape[1]):
        # Hoist: preload all RHS K-tiles for this N column.
        rhs_loaded = rhs_tiles[:, n].load()  # SBUF view: K tiles
        for m in range(C_tiles.shape[0]):
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(acc, 0.0)
            for k in range(K_TILES):
                rhs_tile = rhs_loaded[k]  # tile: [128, 512]
                lhs_tile = lhsT_tiles[k, m].load()  # tile: [128, 128]
                nisa.nc_matmul(acc, lhs_tile.data, rhs_tile.data)
            sbuf = nl.ndarray((TILE_M, TILE_N), dtype=acc.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(sbuf, acc)
            C_tiles[m, n].store(sbuf)

    return C


@nki.jit
def matmul_hoist_both(lhsT_hbm, rhs_hbm):
    """Hoist both operands -- each tile DMA'd exactly once.
    Maximum on-chip footprint; minimum HBM traffic."""
    # lhsT_hbm: [K, M],  rhs_hbm: [K, N]  ->  C: [M, N]
    K, M = lhsT_hbm.shape
    _, N = rhs_hbm.shape
    C = nl.ndarray((M, N), dtype=lhsT_hbm.dtype, buffer=nl.shared_hbm)

    lhsT_tiles = nt.tiles(lhsT_hbm, tile_size=(TILE_K, TILE_M))
    rhs_tiles = nt.tiles(rhs_hbm, tile_size=(TILE_K, TILE_N))
    C_tiles = nt.tiles(C, tile_size=(TILE_M, TILE_N))
    K_TILES = lhsT_tiles.shape[0]

    for m in range(C_tiles.shape[0]):
        lhs_tiles = lhsT_tiles[:, m].load()
        for n in range(C_tiles.shape[1]):
            rhs_loaded = rhs_tiles[:, n].load()
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(acc, 0.0)
            for k in range(K_TILES):
                lhs_tile = lhs_tiles[k]
                rhs_tile = rhs_loaded[k]
                nisa.nc_matmul(acc, lhs_tile.data, rhs_tile.data)
            sbuf = nl.ndarray((TILE_M, TILE_N), dtype=acc.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(sbuf, acc)
            C_tiles[m, n].store(sbuf)

    return C


@nki.jit
def matmul_hoist_stream(lhsT_hbm, rhs_hbm):
    """Hoist LHS, stream RHS with double-buffering -- DMA/compute overlap."""
    # lhsT_hbm: [K, M],  rhs_hbm: [K, N]  ->  C: [M, N]
    K, M = lhsT_hbm.shape
    _, N = rhs_hbm.shape
    C = nl.ndarray((M, N), dtype=lhsT_hbm.dtype, buffer=nl.shared_hbm)

    lhsT_tiles = nt.tiles(lhsT_hbm, tile_size=(TILE_K, TILE_M))
    rhs_tiles = nt.tiles(rhs_hbm, tile_size=(TILE_K, TILE_N))
    C_tiles = nt.tiles(C, tile_size=(TILE_M, TILE_N))
    K_TILES = lhsT_tiles.shape[0]

    for m in range(C_tiles.shape[0]):
        lhs_tiles = lhsT_tiles[:, m].load()
        for n in range(C_tiles.shape[1]):
            # Stream n'th RHS column with double buffering -- each buffer is one tile.
            rhs_stream = rhs_tiles[:, n].stream(buffer_count=2)
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(acc, 0.0)
            for k in nl.affine_range(K_TILES):
                lhs_tile = lhs_tiles[k]
                rhs_tile = rhs_stream.load(k)  # tile: [128, 512]
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
    lhs = torch.rand(M, K, dtype=torch.bfloat16)
    rhs = torch.rand(K, N, dtype=torch.bfloat16)
    return lhs.T.contiguous(), rhs, lhs @ rhs


def _check(kernel_fn, M, K, N, name):
    lhsT, rhs, expected = _setup(M, K, N)
    result = kernel_fn(to_device(lhsT), to_device(rhs))
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=1e-3, atol=1e-3)
    print(f"{name}: PASSED")


# ============================================================================
# Tests
# ============================================================================


def test_baseline():
    _check(matmul_baseline, 256, 256, 512, "baseline")


def test_hoist_lhs():
    _check(matmul_hoist_lhs, 256, 256, 512, "hoist_lhs")


def test_hoist_rhs():
    _check(matmul_hoist_rhs, 256, 256, 512, "hoist_rhs")


def test_hoist_both():
    _check(matmul_hoist_both, 256, 256, 512, "hoist_both")


def test_hoist_stream():
    _check(matmul_hoist_stream, 256, 256, 512, "hoist_stream")


def main():
    test_baseline()
    test_hoist_lhs()
    test_hoist_rhs()
    test_hoist_both()
    test_hoist_stream()


if __name__ == "__main__":
    main()
