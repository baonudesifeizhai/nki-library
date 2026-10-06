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
Software pipelining: explicit prefetch with prologue / steady-state /
epilogue, and triple buffering (buffer_count=3) for deeper overlap.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from nkilib_src.nkilib.experimental import neurotile as nt

# ============================================================================
# Kernels
# ============================================================================


@nki.jit
def explicit_prefetch_matmul(AT_hbm, B_hbm):
    """Matmul with explicit prologue / steady-state / epilogue prefetching.
    Hoists the LHS operand and prefetches the next RHS tile while computing
    the current one."""
    TILE_K, TILE_M, TILE_N = 128, 128, 128
    K, M = AT_hbm.shape
    _, N = B_hbm.shape

    C = nl.ndarray((M, N), dtype=AT_hbm.dtype, buffer=nl.shared_hbm)

    AT_tiles = nt.tiles(AT_hbm, tile_size=(TILE_K, TILE_M))
    B_tiles = nt.tiles(B_hbm, tile_size=(TILE_K, TILE_N))
    C_tiles = nt.tiles(C, tile_size=(TILE_M, TILE_N))
    K_TILES = AT_tiles.shape[0]

    for m in range(C_tiles.shape[0]):
        # Hoist: preload all K LHS tiles for this M row.
        A_tiles = AT_tiles[:, m].load()

        for n in range(C_tiles.shape[1]):
            B_stream = B_tiles[:, n].stream(buffer_count=2)
            acc = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(acc, 0.0)

            # PROLOGUE
            B_stream.load(0)

            # STEADY STATE: prefetch k+1 while computing k
            for k in nl.sequential_range(K_TILES - 1):
                B_stream.load(k + 1)
                B_tile = B_stream[k]
                A_tile = A_tiles[k]
                nisa.nc_matmul(acc, A_tile.data, B_tile.data)

            # EPILOGUE
            B_tile = B_stream[K_TILES - 1]
            A_tile = A_tiles[K_TILES - 1]
            nisa.nc_matmul(acc, A_tile.data, B_tile.data)

            sbuf = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(sbuf, acc)
            C_tiles[m, n].store(sbuf)

    return C


@nki.jit
def triple_buffer_pipeline(src):
    """Triple-buffered pipeline (buffer_count=3): load(k), compute(k-1),
    store(k-2) per steady-state iteration."""
    TILE_P, TILE_F = 128, 256

    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(TILE_P, TILE_F))
    dst_tiles = nt.tiles(dst, tile_size=(TILE_P, TILE_F))
    N_TILES = src_tiles.shape[0]

    src_stream = src_tiles[:, 0].stream(buffer_count=3)
    dst_stream = dst_tiles[:, 0].stream(buffer_count=3)

    # PROLOGUE: load tiles 0 + 1, compute tile 0
    src_stream.load(0)
    src_stream.load(1)
    in0 = src_stream[0]
    out0 = dst_stream[0]
    nisa.tensor_scalar(out0.data, in0.data, nl.multiply, 4.0)

    # STEADY STATE
    for k in nl.sequential_range(2, N_TILES):
        src_stream.load(k)
        in_tile = src_stream[k - 1]
        out_tile = dst_stream[k - 1]
        nisa.tensor_scalar(out_tile.data, in_tile.data, nl.multiply, 4.0)
        dst_stream.store(k - 2)

    # EPILOGUE: drain the last two stores
    dst_stream.store(0)
    in_last = src_stream[N_TILES - 1]
    out_last = dst_stream[N_TILES - 1]
    nisa.tensor_scalar(out_last.data, in_last.data, nl.multiply, 4.0)
    dst_stream.store(N_TILES - 1)
    dst_stream.store(N_TILES - 2)

    return dst


# ============================================================================
# Helpers
# ============================================================================


def to_device(t):
    import torch_xla.core.xla_model as xm

    return t.to(xm.xla_device())


def to_cpu(t):
    return t.cpu() if isinstance(t, torch.Tensor) else t


# ============================================================================
# Tests
# ============================================================================


def test_explicit_prefetch_matmul():
    torch.manual_seed(42)
    AT = torch.rand(256, 256, dtype=torch.bfloat16)
    B = torch.rand(256, 256, dtype=torch.bfloat16)
    result = explicit_prefetch_matmul(to_device(AT), to_device(B))
    expected = AT.T @ B
    assert torch.allclose(to_cpu(result).to(torch.bfloat16), expected, rtol=1e-3, atol=1e-3)
    print("explicit_prefetch_matmul: PASSED")


def test_triple_buffer_pipeline():
    torch.manual_seed(42)
    src = torch.randn(512, 256, dtype=torch.bfloat16)
    result = triple_buffer_pipeline(to_device(src))
    assert torch.allclose(to_cpu(result), src * 4.0, rtol=1e-3, atol=1e-3)
    print("triple_buffer_pipeline: PASSED")


def main():
    test_explicit_prefetch_matmul()
    test_triple_buffer_pipeline()


if __name__ == "__main__":
    main()
