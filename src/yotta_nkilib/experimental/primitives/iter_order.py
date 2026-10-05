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

"""Iteration order strategies for TileStream."""

from __future__ import annotations

from typing import Tuple, Union

import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from .view_spec import Permute, ReshapeDim, ViewSpec


class RowMajor(nl.NKIObject):
    """Row-major iteration: rightmost dimension changes fastest."""

    def advance(self, cur_pos: list, grid: Tuple[int, ...]) -> None:
        """Advance to next position in row-major order."""
        ndim = len(grid)
        cur_pos[ndim - 1] = cur_pos[ndim - 1] + 1
        for i in range(ndim - 1, 0, -1):
            if cur_pos[i] >= grid[i]:
                cur_pos[i] = 0
                cur_pos[i - 1] = cur_pos[i - 1] + 1
            else:
                break

    def reset(self) -> None:
        pass

    def get_iter_grid(self, grid: Tuple[int, ...]) -> Tuple[int, ...]:
        return grid

    def to_grid_pos(self, iter_pos, grid: Tuple[int, ...]):
        return iter_pos


class ColMajor(nl.NKIObject):
    """Column-major iteration: partition dimension (index 0) changes fastest."""

    def advance(self, cur_pos: list, grid: Tuple[int, ...]) -> None:
        """Advance to next position in column-major order."""
        ndim = len(grid)
        if ndim < 2:
            if ndim == 1:
                cur_pos[0] = cur_pos[0] + 1
            return

        cur_pos[0] = cur_pos[0] + 1

        if cur_pos[0] >= grid[0]:
            cur_pos[0] = 0
            for i in range(ndim - 1, 0, -1):
                cur_pos[i] = cur_pos[i] + 1
                if cur_pos[i] >= grid[i]:
                    cur_pos[i] = 0
                    if i == 1:
                        cur_pos[0] = grid[0]
                        break
                else:
                    break

    def reset(self) -> None:
        pass

    def get_iter_grid(self, grid: Tuple[int, ...]) -> Tuple[int, ...]:
        return grid

    def to_grid_pos(self, iter_pos, grid: Tuple[int, ...]):
        return iter_pos


class ViewOrder(nl.NKIObject):
    """Iteration order defined by ViewSpec operations on the tile grid.

    Uses reshape_dim and permute to transform the tile grid into a virtual
    iteration space. Convention: rightmost virtual dim is fastest-changing.

    Only reshape_dim and permute are supported. All other ViewSpec ops
    (slice, select, broadcast, stride, expand) will raise an error.

    Args:
        view: ViewSpec chain containing only reshape_dim and permute ops.

    Examples:
        # Simple permute: dim 1 fastest, dim 0 slowest
        ViewOrder(ViewSpec().permute(dims=(0, 2, 1)))

        # Split dim 0 into factors, swap iteration of factors 1 and 2
        ViewOrder(ViewSpec()
            .reshape_dim(0, (T_div_4, 4, 4))
            .permute(dims=(0, 2, 1)))
    """

    def __init__(self, view: ViewSpec) -> None:
        for op in view.get_ops():
            if not isinstance(op, (ReshapeDim, Permute)):
                kernel_assert(False, f"ViewOrder only supports reshape_dim and permute, got {type(op).__name__}")
        self._view = view
        self._built = False
        self._has_splits = False
        self._virt_pos = None
        self._vmap_orig = None
        self._vmap_iter = None
        self._real_dim_keys = None
        self._real_dim_vdims = None
        self._perm = None
        self._inv_perm = None

    def _ensure_built(self, grid: Tuple[int, ...]) -> None:
        """Process ViewSpec ops against the tile grid to build internal state."""
        if self._built:
            return

        vmap = []
        for d in range(len(grid)):
            vmap.append((d, None))

        perm = None
        for op in self._view.get_ops():
            if isinstance(op, ReshapeDim):
                d = op.dim
                real_dim = vmap[d][0]
                new_entries = []
                for i in range(len(op.shape)):
                    new_entries.append((real_dim, op.shape[i]))
                vmap = vmap[:d] + new_entries + vmap[d + 1 :]
                self._has_splits = True
            elif isinstance(op, Permute):
                perm = tuple(op.dims)

        self._vmap_orig = list(vmap)

        self._real_dim_keys = []
        self._real_dim_vdims = []
        for vi in range(len(vmap)):
            real_dim = vmap[vi][0]
            found = -1
            for ri in range(len(self._real_dim_keys)):
                if self._real_dim_keys[ri] == real_dim:
                    found = ri
                    break
            if found < 0:
                self._real_dim_keys.append(real_dim)
                self._real_dim_vdims.append([vi])
            else:
                self._real_dim_vdims[found].append(vi)

        if perm is not None:
            self._perm = perm
            inv = []
            for _ in range(len(perm)):
                inv.append(0)
            for i in range(len(perm)):
                inv[perm[i]] = i
            self._inv_perm = tuple(inv)
            permuted = []
            for i in range(len(perm)):
                permuted.append(vmap[perm[i]])
            self._vmap_iter = permuted
        else:
            self._vmap_iter = list(vmap)

        self._built = True

    def reset(self) -> None:
        """Reset internal virtual state for split iteration."""
        self._virt_pos = None

    def get_iter_grid(self, grid: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return the virtual iteration grid (post-permute)."""
        self._ensure_built(grid)
        if not self._has_splits and self._perm is None:
            return grid
        result = []
        for vi in range(len(self._vmap_iter)):
            entry = self._vmap_iter[vi]
            if entry[1] is not None:
                result.append(entry[1])
            else:
                result.append(grid[entry[0]])
        return tuple(result)

    def to_grid_pos(self, iter_pos, grid: Tuple[int, ...]):
        """Convert iteration pos to real grid pos. Returns None if any dim OOB."""
        self._ensure_built(grid)
        if not self._has_splits and self._perm is None:
            return iter_pos

        if not self._has_splits and self._perm is not None:
            pos = []
            for _ in range(len(grid)):
                pos.append(0)
            for vi in range(len(self._vmap_orig)):
                real_dim = self._vmap_orig[vi][0]
                pos[real_dim] = iter_pos[self._inv_perm[vi]]
            return tuple(pos)

        if self._perm is not None:
            orig_pos = []
            for j in range(len(iter_pos)):
                orig_pos.append(iter_pos[self._inv_perm[j]])
        else:
            orig_pos = iter_pos

        pos = []
        for _ in range(len(grid)):
            pos.append(0)
        for ri in range(len(self._real_dim_keys)):
            real_dim = self._real_dim_keys[ri]
            vdims = self._real_dim_vdims[ri]
            idx = 0
            for vi in vdims:
                bound = self._vmap_orig[vi][1]
                if bound is None:
                    bound = grid[real_dim]
                idx = idx * bound + orig_pos[vi]
            pos[real_dim] = idx
            if idx >= grid[real_dim]:
                return None
        return tuple(pos)

    def advance(self, cur_pos: list, grid: Tuple[int, ...]) -> None:
        """Advance to next position in the virtual iteration grid."""
        self._ensure_built(grid)
        if not self._has_splits and self._perm is None:
            ndim = len(grid)
            cur_pos[ndim - 1] = cur_pos[ndim - 1] + 1
            for i in range(ndim - 1, 0, -1):
                if cur_pos[i] >= grid[i]:
                    cur_pos[i] = 0
                    cur_pos[i - 1] = cur_pos[i - 1] + 1
                else:
                    break
            return

        n = len(self._vmap_iter)
        if self._virt_pos is None:
            self._virt_pos = []
            for _ in range(n):
                self._virt_pos.append(0)
        vp = self._virt_pos
        vp[n - 1] = vp[n - 1] + 1
        for i in range(n - 1, 0, -1):
            if vp[i] >= grid[i]:
                vp[i] = 0
                vp[i - 1] = vp[i - 1] + 1
            else:
                break
        for i in range(n):
            cur_pos[i] = vp[i]


def DimOrder(dims: Tuple[int, ...]) -> ViewOrder:
    """Create a ViewOrder that permutes grid dimensions.

    Shorthand for ViewOrder(ViewSpec().permute(dims)).
    Only supports permutation — use ViewOrder directly for reshape_dim.

    Args:
        dims: Permutation of grid dimension indices.

    Examples:
        DimOrder((0, 2, 1))   # dim 1 fastest, dim 0 slowest
        DimOrder((1, 0))      # column-major for 2D grid
    """
    return ViewOrder(ViewSpec().permute(dims=dims))


IterOrder = Union[ColMajor, RowMajor, ViewOrder]
