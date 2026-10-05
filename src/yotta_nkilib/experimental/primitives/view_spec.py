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
ViewSpec: Lazy composition of view operations for TileStream.

Usage:
    from nkiprimitives.view_spec import view

    # Compose operations
    spec = view().slice(0, 0, 4).broadcast(1, 4)

    # Apply to TileStream
    ts_view = ts.view_logical(spec)
"""

from dataclasses import dataclass
from typing import Union

import nki.language as nl


# View operation classes
@dataclass
class Slice(nl.NKIObject):
    """Slice a dimension: narrow to [start, end)."""

    dim: int
    start: int
    end: int


@dataclass
class Select(nl.NKIObject):
    """Select single index, reducing dimensionality."""

    dim: int
    index: int


@dataclass
class ReshapeDim(nl.NKIObject):
    """Reshape one dimension into multiple."""

    dim: int
    shape: tuple[int, ...]


@dataclass
class Permute(nl.NKIObject):
    """Permute dimensions."""

    dims: tuple[int, ...]


@dataclass
class Broadcast(nl.NKIObject):
    """Broadcast dimension: stride=0, expand to size."""

    dim: int
    size: int


@dataclass
class Stride(nl.NKIObject):
    """Override stride for a dimension."""

    dim: int
    stride: int


@dataclass
class Expand(nl.NKIObject):
    """Add a new size-1 dimension (unsqueeze). Only valid on free dim.
    Merges existing f_tile with f_grid, then adds new dim as f_tile."""

    dim: int


ViewOp = Union[Slice, Select, ReshapeDim, Permute, Broadcast, Stride, Expand]


class ViewSpec(nl.NKIObject):
    """Lazily composes view operations.

    Operations are stored and applied when the spec is used with
    TileStream.view_logical(). Each method returns a new ViewSpec
    with the operation appended (immutable).
    """

    def __init__(self, ops: list[ViewOp] = None):
        self._ops = list(ops) if ops else []

    def slice(self, dim: int, start: int, end: int) -> 'ViewSpec':
        """Slice dimension to [start, end)."""
        return ViewSpec(self._ops + [Slice(dim, start, end)])

    def select(self, dim: int, index: int) -> 'ViewSpec':
        """Select single index, reducing dimensionality."""
        return ViewSpec(self._ops + [Select(dim, index)])

    def reshape_dim(self, dim: int, shape: tuple[int, ...]) -> 'ViewSpec':
        """Reshape one dimension into multiple."""
        return ViewSpec(self._ops + [ReshapeDim(dim, shape)])

    def permute(self, dims: tuple[int, ...]) -> 'ViewSpec':
        """Permute dimensions."""
        return ViewSpec(self._ops + [Permute(dims)])

    def broadcast(self, dim: int, size: int) -> 'ViewSpec':
        """Broadcast dimension with stride=0."""
        return ViewSpec(self._ops + [Broadcast(dim, size)])

    def stride(self, dim: int, stride: int) -> 'ViewSpec':
        """Override stride for dimension."""
        return ViewSpec(self._ops + [Stride(dim, stride)])

    def expand(self, dim: int) -> 'ViewSpec':
        """Add a new size-1 dimension at dim (unsqueeze). Only valid on free dim."""
        return ViewSpec(self._ops + [Expand(dim)])

    def get_ops(self) -> list[ViewOp]:
        """Get list of operations."""
        return self._ops


def view() -> ViewSpec:
    """Factory function to start a view specification chain.

    Usage:
        spec = view().slice(0, 0, 4).broadcast(1, 4)
    """
    return ViewSpec()
