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

"""Padding parameters.

Stores per-dimension ``(before, after)`` padding amounts and the padding mode.
Indexed ``0 = D, 1 = H, 2 = W`` (outermost to innermost spatial dimension).
"""

from typing import Tuple

import nki.language as nl

PadTuple = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


class PadParams(nl.NKIObject):
    """Padding specification for up to 3 spatial dimensions.

    Args:
        pad: Tuple of 3 ``(before, after)`` pairs.
        mode: Padding mode string.
    """

    def __init__(self, pad: PadTuple, mode: str):
        self.pad = pad
        self.mode = mode

    def before(self, dim: int) -> int:
        """Padding added before dimension *dim*."""
        return self.pad[dim][0]

    def after(self, dim: int) -> int:
        """Padding added after dimension *dim*."""
        return self.pad[dim][1]

    def total(self, dim: int) -> int:
        """Total padding on dimension *dim*."""
        return self.before(dim) + self.after(dim)
