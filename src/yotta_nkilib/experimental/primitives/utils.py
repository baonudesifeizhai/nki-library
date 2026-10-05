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
NKI Primitives utility functions.
"""

from ...core.utils.kernel_helpers import get_floor_aligned_size


def max_tile(total: int, unit: int, max_size: int) -> int:
    """
    Compute largest tile size that fits constraints.

    Returns the largest tile size that is:
    1. Aligned to unit (divisible by unit)
    2. Does not exceed max_size
    3. Does not exceed total

    Args:
        total (int): Total size to tile (e.g., h1 * bxs)
        unit (int): Alignment unit (e.g., bxs for F dimension)
        max_size (int): Maximum allowed tile size (e.g., F_MAX = 512)

    Returns:
        int: Largest valid tile size

    Pseudocode:
        result = min(floor_align(max_size, unit), floor_align(total, unit))
        return result
    """
    return min(get_floor_aligned_size(max_size, unit), get_floor_aligned_size(total, unit))
