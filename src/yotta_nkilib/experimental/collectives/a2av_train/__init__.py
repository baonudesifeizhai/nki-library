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
"""All-to-all-v training kernels for MoE.

Public kernels:
    permute_a2av            - capacity-bounded gather + all-to-all-v
    unpermute_a2av          - capacity-bounded all-to-all-v + scatter-add

Internal helpers (module-private, not part of the public API; see
``a2av_train_utils``):
    _exclusive_cumsum_u32(counts_hbm, EP)
    _write_a2av_v_metadata(metadata_hbm, counts_sb, sdispls_sb, H, EP)
    _validate_a2av_indices_counts(send_indices, counts, T, EP)
"""

from .permute_a2av import permute_a2av
from .unpermute_a2av import unpermute_a2av

__all__ = [
    "permute_a2av",
    "unpermute_a2av",
]
