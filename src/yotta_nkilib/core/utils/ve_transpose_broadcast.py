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


import nki.isa as nisa

from .stream_shuffle_broadcast import stream_shuffle_broadcast


def ve_transpose_broadcast(src, dst, num_free, num_par):
    """Transpose a column from partition→free dimension via vector-engine nc_transpose, then broadcast.

    Transposes src (num_free, 1) in partition dim to dst[0:1, 0:num_free] in free dim
    using 32-element chunks (vector-engine limit), then broadcasts partition 0
    across all num_par partitions: dst[0:num_par, 0:num_free].

    Args:
        src: Source SBUF tensor slice with shape (num_free, 1) — column to transpose.
        dst: Destination SBUF tensor with shape (num_par, num_free) — receives broadcast result.
        num_free: Number of elements to transpose (free dimension size).
        num_par: Number of partitions to broadcast across.
    """
    _CHUNK = 32
    for chunk_i in range(0, num_free, _CHUNK):
        chunk_end = min(chunk_i + _CHUNK, num_free)
        nisa.nc_transpose(
            dst[0:1, chunk_i:chunk_end],
            src[chunk_i:chunk_end],
            engine=nisa.vector_engine,
        )
    stream_shuffle_broadcast(
        src=dst[0:1, :num_free],
        dst=dst[:num_par, :num_free],
    )
