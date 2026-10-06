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
"""Torch references for 03_streaming.py."""

_TILE_P = 128
_TILE_F = 128


def stream_input_output_scale_torch_ref(src):
    return src * 2.0


def stream_with_dtype_conversion_torch_ref(src):
    # Tutorial expected: src.view(4, 128, 8, 128).sum(dim=0).reshape(128, 1024)
    return src.view(4, 128, 8, 128).sum(dim=0).reshape(128, 1024)


def stream_dim_walks_columns_torch_ref(src):
    return src * 2.0


def stream_2d_blocks_torch_ref(src):
    return src * 2.0
