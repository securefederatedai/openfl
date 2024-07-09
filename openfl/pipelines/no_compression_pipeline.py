# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NoCompressionPipeline module."""

from openfl.pipelines.pipeline import (
    Float32NumpyArrayToBytes,
    TransformationPipeline,
)


class NoCompressionPipeline(TransformationPipeline):
    """The data pipeline without any compression."""

    def __init__(self, **kwargs):
        """Initialize."""
        super().__init__(
            transformers=[Float32NumpyArrayToBytes()], **kwargs)
