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

"""RandomShiftPipeline module."""

import numpy as np

from openfl.pipelines.pipeline import Float32NumpyArrayToBytes, TransformationPipeline, Transformer


class RandomShiftTransformer(Transformer):
    """Random Shift Transformer."""

    def __init__(self):
        """Initialize."""
        self.lossy = False

        return

    def forward(self, data, **kwargs):
        """Forward pass - compress data.

        Implement the data transformation.

        Args:
            data: an array value from a model tensor_dict

        Returns:
            transformed_data:
            metadata:

        """
        shape = data.shape
        random_shift = np.random.uniform(low=-20, high=20, size=shape).astype(np.float32)
        transformed_data = data + random_shift

        # construct metadata
        metadata = {"int_to_float": {}, "int_list": list(shape)}
        for idx, val in enumerate(random_shift.flatten(order="C")):
            metadata["int_to_float"][idx] = val

        return transformed_data, metadata

    def backward(self, data, metadata, **kwargs):
        """Backward pass - Decompress data.

        Implement the data transformation needed when going the oppposite
        direction to the forward method.

        Args:
            data:
            metadata:

        Returns:
            transformed_data:
        """
        shape = tuple(metadata["int_list"])
        # this is an awkward use of the metadata into to float dict, usually
        # it will trully be treated as a dict. Here (and in 'forward' above)
        # we use it essentially as an array.
        shift = np.reshape(
            np.array(
                [metadata["int_to_float"][idx] for idx in range(len(metadata["int_to_float"]))]
            ),
            newshape=shape,
            order="C",
        )
        return data - shift


class RandomShiftPipeline(TransformationPipeline):
    """Random Shift Pipeline."""

    def __init__(self, **kwargs):
        """Initialize."""
        transformers = [RandomShiftTransformer(), Float32NumpyArrayToBytes()]
        super().__init__(transformers=transformers)
