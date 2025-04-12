# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Pipeline module."""

import numpy as np


class Transformer:
    """Base class for data transformation.

    This class defines the basic structure of a data transformer. It should be
    subclassed when implementing new types of data transformations.
    """

    def forward(self, data, **kwargs):
        """Forward pass data transformation.

        Implement the data transformation.
        This method should be overridden by all subclasses.

        Args:
            data: The data to be transformed.
            **kwargs: Additional parameters to pass to the function

        Returns:
            transformed_data: The transformed data.
            metadata: The metadata for the transformation.
        """
        raise NotImplementedError

    def backward(self, data, metadata, **kwargs):
        """Backward pass data transformation.

        Implement the data transformation needed when going the opposite
        direction to the forward method.
        This method should be overridden by all subclasses.

        Args:
            data: The transformed data.
            metadata: The metadata for the transformation.
            **kwargs: Additional keyword arguments for the transformation.

        Returns:
            transformed_data: The original data before the transformation.
        """
        raise NotImplementedError


class Float32NumpyArrayToBytes(Transformer):
    """Transformer class for converting float32 Numpy arrays to bytes
    arrays."""

    def __init__(self):
        """Initialize Float32NumpyArrayToBytes."""
        self.lossy = False

    def forward(self, data: np.ndarray, **kwargs):
        """Convert a float32 Numpy array to bytes.

        Args:
            data: The float32 Numpy array to be converted.
            **kwargs: Additional keyword arguments for the conversion.

        Returns:
            data_bytes: The data converted to bytes.
            metadata: The metadata for the conversion.
        """
        # TODO: Warn when this casting is being performed.
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        array_shape = data.shape
        # Better call it array_shape?
        metadata = {"int_list": list(array_shape)}
        data_bytes = data.tobytes(order="C")
        return data_bytes, metadata

    def backward(self, data, metadata, **kwargs):
        """Convert bytes back to a float32 Numpy array.

        Args:
            data: The data in bytes.
            metadata: The metadata for the conversion.

        Returns:
            The data converted back to a float32 Numpy array.
        """
        array_shape = tuple(metadata["int_list"])
        flat_array = np.frombuffer(data, dtype=np.float32)
        # For integer parameters we probably should unpack arrays
        # with shape (1,)
        return np.reshape(flat_array, newshape=array_shape, order="C")


class TransformationPipeline:
    """Data Transformer Pipeline Class.

    This class is a pipeline of transformers that transform data in a
    sequential manner.

    A sequential pipeline to transform (e.x. compress) data (e.x. layer of
    model_weights) as well as return metadata (if needed) for the
    reconstruction process carried out by the backward method.

    Attributes:
        transformers (list): The list of transformers in the pipeline.
    """

    def __init__(self, transformers, **kwargs):
        """Initialize TransformationPipeline.

        Args:
            transformers (list): The list of transformers in the pipeline.
            **kwargs: Additional keyword arguments for the pipeline.
        """
        self.transformers = transformers

    def forward(self, data, **kwargs):
        """Forward pass of pipeline data transformer.

        Args:
            data: The data to be transformed.
            **kwargs: Additional keyword arguments for the transformation.

        Returns:
            data: The transformed data.
            transformer_metadata: The metadata for the transformation.
        """
        # dataformat::numpy::float.32
        # model proto:: a collection of tensor_dict proto
        # protobuff::-> a layer of weights
        # input::tensor_dict:{"layer1":np.array(float32, [128,3,28,28]),
        # "layer2": np.array()}
        # output::meta data::numpy::int array
        # (data, transformer_metadata)::(float32, dictionary o
        # key+float32 values)
        # input:: numpy_data (float32)
        # input:: (data(bytes), transformer_metadata_list::a list of dictionary
        # from int to float)

        transformer_metadata = []
        for transformer in self.transformers:
            data, metadata = transformer.forward(data=data, **kwargs)
            transformer_metadata.append(metadata)
        return data, transformer_metadata

    def backward(self, data, transformer_metadata, **kwargs):
        """Backward pass of pipeline data transformer.

        Args:
            data: The transformed data.
            transformer_metadata: The metadata for the transformation.
            **kwargs: Additional keyword arguments for the transformation.

        Returns:
            The original data before the transformation.
        """
        for transformer in self.transformers[::-1]:
            data = transformer.backward(data=data, metadata=transformer_metadata.pop(), **kwargs)
        return data

    def is_lossy(self):
        """If any of the transformers are lossy, then the pipeline is lossy.

        Returns:
            True if any of the transformers in the pipeline are lossy, False
                otherwise.
        """
        return any(transformer.lossy for transformer in self.transformers)
