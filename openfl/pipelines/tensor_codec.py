# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""TensorCodec module."""

import numpy as np

from openfl.pipelines import NoCompressionPipeline
from openfl.utilities import TensorKey, change_tags


class TensorCodec:
    """TensorCodec is responsible for the following.

    1. Tracking the compression/decompression related dependencies of a given
    tensor.
    2. Acting as a TensorKey aware wrapper for the compression_pipeline
    functionality.

    Attributes:
        compression_pipeline: The pipeline used for compression.
        lossless_pipeline: The pipeline used for lossless compression.
    """

    def __init__(self, compression_pipeline):
        """Initialize the TensorCodec.

        Args:
            compression_pipeline: The pipeline used for compression.
        """
        self.compression_pipeline = compression_pipeline
        if self.compression_pipeline.is_lossy():
            self.lossless_pipeline = NoCompressionPipeline()
        else:
            self.lossless_pipeline = compression_pipeline

    def set_lossless_pipeline(self, lossless_pipeline):
        """
        Set lossless pipeline.

        Args:
            lossless_pipeline: The pipeline to be set as the lossless pipeline.
                It should be a pipeline that is not lossy.

        Raises:
            AssertionError: If the provided pipeline is not lossless.
        """
        assert lossless_pipeline.is_lossy() is False, "The provided pipeline is not lossless"
        self.lossless_pipeline = lossless_pipeline

    def compress(self, tensor_key, data, require_lossless=False, **kwargs):
        """Function-wrapper around the tensor_pipeline.forward function.

        It also keeps track of the tensorkeys associated with the compressed
        nparray.

        Args:
            tensor_key: TensorKey is provided to verify it should be
                compressed, and new TensorKeys returned will be derivatives of
                the existing tensor_name.
            data: (uncompressed) numpy array associated with the tensor_key.
            require_lossless: boolean. Does tensor require compression.

        Returns:
            compressed_tensor_key: Tensorkey corresponding to the decompressed
                tensor.
            compressed_nparray: The compressed tensor.
            metadata: metadata associated with compressed tensor.
        """
        if require_lossless:
            compressed_nparray, metadata = self.lossless_pipeline.forward(data, **kwargs)
        else:
            compressed_nparray, metadata = self.compression_pipeline.forward(data, **kwargs)
        # Define the compressed tensorkey that should be
        # returned ('trained.delta'->'trained.delta.lossy_compressed')
        tensor_name, origin, round_number, report, tags = tensor_key
        if not self.compression_pipeline.is_lossy() or require_lossless:
            new_tags = change_tags(tags, add_field="compressed")
        else:
            new_tags = change_tags(tags, add_field="lossy_compressed")
        compressed_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
        return compressed_tensor_key, compressed_nparray, metadata

    def decompress(
        self,
        tensor_key,
        data,
        transformer_metadata,
        require_lossless=False,
        **kwargs,
    ):
        """
        Function-wrapper around the tensor_pipeline.backward function.

        It also keeps track of the tensorkeys associated with the decompressed
        nparray.

        Args:
            tensor_key: TensorKey is provided to verify it should be
                decompressed, and new TensorKeys returned will be derivatives
                of the existing tensor_name.
            data: (compressed) numpy array associated with the tensor_key.
            transformer_metadata: metadata associated with the compressed
                tensor.
            require_lossless: boolean, does data require lossless
                decompression.

        Returns:
            decompressed_tensor_key: Tensorkey corresponding to the
                decompressed tensor.
            decompressed_nparray: The decompressed tensor.
        """
        tensor_name, origin, round_number, report, tags = tensor_key

        assert len(transformer_metadata) > 0, "metadata must be included for decompression"
        assert ("compressed" in tags) or (
            "lossy_compressed" in tags
        ), "Cannot decompress an uncompressed tensor"
        if require_lossless:
            assert "compressed" in tags, "Cannot losslessly decompress lossy tensor"

        if require_lossless or "compressed" in tags:
            decompressed_nparray = self.lossless_pipeline.backward(
                data, transformer_metadata, **kwargs
            )
        else:
            decompressed_nparray = self.compression_pipeline.backward(
                data, transformer_metadata, **kwargs
            )
        # Define the decompressed tensorkey that should be returned
        if "lossy_compressed" in tags:
            new_tags = change_tags(
                tags,
                add_field="lossy_decompressed",
                remove_field="lossy_compressed",
            )
            decompressed_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
        elif "compressed" in tags:
            # 'compressed' == lossless compression; no need for
            # compression related tag after decompression
            new_tags = change_tags(tags, remove_field="compressed")
            decompressed_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
        else:
            raise NotImplementedError("Decompression is only supported on compressed data")

        return decompressed_tensor_key, decompressed_nparray

    @staticmethod
    def generate_delta(tensor_key, nparray, base_model_nparray):
        """Create delta from the updated layer and base layer.

        Args:
            tensor_key: This is the tensor_key associated with the nparray.
                Should have a tag of 'trained' or 'aggregated'
            nparray: The nparray that corresponds to the tensorkey.
            base_model_nparray: The base model tensor that will be subtracted
                from the new weights.

        Returns:
            delta_tensor_key: Tensorkey that corresponds to the delta weight
                array.
            delta: Difference between the provided tensors.
        """
        tensor_name, origin, round_number, report, tags = tensor_key
        if not np.isscalar(nparray):
            assert nparray.shape == base_model_nparray.shape, (
                f"Shape of updated layer ({nparray.shape}) is not equal to base "
                f"layer shape of ({base_model_nparray.shape})"
            )
        assert "model" not in tags, (
            "The tensorkey should be provided "
            "from the layer with new weights, not the base model"
        )
        new_tags = change_tags(tags, add_field="delta")
        delta_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
        return delta_tensor_key, nparray - base_model_nparray

    @staticmethod
    def apply_delta(tensor_key, delta, base_model_nparray, creates_model=False):
        """Add delta to the nparray.

        Args:
            tensor_key: This is the tensor_key associated with the delta.
                Should have a tag of 'trained' or 'aggregated'.
            delta: Weight delta between the new model and old model.
            base_model_nparray: The nparray that corresponds to the prior
                weights.
            creates_model: If flag is set, the tensorkey returned will
                correspond to the aggregator model.

        Returns:
            new_model_tensor_key: Latest model layer tensorkey.
            new_model_nparray: Latest layer weights.
        """
        tensor_name, origin, round_number, report, tags = tensor_key
        if not np.isscalar(base_model_nparray):
            assert delta.shape == base_model_nparray.shape, (
                f"Shape of delta ({delta.shape}) is not equal to shape of model"
                f" layer ({base_model_nparray.shape})"
            )
        # assert('model' in tensor_key[3]), 'The tensorkey should be provided
        # from the base model'
        # Aggregator UUID has the prefix 'aggregator'
        if "aggregator" in origin and not creates_model:
            new_tags = change_tags(tags, remove_field="delta")
            new_model_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
        else:
            new_model_tensor_key = TensorKey(tensor_name, origin, round_number, report, ("model",))

        return new_model_tensor_key, base_model_nparray + delta

    def find_dependencies(self, tensor_key, send_model_deltas):
        """Resolve the tensors required to do the specified operation.

        Args:
            tensor_key: A tuple containing the tensor name, origin, round
                number, report, and tags.
            send_model_deltas: A boolean flag indicating whether to send model
                deltas.

        Returns:
            tensor_key_dependencies: A list of tensor keys that are
                dependencies of the given tensor key.
        """
        tensor_key_dependencies = []

        tensor_name, origin, round_number, report, tags = tensor_key

        if "model" in tags and send_model_deltas:
            if round_number >= 1:
                # The new model can be generated by previous model + delta
                tensor_key_dependencies.append(
                    TensorKey(tensor_name, origin, round_number - 1, report, tags)
                )
                if self.compression_pipeline.is_lossy():
                    new_tags = ("aggregated", "delta", "lossy_compressed")
                else:
                    new_tags = ("aggregated", "delta", "compressed")
                tensor_key_dependencies.append(
                    TensorKey(tensor_name, origin, round_number, report, new_tags)
                )

        return tensor_key_dependencies
