# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""TensorCodec module."""

from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import utils
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
        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()
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
        assert ("compressed" in tags) or ("lossy_compressed" in tags), (
            "Cannot decompress an uncompressed tensor"
        )
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

    def deserialise(self, named_tensor, collaborator_name):
        """Convert named tensor to a numpy array.

        Args:
            named_tensor (protobuf): The tensor to convert to nparray.
            collaborator_name (str): Name of teh collaborator for which the named tensor is to
                be deserialised.

        Returns:
            decompressed_nparray (nparray): The nparray converted.
        """
        # do the stuff we do now for decompression and frombuffer and stuff
        # This should probably be moved back to protoutils
        raw_bytes = named_tensor.data_bytes
        metadata = [
            {
                "int_to_float": proto.int_to_float,
                "int_list": proto.int_list,
                "bool_list": proto.bool_list,
            }
            for proto in named_tensor.transformer_metadata
        ]
        # The tensor has already been transferred to collaborator, so
        # the newly constructed tensor should have the collaborator origin
        tensor_key = TensorKey(
            named_tensor.name,
            collaborator_name,
            named_tensor.round_number,
            named_tensor.report,
            tuple(named_tensor.tags),
        )
        *_, tags = tensor_key
        if "compressed" in tags:
            decompressed_tensor_key, decompressed_nparray = self.decompress(
                tensor_key,
                data=raw_bytes,
                transformer_metadata=metadata,
                require_lossless=True,
            )
        elif "lossy_compressed" in tags:
            decompressed_tensor_key, decompressed_nparray = self.decompress(
                tensor_key,
                data=raw_bytes,
                transformer_metadata=metadata,
                require_lossless=False,
            )
        else:
            # There could be a case where the compression pipeline is bypassed
            # entirely
            decompressed_tensor_key = tensor_key
            decompressed_nparray = raw_bytes

        return decompressed_tensor_key, decompressed_nparray

    def serialise(self, tensor_key, nparray, lossless=True):
        """Construct the NamedTensor Protobuf.

        Includes logic to create delta, compress tensors with the TensorCodec,
        etc.

        Args:
            tensor_key (namedtuple): Tensorkey that will be resolved locally or
                remotely. May be the product of other tensors.
            nparray: The decompressed tensor associated with the requested
                tensor key.

        Returns:
            named_tensor (protobuf) : The tensor constructed from the nparray.
        """
        # Secure aggregation setup tensor.
        if "secagg" in tensor_key.tags:
            import json

            import numpy as np

            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)

            compressed_tensor_key, compressed_nparray = (
                tensor_key,
                str.encode(json.dumps(nparray, cls=NumpyEncoder)),
            )
        else:
            compressed_tensor_key, compressed_nparray, metadata = self.compress(
                tensor_key, nparray, require_lossless=lossless
            )

        named_tensor = utils.construct_named_tensor(
            compressed_tensor_key, compressed_nparray, metadata, lossless=lossless
        )

        return named_tensor
