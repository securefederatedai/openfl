# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tensor codec tests module."""


from unittest import mock

import numpy as np
import pytest

from openfl.pipelines import NoCompressionPipeline
from openfl.pipelines import SKCPipeline
from openfl.pipelines import TensorCodec
from openfl.protocols import base_pb2
from openfl.utilities.types import TensorKey


@pytest.fixture
def named_tensor():
    """Initialize the named_tensor mock."""
    tensor = base_pb2.NamedTensor(
        name='tensor_name',
        round_number=0,
        lossless=False,
        report=False,
        data_bytes=32 * b'1'
    )
    metadata = tensor.transformer_metadata.add()
    metadata.int_to_float[1] = 1.
    metadata.int_list.extend([1, 8])
    metadata.bool_list.append(True)

    return tensor


@pytest.fixture
def tensor_key(named_tensor):
    """Initialize the tensor_key mock."""
    tensor_key = TensorKey(
        named_tensor.name,
        'col1',
        named_tensor.round_number,
        named_tensor.report,
        tuple(named_tensor.tags)
    )
    return tensor_key


@pytest.fixture
def tensor_key_trained(named_tensor):
    """Initialize the tensor_key_trained mock."""
    named_tensor.tags.append('trained')
    # named_tensor.tags.remove('model')
    tensor_key = TensorKey(
        named_tensor.name,
        'col1',
        named_tensor.round_number,
        named_tensor.report,
        tuple(named_tensor.tags)
    )
    return tensor_key


def test_compress(tensor_key, named_tensor):
    """Test that compress works correctly."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    array_shape = tuple(metadata[0]['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')
    compressed_tensor_key, compressed_nparray, metadata = tensor_codec.compress(
        tensor_key, nparray)

    assert 'compressed' in compressed_tensor_key.tags
    assert compressed_tensor_key.tensor_name == tensor_key.tensor_name
    assert compressed_tensor_key.origin == tensor_key.origin
    assert compressed_tensor_key.round_number == tensor_key.round_number


def test_compress_lossless(tensor_key, named_tensor):
    """Test that compress works correctly with require_lossless flag."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    array_shape = tuple(metadata[0]['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')
    compressed_tensor_key, compressed_nparray, metadata = tensor_codec.compress(
        tensor_key, nparray, require_lossless=True)

    assert 'compressed' in compressed_tensor_key.tags
    assert compressed_tensor_key.tensor_name == tensor_key.tensor_name
    assert compressed_tensor_key.origin == tensor_key.origin
    assert compressed_tensor_key.round_number == tensor_key.round_number


def test_compress_not_lossy_lossless(tensor_key, named_tensor):
    """Test that compress works correctly with require_lossless flag and lossless pipeline."""
    tensor_codec = TensorCodec(SKCPipeline())
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    array_shape = tuple(metadata[0]['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')
    compressed_tensor_key, compressed_nparray, metadata = tensor_codec.compress(
        tensor_key, nparray, require_lossless=True)

    assert 'compressed' in compressed_tensor_key.tags
    assert compressed_tensor_key.tensor_name == tensor_key.tensor_name
    assert compressed_tensor_key.origin == tensor_key.origin
    assert compressed_tensor_key.round_number == tensor_key.round_number


def test_compress_not_require_lossless(tensor_key, named_tensor):
    """Test that compress works correctly flag with lossless pipeline without require_lossless."""
    tensor_codec = TensorCodec(SKCPipeline())
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    array_shape = tuple(metadata[0]['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')
    tensor_codec.compression_pipeline.forward = mock.Mock(return_value=(nparray, metadata[0]))
    compressed_tensor_key, compressed_nparray, metadata = tensor_codec.compress(
        tensor_key, nparray)

    assert 'lossy_compressed' in compressed_tensor_key.tags
    assert compressed_tensor_key.tensor_name == tensor_key.tensor_name
    assert compressed_tensor_key.origin == tensor_key.origin
    assert compressed_tensor_key.round_number == tensor_key.round_number


def test_decompress_no_metadata(tensor_key, named_tensor):
    """Test that decompress raises exception without metadata."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    metadata = []
    with pytest.raises(AssertionError):
        tensor_codec.decompress(
            tensor_key, named_tensor.data_bytes, metadata
        )


def test_decompress_no_tags(tensor_key, named_tensor):
    """Test that decompress raises exception without tags."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    with pytest.raises(AssertionError):
        tensor_codec.decompress(
            tensor_key, named_tensor.data_bytes, metadata
        )


def test_decompress_require_lossless_no_compressed_in_tags(tensor_key, named_tensor):
    """Test that decompress raises error when require_lossless is True and is no compressed tag."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, origin, round_number, report, ('lossy_compressed',)
    )
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    with pytest.raises(AssertionError):
        tensor_codec.decompress(
            tensor_key, named_tensor.data_bytes, metadata, require_lossless=True
        )


def test_decompress_call_lossless_pipeline_with_require_lossless(tensor_key, named_tensor):
    """Test that decompress calls lossless pipeline when require_lossless is True."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, origin, round_number, report, ('compressed',)
    )
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    tensor_codec.lossless_pipeline = mock.Mock()
    tensor_codec.decompress(
        tensor_key, named_tensor.data_bytes, metadata, require_lossless=True
    )
    tensor_codec.lossless_pipeline.backward.assert_called_with(
        named_tensor.data_bytes, metadata)


def test_decompress_call_compression_pipeline(tensor_key, named_tensor):
    """Test that decompress calls compression pipeline when there is no compressed tag."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, origin, round_number, report, ('lossy_compressed',)
    )
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    tensor_codec.compression_pipeline = mock.Mock()
    tensor_codec.decompress(
        tensor_key, named_tensor.data_bytes, metadata
    )
    tensor_codec.compression_pipeline.backward.assert_called_with(
        named_tensor.data_bytes, metadata)


def test_decompress_lossy_compressed_in_tags(tensor_key, named_tensor):
    """Test that decompress works correctly when there is lossy_compressed tag."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, origin, round_number, report, ('lossy_compressed',)
    )
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    decompressed_tensor_key, decompressed_nparray = tensor_codec.decompress(
        tensor_key, named_tensor.data_bytes, metadata
    )
    assert 'lossy_decompressed' in decompressed_tensor_key.tags


def test_decompress_compressed_in_tags(tensor_key, named_tensor):
    """Test that decompress works correctly when there is compressed tag."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, origin, round_number, report, ('compressed',)
    )
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    decompressed_tensor_key, decompressed_nparray = tensor_codec.decompress(
        tensor_key, named_tensor.data_bytes, metadata
    )
    assert 'compressed' not in decompressed_tensor_key.tags


def test_deserialise_without_tags(named_tensor):
    """Test that deserialise works correctly for tensor without tags."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    _, nparray = tensor_codec.deserialise(named_tensor, 'col1')

    assert named_tensor.data_bytes == nparray


@pytest.mark.parametrize('tag', ['compressed', 'lossy_compressed'])
def test_deserialise_compressed_tag(named_tensor, tag):
    """Test that deserialise works correctly for tensor with tags."""
    named_tensor.tags.append(tag)
    tensor_codec = TensorCodec(NoCompressionPipeline())
    _, nparray = tensor_codec.deserialise(named_tensor, 'col1')

    assert isinstance(nparray, np.ndarray)


def test_serialise(tensor_key, named_tensor):
    """Test that serialise works correctly."""
    named_tensor.tags.append('compressed')
    tensor_codec = TensorCodec(NoCompressionPipeline())
    _, nparray = tensor_codec.deserialise(named_tensor, 'col1')
    tensor = tensor_codec.serialise(tensor_key, nparray)

    assert tensor.data_bytes == named_tensor.data_bytes
    assert tensor.lossless is True
