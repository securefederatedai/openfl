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


def test_generate(tensor_key, named_tensor):
    """Test that generate_delta works correctly."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    array_shape = tuple(metadata[0]['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')

    delta_tensor_key, delta_nparray = tensor_codec.generate_delta(tensor_key, nparray, nparray)

    assert np.array_equal(delta_nparray, nparray - nparray)
    assert 'delta' in delta_tensor_key.tags


def test_generate_delta_assert_model_in_tags(tensor_key, named_tensor):
    """Test that generate_delta raises exception when there is model tag."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, origin, round_number, report, ('model',)
    )
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    array_shape = tuple(metadata[0]['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')

    with pytest.raises(AssertionError):
        tensor_codec.generate_delta(tensor_key, nparray, nparray)


def test_apply_delta_agg(tensor_key, named_tensor):
    """Test that apply_delta works for aggregator tensor_key."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, 'aggregator_1', round_number, report, ('delta',)
    )
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    array_shape = tuple(metadata[0]['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')

    new_model_tensor_key, nparray_with_delta = tensor_codec.apply_delta(
        tensor_key, nparray, nparray)

    assert 'delta' not in new_model_tensor_key.tags
    assert np.array_equal(nparray_with_delta, nparray + nparray)


def test_apply_delta_col(tensor_key, named_tensor):
    """Test that apply_delta works for collaborator tensor_key."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, origin, round_number, report, ('delta',)
    )
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    array_shape = tuple(metadata[0]['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')

    new_model_tensor_key, nparray_with_delta = tensor_codec.apply_delta(
        tensor_key, nparray, nparray)

    assert 'model' in new_model_tensor_key.tags
    assert 'delta' not in new_model_tensor_key.tags
    assert np.array_equal(nparray_with_delta, nparray + nparray)


def test_find_dependencies_without_send_model_deltas(tensor_key):
    """Test that find_dependencies returns empty list when send_model_deltas = False."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, origin, 5, report, ('model',)
    )
    tensor_key_dependencies = tensor_codec.find_dependencies(tensor_key, False)

    assert len(tensor_key_dependencies) == 0


def test_find_dependencies_without_model_in_tags(tensor_key):
    """Test that find_dependencies returns empty list when there is no model tag."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_key_dependencies = tensor_codec.find_dependencies(tensor_key, True)

    assert len(tensor_key_dependencies) == 0


def test_find_dependencies_with_zero_round(tensor_key):
    """Test that find_dependencies returns empty list when round number is 0."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, origin, round_number, report, ('model',)
    )
    tensor_key_dependencies = tensor_codec.find_dependencies(tensor_key, True)

    assert len(tensor_key_dependencies) == 0


def test_find_dependencies(tensor_key):
    """Test that find_dependencies works correctly."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_name, origin, round_number, report, tags = tensor_key
    round_number = 2
    tensor_key = TensorKey(
        tensor_name, origin, round_number, report, ('model',)
    )
    tensor_key_dependencies = tensor_codec.find_dependencies(tensor_key, True)

    assert len(tensor_key_dependencies) == 2
    tensor_key_dependency_0, tensor_key_dependency_1 = tensor_key_dependencies
    assert tensor_key_dependency_0.round_number == round_number - 1
    assert tensor_key_dependency_0.tags == tensor_key.tags
    assert tensor_key_dependency_1.tags == ('aggregated', 'delta', 'compressed')


def test_find_dependencies_is_lossy(tensor_key):
    """Test that find_dependencies works correctly with lossy_compressed."""
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_codec.compression_pipeline.is_lossy = mock.Mock(return_value=True)
    tensor_name, origin, round_number, report, tags = tensor_key
    round_number = 2
    tensor_key = TensorKey(
        tensor_name, origin, round_number, report, ('model',)
    )
    tensor_key_dependencies = tensor_codec.find_dependencies(tensor_key, True)

    assert len(tensor_key_dependencies) == 2
    tensor_key_dependency_0, tensor_key_dependency_1 = tensor_key_dependencies
    assert tensor_key_dependency_0.round_number == round_number - 1
    assert tensor_key_dependency_0.tags == tensor_key.tags
    assert tensor_key_dependency_1.tags == ('aggregated', 'delta', 'lossy_compressed')
