# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pipeline tests module."""

import numpy as np
import pytest

from openfl.pipelines.pipeline import Float32NumpyArrayToBytes
from openfl.pipelines.pipeline import TransformationPipeline
from openfl.pipelines.pipeline import Transformer
from openfl.protocols import base_pb2


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


def test_transformer_forward():
    """Test that Transformer.forward is declared and is not implemented."""
    t = Transformer()

    with pytest.raises(NotImplementedError):
        t.forward(None)


def test_transformer_backward():
    """Test that Transformer.backward is declared and is not implemented."""
    t = Transformer()

    with pytest.raises(NotImplementedError):
        t.backward(None, None)


def test_f32natb_is_lossy():
    """Test that Float32NumpyArrayToBytes object creates with lossy = False."""
    t = Float32NumpyArrayToBytes()
    assert t.lossy is False


def test_f32natb_forward(named_tensor):
    """Test that Float32NumpyArrayToBytes.forward works correctly."""
    t = Float32NumpyArrayToBytes()
    proto = named_tensor.transformer_metadata.pop()
    metadata = {'int_to_float': proto.int_to_float,
                'int_list': proto.int_list,
                'bool_list': proto.bool_list
                }
    array_shape = tuple(metadata['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')

    data_bytes, t_metadata = t.forward(nparray)
    assert t_metadata['int_list'] == metadata['int_list']


def test_f32natb_backward(named_tensor):
    """Test that Float32NumpyArrayToBytes.backward works correctly."""
    t = Float32NumpyArrayToBytes()
    proto = named_tensor.transformer_metadata.pop()
    metadata = {'int_to_float': proto.int_to_float,
                'int_list': proto.int_list,
                'bool_list': proto.bool_list
                }
    array_shape = tuple(metadata['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')

    data = t.backward(nparray, metadata)

    assert data.shape == tuple(metadata['int_list'])


def test_transformation_pipeline_forward(named_tensor):
    """Test that TransformationPipeline.forward works correctly."""
    transformer = Float32NumpyArrayToBytes()
    tp = TransformationPipeline([transformer])
    proto = named_tensor.transformer_metadata.pop()
    metadata = {'int_to_float': proto.int_to_float,
                'int_list': proto.int_list,
                'bool_list': proto.bool_list
                }
    array_shape = tuple(metadata['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')

    data, transformer_metadata = tp.forward(nparray)

    assert len(transformer_metadata) == 1
    assert isinstance(data, bytes)


def test_transformation_pipeline_backward(named_tensor):
    """Test that TransformationPipeline.backward works correctly."""
    transformer = Float32NumpyArrayToBytes()
    tp = TransformationPipeline([transformer])
    proto = named_tensor.transformer_metadata.pop()
    metadata = {'int_to_float': proto.int_to_float,
                'int_list': proto.int_list,
                'bool_list': proto.bool_list
                }
    array_shape = tuple(metadata['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')

    data = tp.backward(nparray, [metadata])

    assert data.shape == tuple(metadata['int_list'])


def test_transformation_pipeline_is_lossy_false(named_tensor):
    """Test that TransformationPipeline.is_lossy returns False if all transformers is not lossy."""
    transformer = Float32NumpyArrayToBytes()
    tp = TransformationPipeline([transformer])

    is_lossy = tp.is_lossy()

    assert is_lossy is False


def test_transformation_pipeline_is_lossy(named_tensor):
    """Test that TransformationPipeline.is_lossy returns False if any transformer is lossy."""
    transformer1 = Float32NumpyArrayToBytes()
    transformer2 = Float32NumpyArrayToBytes()
    transformer2.lossy = True
    tp = TransformationPipeline([transformer1, transformer2])

    is_lossy = tp.is_lossy()

    assert is_lossy is True
