# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tensor codec tests module."""


pass

import numpy as np
import pytest

from openfl.protocols import base_pb2
from openfl.utilities.utils import apply_delta, generate_delta
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


def test_generate(tensor_key, named_tensor):
    """Test that generate_delta works correctly."""
    metadata = [{'int_to_float': proto.int_to_float,
                 'int_list': proto.int_list,
                 'bool_list': proto.bool_list
                 } for proto in named_tensor.transformer_metadata]
    array_shape = tuple(metadata[0]['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')

    delta_tensor_key, delta_nparray = generate_delta(tensor_key, nparray, nparray)

    assert np.array_equal(delta_nparray, nparray - nparray)
    assert 'delta' in delta_tensor_key.tags


def test_apply_delta_agg(tensor_key, named_tensor):
    """Test that apply_delta works for aggregator tensor_key."""
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

    new_model_tensor_key, nparray_with_delta = apply_delta(
        tensor_key, nparray, nparray)

    assert 'delta' not in new_model_tensor_key.tags
    assert np.array_equal(nparray_with_delta, nparray + nparray)


def test_apply_delta_col(tensor_key, named_tensor):
    """Test that apply_delta works for collaborator tensor_key."""
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

    new_model_tensor_key, nparray_with_delta = apply_delta(
        tensor_key, nparray, nparray)

    assert 'model' in new_model_tensor_key.tags
    assert 'delta' not in new_model_tensor_key.tags
    assert np.array_equal(nparray_with_delta, nparray + nparray)
