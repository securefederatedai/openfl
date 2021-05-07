# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Collaborator tests module."""

import pytest
import numpy as np

from openfl.databases.tensor_db import TensorDB
from openfl.utilities.types import TensorKey
from openfl.protocols import NamedTensor
from openfl.component.aggregation_functions import AggregationFunctionInterface


@pytest.fixture
def named_tensor():
    """Initialize the named_tensor mock."""
    tensor = NamedTensor(
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
def nparray(named_tensor):
    """Initialize the nparray."""
    proto = named_tensor.transformer_metadata.pop()
    metadata = {
        'int_to_float': proto.int_to_float,
        'int_list': proto.int_list,
        'bool_list': proto.bool_list
    }
    array_shape = tuple(metadata['int_list'])
    flat_array = np.frombuffer(named_tensor.data_bytes, dtype=np.float32)

    nparray = np.reshape(flat_array, newshape=array_shape, order='C')

    return nparray


def test_cache_and_get_tensor(nparray, tensor_key):
    """Test that cash and get work correctly."""
    db = TensorDB()
    db.cache_tensor({tensor_key: nparray})
    cached_nparray = db.get_tensor_from_cache(tensor_key)

    assert np.array_equal(nparray, cached_nparray)


def test_tensor_from_cache_empty(tensor_key):
    """Test get works returns None if tensor key is not in the db."""
    db = TensorDB()
    cached_nparray = db.get_tensor_from_cache(tensor_key)
    assert cached_nparray is None


def test_clean_up(nparray, tensor_key):
    """Test that clean_up remove old records."""
    db = TensorDB()

    db.cache_tensor({tensor_key: nparray})
    db.tensor_db['round'] = 2
    db.clean_up()
    cached_nparray = db.get_tensor_from_cache(tensor_key)

    assert cached_nparray is None


def test_clean_up_not_old(nparray, tensor_key):
    """Test that clean_up don't remove not old records."""
    db = TensorDB()

    db.cache_tensor({tensor_key: nparray})
    db.clean_up()
    cached_nparray = db.get_tensor_from_cache(tensor_key)

    assert np.array_equal(nparray, cached_nparray)


def test_clean_up_not_clean_up_with_negative_argument(nparray, tensor_key):
    """Test that clean_up don't remove if records remove_older_than is negative."""
    db = TensorDB()

    db.cache_tensor({tensor_key: nparray})
    db.tensor_db['round'] = 2
    db.clean_up(remove_older_than=-1)
    db.tensor_db['round'] = 0
    cached_nparray = db.get_tensor_from_cache(tensor_key)

    assert np.array_equal(nparray, cached_nparray)


def test_get_aggregated_tensor_directly(nparray, tensor_key):
    """Test that get_aggregated_tensor returns tensors directly."""
    db = TensorDB()
    db.cache_tensor({tensor_key: nparray})
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, 'col2', round_number, report, ('model',)
    )

    db.cache_tensor({tensor_key: nparray})
    agg_nparray, agg_metadata_dict = db.get_aggregated_tensor(tensor_key, {}, None)

    assert np.array_equal(nparray, agg_nparray)


def test_get_aggregated_tensor_only_col(nparray, tensor_key):
    """Test that get_aggregated_tensor returns None if data presents for only collaborator."""
    db = TensorDB()
    db.cache_tensor({tensor_key: nparray})
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, 'col2', round_number, report, ('model',)
    )

    collaborator_weight_dict = {'col1': 0.5, 'col2': 0.5}
    agg_nparray = db.get_aggregated_tensor(
        tensor_key, collaborator_weight_dict, None)

    assert agg_nparray is None


def test_get_aggregated_tensor(nparray, tensor_key):
    """Test that get_aggregated_tensor returns tensors directly."""
    db = TensorDB()
    db.cache_tensor({tensor_key: nparray})
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, 'col2', round_number, report, ('model',)
    )
    db.cache_tensor({tensor_key: nparray})

    collaborator_weight_dict = {'col1': 0.5, 'col2': 0.5}
    agg_nparray, agg_metadata_dict = db.get_aggregated_tensor(
        tensor_key, collaborator_weight_dict, None)

    assert np.array_equal(nparray, agg_nparray)


def test_get_aggregated_tensor_raise_wrong_weights(nparray, tensor_key):
    """Test that get_aggregated_tensor raises if collaborator weights do not sum to 1.0."""
    db = TensorDB()
    db.cache_tensor({tensor_key: nparray})

    collaborator_weight_dict = {'col1': 0.5, 'col2': 0.8}
    with pytest.raises(AssertionError):
        db.get_aggregated_tensor(
            tensor_key, collaborator_weight_dict, None)


@pytest.fixture
def tensor_db():
    """Prepare tensor db."""
    db = TensorDB()
    array_1 = np.array([0, 1, 2, 3, 4])
    tensor_key_1 = TensorKey('tensor_name', 'agg', 0, False, ('col1',))
    array_2 = np.array([2, 3, 4, 5, 6])
    tensor_key_2 = TensorKey('tensor_name', 'agg', 0, False, ('col2',))
    db.cache_tensor({
        tensor_key_1: array_1,
        tensor_key_2: array_2
    })
    return db


def test_get_aggregated_tensor_weights(tensor_db):
    """Test that get_aggregated_tensor calculates correctly."""
    collaborator_weight_dict = {'col1': 0.1, 'col2': 0.9}
    tensor_key = TensorKey('tensor_name', 'agg', 0, False, ())
    agg_nparray = tensor_db.get_aggregated_tensor(
        tensor_key, collaborator_weight_dict, None)

    control_nparray = np.average(
        [np.array([0, 1, 2, 3, 4]), np.array([2, 3, 4, 5, 6])],
        weights=np.array(list(collaborator_weight_dict.values())),
        axis=0
    )

    assert np.array_equal(agg_nparray, control_nparray)


def test_get_aggregated_tensor_error_aggregation_function(tensor_db):
    """Test that get_aggregated_tensor raise error if aggregation function is not callable."""
    collaborator_weight_dict = {'col1': 0.1, 'col2': 0.9}
    tensor_key = TensorKey('tensor_name', 'agg', 0, False, ())
    with pytest.raises(NotImplementedError):
        tensor_db.get_aggregated_tensor(
            tensor_key, collaborator_weight_dict, 'fake_agg_function')


def test_get_aggregated_tensor_new_aggregation_function(tensor_db):
    """Test that get_aggregated_tensor works correctly with a given agg function."""
    collaborator_weight_dict = {'col1': 0.1, 'col2': 0.9}

    class Sum(AggregationFunctionInterface):
        def call(self, agg_tensor_dict, *_):
            tensors = np.array(list(agg_tensor_dict.values()))
            return np.sum(tensors, axis=0)

    tensor_key = TensorKey('tensor_name', 'agg', 0, False, ())

    agg_nparray = tensor_db.get_aggregated_tensor(
        tensor_key, collaborator_weight_dict, Sum())

    assert np.array_equal(agg_nparray, np.array([2, 4, 6, 8, 10]))
