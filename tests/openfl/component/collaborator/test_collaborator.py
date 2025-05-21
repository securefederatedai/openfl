# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Collaborator tests module."""

from unittest import mock

import numpy
import pytest

from openfl.component.collaborator import Collaborator
from openfl.protocols import base_pb2
from openfl.utilities.types import TensorKey


@pytest.fixture
def collaborator_mock():
    """Initialize the collaborator mock."""
    col = Collaborator('col1', 'some_uuid', 'federation_uuid',
                       mock.Mock(), mock.Mock(), mock.Mock(), opt_treatment='RESET')
    col.tensor_db = mock.Mock()

    return col


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
    tensor.tags.append('model')
    metadata = tensor.transformer_metadata.add()
    metadata.int_to_float[1] = 1.
    metadata.int_list.extend([1, 8])
    metadata.bool_list.append(True)

    return tensor


@pytest.fixture
def tensor_key(collaborator_mock, named_tensor):
    """Initialize the tensor_key mock."""
    tensor_key = TensorKey(
        named_tensor.name,
        collaborator_mock.collaborator_name,
        named_tensor.round_number,
        named_tensor.report,
        tuple(named_tensor.tags)
    )
    return tensor_key


@pytest.fixture
def tensor_key_trained(collaborator_mock, named_tensor):
    """Initialize the tensor_key_trained mock."""
    named_tensor.tags.append('trained')
    named_tensor.tags.remove('model')
    tensor_key = TensorKey(
        named_tensor.name,
        collaborator_mock.collaborator_name,
        named_tensor.round_number,
        named_tensor.report,
        tuple(named_tensor.tags)
    )
    return tensor_key


def test_do_task(collaborator_mock, tensor_key):
    """Test that do_task works correctly."""
    round_number = 0
    nparray = numpy.array([0, 1, 2, 3, 4])
    tensor_key = tensor_key._replace(origin='GLOBAL')
    result = {tensor_key: nparray}, {tensor_key: nparray}

    task = mock.MagicMock()
    task.function_name = 'func_name'
    task.name = 'task_name'
    task.task_type = 'validate'
    task.__getitem__ = mock.Mock(side_effect=[task.function_name, {}])

    collaborator_mock.task_config = mock.MagicMock()
    collaborator_mock.task_config.__getitem__ = mock.MagicMock(return_value=task)
    collaborator_mock.task_runner.get_required_tensorkeys_for_function = mock.Mock(
        return_value=[tensor_key])
    collaborator_mock.task_runner.func_name = mock.Mock(return_value=result)
    collaborator_mock.send_task_results = mock.Mock()
    collaborator_mock.do_task(task, round_number)

    collaborator_mock.send_task_results.assert_called_with(result[0], round_number, task.name)


def test_send_task_results(collaborator_mock, tensor_key):
    """Test that send_task_results works correctly."""
    task_name = 'task_name'
    tensor_key = tensor_key._replace(report=True)
    tensor_dict = {}
    round_number = 0
    data_size = -1
    collaborator_mock.client.send_local_task_results = mock.Mock()
    collaborator_mock.send_task_results(tensor_dict, round_number, task_name)
    collaborator_mock.client.send_local_task_results.assert_called_with(
        round_number, task_name, data_size, [])


def test_send_task_results_train(collaborator_mock):
    """Test that send_task_results for train tasks works correctly."""
    task_name = 'train_task'
    tensor_dict = {}
    round_number = 0
    data_size = 200
    collaborator_mock.nparray_to_named_tensor = mock.Mock()
    collaborator_mock.task_runner.get_train_data_size = mock.Mock(return_value=data_size)
    collaborator_mock.client.send_local_task_results = mock.Mock()
    collaborator_mock.send_task_results(tensor_dict, round_number, task_name)

    collaborator_mock.client.send_local_task_results.assert_called_with(
        round_number, task_name, data_size, [])


def test_send_task_results_valid(collaborator_mock):
    """Test that send_task_results for validation tasks works correctly."""
    task_name = 'valid_task'
    tensor_dict = {}
    round_number = 0
    data_size = 400
    collaborator_mock.nparray_to_named_tensor = mock.Mock()
    collaborator_mock.task_runner.get_valid_data_size = mock.Mock(return_value=data_size)
    collaborator_mock.client.send_local_task_results = mock.Mock()
    collaborator_mock.send_task_results(tensor_dict, round_number, task_name)

    collaborator_mock.client.send_local_task_results.assert_called_with(
        round_number, task_name, data_size, [])


def test_fetch_tensors_from_aggregator(collaborator_mock, tensor_key, named_tensor):
    """Test that fetch_tensors_from_aggregator works correctly."""
    # Simulate tensor not in cache
    collaborator_mock.tensor_db.get_tensor_from_cache.return_value = None
    collaborator_mock.client.get_aggregated_tensors = mock.Mock(return_value=[named_tensor])
    collaborator_mock.tensor_db.cache_tensor = mock.Mock()
    # Patch utils.deserialize_tensor to avoid side effects
    with mock.patch("openfl.protocols.utils.deserialize_tensor", return_value=(tensor_key, "nparray")):
        collaborator_mock.fetch_tensors_from_aggregator([tensor_key])
    collaborator_mock.client.get_aggregated_tensors.assert_called_with(
        [tensor_key], require_lossless=True)
    collaborator_mock.tensor_db.cache_tensor.assert_called()
