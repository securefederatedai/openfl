# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Collaborator tests module."""

from unittest import mock

import numpy
import pytest

from openfl.component.collaborator import Collaborator
from openfl.pipelines import NoCompressionPipeline
from openfl.protocols import base_pb2
from openfl.transport.serialiser import CollaboratorSerialiser
from openfl.utilities.types import TensorKey

@pytest.fixture
def collaborator_mock():
    """Initialize the collaborator mock."""
    col = Collaborator('col1', 'some_uuid', 'federation_uuid',
                       mock.Mock(), mock.Mock(), mock.Mock(), opt_treatment='RESET')
    col.tensor_db = mock.Mock()
    col._serialisation_middleware = CollaboratorSerialiser(
        col.collaborator_name, mock.Mock(), NoCompressionPipeline()
    )

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


def test_send_task_results_train(collaborator_mock):
    """Test that send_task_results for train tasks works correctly."""
    task_name = 'train_task'
    tensor_dict = {}
    round_number = 0
    data_size = 200
    collaborator_mock.task_runner.get_train_data_size = mock.Mock(return_value=data_size)
    collaborator_mock._serialisation_middleware._aggregator_client.send_local_task_results = mock.Mock()
    collaborator_mock.send_task_results(tensor_dict, round_number, task_name)

    collaborator_mock._serialisation_middleware._aggregator_client.send_local_task_results.assert_called_with(
        round_number, task_name, data_size, [])


def test_send_task_results_valid(collaborator_mock):
    """Test that send_task_results for validation tasks works correctly."""
    task_name = 'valid_task'
    tensor_dict = {}
    round_number = 0
    data_size = 400
    collaborator_mock.task_runner.get_valid_data_size = mock.Mock(return_value=data_size)
    collaborator_mock._serialisation_middleware._aggregator_client.send_local_task_results = mock.Mock()
    collaborator_mock.send_task_results(tensor_dict, round_number, task_name)

    collaborator_mock._serialisation_middleware._aggregator_client.send_local_task_results.assert_called_with(
        round_number, task_name, data_size, [])



@pytest.mark.parametrize('require_lossless', [True, False])
def test_get_aggregated_tensor_from_aggregator(collaborator_mock, tensor_key,
                                               named_tensor, require_lossless):
    """Test that get_aggregated_tensor works correctly."""
    collaborator_mock._serialisation_middleware._aggregator_client.get_aggregated_tensor = mock.Mock(return_value=named_tensor)
    nparray = collaborator_mock.get_aggregated_tensor_from_aggregator(tensor_key, require_lossless)

    collaborator_mock._serialisation_middleware._aggregator_client.get_aggregated_tensor.assert_called_with(
        tensor_key.tensor_name, tensor_key.round_number,
        tensor_key.report, tensor_key.tags, require_lossless)
    assert nparray == named_tensor.data_bytes


def test_get_data_for_tensorkey_from_db(collaborator_mock, tensor_key):
    """Test that get_data_for_tensorkey works correctly for data form db."""
    expected_nparray = 'some_data'
    collaborator_mock.tensor_db.get_tensor_from_cache = mock.Mock(
        return_value='some_data')
    nparray = collaborator_mock.get_data_for_tensorkey(tensor_key)

    assert nparray == expected_nparray


def test_get_data_for_tensorkey(collaborator_mock, tensor_key):
    """Test that get_data_for_tensorkey works correctly if data is not in db."""
    collaborator_mock.tensor_db.get_tensor_from_cache = mock.Mock(
        return_value=None)
    collaborator_mock.get_aggregated_tensor_from_aggregator = mock.Mock()
    collaborator_mock.get_data_for_tensorkey(tensor_key)
    collaborator_mock.get_aggregated_tensor_from_aggregator.assert_called_with(
        tensor_key, require_lossless=True)


def test_get_data_for_tensorkey_locally(collaborator_mock, tensor_key):
    """Test that get_data_for_tensorkey works correctly if found tensor locally."""
    tensor_key = tensor_key._replace(round_number=1)
    nparray = numpy.array([0, 1, 2, 3, 4])
    collaborator_mock.tensor_db.get_tensor_from_cache = mock.Mock(
        side_effect=[None, nparray])
    ret = collaborator_mock.get_data_for_tensorkey(tensor_key)

    assert numpy.array_equal(ret, nparray)


def test_get_data_for_tensorkey_dependencies(collaborator_mock, tensor_key):
    """Test that get_data_for_tensorkey works correctly if additional dependencies."""
    tensor_key = tensor_key._replace(round_number=1)
    collaborator_mock.tensor_db.get_tensor_from_cache = mock.Mock(
        return_value=None)
    collaborator_mock._find_dependencies = mock.Mock(return_value=[tensor_key])
    collaborator_mock.get_aggregated_tensor_from_aggregator = mock.Mock()
    collaborator_mock.get_data_for_tensorkey(tensor_key)
    collaborator_mock.get_aggregated_tensor_from_aggregator.assert_called_with(
        tensor_key, require_lossless=True)

def test_find_dependencies_without_send_model_deltas(collaborator_mock, tensor_key):
    """Test that find_dependencies returns empty list when send_model_deltas = False."""
    tensor_name, origin, _, report, _ = tensor_key
    tensor_key = TensorKey(
        tensor_name, origin, 5, report, ('model',)
    )
    tensor_key_dependencies = collaborator_mock._find_dependencies(tensor_key)

    assert len(tensor_key_dependencies) == 0


def test_find_dependencies_without_model_in_tags(collaborator_mock, tensor_key):
    """Test that find_dependencies returns empty list when there is no model tag."""
    collaborator_mock.use_delta_updates = True
    tensor_key_dependencies = collaborator_mock._find_dependencies(tensor_key)

    assert len(tensor_key_dependencies) == 0


def test_find_dependencies_with_zero_round(collaborator_mock, tensor_key):
    """Test that find_dependencies returns empty list when round number is 0."""
    collaborator_mock.use_delta_updates = True
    tensor_name, origin, round_number, report, tags = tensor_key
    tensor_key = TensorKey(
        tensor_name, origin, round_number, report, ('model',)
    )
    tensor_key_dependencies = collaborator_mock._find_dependencies(tensor_key)

    assert len(tensor_key_dependencies) == 0


# def test_find_dependencies(collaborator_mock, tensor_key):
#     """Test that find_dependencies works correctly."""
#     collaborator_mock.use_delta_updates = True
#     tensor_name, origin, round_number, report, tags = tensor_key
#     round_number = 2
#     tensor_key = TensorKey(
#         tensor_name, origin, round_number, report, ('model',)
#     )
#     tensor_key_dependencies = collaborator_mock._find_dependencies(tensor_key)

#     assert len(tensor_key_dependencies) == 2
#     tensor_key_dependency_0, tensor_key_dependency_1 = tensor_key_dependencies
#     assert tensor_key_dependency_0.round_number == round_number - 1
#     assert tensor_key_dependency_0.tags == tensor_key.tags
#     assert tensor_key_dependency_1.tags == ('aggregated', 'delta', 'compressed')


# def test_find_dependencies_is_lossy(collaborator_mock, tensor_key):
#     """Test that find_dependencies works correctly with lossy_compressed."""
#     collaborator_mock.use_delta_updates = True
#     collaborator_mock.compression_pipeline.is_lossy = mock.Mock(return_value=True)
#     tensor_name, origin, round_number, report, tags = tensor_key
#     round_number = 2
#     tensor_key = TensorKey(
#         tensor_name, origin, round_number, report, ('model',)
#     )
#     tensor_key_dependencies = collaborator_mock._find_dependencies(tensor_key)

#     assert len(tensor_key_dependencies) == 2
#     tensor_key_dependency_0, tensor_key_dependency_1 = tensor_key_dependencies
#     assert tensor_key_dependency_0.round_number == round_number - 1
#     assert tensor_key_dependency_0.tags == tensor_key.tags
#     assert tensor_key_dependency_1.tags == ('aggregated', 'delta', 'lossy_compressed')
