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


def test_get_tasks(collaborator_mock):
    """Test that get_tasks works correctly."""
    results = (['task_name'], 0, 0, True)
    collaborator_mock.client.get_tasks = mock.Mock(return_value=results)
    tasks, round_number, sleep_time, time_to_quit = collaborator_mock.get_tasks()
    assert results == (tasks, round_number, sleep_time, time_to_quit)


def test_do_task(collaborator_mock, tensor_key):
    """Test that do_task works correctly."""
    round_number = 0
    nparray = numpy.array([0, 1, 2, 3, 4])
    result = {tensor_key: nparray}, {tensor_key: nparray}

    task = mock.Mock()
    task.function_name = 'func_name'
    task.name = 'task_name'
    task.task_type = 'validate'

    collaborator_mock.task_runner.TASK_REGISTRY = mock.MagicMock()
    collaborator_mock.task_runner.TASK_REGISTRY.__getitem__.return_value = mock.Mock(
        return_value=result)
    collaborator_mock.task_runner.get_required_tensorkeys_for_function = mock.Mock(
        return_value=[tensor_key])
    collaborator_mock.send_task_results = mock.Mock()
    collaborator_mock.do_task(task, round_number)

    collaborator_mock.send_task_results.assert_called_with(result[0], round_number, task.name)


def test_do_task_no_registry(collaborator_mock, tensor_key):
    """Test that do_task works correctly when no TASK_REGISTRY in task_runner."""
    round_number = 0
    nparray = numpy.array([0, 1, 2, 3, 4])
    tensor_key = tensor_key._replace(origin='GLOBAL')
    result = {tensor_key: nparray}, {tensor_key: nparray}

    task = mock.MagicMock()
    task.function_name = 'func_name'
    task.name = 'task_name'
    task.task_type = 'validate'
    task.__getitem__ = mock.Mock(side_effect=[task.function_name, {}])

    del collaborator_mock.task_runner.TASK_REGISTRY
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
    tensor_dict = {tensor_key: 0}
    round_number = 0
    data_size = -1
    collaborator_mock.nparray_to_named_tensor = mock.Mock(return_value=None)
    collaborator_mock.client.send_local_task_results = mock.Mock()
    collaborator_mock.send_task_results(tensor_dict, round_number, task_name)

    collaborator_mock.client.send_local_task_results.assert_called_with(
        collaborator_mock.collaborator_name, round_number, task_name, data_size, [None])


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
        collaborator_mock.collaborator_name, round_number, task_name, data_size, [])


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
        collaborator_mock.collaborator_name, round_number, task_name, data_size, [])


def test_named_tensor_to_nparray_without_tags(collaborator_mock, named_tensor):
    """Test that named_tensor_to_nparray works correctly for tensor without tags."""
    nparray = collaborator_mock.named_tensor_to_nparray(named_tensor)

    assert named_tensor.data_bytes == nparray


@pytest.mark.parametrize('tag', ['compressed', 'lossy_compressed'])
def test_named_tensor_to_nparray_compressed_tag(collaborator_mock, named_tensor, tag):
    """Test that named_tensor_to_nparray works correctly for tensor with tags."""
    named_tensor.tags.append(tag)
    nparray = collaborator_mock.named_tensor_to_nparray(named_tensor)

    assert isinstance(nparray, numpy.ndarray)


def test_nparray_to_named_tensor(collaborator_mock, tensor_key, named_tensor):
    """Test that nparray_to_named_tensor works correctly."""
    named_tensor.tags.append('compressed')
    nparray = collaborator_mock.named_tensor_to_nparray(named_tensor)
    tensor = collaborator_mock.nparray_to_named_tensor(tensor_key, nparray)
    assert tensor.data_bytes == named_tensor.data_bytes
    assert tensor.lossless is True


def test_nparray_to_named_tensor_trained(collaborator_mock, tensor_key_trained, named_tensor):
    """Test that nparray_to_named_tensor works correctly for trained tensor."""
    named_tensor.tags.append('compressed')
    collaborator_mock.delta_updates = True
    nparray = collaborator_mock.named_tensor_to_nparray(named_tensor)
    collaborator_mock.tensor_db.get_tensor_from_cache = mock.Mock(
        return_value=nparray)
    tensor = collaborator_mock.nparray_to_named_tensor(tensor_key_trained, nparray)
    assert len(tensor.data_bytes) == 32
    assert tensor.lossless is False
    assert 'delta' in tensor.tags


@pytest.mark.parametrize('require_lossless', [True, False])
def test_get_aggregated_tensor_from_aggregator(collaborator_mock, tensor_key,
                                               named_tensor, require_lossless):
    """Test that get_aggregated_tensor works correctly."""
    collaborator_mock.client.get_aggregated_tensor = mock.Mock(return_value=named_tensor)
    nparray = collaborator_mock.get_aggregated_tensor_from_aggregator(tensor_key, require_lossless)

    collaborator_mock.client.get_aggregated_tensor.assert_called_with(
        collaborator_mock.collaborator_name, tensor_key.tensor_name, tensor_key.round_number,
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
    collaborator_mock.tensor_codec.find_dependencies = mock.Mock(return_value=[tensor_key])
    collaborator_mock.get_aggregated_tensor_from_aggregator = mock.Mock()
    collaborator_mock.get_data_for_tensorkey(tensor_key)
    collaborator_mock.get_aggregated_tensor_from_aggregator.assert_called_with(
        tensor_key, require_lossless=True)


def test_get_numpy_dict_for_tensorkeys(collaborator_mock, tensor_key):
    """Test that get_numpy_dict_for_tensorkeys works."""
    expected_nparray = 'some_data'
    collaborator_mock.tensor_db.get_tensor_from_cache = mock.Mock(
        return_value='some_data')
    numpy_dict = collaborator_mock.get_numpy_dict_for_tensorkeys([tensor_key])

    assert numpy_dict == {tensor_key.tensor_name: expected_nparray}


def test_run_time_to_quit(collaborator_mock):
    """Test that run works correctly if is time to quit."""
    collaborator_mock.get_tasks = mock.Mock(return_value=([], 0, 0, True))
    collaborator_mock.run()


def test_run(collaborator_mock):
    """Test that run works correctly."""
    round_number = 0
    collaborator_mock.get_tasks = mock.Mock()
    collaborator_mock.get_tasks.side_effect = [(['task'], round_number, 0, False),
                                               (['task'], round_number, 0, True)]
    collaborator_mock.do_task = mock.Mock(return_value={'metric': 0.0})
    collaborator_mock.run()
    collaborator_mock.do_task.assert_called_with('task', round_number)


def test_run_simulation_time_to_quit(collaborator_mock):
    """Test that run_simulation works correctly if is time to quit."""
    round_number = 0
    collaborator_mock.get_tasks = mock.Mock(return_value=([], round_number, 0, True))
    collaborator_mock.run_simulation()


def test_run_simulation(collaborator_mock):
    """Test that run_simulation works correctly."""
    round_number = 0
    collaborator_mock.get_tasks = mock.Mock(return_value=(['task'], round_number, 0, False))

    collaborator_mock.do_task = mock.Mock()
    collaborator_mock.run_simulation()
    collaborator_mock.do_task.assert_called_with('task', round_number)
