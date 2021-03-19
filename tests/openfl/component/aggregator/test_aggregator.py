# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Aggregator tests module."""

from unittest import mock
import pytest

from openfl.protocols import ModelProto, utils
from openfl.component import aggregator
from openfl.component.assigner import Assigner
from openfl.utilities import TaskResultKey


@pytest.fixture
def model():
    """Initialize the model."""
    model = ModelProto()
    tensor = model.tensors.add()
    tensor.name = 'test-tensor-name'
    tensor.round_number = 0
    tensor.lossless = True
    tensor.report = True
    tensor.tags.append('some_tag')
    metadata = tensor.transformer_metadata.add()
    metadata.int_to_float[1] = 1.
    metadata.int_list.extend([1, 8])
    metadata.bool_list.append(True)
    tensor.data_bytes = 32 * b'1'

    return utils.model_proto_to_dict(model)


@pytest.fixture()
def assigner():
    """Initialize the assigner."""
    Assigner.define_task_assignments = mock.Mock()
    assigner = Assigner(None, None, None)
    assigner.define_task_assignments = mock.Mock()
    return assigner


@pytest.fixture
def agg(mocker, model, assigner):
    """Initialize the aggregator."""
    mocker.patch('openfl.protocols.utils.load_proto', return_value=model)
    agg = aggregator.Aggregator(
        'some_uuid',
        'federation_uuid',
        ['col1', 'col2'],

        'init_state_path',
        'best_state_path',
        'last_state_path',

        assigner,
    )
    return agg


@pytest.mark.parametrize(
    'cert_common_name,collaborator_common_name,authorized_cols,single_cccn,expected_is_valid', [
        ('col1', 'col1', ['col1', 'col2'], '', True),
        ('col2', 'col2', ['col1', 'col2'], '', True),
        ('col3', 'col3', ['col1', 'col2'], '', False),
        ('col3', 'col3', ['col1', 'col2'], '', False),
        ('col1', 'col2', ['col1', 'col2'], '', False),
        ('col2', 'col1', ['col1', 'col2'], '', False),
        ('col1', 'col1', [], '', False),
        ('col1', 'col1', ['col1', 'col2'], 'col1', True),
        ('col1', 'col1', ['col1', 'col2'], 'col2', False),
        ('col3', 'col3', ['col1', 'col2'], 'col3', False),
        ('col1', 'col1', ['col1', 'col2'], 'col3', False),
    ])
def test_valid_collaborator_cn_and_id(agg, cert_common_name, collaborator_common_name,
                                      authorized_cols, single_cccn, expected_is_valid):
    """Test that valid_collaborator_cn_and_id works correctly."""
    ac = agg.authorized_cols
    agg.authorized_cols = authorized_cols
    agg.single_col_cert_common_name = single_cccn
    is_valid = agg.valid_collaborator_cn_and_id(cert_common_name, collaborator_common_name)
    agg.authorized_cols = ac
    agg.single_col_cert_common_name = ''

    assert is_valid == expected_is_valid


@pytest.mark.parametrize('quit_job_sent_to,authorized_cols,expected', [
    (['col1', 'col2'], ['col1', 'col2'], True),
    (['col1'], ['col1', 'col2'], False),
    ([], [], True),
])
def test_all_quit_jobs_sent(agg, quit_job_sent_to, authorized_cols, expected):
    """Test that valid_collaborator_cn_and_id works correctly."""
    ac = agg.authorized_cols
    agg.authorized_cols = authorized_cols
    agg.quit_job_sent_to = quit_job_sent_to
    all_quit_jobs_sent = agg.all_quit_jobs_sent()
    agg.authorized_cols = ac
    agg.quit_job_sent_to = []

    assert all_quit_jobs_sent == expected


def test_get_sleep_time(agg):
    """Test that get_sleep_time returns 10."""
    assert 10 == agg._get_sleep_time()


@pytest.mark.parametrize('round_number,rounds_to_train,expected', [
    (0, 10, False), (10, 10, True), (9, 10, False), (10, 0, True)
])
def test_time_to_quit(agg, round_number, rounds_to_train, expected):
    """Test that test_time_to_quit works correctly."""
    rn = agg.round_number
    rtt = agg.rounds_to_train
    agg.round_number = round_number
    agg.rounds_to_train = rounds_to_train
    time_to_quit = agg._time_to_quit()
    assert expected == time_to_quit

    agg.round_number = rn
    agg.rounds_to_train = rtt


@pytest.mark.parametrize(
    'col_name,tasks,time_to_quit,exp_tasks,exp_sleep_time,exp_time_to_quit', [
        ('col1', ['task_name'], True, None, 0, True),
        ('col1', [], False, None, 10, False),
        ('col1', ['task_name'], False, ['task_name'], 0, False),
    ])
def test_get_tasks(agg, col_name, tasks, time_to_quit,
                   exp_tasks, exp_sleep_time, exp_time_to_quit):
    """Test that test_get_tasks works correctly."""
    agg.assigner.get_tasks_for_collaborator = mock.Mock(return_value=tasks)
    agg._time_to_quit = mock.Mock(return_value=time_to_quit)
    tasks, round_number, sleep_time, time_to_quit = agg.get_tasks('col1')
    assert (tasks, sleep_time, time_to_quit) == (exp_tasks, exp_sleep_time, exp_time_to_quit)


def test_get_aggregated_tensor(agg):
    """Test that test_get_tasks is failed without a correspond data."""
    collaborator_name = 'col1'
    tensor_name = 'test_tensor_name'
    require_lossless = False
    round_number = 0
    report = False
    tags = ['compressed']
    with pytest.raises(ValueError):
        agg.get_aggregated_tensor(
            collaborator_name, tensor_name, round_number, report, tags, require_lossless)


def test_collaborator_task_completed_none(agg):
    """Test that returns False if there are not collaborator tasks results."""
    round_num = 0
    is_completed = agg._collaborator_task_completed(
        'col1', 'task_name', round_num)
    assert is_completed is False


def test_collaborator_task_completed_true(agg):
    """Test that returns True if there are collaborator tasks results."""
    round_num = 0
    task_name = 'test_task_name'
    col1 = 'one'
    agg.collaborator_tasks_results = {
        TaskResultKey(task_name, col1, round_num): 1
    }
    is_completed = agg._collaborator_task_completed(
        col1, task_name, round_num)

    assert is_completed is True


def test_is_task_done_no_cols(agg):
    """Test that is_task_done returns True without corresponded collaborators."""
    task_name = 'test_task_name'
    agg.assigner.get_collaborators_for_task = mock.Mock(return_value=[])
    is_task_done = agg._is_task_done(task_name)

    assert is_task_done is True


def test_is_task_done_not_done(agg):
    """Test that is_task_done returns False in the corresponded case."""
    task_name = 'test_task_name'
    col1 = 'one'
    col2 = 'two'
    agg.assigner.get_collaborators_for_task = mock.Mock(return_value=[col1, col2])
    is_task_done = agg._is_task_done(task_name)

    assert is_task_done is False


def test_is_task_done_done(agg):
    """Test that is_task_done returns True in the corresponded case."""
    round_num = 0
    task_name = 'test_task_name'
    col1 = 'one'
    col2 = 'two'
    agg.assigner.get_collaborators_for_task = mock.Mock(return_value=[col1, col2])
    agg.collaborator_tasks_results = {
        TaskResultKey(task_name, col1, round_num): 1,
        TaskResultKey(task_name, col2, round_num): 1
    }
    is_task_done = agg._is_task_done(task_name)

    assert is_task_done is True


def test_is_round_done_no_tasks(agg):
    """Test that is_round_done returns True in the corresponded case."""
    agg.assigner.get_all_tasks_for_round = mock.Mock(return_value=[])
    is_round_done = agg._is_round_done()

    assert is_round_done is True


def test_is_round_done_not_done(agg):
    """Test that is_round_done returns False in the corresponded case."""
    round_num = 0
    task_name = 'test_task_name'
    col1 = 'one'
    col2 = 'two'
    agg.assigner.get_all_tasks_for_round = mock.Mock(return_value=[task_name])
    agg.assigner.get_collaborators_for_task = mock.Mock(return_value=[col1, col2])
    agg.collaborator_tasks_results = {
        TaskResultKey(task_name, col1, round_num): 1,
    }
    is_round_done = agg._is_round_done()

    assert is_round_done is False


def test_is_round_done_done(agg):
    """Test that is_round_done returns True in the corresponded case."""
    round_num = 0
    task_name = 'test_task_name'
    col1 = 'one'
    col2 = 'two'
    agg.assigner.get_all_tasks_for_round = mock.Mock(return_value=[task_name])
    agg.assigner.get_collaborators_for_task = mock.Mock(return_value=[col1, col2])
    agg.collaborator_tasks_results = {
        TaskResultKey(task_name, col1, round_num): 1,
        TaskResultKey(task_name, col2, round_num): 1
    }
    is_round_done = agg._is_round_done()

    assert is_round_done is True
