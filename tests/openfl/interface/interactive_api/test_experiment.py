# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Experiment tests."""

from unittest import mock

import pytest

from openfl.component.assigner.tasks import TrainTask
from openfl.component.assigner.tasks import ValidateTask
from openfl.interface.interactive_api.experiment import FLExperiment
from openfl.interface.interactive_api.experiment import TaskKeeper


@pytest.fixture()
def all_registered_tasks():
    """Return all registered tasks fixture."""
    return {
        'train': TrainTask(
            name='train',
            function_name='train_func',
        ),
        'locally_tuned_model_validate': ValidateTask(
            name='locally_tuned_model_validate',
            function_name='validate',
            apply_local=True,
        ),
        'aggregated_model_validate': ValidateTask(
            name='aggregated_model_validate',
            function_name='validate',
        ),
    }


def test_define_task_assigner_all_tasks(all_registered_tasks):
    """Test define_task_assigner if all task types are registered."""
    task_keeper = TaskKeeper()
    task_keeper.get_registered_tasks = mock.Mock(return_value=all_registered_tasks)
    rounds_to_train = 10
    task_assigner_fn = FLExperiment(None).define_task_assigner(task_keeper, rounds_to_train)
    tasks_by_collaborator = task_assigner_fn(['one', 'two'], 1)
    assert tasks_by_collaborator['one'] == list(all_registered_tasks.values())


def test_define_task_assigner_val_tasks():
    """Test define_task_assigner if only validate types are registered."""
    task_keeper = TaskKeeper()
    agg_task = ValidateTask(
        name='aggregated_model_validate',
        function_name='validate',
    )
    task_keeper.get_registered_tasks = mock.Mock(return_value={
        'aggregated_model_validate': agg_task
    })
    rounds_to_train = 1
    task_assigner_fn = FLExperiment(None).define_task_assigner(task_keeper, rounds_to_train)
    tasks_by_collaborator = task_assigner_fn(['one', 'two'], 1)
    assert tasks_by_collaborator['one'] == [agg_task]


def test_define_task_assigner_exception_validate():
    """Test define_task_assigner if only validate tasks are registered and rounds more than 1."""
    task_keeper = TaskKeeper()
    agg_task = ValidateTask(
        name='aggregated_model_validate',
        function_name='validate',
    )
    task_keeper.get_registered_tasks = mock.Mock(return_value={
        'aggregated_model_validate': agg_task
    })
    rounds_to_train = 10
    with pytest.raises(Exception):
        FLExperiment(None).define_task_assigner(task_keeper, rounds_to_train)


def test_define_task_assigner_exception_only_train():
    """Test define_task_assigner if only train task types are registered."""
    task_keeper = TaskKeeper()
    train_task = TrainTask(
        name='train',
        function_name='train',
    )
    task_keeper.get_registered_tasks = mock.Mock(return_value={
        'train': train_task
    })
    rounds_to_train = 10
    with pytest.raises(Exception):
        FLExperiment(None).define_task_assigner(task_keeper, rounds_to_train)


def test_define_task_assigner_exception_no_tasks():
    """Test define_task_assigner if no tasks are registered."""
    task_keeper = TaskKeeper()
    task_keeper.get_registered_tasks = mock.Mock(return_value={})
    rounds_to_train = 1
    with pytest.raises(Exception):
        FLExperiment(None).define_task_assigner(task_keeper, rounds_to_train)
