# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""CustomAssigner tests."""

from unittest import mock

import pytest

from openfl.interface.aggregation_functions import GeometricMedian
from openfl.interface.aggregation_functions import WeightedAverage
from openfl.component.assigner.custom_assigner import Assigner
from openfl.component.assigner.tasks import TrainTask
from openfl.component.assigner.tasks import ValidateTask


default_tasks = [
    TrainTask(
        name='train',
        function_name='train_func',
    ),
    ValidateTask(
        name='locally_tuned_model_validate',
        function_name='validate',
        apply_local=True,
    ),
    ValidateTask(
        name='aggregated_model_validate',
        function_name='validate',
    ),
]


def assigner_function(collaborators, round_number, **kwargs):
    """Return tasks by collaborator."""
    tasks_by_collaborator = {}
    for collaborator in collaborators:
        tasks_by_collaborator[collaborator] = default_tasks
    return tasks_by_collaborator


@pytest.fixture()
def assigner():
    """Return Assigner fixture."""
    assigner = Assigner(
        assigner_function=assigner_function,
        aggregation_functions_by_task={
            'train_func': GeometricMedian()
        },
        authorized_cols=['one', 'two'],
        rounds_to_train=10,
    )
    assigner.define_task_assignments = mock.Mock()
    return assigner


def test_define_task_assignments(assigner):
    """Test `define_task_assignments` is working."""
    assigner.define_task_assignments()


def test_get_tasks_for_collaborator(assigner):
    """Test `get_tasks_for_collaborator` base working."""
    tasks = assigner.get_tasks_for_collaborator('one', 2)

    assert tasks == default_tasks
    assert len(tasks) == 3
    assert isinstance(tasks[0], TrainTask)
    assert isinstance(tasks[1], ValidateTask)


def test_get_collaborators_for_task(assigner):
    """Test `get_collaborators_for_task` base working."""
    collaborators = assigner.get_collaborators_for_task('train', 2)

    assert collaborators == ['one', 'two']


def test_get_all_tasks_for_round(assigner):
    """Test `get_all_tasks_for_round` base working."""
    all_tasks = assigner.get_all_tasks_for_round(2)

    assert all_tasks == [task.name for task in default_tasks]


def test_get_aggregation_type_for_task(assigner):
    """Test `get_aggregation_type_for_task` base working."""
    agg_fn = assigner.get_aggregation_type_for_task('train')

    assert isinstance(agg_fn, GeometricMedian)


def test_get_aggregation_type_for_task_by_default():
    """Test get_aggregation_type_for_task working without assigned agg functions."""
    assigner = Assigner(
        assigner_function=assigner_function,
        aggregation_functions_by_task={},
        authorized_cols=['one', 'two'],
        rounds_to_train=10,
    )
    agg_fn = assigner.get_aggregation_type_for_task('train')

    assert isinstance(agg_fn, WeightedAverage)
