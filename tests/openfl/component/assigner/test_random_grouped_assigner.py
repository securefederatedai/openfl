# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""RandomGroupedAssigner tests."""

import pytest

from openfl.component.assigner import RandomGroupedAssigner, Assigner

ROUNDS_TO_TRAIN = 10


@pytest.fixture
def task_groups():
    """Initialize task groups."""
    task_groups = [
        {
            'name': 'learning',
            'percentage': 1.0,
            'tasks': [
                'aggregated_model_validation',
                'train',
                'locally_tuned_model_validation'
            ]
        },
        {
            'name': 'evaluation',
            'percentage': 0,
            'tasks': [
                'aggregated_model_validation'
            ]
        },
    ]
    return task_groups


@pytest.fixture
def authorized_cols():
    """Initialize authorized cols."""
    return ['one', 'two']


@pytest.fixture
def assigner(task_groups, authorized_cols):
    """Initialize assigner."""
    assigner = RandomGroupedAssigner(
        task_groups=task_groups,  # Pass task_groups here
        tasks=None,
        authorized_cols=authorized_cols,
        rounds_to_train=ROUNDS_TO_TRAIN
    )
    return assigner


def test_define_task_assignments(assigner):
    """Test `define_task_assignments` is working."""
    assigner.define_task_assignments()

def test_check_default_task_group(assigner):
    """Assert that by default learning task_group is assigned."""
    assert assigner.selected_task_group == None

@pytest.mark.parametrize('round_number', range(ROUNDS_TO_TRAIN))
def test_get_default_tasks_for_collaborator(assigner, task_groups,
                                    authorized_cols, round_number):
    """Test that assigner tasks correspond to task groups defined."""
    tasks = assigner.get_tasks_for_collaborator(
        authorized_cols[0], round_number)
    assert tasks == task_groups[0]['tasks']
    assert assigner.selected_task_group == None

@pytest.mark.parametrize('round_number', range(ROUNDS_TO_TRAIN))
def test_get_tasks_for_collaborator(assigner, task_groups,
                                    authorized_cols, round_number):
    """Test that assigner tasks correspond to task groups defined."""
    tasks = assigner.get_tasks_for_collaborator(
        authorized_cols[0], round_number)
    assert tasks == task_groups[0]['tasks']

@pytest.mark.parametrize('round_number', range(ROUNDS_TO_TRAIN))
def test_get_collaborators_for_task(
        assigner, task_groups, round_number, authorized_cols):
    """Check that assigner collaborators set is equal to authorized collaborators set."""
    for task_name in task_groups[0]['tasks']:
        cols = assigner.get_collaborators_for_task(task_name, round_number)
        assert set(cols) == set(authorized_cols)

@pytest.mark.parametrize('round_number', range(ROUNDS_TO_TRAIN))
def test_get_assigned_tasks_for_collaborator(task_groups,
                                    authorized_cols, round_number):
    """Test that assigner tasks correspond to task groups defined."""
    eval_assigner = RandomGroupedAssigner(
        task_groups=task_groups,
        tasks=None,
        authorized_cols=authorized_cols,
        rounds_to_train=ROUNDS_TO_TRAIN,
        selected_task_group="evaluation"
    )
    eval_assigner.define_task_assignments()
    assert eval_assigner.selected_task_group == "evaluation"
    tasks = eval_assigner.get_tasks_for_collaborator(
        authorized_cols[0], round_number)
    assert tasks == task_groups[1]['tasks']
    assert eval_assigner.selected_task_group == task_groups[1]['name']
    assert len(eval_assigner.task_groups) == 1
    assert eval_assigner.task_groups[0]['percentage'] == 1.0
