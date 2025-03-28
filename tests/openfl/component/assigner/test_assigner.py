# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Assigner tests module."""

from unittest import mock, TestCase

import pytest

from openfl.component.assigner import Assigner


@pytest.fixture()
def assigner():
    """Initialize the assigner."""
    assigner = Assigner
    assigner.define_task_assignments = mock.Mock()
    return assigner


def test_get_aggregation_type_for_task_none(assigner):
    """Assert that aggregation type of custom task is None."""
    task_name = "test_name"
    tasks = {task_name: {}}

    assigner = assigner(tasks, None, None)

    aggregation_type = assigner.get_aggregation_type_for_task(task_name)

    assert aggregation_type is None


def test_get_aggregation_type_for_task(assigner):
    """Assert that aggregation type of task is getting correctly."""
    task_name = "test_name"
    test_aggregation_type = "test_aggregation_type"
    tasks = {task_name: {"aggregation_type": test_aggregation_type}}
    assigner = assigner(tasks, None, None)

    aggregation_type = assigner.get_aggregation_type_for_task(task_name)

    assert aggregation_type == test_aggregation_type


def test_get_all_tasks_for_round(assigner):
    """Assert that assigner tasks object is list."""
    assigner = Assigner(None, None, None)
    tasks = assigner.get_all_tasks_for_round("test")

    assert isinstance(tasks, list)

def test_default_task_group(assigner):
    """Assert that by default learning task_group is assigned."""
    assigner = Assigner(None,None,None)
    assert assigner.selected_task_group == None

def test_task_group_filtering_no_task_groups(assigner):
    """Assert that task_group_filtering does not filter when no task_groups are defined."""
    assigner = Assigner(None,None,None)
    assigner.selected_task_group = "test_group"
    assigner.define_task_assignments()
    assert not hasattr(assigner, "task_groups")

class TestNotImplError(TestCase):
    def test_define_task_assignments(self):
        # TODO: define_task_assignments is defined as a mock in multiple fixtures,
        # which leads the function to behave as a mock here and other tests.
        pass

    def test_get_tasks_for_collaborator(self):
        with self.assertRaises(NotImplementedError):
            assigner = Assigner(None, None, None)
            assigner.get_tasks_for_collaborator("col1", 0)

    def test_get_collaborators_for_task(self):
        with self.assertRaises(NotImplementedError):
            assigner = Assigner(None, None, None)
            assigner.get_collaborators_for_task("task_name", 0)
