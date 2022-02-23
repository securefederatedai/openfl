# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom Assigner module."""


import logging
from collections import defaultdict

from openfl.component.aggregation_functions import WeightedAverage

logger = logging.getLogger(__name__)


class Assigner:
    """Custom assigner class."""

    def __init__(self, *, assigner_function, aggregation_functions_by_task, authorized_cols):
        """Initialize."""
        self.aggregation_functions_by_task = aggregation_functions_by_task
        self.authorized_cols = authorized_cols
        self.all_tasks_for_round = {}
        self.collaborators_for_task = defaultdict(list)
        self.collaborator_tasks = defaultdict(list)
        self.assigner_function = assigner_function

        self.define_task_assignments_for_round(0)

    def define_task_assignments_for_round(self, round_number):
        """Abstract method."""
        self.all_tasks_for_round = {}
        self.collaborators_for_task = defaultdict(list)
        self.collaborator_tasks = defaultdict(list)
        tasks_by_collaborator = self.assigner_function(
            self.authorized_cols,
            round_number,
            number_of_callaborators=len(self.authorized_cols)
        )
        for collaborator_name, tasks in tasks_by_collaborator.items():
            self.collaborator_tasks[collaborator_name].extend(tasks)
            for task in tasks:
                self.all_tasks_for_round[task.name] = task
                self.collaborators_for_task[task.name].append(collaborator_name)

    def get_tasks_for_collaborator(self, collaborator_name):
        """Abstract method."""
        return self.collaborator_tasks[collaborator_name]

    def get_collaborators_for_task(self, task_name):
        """Abstract method."""
        return self.collaborators_for_task[task_name]

    def get_all_tasks_for_round(self):
        """
        Return tasks for the current round.

        Currently all tasks are performed on each round,
        But there may be a reason to change this.
        """
        return self.all_tasks_for_round.values()

    def get_aggregation_type_for_task(self, function_name):
        """Extract aggregation type from self.tasks."""
        agg_fn = self.aggregation_functions_by_task.get(function_name, WeightedAverage())
        return agg_fn
