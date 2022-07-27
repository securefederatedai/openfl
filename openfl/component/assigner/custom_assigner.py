# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom Assigner module."""


import logging
from collections import defaultdict

from openfl.interface.aggregation_functions import WeightedAverage

logger = logging.getLogger(__name__)


class Assigner:
    """Custom assigner class."""

    def __init__(
            self,
            *,
            assigner_function,
            aggregation_functions_by_task,
            authorized_cols,
            rounds_to_train
    ):
        """Initialize."""
        self.agg_functions_by_task = aggregation_functions_by_task
        self.agg_functions_by_task_name = {}
        self.authorized_cols = authorized_cols
        self.rounds_to_train = rounds_to_train
        self.all_tasks_for_round = defaultdict(dict)
        self.collaborators_for_task = defaultdict(lambda: defaultdict(list))
        self.collaborator_tasks = defaultdict(lambda: defaultdict(list))
        self.assigner_function = assigner_function

        self.define_task_assignments()

    def define_task_assignments(self):
        """Abstract method."""
        for round_number in range(self.rounds_to_train):
            tasks_by_collaborator = self.assigner_function(
                self.authorized_cols,
                round_number,
                number_of_callaborators=len(self.authorized_cols)
            )
            for collaborator_name, tasks in tasks_by_collaborator.items():
                self.collaborator_tasks[round_number][collaborator_name].extend(tasks)
                for task in tasks:
                    self.all_tasks_for_round[round_number][task.name] = task
                    self.collaborators_for_task[round_number][task.name].append(collaborator_name)
                    if self.agg_functions_by_task:
                        self.agg_functions_by_task_name[
                            task.name
                        ] = self.agg_functions_by_task.get(task.function_name, WeightedAverage())

    def get_tasks_for_collaborator(self, collaborator_name, round_number):
        """Abstract method."""
        return self.collaborator_tasks[round_number][collaborator_name]

    def get_collaborators_for_task(self, task_name, round_number):
        """Abstract method."""
        return self.collaborators_for_task[round_number][task_name]

    def get_all_tasks_for_round(self, round_number):
        """
        Return tasks for the current round.

        Currently all tasks are performed on each round,
        But there may be a reason to change this.
        """
        return [task.name for task in self.all_tasks_for_round[round_number].values()]

    def get_aggregation_type_for_task(self, task_name):
        """Extract aggregation type from self.tasks."""
        agg_fn = self.agg_functions_by_task_name.get(task_name, WeightedAverage())
        return agg_fn
