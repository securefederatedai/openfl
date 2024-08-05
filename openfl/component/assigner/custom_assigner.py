# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Custom Assigner module."""

import logging
from collections import defaultdict

from openfl.interface.aggregation_functions import WeightedAverage

logger = logging.getLogger(__name__)


class Assigner:
    """Custom assigner class.

    Attributes:
        agg_functions_by_task (dict): Dictionary mapping tasks to their
            respective aggregation functions.
        agg_functions_by_task_name (dict): Dictionary mapping task names to
            their respective aggregation functions.
        authorized_cols (list of str): List of authorized collaborators.
        rounds_to_train (int): Number of rounds to train.
        all_tasks_for_round (defaultdict): Dictionary mapping round numbers
            to tasks.
        collaborators_for_task (defaultdict): Dictionary mapping round numbers
            to collaborators for each task.
        collaborator_tasks (defaultdict): Dictionary mapping round numbers
            to tasks for each collaborator.
        assigner_function (function): Function to assign tasks to
            collaborators.
    """

    def __init__(
        self, *, assigner_function, aggregation_functions_by_task, authorized_cols, rounds_to_train
    ):
        """Initialize the Custom assigner object.

        Args:
            assigner_function (function): Function to assign tasks to
                collaborators.
            aggregation_functions_by_task (dict): Dictionary mapping tasks to
                their respective aggregation functions.
            authorized_cols (list of str): List of authorized collaborators.
            rounds_to_train (int): Number of rounds to train.
        """
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
        """Define task assignments for each round and collaborator.

        This method uses the assigner function to assign tasks to
        collaborators for each round. It also maps tasks to their respective
        aggregation functions.

        Abstract method.

        Args:
            None

        Returns:
            None
        """
        for round_number in range(self.rounds_to_train):
            tasks_by_collaborator = self.assigner_function(
                self.authorized_cols,
                round_number,
                number_of_callaborators=len(self.authorized_cols),
            )
            for collaborator_name, tasks in tasks_by_collaborator.items():
                self.collaborator_tasks[round_number][collaborator_name].extend(tasks)
                for task in tasks:
                    self.all_tasks_for_round[round_number][task.name] = task
                    self.collaborators_for_task[round_number][task.name].append(collaborator_name)
                    if self.agg_functions_by_task:
                        self.agg_functions_by_task_name[task.name] = self.agg_functions_by_task.get(
                            task.function_name, WeightedAverage()
                        )

    def get_tasks_for_collaborator(self, collaborator_name, round_number):
        """Get tasks for a specific collaborator in a specific round.

        Abstract method.

        Args:
            collaborator_name (str): Name of the collaborator.
            round_number (int): Round number.

        Returns:
            list: List of tasks for the collaborator in the specified round.
        """
        return self.collaborator_tasks[round_number][collaborator_name]

    def get_collaborators_for_task(self, task_name, round_number):
        """Get collaborators for a specific task in a specific round.

        Abstract method.

        Args:
            task_name (str): Name of the task.
            round_number (int): Round number.

        Returns:
            list: List of collaborators for the task in the specified round.
        """
        return self.collaborators_for_task[round_number][task_name]

    def get_all_tasks_for_round(self, round_number):
        """Get all tasks for a specific round.

        Currently all tasks are performed on each round,
        But there may be a reason to change this.

        Args:
            round_number (int): Round number.

        Returns:
            list: List of all tasks for the specified round.
        """
        return [task.name for task in self.all_tasks_for_round[round_number].values()]

    def get_aggregation_type_for_task(self, task_name):
        """Get the aggregation type for a specific task (from self.tasks).

        Args:
            task_name (str): Name of the task.

        Returns:
            function: Aggregation function for the task.
        """
        agg_fn = self.agg_functions_by_task_name.get(task_name, WeightedAverage())
        return agg_fn
