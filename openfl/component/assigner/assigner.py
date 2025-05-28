# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Assigner module."""

import logging
from functools import wraps

logger = logging.getLogger(__name__)


class Assigner:
    r"""
    The task assigner maintains a list of tasks.

    Also it decides the policy for which collaborator should run those tasks.
    There may be many types of policies implemented, but a natural place to start
    is with a:

        - RandomGroupedTaskAssigner :
            Given a set of task groups, and a percentage,
            assign that task group to that percentage of collaborators in the federation.
            After assigning the tasks to collaborator, those tasks should be carried
            out each round (no reassignment between rounds).
        - GroupedTaskAssigner :
            Given task groups and a list of collaborators that belong to that task group,
            carry out tasks for each round of experiment.

    Attributes:
        tasks* (list of object): List of tasks to assign.
        authorized_cols (list of str): Collaborators.
        rounds (int): Number of rounds to train.
        all_tasks_in_groups (list): All tasks in groups.
        task_group_collaborators (dict): Task group collaborators.
        collaborators_for_task (dict): Collaborators for each task.
        collaborator_tasks (dict): Tasks for each collaborator.

    .. note::
        \* - ``tasks`` argument is taken from ``tasks`` section of FL plan YAML file.
    """

    def __init__(
        self,
        tasks,
        authorized_cols,
        rounds_to_train,
        selected_task_group: str = None,
        **kwargs,
    ):
        """Initializes the Assigner.

        Args:
            tasks (list of object): List of tasks to assign.
            authorized_cols (list of str): Collaborators.
            rounds_to_train (int): Number of training rounds.
            selected_task_group (str, optional): Selected task_group.
            **kwargs: Additional keyword arguments.
        """
        self.tasks = tasks
        self.authorized_cols = authorized_cols
        self.rounds = rounds_to_train
        self.selected_task_group = selected_task_group
        self.all_tasks_in_groups = []

        self.task_group_collaborators = {}
        self.collaborators_for_task = {}
        self.collaborator_tasks = {}

        self.define_task_assignments()

    def define_task_assignments(self):
        """Abstract method."""
        raise NotImplementedError

    def get_tasks_for_collaborator(self, collaborator_name, round_number):
        """Abstract method."""
        raise NotImplementedError

    def get_collaborators_for_task(self, task_name, round_number):
        """Abstract method."""
        raise NotImplementedError

    def is_task_group_evaluation(self):
        """Check if the selected task group is for 'evaluation' run.

        Returns:
            bool: True if the selected task group is 'evaluation', False otherwise.
        """
        if hasattr(self, "selected_task_group"):
            return self.selected_task_group == "evaluation"
        return False

    def get_all_tasks_for_round(self, round_number):
        """Return tasks for the current round.

        Currently all tasks are performed on each round,
        But there may be a reason to change this.

        Args:
            round_number (int): Round number.

        Returns:
            list: List of tasks for the current round.
        """
        return self.all_tasks_in_groups

    def get_aggregation_type_for_task(self, task_name):
        """Extract aggregation type from self.tasks.

        Args:
            task_name (str): Name of the task.

        Returns:
            str: Aggregation type for the task.
        """
        if "aggregation_type" not in self.tasks[task_name]:
            return None
        return self.tasks[task_name]["aggregation_type"]

    @classmethod
    def with_selected_task_group(cls, func):
        """Decorator to filter task groups based on selected_task_group.

        This decorator should be applied to define_task_assignments() method
        in Assigner subclasses to handle task_group filtering.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if selection of task_group is applicable
            if hasattr(self, "selected_task_group") and self.selected_task_group is not None:
                # Verify task_groups exists before attempting filtering
                if not hasattr(self, "task_groups"):
                    logger.warning(
                        "Task group specified for selection but no task_groups found. "
                        "Skipping filtering. This might be intentional for custom assigners."
                    )
                    return func(self, *args, **kwargs)

                assert self.task_groups, "No task_groups defined in assigner."

                # Perform the filtering
                selected_task_groups = [
                    group for group in self.task_groups if group["name"] == self.selected_task_group
                ]

                assert len(selected_task_groups) == 1, (
                    f"Only one task group with name {self.selected_task_group} should exist"
                )

                # Since we have filtered to one of the task_groups, we need to ensure that
                # the selected_task_group percentage compute allocation is defaulted to 1.0
                current_percentage = selected_task_groups[0]["percentage"]
                logger.info(
                    f"`percentage` for task_group {self.selected_task_group} is "
                    f"{current_percentage}, setting it to 1.0"
                )
                selected_task_groups[0]["percentage"] = 1.0

                self.task_groups = selected_task_groups

            # Call the original method
            return func(self, *args, **kwargs)

        return wrapper
