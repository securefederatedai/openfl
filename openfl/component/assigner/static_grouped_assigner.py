# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Static grouped assigner module."""

from openfl.component.assigner.assigner import Assigner


class StaticGroupedAssigner(Assigner):
    r"""The task assigner maintains a list of tasks.

    Also it decides the policy for which collaborator should run those tasks
    There may be many types of policies implemented, but a natural place to
    start is with a:

        - StaticGroupedAssigner :
            Given a set of task groups, and a list of
            collaborators for that group, assign tasks for of collaborators in
            the federation.
            After assigning the tasks to collaborator, those tasks should be
            carried out each round (no reassignment between rounds).
        - GroupedAssigner :
            Given task groups and a list of collaborators that
            belong to that task group, carry out tasks for each round of
            experiment.

    Attributes:
        task_groups* (list of object): Task groups to assign.

    .. note::
        \* - Plan setting.
    """

    with_selected_task_group = Assigner.with_selected_task_group

    def __init__(self, task_groups, **kwargs):
        """Initializes the StaticGroupedAssigner.

        Args:
            task_groups (list of object): Task groups to assign.
            **kwargs: Additional keyword arguments.
        """
        self.task_groups = task_groups
        super().__init__(**kwargs)

    @with_selected_task_group
    def define_task_assignments(self):
        """Define task assignments for each round and collaborator.

        This method uses the assigner function to assign tasks to
        collaborators for each round.
        It also maps tasks to their respective aggregation functions.

        Args:
            None

        Returns:
            None
        """
        cols_amount = sum([len(group["collaborators"]) for group in self.task_groups])
        authorized_cols_amount = len(self.authorized_cols)

        unique_cols = {col for group in self.task_groups for col in group["collaborators"]}
        unique_authorized_cols = set(self.authorized_cols)

        assert cols_amount == authorized_cols_amount and unique_cols == unique_authorized_cols, (
            f"Collaborators in each group must be distinct: {unique_cols}, {unique_authorized_cols}"
        )

        # Start by finding all of the tasks in all specified groups
        self.all_tasks_in_groups = list(
            {task for group in self.task_groups for task in group["tasks"]}
        )

        # Initialize the map of collaborators for a given task on a given round
        for task in self.all_tasks_in_groups:
            self.collaborators_for_task[task] = {i: [] for i in range(self.rounds)}

        for group in self.task_groups:
            group_col_list = group["collaborators"]
            self.task_group_collaborators[group["name"]] = group_col_list
            for col in group_col_list:
                # For now, we assume that collaborators have the same tasks for
                # every round
                self.collaborator_tasks[col] = dict.fromkeys(range(self.rounds), group["tasks"])
            # Now populate reverse lookup of tasks->group
            for task in group["tasks"]:
                for round_ in range(self.rounds):
                    # This should append the list of collaborators performing
                    # that task
                    self.collaborators_for_task[task][round_] += group_col_list

    def get_tasks_for_collaborator(self, collaborator_name, round_number):
        """Get tasks for a specific collaborator in a specific round.

        Args:
            collaborator_name (str): Name of the collaborator.
            round_number (int): Round number.

        Returns:
            list: List of tasks for the collaborator in the specified round.
        """
        return self.collaborator_tasks[collaborator_name][round_number]

    def get_collaborators_for_task(self, task_name, round_number):
        """Get collaborators for a specific task in a specific round.

        Args:
            task_name (str): Name of the task.
            round_number (int): Round number.

        Returns:
            list: List of collaborators for the task in the specified round.
        """
        return self.collaborators_for_task[task_name][round_number]
