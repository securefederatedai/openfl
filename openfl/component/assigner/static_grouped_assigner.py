# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Static grouped assigner module."""

from .assigner import Assigner


class StaticGroupedAssigner(Assigner):
    r"""
    The task assigner maintains a list of tasks.

    Also it decides the policy for
    which collaborator should run those tasks
    There may be many types of policies implemented, but a natural place to
    start is with a:

    StaticGroupedAssigner  - Given a set of task groups, and a list of
                             collaborators for that group, assign tasks for
                             of collaborators in the federation. After assigning
                             the tasks to collaborator, those tasks
                             should be carried out each round (no reassignment
                             between rounds)
    GroupedAssigner -        Given task groups and a list of collaborators that
                             belong to that task group, carry out tasks for
                             each round of experiment

    Args:
        task_groups* (list of obj): task groups to assign.

    Note:
        \* - Plan setting.
    """

    def __init__(self, task_groups, **kwargs):
        """Initialize."""
        self.task_groups = task_groups
        super().__init__(**kwargs)

    def define_task_assignments(self):
        """All of the logic to set up the map of tasks to collaborators is done here."""
        assert (sum(
            [len(group['collaborators']) for group in self.task_groups]
        ) == len(self.authorized_cols) and set(
            [col for group in self.task_groups for col in group['collaborators']
             ]) == set(self.authorized_cols)), (
            'Collaborators in each group must be distinct: {}, {}'.format(
                set([col for group in self.task_groups
                     for col in group['collaborators']]),
                set(self.authorized_cols))
        )

        # Start by finding all of the tasks in all specified groups
        self.all_tasks_in_groups = list(set(
            [task for group in self.task_groups for task in group['tasks']]
        ))

        # Initialize the map of collaborators for a given task on a given round
        for task in self.all_tasks_in_groups:
            self.collaborators_for_task[task] = {
                i: [] for i in range(self.rounds)
            }

        # col_list_size = len(self.authorized_cols)
        for group in self.task_groups:
            group_col_list = group['collaborators']
            self.task_group_collaborators[group['name']] = group_col_list
            for col in group_col_list:
                # For now, we assume that collaborators have the same tasks for
                # every round
                self.collaborator_tasks[col] = {
                    i: group['tasks'] for i in range(self.rounds)
                }
            # Now populate reverse lookup of tasks->group
            for task in group['tasks']:
                for round_ in range(self.rounds):
                    # This should append the list of collaborators performing
                    # that task
                    self.collaborators_for_task[task][round_] += group_col_list

    def get_tasks_for_collaborator(self, collaborator_name, round_number):
        """Get tasks for the collaborator specified."""
        return self.collaborator_tasks[collaborator_name][round_number]

    def get_collaborators_for_task(self, task_name, round_number):
        """Get collaborators for the task specified."""
        return self.collaborators_for_task[task_name][round_number]
