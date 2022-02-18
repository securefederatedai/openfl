import logging
from collections import defaultdict

from openfl.component.aggregation_functions import WeightedAverage

logger = logging.getLogger(__name__)


class Assigner:
    def __init__(self, *, assigner_function, aggregation_functions_by_task, authorized_cols):
        """Initialize."""
        self.aggregation_functions_by_task = aggregation_functions_by_task
        self.authorized_cols = authorized_cols
        self.all_tasks_for_round = []
        self.collaborators_for_task = defaultdict(list)
        self.collaborator_tasks = defaultdict(list)
        self.assigner_function = assigner_function

        self.define_task_assignments_for_round(0)

    def define_task_assignments_for_round(self, round_number):
        """Abstract method."""
        self.all_tasks_for_round = []
        self.collaborators_for_task = defaultdict(list)
        self.collaborator_tasks = defaultdict(list)
        for collaborator_name in self.authorized_cols:
            tasks = self.assigner_function(
                collaborator_name,
                round_number,
                number_of_callaborators=len(self.authorized_cols)
            )
            self.all_tasks_for_round.extend(tasks)
            self.collaborator_tasks[collaborator_name].extend(tasks)
            for task in tasks:
                self.collaborators_for_task[task.name].append(collaborator_name)

    def get_tasks_for_collaborator(self, collaborator_name):
        """Abstract method."""
        logger.info(f'{collaborator_name=}')
        logger.info(f'{self.collaborator_tasks=}')
        logger.info(f'{self.collaborator_tasks[collaborator_name]}')
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
        return self.all_tasks_for_round

    def get_aggregation_type_for_task(self, function_name):
        """Extract aggregation type from self.tasks."""
        agg_fn = self.aggregation_functions_by_task.get(function_name, WeightedAverage())
        return agg_fn
