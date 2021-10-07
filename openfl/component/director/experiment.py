# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experiment module."""

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

logger = logging.getLogger(__name__)


class Status:
    """Experiment's statuses."""

    PENDING = 'PENDING'
    FINISHED = 'FINISHED'
    IN_PROGRESS = 'IN_PROGRESS'
    FAILED = 'FAILED'


class Experiment:
    """Experiment class."""

    def __init__(
            self, *,
            name: str,
            archive_path: Union[Path, str],
            collaborators: List[str],
            sender: str,
            init_tensor_dict: dict,
            plan_path: Union[Path, str] = 'plan/plan.yaml',
            users: Iterable[str] = None,
    ) -> None:
        """Initialize an experiment object."""
        self.name = name
        if isinstance(archive_path, str):
            archive_path = Path(archive_path)
        self.archive_path = archive_path
        self.collaborators = collaborators
        self.sender = sender
        self.init_tensor_dict = init_tensor_dict
        if isinstance(plan_path, str):
            plan_path = Path(plan_path)
        self.plan_path = plan_path
        self.users = set() if users is None else set(users)
        self.status = Status.PENDING
        self.aggregator = None


class ExperimentsRegistry:
    """ExperimentsList class."""

    def __init__(self) -> None:
        """Initialize an experiments list object."""
        self.__active_experiment_name = None
        self.__col_exp_queues = defaultdict(asyncio.Queue)
        self.__experiments_queue = []
        self.__archived_experiments = []
        self.__dict = {}

    @property
    def active_experiment(self) -> Union[Experiment, None]:
        """Get active experiment."""
        if self.__active_experiment_name is None:
            return None
        return self.__dict[self.__active_experiment_name]

    @property
    def queue(self) -> List[str]:
        """Get queue of not started experiments."""
        return self.__experiments_queue

    def add(self, experiment: Experiment) -> None:
        """Add experiment to queue of not started experiments."""
        self.__dict[experiment.name] = experiment
        self.__experiments_queue.append(experiment.name)

    def remove(self, name: str) -> None:
        """Remove experiment from everywhere."""
        if self.__active_experiment_name == name:
            self.__active_experiment_name = None
        if name in self.__experiments_queue:
            self.__experiments_queue.remove(name)
        if name in self.__archived_experiments:
            self.__archived_experiments.remove(name)
        if name in self.__dict:
            del self.__dict[name]

    def __getitem__(self, key: str) -> Experiment:
        """Get experiment by name."""
        return self.__dict[key]

    def get(self, key: str, default=None) -> Experiment:
        """Get experiment by name."""
        return self.__dict.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if experiment exists."""
        return key in self.__dict

    def finish_active(self) -> None:
        """Finish active experiment."""
        self.__dict[self.__active_experiment_name].__aggregator_grpc_server = None
        self.__archived_experiments.insert(0, self.__active_experiment_name)
        self.__active_experiment_name = None

    @asynccontextmanager
    async def get_next_experiment(self):
        """Context manager.

        On enter get experiment from queue.
        On exit put finished experiment to archive.
        """
        while True:
            if self.active_experiment is None and self.queue:
                break
            await asyncio.sleep(10)

        try:
            self.__active_experiment_name = self.queue.pop(0)
            yield self.active_experiment
        finally:
            self.finish_active()
