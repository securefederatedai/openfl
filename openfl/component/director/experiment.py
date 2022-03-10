# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experiment module."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

from openfl.federated import Plan
from openfl.transport import AggregatorGRPCServer
from openfl.utilities.workspace import ExperimentWorkspace

logger = logging.getLogger(__name__)


class Status:
    """Experiment's statuses."""

    PENDING = 'pending'
    FINISHED = 'finished'
    IN_PROGRESS = 'in_progress'
    FAILED = 'failed'


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

    async def start(
            self, *,
            tls: bool = True,
            root_certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
    ):
        """Run experiment."""
        self.status = Status.IN_PROGRESS
        try:
            logger.info(f'New experiment {self.name} for '
                        f'collaborators {self.collaborators}')

            with ExperimentWorkspace(self.name, self.archive_path):
                aggregator_grpc_server = self._create_aggregator_grpc_server(
                    tls=tls,
                    root_certificate=root_certificate,
                    private_key=private_key,
                    certificate=certificate,
                )
                self.aggregator = aggregator_grpc_server.aggregator
                await self._run_aggregator_grpc_server(
                    aggregator_grpc_server=aggregator_grpc_server,
                )
            self.status = Status.FINISHED
            logger.info(f'Experiment "{self.name}" was finished successfully.')
        except Exception as e:
            self.status = Status.FAILED
            logger.exception(f'Experiment "{self.name}" was failed with error: {e}.')

    def _create_aggregator_grpc_server(
            self, *,
            tls: bool = True,
            root_certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
    ) -> AggregatorGRPCServer:
        plan = Plan.parse(plan_config_path=Path(self.plan_path))
        plan.authorized_cols = list(self.collaborators)

        logger.info('🧿 Starting the Aggregator Service.')
        aggregator_grpc_server = plan.interactive_api_get_server(
            tensor_dict=self.init_tensor_dict,
            root_certificate=root_certificate,
            certificate=certificate,
            private_key=private_key,
            tls=tls,
        )
        return aggregator_grpc_server

    @staticmethod
    async def _run_aggregator_grpc_server(aggregator_grpc_server: AggregatorGRPCServer) -> None:
        """Run aggregator."""
        logger.info('🧿 Starting the Aggregator Service.')
        grpc_server = aggregator_grpc_server.get_server()
        grpc_server.start()
        logger.info('Starting Aggregator gRPC Server')

        try:
            while not aggregator_grpc_server.aggregator.all_quit_jobs_sent():
                # Awaiting quit job sent to collaborators
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            grpc_server.stop(0)
            # Temporary solution to free RAM used by TensorDB
            aggregator_grpc_server.aggregator.tensor_db.clean_up(0)


class ExperimentsRegistry:
    """ExperimentsList class."""

    def __init__(self) -> None:
        """Initialize an experiments list object."""
        self.__active_experiment_name = None
        self.__pending_experiments = []
        self.__archived_experiments = []
        self.__dict = {}

    @property
    def active_experiment(self) -> Union[Experiment, None]:
        """Get active experiment."""
        if self.__active_experiment_name is None:
            return None
        return self.__dict[self.__active_experiment_name]

    @property
    def pending_experiments(self) -> List[str]:
        """Get queue of not started experiments."""
        return self.__pending_experiments

    def add(self, experiment: Experiment) -> None:
        """Add experiment to queue of not started experiments."""
        self.__dict[experiment.name] = experiment
        self.__pending_experiments.append(experiment.name)

    def remove(self, name: str) -> None:
        """Remove experiment from everywhere."""
        if self.__active_experiment_name == name:
            self.__active_experiment_name = None
        if name in self.__pending_experiments:
            self.__pending_experiments.remove(name)
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

    def get_user_experiments(self, user: str) -> List[Experiment]:
        """Get list of experiments for specific user."""
        return [
            exp
            for exp in self.__dict.values()
            if user in exp.users
        ]

    def __contains__(self, key: str) -> bool:
        """Check if experiment exists."""
        return key in self.__dict

    def finish_active(self) -> None:
        """Finish active experiment."""
        self.__archived_experiments.insert(0, self.__active_experiment_name)
        self.__active_experiment_name = None

    @asynccontextmanager
    async def get_next_experiment(self):
        """Context manager.

        On enter get experiment from pending_experiments.
        On exit put finished experiment to archive_experiments.
        """
        while True:
            if self.active_experiment is None and self.pending_experiments:
                break
            await asyncio.sleep(10)

        try:
            self.__active_experiment_name = self.pending_experiments.pop(0)
            yield self.active_experiment
        finally:
            self.finish_active()
