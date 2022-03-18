# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experiment module."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tarfile import TarFile
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from openfl.docker import docker
from openfl.federated import Plan
from openfl.transport import AggregatorGRPCServer
from openfl.transport import AsyncAggregatorGRPCClient
from openfl.utilities.workspace import ExperimentWorkspace

logger = logging.getLogger(__name__)


class Status:
    """Experiment's statuses."""

    PENDING = 'pending'
    FINISHED = 'finished'
    IN_PROGRESS = 'in_progress'
    FAILED = 'failed'
    DOCKER_BUILD_IN_PROGRESS = 'docker_build_in_progress'
    DOCKER_CREATE_CONTAINER = 'docker_create_container'
    DOCKER_RUN_CONTAINER = 'docker_run_container'


class Experiment:
    """Experiment class."""

    def __init__(
            self, *,
            name: str,
            archive_path: Union[Path, str],
            collaborators: List[str],
            sender: str,
            init_tensor_dict_path: Union[Path, str],
            director_host: str,
            director_port: str,
            aggregator_client: AsyncAggregatorGRPCClient,
            plan: Plan,
            plan_path: Union[Path, str] = 'plan/plan.yaml',
            users: Iterable[str] = None,
            use_docker: bool = False,
            docker_env: Optional[Dict[str, str]] = None,
            docker_buildargs: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize an experiment object."""
        self.name = name
        if isinstance(archive_path, str):
            archive_path = Path(archive_path)
        self.archive_path = archive_path
        self.collaborators = collaborators
        self.sender = sender
        self.init_tensor_dict_path = Path(init_tensor_dict_path)
        if isinstance(plan_path, str):
            plan_path = Path(plan_path)
        self.plan_path = plan_path
        self.plan = plan
        self.users = set() if users is None else set(users)
        self.status = Status.PENDING
        self.aggregator_container = None  # ???
        self.aggregator_client = aggregator_client
        self._use_docker = use_docker
        self.director_host = director_host
        self.director_port = director_port
        self.last_tensor_dict = {}
        self.best_tensor_dict = {}
        if docker_env is None:
            docker_env = {}
        self.docker_env = docker_env
        if docker_buildargs is None:
            docker_buildargs = {}
        self.docker_buildargs = docker_buildargs

    async def start(
            self, *,
            tls: bool = True,
            root_certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
    ) -> None:
        """Run experiment."""
        self.status = Status.IN_PROGRESS
        try:
            logger.info(f'New experiment {self.name} for '
                        f'collaborators {self.collaborators}')

            if self._use_docker:
                await self._run_aggregator_in_docker(
                    data_file_path=self.archive_path,
                    tls=tls,
                    root_certificate=root_certificate,
                    private_key=private_key,
                    certificate=certificate,
                )
            else:
                with ExperimentWorkspace(self.name, self.archive_path):
                    aggregator_grpc_server = self._create_aggregator_grpc_server(
                        director_host=self.director_host,
                        director_port=self.director_port,
                        tls=tls,
                        root_certificate=root_certificate,
                        private_key=private_key,
                        certificate=certificate,
                    )
                    await self._run_aggregator_grpc_server(
                        aggregator_grpc_server=aggregator_grpc_server,
                    )

            self.status = Status.FINISHED
            logger.info(f'Experiment "{self.name}" was finished successfully.')
        except Exception as e:
            self.status = Status.FAILED
            logger.exception(f'Experiment "{self.name}" was failed with error: {e}.')

    async def stop(self, failed_collaborator: str = None):
        await self.aggregator_client.stop(failed_collaborator=failed_collaborator)

    async def _run_aggregator_in_docker(
            self, *,
            data_file_path: Path,
            tls: bool = False,
            root_certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
    ) -> None:
        docker_client = docker.Docker()
        docker_context_path = docker.create_aggregator_context(
            data_file_path=data_file_path,
            init_tensor_dict_path=self.init_tensor_dict_path,
        )
        self.status = Status.DOCKER_BUILD_IN_PROGRESS
        image_tag = await docker_client.build_image(
            context_path=docker_context_path,
            tag='aggregator',
            buildargs=self.docker_buildargs,
        )
        cmd = (
            f'python run.py '
            f'--experiment_name {self.name} '
            f'--director_host {self.director_host} '
            f'--director_port {self.director_port} '
            f'--init_tensor_dict_path init_tensor_dict.pickle '
            f'--collaborators {" ".join(self.collaborators)} '
            f'--root_certificate {root_certificate} '
            f'--private_key {private_key} '
            f'--certificate {certificate} '
            f'{"--tls " if tls else "--no-tls "}'
        )
        self.status = Status.DOCKER_CREATE_CONTAINER
        self.container = await docker_client.create_container(
            name=f'{self.name.lower()}_aggregator',
            image_tag=image_tag,
            cmd=cmd,
            env=self.docker_env,
        )
        self.status = Status.DOCKER_RUN_CONTAINER
        await docker_client.start_and_monitor_container(container=self.container)
        await self.container.delete(force=True)

    def _create_aggregator_grpc_server(
            self, *,
            director_host,
            director_port,
            tls: bool = True,
            root_certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
    ) -> AggregatorGRPCServer:
        self.plan.authorized_cols = list(self.collaborators)

        logger.info('🧿 Starting the Aggregator Service.')
        init_tensor_dict = np.load(str(self.init_tensor_dict_path), allow_pickle=True)
        aggregator_grpc_server = self.plan.interactive_api_get_server(
            experiment_name=self.name,
            director_host=director_host,
            director_port=director_port,
            tensor_dict=init_tensor_dict,
            root_certificate=root_certificate,
            certificate=certificate,
            private_key=private_key,
            tls=tls,
        )
        return aggregator_grpc_server

    def _parse_plan(self):
        with TarFile(name=self.archive_path, mode='r') as tar_file:
            plan_buffer = tar_file.extractfile(f'./{self.plan_path}')
            if plan_buffer is None:
                raise Exception(f'No {self.plan_path} in workspace.')
            plan_data = plan_buffer.read()
        local_plan_path = Path(self.name) / self.plan_path
        local_plan_path.parent.mkdir(parents=True, exist_ok=True)
        with local_plan_path.open('wb') as plan_f:
            plan_f.write(plan_data)
        plan = Plan.parse(plan_config_path=local_plan_path)
        return plan

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
        self.last_tensor_dict = {}
        self.best_tensor_dict = {}

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
