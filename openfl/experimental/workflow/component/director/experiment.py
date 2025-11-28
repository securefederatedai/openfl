# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Experimental Experiment module."""

import asyncio
import logging
from contextlib import asynccontextmanager
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

from openfl.experimental.workflow.federated import Plan
from openfl.experimental.workflow.transport import AggregatorGRPCServer
from openfl.utilities.workspace import ExperimentWorkspace

logger = logging.getLogger(__name__)


class Status(Enum):
    """Experiment's statuses."""

    PENDING = auto()
    FINISHED = auto()
    IN_PROGRESS = auto()
    FAILED = auto()
    REJECTED = auto()


class Experiment:
    """Experiment class.

    Attributes:
            name (str): The name of the experiment.
            archive_path (Union[Path, str]): The path to the experiment
                archive.
            collaborators (List[str]): The list of collaborators.
            sender (str): The name of the sender.
            init_tensor_dict (dict): The initial tensor dictionary.
            plan_path (Union[Path, str]): The path to the plan.
            users (Iterable[str]): The list of users.
            status (str): The status of the experiment.
            aggregator (Aggregator): The aggregator instance.
            updated_flow (FLSpec): Updated flow instance.
    """

    def __init__(
        self,
        *,
        name: str,
        archive_path: Union[Path, str],
        collaborators: List[str],
        sender: str,
        plan_path: Union[Path, str] = "plan/plan.yaml",
        users: Iterable[str] = None,
    ) -> None:
        """Initialize an experiment object.

        Args:
            name (str): The name of the experiment.
            archive_path (Union[Path, str]): The path to the experiment
                archive.
            collaborators (List[str]): The list of collaborators.
            sender (str): The name of the sender.
            plan_path (Union[Path, str], optional): The path to the plan.
                Defaults to 'plan/plan.yaml'.
            users (Iterable[str], optional): The list of users. Defaults to
                None.
        """
        self.name = name
        self.archive_path = Path(archive_path).absolute()
        self.collaborators = collaborators
        self.sender = sender
        # This plan path ("plan/plan.yaml") originates from the
        # experiment workspace provided by the director
        self.plan_path = Path(plan_path)
        self.users = set() if users is None else set(users)
        self.status = Status.PENDING
        self.aggregator = None
        self.updated_flow = None

    async def start(
        self,
        *,
        tls: bool = True,
        root_certificate: Optional[Union[Path, str]] = None,
        private_key: Optional[Union[Path, str]] = None,
        certificate: Optional[Union[Path, str]] = None,
        director_config: Path = None,
        install_requirements: bool = True,
    ) -> Tuple[bool, Any]:
        """Run experiment.

        Args:
            tls (bool, optional): A flag indicating if TLS should be used for
                connections. Defaults to True.
            root_certificate (Optional[Union[Path, str]], optional): The path to the
                root certificate for TLS. Defaults to None.
            private_key (Optional[Union[Path, str]], optional): The path to the private
                key for TLS. Defaults to None.
            certificate (Optional[Union[Path, str]], optional): The path to the
                certificate for TLS. Defaults to None.
            director_config (Path): Path to director's config file
            install_requirements (bool, optional): A flag indicating if the
                requirements should be installed. Defaults to True.

        Returns:
            List[Union[bool, Any]]:
                - status: status of the experiment.
                - updated_flow: The updated flow object.
        """
        self.status = Status.IN_PROGRESS
        try:
            logger.info(f"New experiment {self.name} for collaborators {self.collaborators}")

            with ExperimentWorkspace(
                experiment_name=self.name,
                data_file_path=self.archive_path,
                install_requirements=install_requirements,
            ):
                aggregator_grpc_server = self._create_aggregator_grpc_server(
                    tls=tls,
                    root_certificate=root_certificate,
                    private_key=private_key,
                    certificate=certificate,
                    director_config=director_config,
                )
                self.aggregator = aggregator_grpc_server.aggregator
                _, self.updated_flow = await asyncio.gather(
                    self._run_aggregator_grpc_server(
                        aggregator_grpc_server,
                    ),
                    self.aggregator.run_flow(),
                )
            self.status = Status.FINISHED
            logger.info("Experiment %s was finished successfully.", self.name)
        except Exception as e:
            self.status = Status.FAILED
            logger.error("Experiment %s failed with error: %s.", self.name, e)
            raise

        return self.status == Status.FINISHED, self.updated_flow

    def _create_aggregator_grpc_server(
        self,
        *,
        tls: bool = True,
        root_certificate: Optional[Union[Path, str]] = None,
        private_key: Optional[Union[Path, str]] = None,
        certificate: Optional[Union[Path, str]] = None,
        director_config: Path = None,
    ) -> AggregatorGRPCServer:
        """Create an aggregator gRPC server.

        Args:
            tls (bool, optional): A flag indicating if TLS should be used for
                connections. Defaults to True.
            root_certificate (Optional[Union[Path, str]]): The path to the
                root certificate for TLS. Defaults to None.
            private_key (Optional[Union[Path, str]]): The path to the private
                key for TLS. Defaults to None.
            certificate (Optional[Union[Path, str]]): The path to the
                certificate for TLS. Defaults to None.
            director_config (Path): Path to director's config file.
                Defaults to None.
        Returns:
            AggregatorGRPCServer: The created aggregator gRPC server.
        """
        plan = Plan.parse(plan_config_path=self.plan_path)
        plan.authorized_cols = list(self.collaborators)

        logger.info("🧿 Created an Aggregator Server for %s experiment.", self.name)
        aggregator_grpc_server = plan.get_server(
            director_config,
            root_certificate=root_certificate,
            certificate=certificate,
            private_key=private_key,
            tls=tls,
        )
        return aggregator_grpc_server

    @staticmethod
    async def _run_aggregator_grpc_server(
        aggregator_grpc_server: AggregatorGRPCServer,
    ) -> None:
        """Run aggregator.

        Args:
            aggregator_grpc_server (AggregatorGRPCServer): The aggregator gRPC
                server to run.
        """
        logger.info("🧿 Starting the Aggregator Service.")
        grpc_server = aggregator_grpc_server.get_server()
        grpc_server.start()
        logger.info("Starting Aggregator gRPC Server")

        try:
            while not aggregator_grpc_server.aggregator.all_quit_jobs_sent():
                # Awaiting quit job sent to collaborators
                await asyncio.sleep(10)
            logger.debug("Aggregator sent quit jobs calls to all collaborators")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping the server.")
        finally:
            grpc_server.stop(0)


class ExperimentsRegistry:
    """ExperimentsList class."""

    def __init__(self) -> None:
        """Initialize an experiments registry object."""
        self.__active_experiment_name = None
        self.__pending_experiments = []
        self.__archived_experiments = []
        self.__dict = {}

    @property
    def active(self) -> Union[Experiment, None]:
        """Get active experiment.

        Returns:
            Union[Experiment, None]: The active experiment if exists, None
                otherwise.
        """
        if self.__active_experiment_name is None:
            return None
        return self.__dict[self.__active_experiment_name]

    @property
    def pending(self) -> List[str]:
        """Get queue of not started experiments.

        Returns:
            List[str]: The list of pending experiments.
        """
        return self.__pending_experiments

    def add(self, experiment: Experiment) -> None:
        """Add experiment to queue of not started experiments.

        Args:
           experiment (Experiment): The experiment to add.
        """
        self.__dict[experiment.name] = experiment
        self.__pending_experiments.append(experiment.name)

    def remove(self, name: str) -> None:
        """Remove experiment from everywhere.

        Args:
            name (str): The name of the experiment to remove.
        """
        if self.__active_experiment_name == name:
            self.__active_experiment_name = None
        if name in self.__pending_experiments:
            self.__pending_experiments.remove(name)
        if name in self.__archived_experiments:
            self.__archived_experiments.remove(name)
        if name in self.__dict:
            del self.__dict[name]

    def __getitem__(self, key: str) -> Experiment:
        """Get experiment by name.

        Args:
            key (str): The name of the experiment.

        Returns:
            Experiment: The experiment with the given name.
        """
        return self.__dict[key]

    def get(self, key: str, default=None) -> Experiment:
        """Get experiment by name.

        Args:
            key (str): The name of the experiment.
            default (optional): The default value to return if the experiment
                does not exist.

        Returns:
            Experiment: The experiment with the given name, or the default
                value if the experiment does not exist.
        """
        return self.__dict.get(key, default)

    def get_user_experiments(self, user: str) -> List[Experiment]:
        """Get list of experiments for specific user.

        Args:
            user (str): The name of the user.

        Returns:
            List[Experiment]: The list of experiments for the specific user.
        """
        return [exp for exp in self.__dict.values() if user in exp.users]

    def __contains__(self, key: str) -> bool:
        """Check if experiment exists.

        Args:
            key (str): The name of the experiment.

        Returns:
            bool: True if the experiment exists, False otherwise.
        """
        return key in self.__dict

    def finish_active(self) -> None:
        """Finish active experiment."""
        self.__archived_experiments.insert(0, self.__active_experiment_name)
        self.__active_experiment_name = None

    @asynccontextmanager
    async def get_next_experiment(self):
        """Context manager.

        On enter get experiment from pending experiments. On exit put finished
        experiment to archive_experiments.
        """
        while True:
            if self.active is None and self.pending:
                break
            await asyncio.sleep(10)

        try:
            self.__active_experiment_name = self.pending.pop(0)
            yield self.active
        finally:
            self.finish_active()
