# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Experiment module."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Iterable, List, Union

from openfl.federated import Plan
from openfl.transport import AggregatorGRPCServer
from openfl.utilities.workspace import ExperimentWorkspace

logger = logging.getLogger(__name__)


class Status:
    """Experiment's statuses."""

    PENDING = "pending"
    FINISHED = "finished"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    REJECTED = "rejected"


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
            aggregator (object): The aggregator object.
            run_aggregator_atask (object): The run aggregator async task
                object.
    """

    def __init__(
        self,
        *,
        name: str,
        archive_path: Union[Path, str],
        collaborators: List[str],
        sender: str,
        init_tensor_dict: dict,
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
            init_tensor_dict (dict): The initial tensor dictionary.
            plan_path (Union[Path, str], optional): The path to the plan.
                Defaults to 'plan/plan.yaml'.
            users (Iterable[str], optional): The list of users. Defaults to
                None.
        """
        self.name = name
        self.archive_path = Path(archive_path).absolute()
        self.collaborators = collaborators
        self.sender = sender
        self.init_tensor_dict = init_tensor_dict
        self.plan_path = Path(plan_path)
        self.users = set() if users is None else set(users)
        self.status = Status.PENDING
        self.aggregator = None
        self.run_aggregator_atask = None

    async def start(
        self,
        *,
        tls: bool = True,
        root_certificate: Union[Path, str] = None,
        private_key: Union[Path, str] = None,
        certificate: Union[Path, str] = None,
        install_requirements: bool = False,
    ):
        """Run experiment.

        Args:
            tls (bool, optional): A flag indicating if TLS should be used for
                connections. Defaults to True.
            root_certificate (Union[Path, str], optional): The path to the
                root certificate for TLS. Defaults to None.
            private_key (Union[Path, str], optional): The path to the private
                key for TLS. Defaults to None.
            certificate (Union[Path, str], optional): The path to the
                certificate for TLS. Defaults to None.
            install_requirements (bool, optional): A flag indicating if the
                requirements should be installed. Defaults to False.
        """
        self.status = Status.IN_PROGRESS
        try:
            logger.info(f"New experiment {self.name} for " f"collaborators {self.collaborators}")

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
                )
                self.aggregator = aggregator_grpc_server.aggregator

                self.run_aggregator_atask = asyncio.create_task(
                    self._run_aggregator_grpc_server(
                        aggregator_grpc_server=aggregator_grpc_server,
                    )
                )
                await self.run_aggregator_atask
            self.status = Status.FINISHED
            logger.info("Experiment %s was finished successfully.", self.name)
        except Exception as e:
            self.status = Status.FAILED
            logger.exception("Experiment %s failed with error: %s.", self.name, e)

    async def review_experiment(self, review_plan_callback: Callable) -> bool:
        """Get plan approve in console.

        Args:
            review_plan_callback (Callable): A callback function for reviewing the plan.

        Returns:
            bool: True if the plan was approved, False otherwise.
        """
        logger.debug("Experiment Review starts")
        # Extract the workspace for review (without installing requirements)
        with ExperimentWorkspace(
            self.name,
            self.archive_path,
            is_install_requirements=False,
            remove_archive=False,
        ):
            loop = asyncio.get_event_loop()
            # Call for a review in a separate thread (server is not blocked)
            review_approve = await loop.run_in_executor(
                None, review_plan_callback, self.name, self.plan_path
            )
            if not review_approve:
                self.status = Status.REJECTED
                self.archive_path.unlink(missing_ok=True)
                return False

        logger.debug("Experiment Review succeeded")
        return True

    def _create_aggregator_grpc_server(
        self,
        *,
        tls: bool = True,
        root_certificate: Union[Path, str] = None,
        private_key: Union[Path, str] = None,
        certificate: Union[Path, str] = None,
    ) -> AggregatorGRPCServer:
        """Create an aggregator gRPC server.

        Args:
            tls (bool, optional): A flag indicating if TLS should be used for
                connections. Defaults to True.
            root_certificate (Union[Path, str], optional): The path to the
                root certificate for TLS. Defaults to None.
            private_key (Union[Path, str], optional): The path to the private
                key for TLS. Defaults to None.
            certificate (Union[Path, str], optional): The path to the
                certificate for TLS. Defaults to None.

        Returns:
            AggregatorGRPCServer: The created aggregator gRPC server.
        """
        plan = Plan.parse(plan_config_path=self.plan_path)
        plan.authorized_cols = list(self.collaborators)

        logger.info("🧿 Created an Aggregator Server for %s experiment.", self.name)
        aggregator_grpc_server = plan.interactive_api_get_server(
            tensor_dict=self.init_tensor_dict,
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
        """Get active experiment.

        Returns:
            Union[Experiment, None]: The active experiment if exists, None
                otherwise.
        """
        if self.__active_experiment_name is None:
            return None
        return self.__dict[self.__active_experiment_name]

    @property
    def pending_experiments(self) -> List[str]:
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

        On enter get experiment from pending_experiments. On exit put finished
        experiment to archive_experiments.
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
