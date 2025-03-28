# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Experimental Director module."""

import asyncio
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterable, Optional, Tuple, Union

import dill

from openfl.experimental.workflow.component.director.experiment import (
    Experiment,
    ExperimentsRegistry,
)
from openfl.experimental.workflow.transport.grpc.exceptions import EnvoyNotFoundError

logger = logging.getLogger(__name__)


class Director:
    """Director class for managing experiments and envoys.

    Attributes:
        tls (bool): A flag indicating if TLS should be used for connections.
        root_certificate (Optional[Union[Path, str]]): The path to the root certificate
            for TLS.
        private_key (Optional[Union[Path, str]]): The path to the private key for TLS.
        certificate (Optional[Union[Path, str]]): The path to the certificate for TLS.
        director_config (Optional[Path]): Path to director_config file
        install_requirements (bool): A flag indicating if the requirements
            should be installed.
        _flow_status (Queue): Stores the flow status
        experiments_registry (ExperimentsRegistry): An object of
            ExperimentsRegistry to store the experiments.
        col_exp (dict): A dictionary to store the experiments for
            collaborators.
        col_exp_queues (defaultdict): A defaultdict to store the experiment
            queues for collaborators.
        _envoy_registry (dict): A dcitionary to store envoy info
        envoy_health_check_period (int): The period for health check of envoys
            in seconds.
        authorized_cols (list): A list of authorized envoys
    """

    def __init__(
        self,
        *,
        tls: bool = True,
        root_certificate: Optional[Union[Path, str]] = None,
        private_key: Optional[Union[Path, str]] = None,
        certificate: Optional[Union[Path, str]] = None,
        director_config: Optional[Path] = None,
        envoy_health_check_period: int = 60,
        install_requirements: bool = True,
    ) -> None:
        """Initialize a Director object.

        Args:
            tls (bool, optional): A flag indicating if TLS should be used for
                connections. Defaults to True.
            root_certificate (Optional[Union[Path, str]]): The path to the
                root certificate for TLS. Defaults to None.
            private_key (Optional[Union[Path, str]]): The path to the private
                key for TLS. Defaults to None.
            certificate (Optional[Union[Path, str]]): The path to the
                certificate for TLS. Defaults to None.
            director_config (Optional[Path]): Path to director_config file
            envoy_health_check_period (int): The period for health check of envoys
            in seconds.
            install_requirements (bool, optional): A flag indicating if the
                requirements should be installed. Defaults to True.
        """
        self.tls = tls
        self.root_certificate = root_certificate
        self.private_key = private_key
        self.certificate = certificate
        self.director_config = director_config
        self.install_requirements = install_requirements
        self._flow_status = asyncio.Queue()

        self.experiments_registry = ExperimentsRegistry()
        self.col_exp = {}
        self.col_exp_queues = defaultdict(asyncio.Queue)
        self._envoy_registry = {}
        self.envoy_health_check_period = envoy_health_check_period
        # authorized_cols refers to envoy & collaborator pair (one to one mapping)
        self.authorized_cols = []

    async def start_experiment_execution_loop(self) -> None:
        """Run tasks and experiments here"""
        loop = asyncio.get_event_loop()
        while True:
            try:
                async with self.experiments_registry.get_next_experiment() as experiment:
                    await self._wait_for_authorized_envoys()
                    run_aggregator_future = loop.create_task(
                        experiment.start(
                            root_certificate=self.root_certificate,
                            certificate=self.certificate,
                            private_key=self.private_key,
                            tls=self.tls,
                            director_config=self.director_config,
                            install_requirements=False,
                        )
                    )
                    # Adding the experiment to collaborators queues
                    for col_name in experiment.collaborators:
                        queue = self.col_exp_queues[col_name]
                        await queue.put(experiment.name)
                    # Wait for the experiment to complete and save the result
                    flow_status = await run_aggregator_future
                    await self._flow_status.put(flow_status)
            except Exception as e:
                logger.error(f"Error while executing experiment: {e}")
                raise

    async def _wait_for_authorized_envoys(self) -> None:
        """Wait until all authorized envoys are connected"""
        while not all(envoy in self.get_envoys().keys() for envoy in self.authorized_cols):
            connected_envoys = len(
                [envoy for envoy in self.authorized_cols if envoy in self.get_envoys().keys()]
            )
            logger.info(
                f"Waiting for {connected_envoys}/{len(self.authorized_cols)} "
                "authorized envoys to connect..."
            )
            await asyncio.sleep(10)

    async def get_flow_state(self) -> Tuple[bool, bytes]:
        """Wait until the experiment flow status indicates completion
        and return the status along with a serialized FLSpec object.

        Returns:
            status (bool): The flow status.
            flspec_obj (bytes): A serialized FLSpec object (in bytes) using dill.
        """
        status, flspec_obj = await self._flow_status.get()
        return status, dill.dumps(flspec_obj)

    async def wait_experiment(self, envoy_name: str) -> str:
        """Waits for an experiment to be ready for a given envoy.

        Args:
            envoy_name (str): The name of the envoy.

        Returns:
            str: The name of the experiment on the queue.
        """
        experiment_name = self.col_exp.get(envoy_name)
        # If any envoy gets disconnected
        if experiment_name and experiment_name in self.experiments_registry:
            experiment = self.experiments_registry[experiment_name]
            if experiment.aggregator.current_round < experiment.aggregator.rounds_to_train:
                return experiment_name

        self.col_exp[envoy_name] = None
        queue = self.col_exp_queues[envoy_name]
        experiment_name = await queue.get()
        self.col_exp[envoy_name] = experiment_name

        return experiment_name

    async def set_new_experiment(
        self,
        experiment_name: str,
        sender_name: str,
        collaborator_names: Iterable[str],
        experiment_archive_path: Path,
    ) -> bool:
        """Set new experiment.

        Args:
            experiment_name (str): String id for experiment.
            sender_name (str): The name of the sender.
            collaborator_names (Iterable[str]): Names of collaborators.
            experiment_archive_path (Path): Path of the experiment.

        Returns:
            bool : Boolean returned if the experiment register was successful.
        """
        experiment = Experiment(
            name=experiment_name,
            archive_path=experiment_archive_path,
            collaborators=collaborator_names,
            users=[sender_name],
            sender=sender_name,
        )

        self.authorized_cols = collaborator_names
        self.experiments_registry.add(experiment)
        return True

    async def stream_experiment_stdout(
        self, experiment_name: str, caller: str
    ) -> AsyncGenerator[Optional[Dict[str, Any]], None]:
        """Stream stdout from the aggregator.

        This method takes next stdout dictionary from the aggregator's queue
        and returns it to the caller.

        Args:
            experiment_name (str): String id for experiment.
            caller (str): String id for experiment owner.

        Yields:
            Optional[Dict[str, str]]: A dictionary containing the keys
            'stdout_origin', 'task_name', and 'stdout_value' if the queue is not empty,
            or None if the queue is empty but the experiment is still running.
        """
        if (
            experiment_name not in self.experiments_registry
            or caller not in self.experiments_registry[experiment_name].users
        ):
            raise Exception(
                f'No experiment name "{experiment_name}" in experiments list, or caller "{caller}"'
                f" does not have access to this experiment"
            )
        while not self.experiments_registry[experiment_name].aggregator:
            await asyncio.sleep(5)
        aggregator = self.experiments_registry[experiment_name].aggregator
        while True:
            if not aggregator.stdout_queue.empty():
                # Yield the next item from the queue
                yield aggregator.stdout_queue.get()
            elif aggregator.all_quit_jobs_sent():
                # Stop Iteration if all jobs have quit and the queue is empty
                break
            else:
                # Yield none if the queue is empty but the experiment is still running.
                yield None

    def get_experiment_data(self, experiment_name: str) -> Path:
        """Get experiment data.

        Args:
            experiment_name (str): String id for experiment.

        Returns:
            str: Path of archive.
        """
        return self.experiments_registry[experiment_name].archive_path

    def ack_envoy_connection_request(self, envoy_name: str) -> bool:
        """Save the envoy info into _envoy_registry

        Args:
            envoy_name (str): Name of the envoy

        Returns:
            bool: Always returns True to indicate the envoy
                has been successfully acknowledged.
        """
        self._envoy_registry[envoy_name] = {
            "name": envoy_name,
            "is_online": True,
            "is_experiment_running": False,
            "last_updated": time.time(),
            "valid_duration": 2 * self.envoy_health_check_period,
        }
        # Currently always returns True, indicating the envoy was added successfully.
        # Future logic might change this to handle conditions.
        return True

    def get_envoys(self) -> Dict[str, Any]:
        """Gets list of connected envoys

        Returns:
            dict: Dictionary with the status information about envoys.
        """
        logger.debug("Envoy registry: %s", self._envoy_registry)
        for envoy in self._envoy_registry.values():
            envoy["is_online"] = time.time() < envoy.get("last_updated", 0) + envoy.get(
                "valid_duration", 0
            )
            envoy["experiment_name"] = self.col_exp.get(envoy["name"], "None")

        return self._envoy_registry

    def update_envoy_status(
        self,
        *,
        envoy_name: str,
        is_experiment_running: bool,
    ) -> int:
        """Accept health check from envoy.

        Args:
            envoy_name (str): String id for envoy.
            is_experiment_running (bool): Boolean value for the status of the
                experiment.

        Raises:
            EnvoyNotFoundError: When Unknown envoy {envoy_name}.

        Returns:
            int: Value of the envoy_health_check_period.
        """
        envoy_info = self._envoy_registry.get(envoy_name)
        if not envoy_info:
            logger.error(f"Unknown envoy {envoy_name}")
            raise EnvoyNotFoundError(f"Unknown envoy {envoy_name}")

        envoy_info.update(
            {
                "is_online": True,
                "is_experiment_running": is_experiment_running,
                "valid_duration": 2 * self.envoy_health_check_period,
                "last_updated": time.time(),
            }
        )

        return self.envoy_health_check_period
