# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experimental Envoy module."""

import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

from openfl.experimental.workflow.federated import Plan
from openfl.experimental.workflow.transport.grpc.director_client import EnvoyDirectorClient
from openfl.experimental.workflow.transport.grpc.exceptions import EnvoyNotFoundError
from openfl.utilities.workspace import ExperimentWorkspace

logger = logging.getLogger(__name__)


class Envoy:
    """Envoy class. The Envoy is a long-lived entity that runs on collaborator
    nodes connected to the Director.

    Attributes:
        name (str): The name of the envoy.
        envoy_config (Optional[Path]): Path to envoy_config.yaml
        tls (bool, optional): A flag indicating if TLS should be used for
                connections. Defaults to True.
        root_certificate (Optional[Union[Path, str]]): The path to the root certificate
            for TLS.
        private_key (Optional[Union[Path, str]]): The path to the private key for TLS.
        certificate (Optional[Union[Path, str]]): The path to the certificate for TLS.
        _envoy_dir_client (EnvoyDirectorClient): The envoy director client.
        install_requirements (bool): A flag indicating if the requirements
            should be installed.
        is_experiment_running (bool): A flag indicating if an experiment is
            running.
        executor (ThreadPoolExecutor): The executor for running tasks.
        plan(str): Path to plan.yaml
        _health_check_future (object): The future object for the health check.
    """

    DEFAULT_RETRY_TIMEOUT_IN_SECONDS = 5

    def __init__(
        self,
        *,
        envoy_name: str,
        director_host: str,
        director_port: int,
        envoy_config: Optional[Path] = None,
        root_certificate: Optional[Union[Path, str]] = None,
        private_key: Optional[Union[Path, str]] = None,
        certificate: Optional[Union[Path, str]] = None,
        tls: bool = True,
        install_requirements: bool = True,
    ) -> None:
        """Initialize a envoy object.

        Args:
            envoy_name (str): The name of the envoy.
            director_host (str): The host of the director.
            director_port (int): The port of the director.
            envoy_config (Optional[Path]): Path to envoy_config.yaml
            root_certificate (Optional[Union[Path, str]]): The path
                to the root certificate for TLS. Defaults to None.
            private_key (Optional[Union[Path, str]]): The path to
                the private key for TLS. Defaults to None.
            certificate (Optional[Union[Path, str]]): The path to
                the certificate for TLS. Defaults to None.
            tls (bool, optional): A flag indicating if TLS should be used for
                connections. Defaults to True.
            install_requirements (bool, optional): A flag indicating if the
                requirements should be installed. Defaults to True.
        """
        self.name = envoy_name
        self.envoy_config = envoy_config
        self.tls = tls
        self._fill_certs(root_certificate, private_key, certificate)
        self.install_requirements = install_requirements
        self._envoy_dir_client = self._create_envoy_dir_client(director_host, director_port)
        self.is_experiment_running = False
        self.executor = ThreadPoolExecutor()
        # This plan path ("plan/plan.yaml") originates from the
        # experiment workspace provided by the director
        self.plan = "plan/plan.yaml"
        self._health_check_future = None

    def _create_envoy_dir_client(
        self, director_host: str, director_port: int
    ) -> EnvoyDirectorClient:
        """Create a EnvoyDirectorClient instance.

        Args:
            director_host (str): The host of the director.
            director_port (int): The port of the director.

        Returns:
            EnvoyDirectorClient: Instance of the client
        """
        return EnvoyDirectorClient(
            director_host=director_host,
            director_port=director_port,
            envoy_name=self.name,
            tls=self.tls,
            root_certificate=self.root_certificate,
            private_key=self.private_key,
            certificate=self.certificate,
        )

    def _fill_certs(self, root_certificate, private_key, certificate) -> None:
        """Fill certificates.

        Args:
            root_certificate (Union[Path, str]): The path to the root
                certificate for the TLS connection.
            private_key (Union[Path, str]): The path to the server's private
                key for the TLS connection.
            certificate (Union[Path, str]): The path to the server's
                certificate for the TLS connection.
        """
        if self.tls:
            if not all([root_certificate, private_key, certificate]):
                raise ValueError("Incomplete certificates provided")

            self.root_certificate = Path(root_certificate).absolute()
            self.private_key = Path(private_key).absolute()
            self.certificate = Path(certificate).absolute()
        else:
            self.root_certificate = self.private_key = self.certificate = None

    def _run(self) -> None:
        """Run of the envoy working cycle."""
        while True:
            try:
                # Wait for experiment from Director server
                experiment_name = self._envoy_dir_client.wait_experiment()
                data_stream = self._envoy_dir_client.get_experiment_data(experiment_name)
            except Exception as exc:
                logger.exception("Failed to get experiment: %s", exc)
                time.sleep(self.DEFAULT_RETRY_TIMEOUT_IN_SECONDS)
                continue
            data_file_path = self._save_data_stream_to_file(data_stream)

            try:
                with ExperimentWorkspace(
                    experiment_name=f"{self.name}_{experiment_name}",
                    data_file_path=data_file_path,
                    install_requirements=self.install_requirements,
                ):
                    self.is_experiment_running = True
                    self._run_collaborator()
            except Exception as exc:
                logger.exception("Collaborator failed with error: %s:", exc)
            finally:
                self.is_experiment_running = False

    @staticmethod
    def _save_data_stream_to_file(data_stream) -> Path:
        """Save data stream to file.

        Args:
            data_stream: The data stream to save.

        Returns:
            Path: The path to the saved data file.
        """
        data_file_path = Path(str(uuid.uuid4())).absolute()
        with open(data_file_path, "wb") as data_file:
            for response in data_stream:
                if response.size == len(response.exp_data):
                    data_file.write(response.exp_data)
                else:
                    raise Exception("Broken archive")
        return data_file_path

    def _send_health_check(self) -> None:
        """Send health check to the director."""
        logger.debug("Sending envoy node status to director.")
        timeout = self.DEFAULT_RETRY_TIMEOUT_IN_SECONDS
        while True:
            try:
                timeout = self._envoy_dir_client.send_health_check(
                    envoy_name=self.name,
                    is_experiment_running=self.is_experiment_running,
                )
            except EnvoyNotFoundError:
                logger.info(
                    "The director has lost information about current envoy. Reconnecting..."
                )
                self._envoy_dir_client.connect_envoy(envoy_name=self.name)
            time.sleep(timeout)

    def _run_collaborator(self) -> None:
        """Run the collaborator for the experiment running."""
        plan = Plan.parse(plan_config_path=Path(self.plan))
        logger.info("🧿 Starting the Collaborator Service.")

        col = plan.get_collaborator(
            self.name,
            self.root_certificate,
            self.private_key,
            self.certificate,
            envoy_config=self.envoy_config,
            tls=self.tls,
        )
        col.run()

    def start(self) -> None:
        """Start the envoy"""
        try:
            is_accepted = self._envoy_dir_client.connect_envoy(envoy_name=self.name)
        except Exception as exc:
            logger.exception("Failed to connect envoy: %s", exc)
            sys.exit(1)
        else:
            if is_accepted:
                logger.info(f"{self.name} is connected to the director")
                self._health_check_future = self.executor.submit(self._send_health_check)
                self._run()
            else:
                # Connection failed
                logger.error(f"{self.name} failed to connect to the director")
                sys.exit(1)
