# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Envoy module."""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from click import echo

from openfl.federated import Plan
from openfl.transport.grpc.director_client import ShardDirectorClient
from openfl.utilities.workspace import ExperimentWorkspace

logger = logging.getLogger(__name__)

DEFAULT_RETRY_TIMEOUT_IN_SECONDS = 5


class Envoy:
    """Envoy class."""

    def __init__(self, *, shard_name, director_host, director_port, shard_descriptor,
                 root_certificate: str = None, private_key: str = None, certificate: str = None,
                 tls: bool = True) -> None:
        """Initialize a envoy object."""
        self.name = shard_name
        self.root_certificate = Path(
            root_certificate).absolute() if root_certificate is not None else None
        self.private_key = Path(private_key).absolute() if root_certificate is not None else None
        self.certificate = Path(certificate).absolute() if root_certificate is not None else None
        self.director_client = ShardDirectorClient(
            director_host=director_host,
            director_port=director_port,
            shard_name=shard_name,
            tls=tls,
            root_certificate=root_certificate,
            private_key=private_key,
            certificate=certificate
        )
        self.shard_descriptor = shard_descriptor
        self.executor = ThreadPoolExecutor()
        self.running_experiments = {}
        self.is_experiment_running = False
        self._health_check_future = None

    def run(self):
        """Run of the envoy working cycle."""
        while True:
            try:
                # Workspace import should not be done by gRPC client!
                experiment_name = self.director_client.wait_experiment()
                data_stream = self.director_client.get_experiment_data(experiment_name)
            except Exception as exc:
                logger.error(f'Failed to get experiment: {exc}')
                time.sleep(DEFAULT_RETRY_TIMEOUT_IN_SECONDS)
                continue
            data_file_path = self._save_data_stream_to_file(data_stream)
            self.is_experiment_running = True
            try:
                with ExperimentWorkspace(
                        experiment_name, data_file_path, is_install_requirements=True
                ):
                    self._run_collaborator(experiment_name)
            except Exception as exc:
                logger.error(f'Collaborator failed: {exc}')
            finally:
                # Workspace cleaning should not be done by gRPC client!
                self.is_experiment_running = False

    @staticmethod
    def _save_data_stream_to_file(data_stream):
        data_file_path = Path(str(uuid.uuid4())).absolute()
        with open(data_file_path, 'wb') as data_file:
            for response in data_stream:
                if response.size == len(response.npbytes):
                    data_file.write(response.npbytes)
                else:
                    raise Exception('Broken archive')
        return data_file_path

    def send_health_check(self):
        """Send health check to the director."""
        logger.info('The health check sender is started.')
        while True:
            timeout = self.director_client.send_health_check(
                collaborator_name=self.name,
                is_experiment_running=self.is_experiment_running
            )
            time.sleep(timeout)

    def _run_collaborator(self, experiment_name, plan='plan/plan.yaml', ):
        """Run the collaborator for the experiment running."""
        plan = Plan.parse(plan_config_path=Path(plan))

        # TODO: Need to restructure data loader config file loader
        echo(f'Data = {plan.cols_data_paths}')
        logger.info('🧿 Starting a Collaborator Service.')

        col = plan.get_collaborator(self.name, self.root_certificate, self.private_key,
                                    self.certificate, shard_descriptor=self.shard_descriptor)
        col.run()

    def start(self):
        """Start the envoy."""
        try:
            is_accepted = self.director_client.report_shard_info(self.shard_descriptor)
        except Exception as exc:
            logger.exception(str(exc))
            logger.exception('Failed to report shard info')
        else:
            if is_accepted:
                # Shard accepted for participation in the federation
                logger.info('Shard accepted')
                self._health_check_future = self.executor.submit(self.send_health_check)
                self.run()
            else:
                # Shut down
                logger.error('Report shard info was not accepted')
