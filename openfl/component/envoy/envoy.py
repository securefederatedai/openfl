# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Envoy module."""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from click import echo

from openfl.federated import Plan
from openfl.transport.grpc.director_client import ShardDirectorClient

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_IN_SECONDS = 60  # TODO: make configurable
DEFAULT_RETRY_TIMEOUT_IN_SECONDS = 5


class Envoy:
    """Envoy class."""

    def __init__(self, shard_name, director_uri, shard_descriptor,
                 root_ca: str = None, key: str = None, cert: str = None,
                 disable_tls: bool = False) -> None:
        """Initialize a envoy object."""
        self.name = shard_name
        self.root_ca = Path(root_ca).absolute() if root_ca is not None else None
        self.key = Path(key).absolute() if root_ca is not None else None
        self.cert = Path(cert).absolute() if root_ca is not None else None
        self.director_client = ShardDirectorClient(director_uri, shard_name=shard_name,
                                                   disable_tls=disable_tls,
                                                   root_ca=root_ca, key=key, cert=cert)
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
                experiment_name = self.director_client.get_experiment_data()
            except Exception as exc:
                logger.error(f'Failed to get experiment: {exc}')
                time.sleep(DEFAULT_RETRY_TIMEOUT_IN_SECONDS)
                continue
            self.is_experiment_running = True
            try:
                self._run_collaborator(experiment_name)
            except Exception as exc:
                logger.error(f'Collaborator failed: {exc}')
            finally:
                # Workspace cleaning should not be done by gRPC client!
                self.director_client.remove_workspace(experiment_name)
                self.is_experiment_running = False

    def send_health_check(self):
        """Send health check to the director."""
        logger.info('The health check sender is started.')
        while True:
            self.director_client.send_health_check(
                self.name,
                self.is_experiment_running,
                DEFAULT_TIMEOUT_IN_SECONDS
            )
            time.sleep(DEFAULT_TIMEOUT_IN_SECONDS / 2)

    def _run_collaborator(self, experiment_name, plan='plan/plan.yaml',):
        """Run the collaborator for the experiment running."""
        cwd = os.getcwd()
        os.chdir(f'{cwd}/{experiment_name}')  # TODO: probably it should be another way

        # This is needed for python module finder
        sys.path.append(os.getcwd())

        plan = Plan.parse(plan_config_path=Path(plan))

        # TODO: Need to restructure data loader config file loader
        echo(f'Data = {plan.cols_data_paths}')
        logger.info('🧿 Starting a Collaborator Service.')

        col = plan.get_collaborator(self.name, self.root_ca, self.key,
                                    self.cert, shard_descriptor=self.shard_descriptor)
        try:
            col.run()
        finally:
            os.chdir(cwd)

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
