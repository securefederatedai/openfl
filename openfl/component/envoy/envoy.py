# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Envoy module."""

import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import check_call
from sys import executable

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
                 tls: bool = True) -> None:
        """Initialize a envoy object."""
        self.name = shard_name
        self.root_ca = Path(root_ca).absolute() if root_ca is not None else None
        self.key = Path(key).absolute() if root_ca is not None else None
        self.cert = Path(cert).absolute() if root_ca is not None else None
        self.director_client = ShardDirectorClient(
            director_uri,
            shard_name=shard_name,
            tls=tls,
            root_ca=root_ca,
            key=key,
            cert=cert
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
                experiment_name, client_id = self.director_client.wait_experiment()
                experiment_work_dir = Path(f'{os.getcwd()}/{client_id}/{experiment_name}')
                data_stream = self.director_client.get_experiment_data(experiment_name)
                self.create_workspace(experiment_name, experiment_work_dir, data_stream)
            except Exception as exc:
                logger.error(f'Failed to get experiment: {exc}')
                time.sleep(DEFAULT_RETRY_TIMEOUT_IN_SECONDS)
                continue
            self.is_experiment_running = True
            try:
                self._run_collaborator(experiment_work_dir=experiment_work_dir)
            except Exception as exc:
                logger.error(f'Collaborator failed: {exc}')
            finally:
                self.remove_workspace(experiment_work_dir)
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

    def _run_collaborator(self, *, experiment_work_dir: Path):
        """Run the collaborator for the experiment running."""
        # This is needed for python module finder
        sys.path.append(str(experiment_work_dir))

        plan = Plan.parse(
            plan_config_path=Path('plan/plan.yaml'),
            working_dir=experiment_work_dir
        )

        # TODO: Need to restructure data loader config file loader
        echo(f'Data = {plan.cols_data_paths}')
        logger.info('🧿 Starting a Collaborator Service.')

        col = plan.get_collaborator(
            self.name,
            self.root_ca,
            self.key,
            self.cert,
            shard_descriptor=self.shard_descriptor
        )
        try:
            col.run()
        finally:
            if experiment_work_dir in sys.path:
                sys.path.remove(str(experiment_work_dir))

    @staticmethod
    def remove_workspace(experiment_name):
        """Remove the workspace."""
        shutil.rmtree(experiment_name, ignore_errors=True)

    @staticmethod
    def create_workspace(experiment_name, experiment_work_dir, response_iter):
        """Create a collaborator workspace for the experiment."""
        if os.path.exists(experiment_work_dir):
            shutil.rmtree(experiment_work_dir)
        os.makedirs(experiment_work_dir)

        arch_name = f'{experiment_work_dir}/{experiment_name}' + '.zip'
        logger.info(f'arch_name: {arch_name}')
        with open(arch_name, 'wb') as content_file:
            for response in response_iter:
                logger.info(f'Size: {response.size}')
                if response.size == len(response.npbytes):
                    content_file.write(response.npbytes)
                else:
                    raise Exception('Broken archive')

        shutil.unpack_archive(arch_name, experiment_work_dir)
        os.remove(arch_name)

        requirements_filename = f'./{experiment_work_dir}/requirements.txt'

        if os.path.isfile(requirements_filename):
            attempts = 3
            for _ in range(attempts):
                try:
                    check_call([
                        executable, '-m', 'pip', 'install', '-r', requirements_filename],
                        shell=False)
                except Exception as exc:
                    logger.error(f'Failed to install requirements: {exc}')
                    time.sleep(3)
                else:
                    break
        else:
            logger.error('No ' + requirements_filename + ' file found.')

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
