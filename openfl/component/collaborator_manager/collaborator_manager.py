# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Collaborator manager module."""

import logging
import os
import sys
import time
from pathlib import Path

from click import echo

from openfl.federated import Plan
from openfl.transport.grpc.director_client import ShardDirectorClient

logger = logging.getLogger(__name__)


class CollaboratorManager:
    """Collaborator manager class."""

    def __init__(self, shard_name, director_uri, shard_descriptor, disable_tls=False) -> None:
        """Initialize a collaborator manager object."""
        self.name = shard_name
        self.director_client = ShardDirectorClient(director_uri, shard_name=shard_name,disable_tls=disable_tls)
        self.shard_descriptor = shard_descriptor

    def run(self):
        """Run of the collaborator manager working cycle."""
        while True:
            try:
                experiment_name = self.director_client.get_experiment_data()
            except Exception as exc:
                time.sleep(1)
                logger.error(f'Error: {exc}')
            try:
                self._run_collaborator(experiment_name)
            except Exception as exc:
                logger.error(f'Experiment running was failed: {exc}')

    def _run_collaborator(self, experiment_name,
                          plan='plan/plan.yaml',):  # TODO: path params, change naming
        """Run the collaborator for the experiment running."""
        cwd = os.getcwd()
        os.chdir(f'{cwd}/{experiment_name}')  # TODO: probably it should be another way

        # This is needed for python module finder
        sys.path.append(os.getcwd())

        plan = Plan.parse(
            plan_config_path=Path(plan)
        )

        # TODO: Need to restructure data loader config file loader

        echo(f'Data = {plan.cols_data_paths}')
        logger.info('🧿 Starting a Collaborator Service.')

        col = plan.get_collaborator(self.name, shard_descriptor=self.shard_descriptor)
        col.run()
        os.chdir(cwd)

    def start(self):
        """Start the collaborator manager."""
        try:
            acknowledgement = self.director_client.report_shard_info(self.shard_descriptor)
        except Exception as exc:
            logger.exception(str(exc))
            logger.exception('Failed to report shard info')
        else:
            if acknowledgement:
                # Shard accepted for participation in the federation
                logger.info('Shard accepted')
                self.run()
            else:
                # Shut down
                logger.error('Report shard info was not accepted')
                exit()
