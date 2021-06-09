import logging
from pathlib import Path
import os
import sys

from openfl.federated import Plan
from click import echo

from openfl.transport.grpc.director_client import ShardDirectorClient

logger = logging.getLogger(__name__)


class CollaboratorManager:

    def __init__(self, shard_name, director_uri, shard_descriptor) -> None:
        self.name = shard_name
        self.director_client = ShardDirectorClient(director_uri, shard_name=shard_name)
        self.shard_descriptor = shard_descriptor

    def run(self):
        while True:
            experiment_name = self.director_client.get_experiment_data()
            self._run_collaborator(experiment_name)

    def _run_collaborator(self, experiment_name,
                          plan='plan/plan.yaml',):  # TODO: path params, change naming
        cwd = os.getcwd()
        os.chdir(f'{cwd}/{experiment_name}')  # TODO: probably it should be another way

        # This is needed for python module finder
        sys.path.append(os.getcwd())

        plan = Plan.Parse(
            plan_config_path=Path(plan)
        )

        # TODO: Need to restructure data loader config file loader

        echo(f'Data = {plan.cols_data_paths}')
        logger.info('🧿 Starting a Collaborator Service.')

        col = plan.get_collaborator(self.name, shard_descriptor=self.shard_descriptor)
        col.run()
        os.chdir(cwd)

    # def load_shard_descriptor(self, data_path):
    #     from shard_descriptor import ShardDescriptor
    #     self.shard_descriptor = ShardDescriptor(data_path)

    def start(self, data_path=None):
        # try:
        #     self.load_shard_descriptor(data_path)
        # except ModuleNotFoundError:
        #     logger.error(f'You should add shard_descriptor.py file to {os.getcwd()}')
        #     exit()
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
                logger.error("Report shard info was not accepted")
                exit()
