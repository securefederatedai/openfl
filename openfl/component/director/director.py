# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Director module."""

import asyncio
import logging
import os
import shutil
import time
import typing
from collections import defaultdict
from pathlib import Path

from openfl.federated import Plan
from openfl.protocols import director_pb2

logger = logging.getLogger(__name__)


class Director:
    """Director class."""

    def __init__(self, *, tls: bool = True,
                 root_ca: Path = None, key: Path = None, cert: Path = None,
                 sample_shape: list = None, target_shape: list = None) -> None:
        """Initialize a director object."""
        # TODO: add working directory
        super().__init__()
        self.sample_shape, self.target_shape = sample_shape, target_shape
        self._shard_registry = {}
        self.col_exp_queues = defaultdict(asyncio.Queue)

        self.experiment_data = {}  # {Experiment name : archive bytes}
        # What if two experiments come with the same name from different users?
        self.experiments_queue = asyncio.Queue()  # experiments waiting to be executed
        self.experiment_stash = defaultdict(dict)  # Running of finished experiments
        # {API name : {experiment name : aggregator}}

        self.tls = tls
        self.root_ca = root_ca
        self.key = key
        self.cert = cert

    def acknowledge_shard(self, shard_info: director_pb2.ShardInfo) -> bool:
        """Save shard info to shard registry if it's acceptable."""
        is_accepted = False
        if (self.sample_shape != shard_info.sample_shape
                or self.target_shape != shard_info.target_shape):
            logger.info('Request was not accepted')
            return is_accepted
        logger.info('Request was accepted')
        self._shard_registry[shard_info.node_info.name] = {
            'shard_info': shard_info,
            'is_online': True,
            'is_experiment_running': False
        }
        is_accepted = True
        return is_accepted

    async def set_new_experiment(self, *, experiment_name: str, sender_name: str,
                                 tensor_dict: dict,
                                 collaborator_names: typing.Iterable[str], data: bytes) -> bool:
        """Set new experiment."""
        # TODO: save to file
        self.experiment_data[experiment_name] = data

        self.create_workspace(experiment_name, data)
        asyncio.create_task(self._run_aggregator(
            sender_name,
            tensor_dict,
            experiment_name,
            collaborator_names
        ))

        logger.info(f'New experiment {experiment_name} for '
                    f'collaborators {collaborator_names}')
        for col_name in collaborator_names:
            queue = self.col_exp_queues[col_name]
            await queue.put(experiment_name)

        return True

    def get_trained_model(self, experiment_name: str, caller: str, model_type: str):
        """Get trained model."""
        if experiment_name not in self.experiment_stash[caller]:
            logger.error('No experiment data in the stash')
            return None

        aggregator = self.experiment_stash[caller][experiment_name].aggregator

        if aggregator.last_tensor_dict is None:
            logger.error('Aggregator have no aggregated model to return')
            return None

        if model_type == 'best':
            return aggregator.best_tensor_dict
        elif model_type == 'last':
            return aggregator.last_tensor_dict
        else:
            logger.error('Unknown model type required.')
            return None

    def get_experiment_data(self, experiment_name: str) -> bytes:
        """Get experiment data."""
        return self.experiment_data.get(experiment_name, b'')

    async def wait_experiment(self, collaborator_name: str) -> str:
        """Wait an experiment."""
        queue = self.col_exp_queues[collaborator_name]
        experiment_name = await queue.get()

        return experiment_name

    def get_dataset_info(self):
        """Get dataset info."""
        return self.sample_shape, self.target_shape

    def get_registered_shards(self) -> list:
        """Get registered shard infos."""
        return [shard_status['shard_info'] for shard_status in self._shard_registry.values()]

    def stream_metrics(self, experiment_name: str, caller: str):
        """
        Stream metrics from the aggregator.

        This method takes next metric dictionary from the aggregator's queue
        and returns it to the caller.

        Inputs:
            experiment_name - string id for experiment
            caller - string id for experiment owner

        Returns:
            metric_dict - {'metric_origin','task_name','metric_name','metric_value','round'}
                if the queue is not empty
            None - f queue is empty but the experiment is still running

        Raises:
            StopIteration - if the experiment is finished and there is no more metrics to report
        """
        aggregator = self.experiment_stash[caller][experiment_name].aggregator
        while True:
            if not aggregator.metric_queue.empty():
                yield aggregator.metric_queue.get()
                continue

            if aggregator.all_quit_jobs_sent() and aggregator.metric_queue.empty():
                return

            yield None

    def remove_experiment_data(self, experiment_name: str, caller: str):
        """Remove experiment data from stash."""
        if experiment_name in self.experiment_stash.get(caller, {}):
            del self.experiment_stash[caller][experiment_name]

    def collaborator_health_check(self, *, collaborator_name: str,
                                  is_experiment_running: bool,
                                  valid_duration: int) -> bool:
        """Accept health check from envoy."""
        is_accepted = False
        shard_info = self._shard_registry.get(collaborator_name)
        if not shard_info:
            logger.error(f'Unknown shard {collaborator_name}')
            return is_accepted
        is_accepted = True

        shard_info['is_online']: True
        shard_info['is_experiment_running'] = is_experiment_running
        shard_info['valid_duration'] = valid_duration
        shard_info['last_updated'] = time.time()

        return is_accepted

    def get_envoys(self) -> list:
        """Get a status information about envoys."""
        logger.info(f'Shard registry: {self._shard_registry}')
        envoy_infos = []
        for envoy in self._shard_registry.values():
            envoy_info = director_pb2.EnvoyInfo(
                shard_info=envoy['shard_info'],
                is_online=time.time() < envoy['last_updated'] + envoy['valid_duration'],
                is_experiment_running=envoy['is_experiment_running']
            )
            envoy_info.valid_duration.seconds = envoy['valid_duration']
            envoy_info.last_updated.seconds = int(envoy['last_updated'])

            envoy_infos.append(envoy_info)

        return envoy_infos

    @staticmethod
    def create_workspace(experiment_name, npbytes):
        """Create the aggregator workspace."""
        if os.path.exists(experiment_name):
            shutil.rmtree(experiment_name)
        os.makedirs(experiment_name)

        arch_name = f'{experiment_name}/{experiment_name}' + '.zip'
        logger.info(f'arch_name: {arch_name}')
        with open(arch_name, 'wb') as content_file:
            content_file.write(npbytes)

        shutil.unpack_archive(arch_name, experiment_name)

    async def _run_aggregator(
            self,
            experiment_sender,
            initial_tensor_dict,
            experiment_name,
            collaborator_names,
            plan='plan/plan.yaml',
    ):  # TODO: path params, change naming
        """Run aggregator."""
        cwd = os.getcwd()
        os.chdir(f'{cwd}/{experiment_name}')
        plan = Plan.parse(plan_config_path=Path(plan))

        plan.authorized_cols = list(collaborator_names)

        logger.info('🧿 Starting the Aggregator Service.')
        aggregator_server = plan.interactive_api_get_server(
            tensor_dict=initial_tensor_dict,
            chain=self.root_ca,
            certificate=self.cert,
            private_key=self.key,
            tls=self.tls,
        )
        self.experiment_stash[experiment_sender][experiment_name] = aggregator_server

        grpc_server = aggregator_server.get_server()
        grpc_server.start()
        logger.info('Starting Aggregator gRPC Server')

        try:
            while not aggregator_server.aggregator.all_quit_jobs_sent():
                # Awaiting quit job sent to collaborators
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            os.chdir(cwd)
            shutil.rmtree(experiment_name)
            grpc_server.stop(0)
            # Temporary solution to free RAM used by TensorDB
            aggregator_server.aggregator.tensor_db.clean_up(0)
