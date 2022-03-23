# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Director module."""

import asyncio
import logging
import pickle
import time
import uuid
from collections import defaultdict
from pathlib import Path
from tarfile import TarFile
from typing import Iterable
from typing import Union
from typing import ValuesView

from openfl.protocols import base_pb2
from openfl.transport import AsyncAggregatorGRPCClient
from .experiment import Experiment
from .experiment import ExperimentsRegistry
from .experiment import Status
from ...docker.docker import DockerConfig
from ...federated import Plan

logger = logging.getLogger(__name__)

ENVOY_HEALTH_CHECK_PERIOD = 60  # in seconds


class Director:
    """Director class."""

    def __init__(
            self, *,
            director_host,
            director_port,
            tls: bool = True,
            root_certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
            sample_shape: list = None,
            target_shape: list = None,
            settings: dict = None,
            docker_config: DockerConfig,
    ) -> None:
        """Initialize a director object."""
        self.sample_shape, self.target_shape = sample_shape, target_shape
        self._shard_registry = {}
        self.tls = tls
        self.root_certificate = root_certificate
        self.private_key = private_key
        self.certificate = certificate
        self.experiments_registry = ExperimentsRegistry()
        self.settings = settings or {}
        self.col_exp_queues = defaultdict(asyncio.Queue)
        self.col_exp = {}
        self.director_host = director_host
        self.director_port = director_port
        self.docker_config = docker_config

    def acknowledge_shard(self, shard_info: dict) -> bool:
        """Save shard info to shard registry if it's acceptable."""
        is_accepted = False
        if (self.sample_shape != shard_info['sample_shape']
                or self.target_shape != shard_info['target_shape']):
            logger.info('Request was not accepted')
            return is_accepted
        logger.info('Request was accepted')
        hc_period = self.settings.get('envoy_health_check_period', ENVOY_HEALTH_CHECK_PERIOD)
        self._shard_registry[shard_info['node_info']['name']] = {
            'shard_info': shard_info,
            'is_online': True,
            'is_experiment_running': False,
            'valid_duration': 2 * hc_period,
            'last_updated': time.time(),
        }
        is_accepted = True
        return is_accepted

    async def set_new_experiment(
            self, *,
            experiment_name: str,
            sender_name: str,
            tensor_dict: dict,
            collaborator_names: Iterable[str],
            experiment_archive_path: Path,
    ) -> bool:
        """Set new experiment."""
        tensor_dict_path = Path(f'{experiment_name}.pickle').absolute()

        with tensor_dict_path.open('wb') as f:
            pickle.dump(tensor_dict, f)
        plan = self._parse_plan(experiment_archive_path)
        aggregator_client = AsyncAggregatorGRPCClient(
            agg_addr=plan.agg_addr,
            agg_port=plan.agg_port,
            tls=self.tls,
            disable_client_auth=not self.tls,
            root_certificate=self.root_certificate,
            certificate=self.certificate,
            private_key=self.private_key
        )
        experiment = Experiment(
            name=experiment_name,
            archive_path=experiment_archive_path,
            collaborators=list(collaborator_names),
            users=[sender_name],
            sender=sender_name,
            init_tensor_dict_path=tensor_dict_path,
            docker_config=self.docker_config,
            director_host=self.director_host,
            director_port=self.director_port,
            plan=plan,
            aggregator_client=aggregator_client,
        )
        self.experiments_registry.add(experiment)
        return True

    async def get_aggregator_client(self, experiment_name):
        """Return an aggregator client for the experiment."""
        exp = self.experiments_registry[experiment_name]
        # while exp.status != Status.IN_PROGRESS:
        #     await asyncio.sleep(1)
        agg_port = exp.plan.agg_port
        agg_addr = exp.plan.agg_addr
        logger.info(f'Aggregator uri: {agg_addr}:{agg_port}')

        aggregator_client = AsyncAggregatorGRPCClient(
            agg_addr,
            agg_port,
            tls=self.tls,
            disable_client_auth=not self.tls,
            root_certificate=self.root_certificate,
            certificate=self.certificate,
            private_key=self.private_key
        )
        return aggregator_client

    async def get_trained_model(self, experiment_name: str, caller: str, model_type: str):
        """Get trained model."""
        if (experiment_name not in self.experiments_registry
                or caller not in self.experiments_registry[experiment_name].users):
            logger.error('No experiment data in the stash')
            return None
        exp = self.experiments_registry[experiment_name]
        # if exp.status != Status.IN_PROGRESS:
        #     return None

        # aggregator_client = await self.get_aggregator_client(experiment_name)
        # trained_model = await aggregator_client.get_trained_model(
        #     experiment_name,
        #     model_type
        # )
        if model_type == 'last':
            return exp.last_tensor_dict
        elif model_type == 'best':
            return exp.best_tensor_dict
        else:
            raise ValueError(
                f'Invalid value {model_type=} in function get_trained_model. '
                f'Allowed values: "last", "best"'
            )

    def get_experiment_data(self, experiment_name: str) -> Path:
        """Get experiment data."""
        return self.experiments_registry[experiment_name].archive_path

    async def wait_experiment(self, envoy_name: str) -> str:
        """Wait an experiment."""
        self.col_exp[envoy_name] = None
        queue = self.col_exp_queues[envoy_name]
        experiment_name = await queue.get()
        self.col_exp[envoy_name] = experiment_name

        return experiment_name

    def get_dataset_info(self):
        """Get dataset info."""
        return self.sample_shape, self.target_shape

    def get_registered_shards(self) -> list:  # Why is it here?
        """Get registered shard infos."""
        return [shard_status['shard_info'] for shard_status in self._shard_registry.values()]

    async def stream_metrics(self, experiment_name: str, caller: str):
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
        if (experiment_name not in self.experiments_registry
                or caller not in self.experiments_registry[experiment_name].users):
            raise Exception(
                f'No experiment name "{experiment_name}" in experiments list, or caller "{caller}"'
                f' does not have access to this experiment'
            )

        aggregator_client = await self.get_aggregator_client(experiment_name)
        async for metric_dict in aggregator_client.get_metric_stream():
            yield metric_dict

    def remove_experiment_data(self, experiment_name: str, caller: str):
        """Remove experiment data from stash."""
        if (experiment_name in self.experiments_registry
                and caller in self.experiments_registry[experiment_name].users):
            self.experiments_registry.remove(experiment_name)

    async def set_experiment_failed(self, *, experiment_name: str, collaborator_name: str):
        """Set experiment failed."""
        exp = self.experiments_registry.get(experiment_name)
        if exp is not None:
            await exp.stop(collaborator_name)

    def update_envoy_status(
            self, *,
            envoy_name: str,
            is_experiment_running: bool,
            cuda_devices_status: list = None,
    ) -> int:
        """Accept health check from envoy."""
        shard_info = self._shard_registry.get(envoy_name)
        if not shard_info:
            raise Exception(f'Unknown shard {envoy_name}')

        hc_period = self.settings.get('envoy_health_check_period', ENVOY_HEALTH_CHECK_PERIOD)
        shard_info['is_online']: True
        shard_info['is_experiment_running'] = is_experiment_running
        shard_info['valid_duration'] = 2 * hc_period
        shard_info['last_updated'] = time.time()

        if cuda_devices_status is not None:
            for i in range(len(cuda_devices_status)):
                shard_info['shard_info']['node_info']['cuda_devices'][i] = cuda_devices_status[i]

        return hc_period

    def get_envoys(self) -> ValuesView:
        """Get a status information about envoys."""
        logger.info(f'Shard registry: {self._shard_registry}')
        for envoy_info in self._shard_registry.values():
            last_updated = envoy_info.get('last_updated', 0)
            valid_duration = envoy_info.get('valid_duration', 0)
            envoy_info['is_online'] = time.time() < last_updated + valid_duration
            envoy_name = envoy_info['shard_info']['node_info']['name']
            envoy_info['experiment_name'] = self.col_exp.get(envoy_name)

        return self._shard_registry.values()

    async def get_experiments_list(self, caller: str) -> list:
        """Get experiments list for specific user."""
        experiments = self.experiments_registry.get_user_experiments(caller)
        result = []
        for exp in experiments:
            exp_data = {
                'name': exp.name,
                'status': exp.status,
                'collaborators_amount': len(exp.collaborators),
            }
            if exp.status == Status.IN_PROGRESS:
                aggregator_client = await self.get_aggregator_client(exp.name)
                experiment_pb2 = await aggregator_client.get_experiment_description()
                exp_data['progress'] = experiment_pb2.progress
                exp_data['tasks_amount'] = len(experiment_pb2.tasks)
            result.append(exp_data)

        return result

    async def get_experiment_description(self, caller: str, experiment_name: str) -> dict:
        """Get a experiment information by name for specific user."""
        exp = self.experiments_registry.get(experiment_name)
        if not exp or caller not in exp.users:
            logger.info(f'Experiment {experiment_name} for user {caller} does not exist.')
            return {}
        if exp.status != Status.IN_PROGRESS:
            return base_pb2.ExperimentDescription(
                name=exp.name,
                status=exp.status,
            )
        aggregator_client = await self.get_aggregator_client(experiment_name)
        experiment_pb2 = await aggregator_client.get_experiment_description()
        experiment_pb2.name = experiment_name
        experiment_pb2.status = exp.status

        return experiment_pb2

    async def start_experiment_execution_loop(self):
        """Run task to monitor and run experiments."""
        while True:
            async with self.experiments_registry.get_next_experiment() as experiment:
                loop = asyncio.get_event_loop()
                run_aggregator_future = loop.create_task(experiment.start(
                    root_certificate=self.root_certificate,
                    certificate=self.certificate,
                    private_key=self.private_key,
                    tls=self.tls,
                ))
                for col_name in experiment.collaborators:
                    queue = self.col_exp_queues[col_name]
                    await queue.put(experiment.name)
                await run_aggregator_future

    async def upload_experiment_model(
            self,
            experiment_name: str,
            tensor_dict: dict,
            model_type: str,
    ) -> None:
        exp = self.experiments_registry[experiment_name]
        if model_type == 'last':
            exp.last_tensor_dict = tensor_dict
        elif model_type == 'best':
            exp.best_tensor_dict = tensor_dict
        else:
            raise ValueError(
                f'Invalid {model_type=} in upload_experiment_model function. '
                f'Allowed values "last", "best"'
            )

    @staticmethod
    def _parse_plan(archive_path):
        plan_path = Path('plan/plan.yaml')
        with TarFile(name=archive_path, mode='r') as tar_file:
            plan_buffer = tar_file.extractfile(f'./{plan_path}')
            if plan_buffer is None:
                raise Exception(f'No {plan_path} in workspace.')
            plan_data = plan_buffer.read()
        tmp_plan_path = Path('tmp') / f'{uuid.uuid4()}.yaml'
        tmp_plan_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_plan_path.open('wb') as plan_f:
            plan_f.write(plan_data)
        plan = Plan.parse(plan_config_path=tmp_plan_path)
        tmp_plan_path.unlink(missing_ok=True)
        return plan
