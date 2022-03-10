# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Director module."""

import asyncio
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

from .experiment import Experiment
from .experiment import ExperimentsRegistry
from .experiment import Status

logger = logging.getLogger(__name__)

ENVOY_HEALTH_CHECK_PERIOD = 60  # in seconds


class Director:
    """Director class."""

    def __init__(
            self, *,
            tls: bool = True,
            root_certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
            sample_shape: list = None,
            target_shape: list = None,
            settings: dict = None
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
        experiment = Experiment(
            name=experiment_name,
            archive_path=experiment_archive_path,
            collaborators=list(collaborator_names),
            users=[sender_name],
            sender=sender_name,
            init_tensor_dict=tensor_dict,
        )
        self.experiments_registry.add(experiment)
        return True

    def get_trained_model(self, experiment_name: str, caller: str, model_type: str):
        """Get trained model."""
        if (experiment_name not in self.experiments_registry
                or caller not in self.experiments_registry[experiment_name].users):
            logger.error('No experiment data in the stash')
            return None

        aggregator = self.experiments_registry[experiment_name].aggregator

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

        while not self.experiments_registry[experiment_name].aggregator:
            await asyncio.sleep(1)
        aggregator = self.experiments_registry[experiment_name].aggregator

        while True:
            if not aggregator.metric_queue.empty():
                yield aggregator.metric_queue.get()
                continue

            if aggregator.all_quit_jobs_sent() and aggregator.metric_queue.empty():
                return

            yield None

    def remove_experiment_data(self, experiment_name: str, caller: str):
        """Remove experiment data from stash."""
        if (experiment_name in self.experiments_registry
                and caller in self.experiments_registry[experiment_name].users):
            self.experiments_registry.remove(experiment_name)

    def set_experiment_failed(self, *, experiment_name: str, collaborator_name: str):
        """Set experiment failed."""
        if experiment_name in self.experiments_registry:
            aggregator = self.experiments_registry[experiment_name].aggregator
            aggregator.stop(failed_collaborator=collaborator_name)

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

    def get_envoys(self) -> list:
        """Get a status information about envoys."""
        logger.info(f'Shard registry: {self._shard_registry}')
        for envoy_info in self._shard_registry.values():
            envoy_info['is_online'] = (
                time.time() < envoy_info.get('last_updated', 0)
                + envoy_info.get('valid_duration', 0)
            )
            envoy_name = envoy_info['shard_info']['node_info']['name']
            envoy_info['experiment_name'] = self.col_exp[envoy_name]

        return self._shard_registry.values()

    def get_experiments_list(self, caller: str) -> list:
        """Get experiments list for specific user."""
        experiments = self.experiments_registry.get_user_experiments(caller)
        result = []
        for exp in experiments:
            exp_data = {
                'name': exp.name,
                'status': exp.status,
                'collaborators_amount': len(exp.collaborators),
            }
            progress = _get_experiment_progress(exp)
            if progress is not None:
                exp_data['progress'] = progress
            if exp.aggregator:
                tasks_amount = len({
                    task['function']
                    for task in exp.aggregator.assigner.tasks.values()
                })
                exp_data['tasks_amount'] = tasks_amount
            result.append(exp_data)

        return result

    def get_experiment_description(self, caller: str, name: str) -> dict:
        """Get a experiment information by name for specific user."""
        exp = self.experiments_registry.get(name)
        if not exp or caller not in exp.users:
            return {}
        progress = _get_experiment_progress(exp)
        model_statuses = _get_model_download_statuses(exp)
        tasks = _get_experiment_tasks(exp)
        collaborators = _get_experiment_collaborators(exp)
        result = {
            'name': name,
            'status': exp.status,
            'current_round': exp.aggregator.round_number,
            'total_rounds': exp.aggregator.rounds_to_train,
            'download_statuses': {
                'models': model_statuses,
                'logs': [{
                    'name': 'aggregator',
                    'status': 'ready'
                }],
            },
            'collaborators': collaborators,
            'tasks': tasks,
            'progress': progress
        }
        return result

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


def _get_model_download_statuses(experiment) -> List[dict]:
    best_model_status = 'ready' if experiment.aggregator.best_tensor_dict else 'pending'
    last_model_status = 'ready' if experiment.aggregator.last_tensor_dict else 'pending'
    model_statuses = [{
        'name': 'best',
        'status': best_model_status,
    }, {
        'name': 'last',
        'status': last_model_status,
    }, {
        'name': 'init',
        'status': 'ready'
    }]
    return model_statuses


def _get_experiment_progress(experiment) -> Union[float, None]:
    if experiment.status == Status.IN_PROGRESS:
        return experiment.aggregator.round_number / experiment.aggregator.rounds_to_train


def _get_experiment_tasks(experiment) -> List[dict]:
    return [{
        'name': task['function'],
        'description': 'Task description Mock',
    } for task in experiment.aggregator.assigner.tasks.values()]


def _get_experiment_collaborators(experiment) -> List[dict]:
    return [{
        'name': name,
        'status': 'pending_mock',
        'progress': 0.0,
        'round': 0,
        'current_task': 'Current Task Mock',
        'next_task': 'Next Task Mock'
    } for name in experiment.aggregator.authorized_cols]
