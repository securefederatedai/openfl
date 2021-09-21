# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Director module."""

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

import time

from openfl.federated import Plan
from openfl.protocols import director_pb2
from openfl.transport import AggregatorGRPCServer
from openfl.utilities.workspace import ExperimentWorkspace

logger = logging.getLogger(__name__)

ENVOY_HEALTH_CHECK_PERIOD = 60  # in seconds


class Status:
    PENDING = 'pending'
    FINISHED = 'finished'
    IN_PROGRESS = 'inProgress'
    FAILED = 'failed'


class Experiment:

    def __init__(
            self, *,
            name: str,
            data_path: Union[Path, str],
            collaborators: List[str],
            sender: str,
            init_tensor_dict: dict,
            plan_path: Union[Path, str] = 'plan/plan.yaml',
            aggregator_grpc_server: AggregatorGRPCServer = None,
            users: Iterable[str] = None,
    ) -> None:
        """Initialize an experiment object."""
        self.name = name
        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data_path = data_path
        self.collaborators = collaborators
        self.sender = sender
        self.init_tensor_dict = init_tensor_dict
        if isinstance(plan_path, str):
            plan_path = Path(plan_path)
        self.plan_path = plan_path
        self._aggregator_grpc_server = aggregator_grpc_server
        self.users = set() if users is None else set(users)
        self.status = Status.PENDING
        self.col_exp_queues = defaultdict(asyncio.Queue)

    async def start(
            self,
            tls=False,
            root_certificate=None,
            certificate=None,
            private_key=None,
    ) -> None:
        self.status = Status.IN_PROGRESS
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._run_aggregator(
                tls=tls,
                root_certificate=root_certificate,
                certificate=certificate,
                private_key=private_key,
            ))
            logger.info(f'New experiment {self.name} for '
                        f'collaborators {self.collaborators}')
            for col_name in self.collaborators:
                queue = self.col_exp_queues[col_name]
                await queue.put(self.name)
        except Exception:
            self.status = Status.FAILED
        self.status = Status.FINISHED

    async def _run_aggregator(
            self,
            tls=False,
            root_certificate=None,
            certificate=None,
            private_key=None,
    ) -> None:
        with ExperimentWorkspace(self.name, self.data_path):
            self._aggregator_grpc_server = self._create_aggregator_grpc_server(
                plan_path=self.plan_path,
                tls=tls,
                root_certificate=root_certificate,
                certificate=certificate,
                private_key=private_key,
            )
            await self._run_aggregator_server()

    def _create_aggregator_grpc_server(
            self,
            plan_path,
            tls=False,
            root_certificate=None,
            certificate=None,
            private_key=None,
    ) -> AggregatorGRPCServer:
        plan = Plan.parse(plan_config_path=Path(plan_path))
        plan.authorized_cols = list(self.collaborators)

        logger.info('🧿 Starting the Aggregator Service.')
        aggregator_grpc_server = plan.interactive_api_get_server(
            tensor_dict=self.init_tensor_dict,
            root_certificate=root_certificate,
            certificate=certificate,
            private_key=private_key,
            tls=tls,
        )
        return aggregator_grpc_server

    async def _run_aggregator_server(self) -> None:
        """Run aggregator."""
        logger.info('🧿 Starting the Aggregator Service.')
        grpc_server = self._aggregator_grpc_server.get_server()
        grpc_server.start()
        logger.info('Starting Aggregator gRPC Server')

        try:
            while not self.aggregator.all_quit_jobs_sent():
                # Awaiting quit job sent to collaborators
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            grpc_server.stop(0)
            # Temporary solution to free RAM used by TensorDB
            self.aggregator.tensor_db.clean_up(0)

    @property
    def aggregator(self):
        if self._aggregator_grpc_server:
            return self._aggregator_grpc_server.aggregator


class ExperimentsList:

    def __init__(self, experiments: List[Experiment] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__active_experiment = None
        if experiments is None:
            self.__experiments_queue = []
            self.__archived_experiments = []
            self.__dict = {}
        else:
            self.__dict = {
                exp.name: exp
                for exp in experiments
            }
            self.__experiments_queue = list(self.__dict.keys())

    @property
    def active_experiment(self) -> Union[Experiment, None]:
        if self.__active_experiment in None:
            return None
        return self.__dict[self.__active_experiment]

    @property
    def queue(self) -> List[str]:
        return self.__experiments_queue

    def add(self, experiment: Experiment) -> None:
        self.__dict[experiment.name] = experiment
        self.__experiments_queue.append(experiment.name)

    def remove(self, name) -> None:
        if self.__active_experiment == name:
            self.__active_experiment = None
        if name in self.__experiments_queue:
            self.__experiments_queue.remove(name)
        if name in self.__archived_experiments:
            self.__archived_experiments.remove(name)
        if name in self.__dict:
            del self.__dict[name]

    def _finish_active(self) -> None:
        self.__archived_experiments.insert(0, self.active_experiment)
        self.__active_experiment = None

    def set_next(self) -> bool:
        if self.active_experiment is not None:
            raise Exception("Finish active experiment before start next.")
        if not self.queue:
            raise Exception("There is no experiments in experiment queue.")
        if self.__experiments_queue:
            self.__active_experiment = self.__experiments_queue.pop(0)
            return True
        return False

    async def start_active(
            self,
            tls: bool,
            root_certificate: Union[Path, str],
            certificate: Union[Path, str],
            private_key: Union[Path, str],
    ) -> None:
        await self.active_experiment.start(
            tls=tls,
            root_certificate=root_certificate,
            certificate=certificate,
            private_key=private_key,
        )
        self._finish_active()

    def __getitem__(self, key) -> Experiment:
        return self.__dict[key]

    def get(self, key, default=None) -> Experiment:
        return self.__dict.get(key, default)

    def __contains__(self, key) -> bool:
        return key in self.__dict


class Director:
    """Director class."""

    def __init__(
            self,
            *,
            tls: bool = True,
            root_certificate: Path = None,
            private_key: Path = None,
            certificate: Path = None,
            sample_shape: list = None,
            target_shape: list = None,
            settings: dict = None
    ) -> None:
        """Initialize a director object."""
        self.sample_shape, self.target_shape = sample_shape, target_shape
        self._shard_registry = {}
        self.experiments = ExperimentsList()

        self.tls = tls
        self.root_certificate = root_certificate
        self.private_key = private_key
        self.certificate = certificate
        self.settings = settings or {}

    async def start(self):
        loop = asyncio.get_event_loop()
        while True:
            if self.experiments.active_experiment is None and not self.experiments.set_next():
                await asyncio.sleep(10)
                continue
            else:
                loop.create_task(self.experiments.start_active(
                    tls=self.tls,
                    root_certificate=self.root_certificate,
                    certificate=self.certificate,
                    private_key=self.private_key,
                ))

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

    async def set_new_experiment(
            self,
            *,
            experiment_name: str,
            sender_name: str,
            tensor_dict: dict,
            collaborator_names: Iterable[str],
            data_file_path: Path,
    ) -> None:
        """Set new experiment."""
        experiment = Experiment(
            name=experiment_name,
            data_path=data_file_path,
            collaborators=list(collaborator_names),
            users=[sender_name],
            sender=sender_name,
            init_tensor_dict=tensor_dict,
        )
        self.experiments.add(experiment)

    def get_trained_model(self, experiment_name: str, caller: str, model_type: str):
        """Get trained model."""
        if (experiment_name not in self.experiments
                or caller not in self.experiments[experiment_name].users):
            logger.error('No experiment data in the stash')
            return None

        aggregator = self.experiments[experiment_name].aggregator

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
        return self.experiments[experiment_name].data_path

    async def wait_experiment(self, collaborator_name: str) -> str:
        """Wait an experiment."""
        while self.experiments.active_experiment is None:
            await asyncio.sleep(10)
        queue = self.experiments.active_experiment.col_exp_queues[collaborator_name]
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
        if (experiment_name not in self.experiments
                or caller not in self.experiments[experiment_name].users):
            raise Exception(
                f'No experiment name "{experiment_name}" in experiments list, or caller "{caller}"'
                f' does not have access to this experiment'
            )
        aggregator = self.experiments[experiment_name].aggregator

        while True:
            if not aggregator.metric_queue.empty():
                yield aggregator.metric_queue.get()
                continue

            if aggregator.all_quit_jobs_sent() and aggregator.metric_queue.empty():
                return

            yield None

    def remove_experiment_data(self, experiment_name: str, caller: str):
        """Remove experiment data from stash."""
        if (experiment_name in self.experiments
                and caller in self.experiments[experiment_name].users):
            self.experiments.remove(experiment_name)

    def collaborator_health_check(
            self, *, collaborator_name: str, is_experiment_running: bool
    ) -> int:
        """Accept health check from envoy."""
        shard_info = self._shard_registry.get(collaborator_name)
        if not shard_info:
            raise Exception(f'Unknown shard {collaborator_name}')

        hc_period = self.settings.get('envoy_health_check_period', ENVOY_HEALTH_CHECK_PERIOD)
        shard_info['is_online']: True
        shard_info['is_experiment_running'] = is_experiment_running
        shard_info['valid_duration'] = 2 * hc_period
        shard_info['last_updated'] = time.time()

        return hc_period

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
