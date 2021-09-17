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
            aggregator: AggregatorGRPCServer = None,
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
        self.aggregator = aggregator
        self.users = set() if users is None else set(users)
        self.status = Status.PENDING

    async def start(
            self,
            tls=False,
            root_certificate=None,
            certificate=None,
            private_key=None,
    ) -> None:
        self.status = Status.IN_PROGRESS
        try:
            with ExperimentWorkspace(self.name, self.data_path):
                self.aggregator = self._create_aggregator(
                    plan_path=self.plan_path,
                    tls=tls,
                    root_certificate=root_certificate,
                    certificate=certificate,
                    private_key=private_key,
                )
                await self._run_aggregator()
        except:
            self.status = Status.FAILED
        self.status = Status.FINISHED

    def _create_aggregator(
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
        aggregator_server = plan.interactive_api_get_server(
            tensor_dict=self.init_tensor_dict,
            root_certificate=root_certificate,
            certificate=certificate,
            private_key=private_key,
            tls=tls,
        )
        return aggregator_server

    async def _run_aggregator(self) -> None:
        """Run aggregator."""
        plan = Plan.parse(plan_config_path=Path(self.plan_path))
        plan.authorized_cols = list(self.collaborators)

        logger.info('🧿 Starting the Aggregator Service.')
        grpc_server = self.aggregator.get_server()
        grpc_server.start()
        logger.info('Starting Aggregator gRPC Server')

        try:
            while not self.aggregator.aggregator.all_quit_jobs_sent():
                # Awaiting quit job sent to collaborators
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            grpc_server.stop(0)
            # Temporary solution to free RAM used by TensorDB
            self.aggregator.aggregator.tensor_db.clean_up(0)


class ExperimentsList:

    def __init__(self, experiments: List[Experiment] = None) -> None:
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

    def finish_active(self) -> None:
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
        # TODO: add working directory
        super().__init__()
        self.sample_shape, self.target_shape = sample_shape, target_shape
        self._shard_registry = {}
        self.col_exp_queues = defaultdict(asyncio.Queue)
        self.experiments = ExperimentsList()
        self.experiment_data = {}  # {Experiment name : archive bytes}
        # What if two experiments come with the same name from different users?
        self.experiment_stash = defaultdict(dict)  # Running of finished experiments
        # {API name : {experiment name : aggregator}}

        self.tls = tls
        self.root_certificate = root_certificate
        self.private_key = private_key
        self.certificate = certificate
        self.settings = settings or {}

    async def start(self):
        loop = asyncio.get_event_loop()
        while True:
            if self.experiments.active_experiment is None:
                if not self.experiments.set_next():
                    await asyncio.sleep(10)
                    continue
            loop.create_task(self.experiments.active_experiment.start(
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
            data_file_path: Path,
            tensor_dict: dict,
    ) -> None:
        """Set new experiment."""
        experiment = Experiment(
            name=experiment_name,
            data_path=data_file_path,
            users=[sender_name],
            sender=sender_name,
            init_tensor_dict=tensor_dict,
        )
        self.experiments.add(experiment)

    async def run_experiment(
            self,
            experiment: Experiment,
            tensor_dict
    ) -> bool:
        loop = asyncio.get_event_loop()  # TODO: refactor after end of support for python3.6
        loop.create_task(self._run_aggregator_in_workspace(
            experiment=experiment,
            initial_tensor_dict=tensor_dict,
        ))

        logger.info(f'New experiment {experiment.name} for '
                    f'collaborators {experiment.collaborators}')
        for col_name in experiment.collaborators:
            queue = self.col_exp_queues[col_name]
            await queue.put(experiment.name)

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
                self.experiments.
                return

            yield None

    def remove_experiment_data(self, experiment_name: str, caller: str):
        """Remove experiment data from stash."""
        if experiment_name in self.experiment_stash.get(caller, {}):
            del self.experiment_stash[caller][experiment_name]

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

    async def _run_aggregator_in_workspace(
            self,
            *,
            experiment: Experiment,
            **kwargs
    ) -> None:
        """Run aggregator in a workspace."""
        with ExperimentWorkspace(experiment.name, experiment.data_path):
            await self._run_aggregator(experiment=experiment, **kwargs)

    # async def _run_aggregator(
    #         self,
    #         *,
    #         experiment: Experiment,
    #         plan_path='plan/plan.yaml'
    # ) -> None:
    #     """Run aggregator."""
    #     plan = Plan.parse(plan_config_path=Path(plan_path))
    #     plan.authorized_cols = list(experiment.collaborators)
    #
    #     logger.info('🧿 Starting the Aggregator Service.')
    #     grpc_server = experiment.aggregator.get_server()
    #     grpc_server.start()
    #     logger.info('Starting Aggregator gRPC Server')
    #
    #     try:
    #         while not experiment.aggregator.aggregator.all_quit_jobs_sent():
    #             # Awaiting quit job sent to collaborators
    #             await asyncio.sleep(10)
    #     except KeyboardInterrupt:
    #         pass
    #     finally:
    #         grpc_server.stop(0)
    #         # Temporary solution to free RAM used by TensorDB
    #         experiment.aggregator.aggregator.tensor_db.clean_up(0)
