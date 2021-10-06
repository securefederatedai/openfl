import asyncio
import logging
from pathlib import Path
from typing import Iterable
from typing import Union
from typing import List
from collections import defaultdict

from openfl.transport import AggregatorGRPCServer
from openfl.utilities.workspace import ExperimentWorkspace
from openfl.component import Aggregator
from openfl.federated import Plan

logger = logging.getLogger(__name__)


class Status:
    """Experiment's statuses."""

    PENDING = 'PENDING'
    FINISHED = 'FINISHED'
    IN_PROGRESS = 'IN_PROGRESS'
    FAILED = 'FAILED'


class Experiment:
    """Experiment class."""

    def __init__(
            self, *,
            name: str,
            archive_path: Union[Path, str],
            collaborators: List[str],
            sender: str,
            init_tensor_dict: dict,
            plan_path: Union[Path, str] = 'plan/plan.yaml',
            users: Iterable[str] = None,
    ) -> None:
        """Initialize an experiment object."""
        self.name = name
        if isinstance(archive_path, str):
            archive_path = Path(archive_path)
        self.data_path = archive_path
        self.collaborators = collaborators
        self.sender = sender
        self.init_tensor_dict = init_tensor_dict
        if isinstance(plan_path, str):
            plan_path = Path(plan_path)
        self.plan_path = plan_path
        self.users = set() if users is None else set(users)
        self.status = Status.PENDING
        self.__aggregator_grpc_server = None

    @property
    def aggregator(self) -> Union[Aggregator, None]:
        """Get aggregator."""
        if self.__aggregator_grpc_server:
            return self.__aggregator_grpc_server.aggregator

    async def start(
            self, *,
            tls: bool = False,
            root_certificate: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
    ) -> None:
        """Start the experiment."""
        self.status = Status.IN_PROGRESS
        try:
            logger.info(f'New experiment {self.name} for '
                        f'collaborators {self.collaborators}')

            await self._run_aggregator(
                tls=tls,
                root_certificate=root_certificate,
                certificate=certificate,
                private_key=private_key,
            )
            self.status = Status.FINISHED
            logger.info(f'Experiment "{self.name}" was finished successfully.')
        except Exception as e:
            self.status = Status.FAILED
            logger.error(f'Experiment "{self.name}" was failed with error: {e}.')

    async def _run_aggregator(
            self, *,
            tls: bool = False,
            root_certificate: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
    ) -> None:
        with ExperimentWorkspace(self.name, self.data_path):
            aggregator_grpc_server = self._create_aggregator_grpc_server(
                plan_path=self.plan_path,
                tls=tls,
                root_certificate=root_certificate,
                certificate=certificate,
                private_key=private_key,
            )
            await self._run_aggregator_grpc_server(aggregator_grpc_server)

    def _create_aggregator_grpc_server(
            self, *,
            plan_path: Union[Path, str],
            tls: bool = False,
            root_certificate: Union[Path, str] = None,
            certificate: Union[Path, str] = None,
            private_key: Union[Path, str] = None,
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

    async def _run_aggregator_grpc_server(self,
                                          aggregator_grpc_server: AggregatorGRPCServer) -> None:
        """Run aggregator."""
        logger.info('🧿 Starting the Aggregator Service.')
        self.__aggregator_grpc_server = aggregator_grpc_server
        grpc_server = self.__aggregator_grpc_server.get_server()
        grpc_server.start()
        logger.info('Starting Aggregator gRPC Server')

        try:
            while not self.aggregator.all_quit_jobs_sent():
                # Awaiting quit job sent to collaborators
                await asyncio.sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            grpc_server.stop(0)
            # Temporary solution to free RAM used by TensorDB
            self.aggregator.tensor_db.clean_up(0)


class ExperimentsRegistry:
    """ExperimentsList class."""

    def __init__(
            self,
            tls: bool,
            root_certificate: Union[Path, str],
            certificate: Union[Path, str],
            private_key: Union[Path, str],
            experiments: List[Experiment] = None,
    ) -> None:
        """Initialize an experiments list object."""
        self.__active_experiment = None
        self.__col_exp_queues = defaultdict(asyncio.Queue)
        self.tls = tls
        self.root_certificate = root_certificate
        self.certificate = certificate
        self.private_key = private_key

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
        """Get active experiment."""
        if self.__active_experiment is None:
            return None
        return self.__dict[self.__active_experiment]

    @property
    def queue(self) -> List[str]:
        """Get queue of not started experiments."""
        return self.__experiments_queue

    def add(self, experiment: Experiment) -> None:
        """Add experiment to queue of not started experiments."""
        self.__dict[experiment.name] = experiment
        self.__experiments_queue.append(experiment.name)

    def remove(self, name: str) -> None:
        """Remove experiment from everywhere."""
        if self.__active_experiment == name:
            self.__active_experiment = None
        if name in self.__experiments_queue:
            self.__experiments_queue.remove(name)
        if name in self.__archived_experiments:
            self.__archived_experiments.remove(name)
        if name in self.__dict:
            del self.__dict[name]

    async def get_envoy_experiment(self, envoy_name: str) -> str:
        """Get experiment name for envoy."""
        queue = self.__col_exp_queues[envoy_name]
        return await queue.get()

    async def run_next_experiment(self) -> None:
        """Set next experiment from the queue."""
        while True:
            if self.active_experiment is not None or not self.queue:
                await asyncio.sleep(10)
                continue
            self.__active_experiment = self.__experiments_queue.pop(0)
            await self.start_active()
            return

    async def start_active(self) -> None:
        """Start active experiment."""
        loop = asyncio.get_event_loop()
        run_aggregator = loop.create_task(self.active_experiment.start(
            tls=self.tls,
            root_certificate=self.root_certificate,
            certificate=self.certificate,
            private_key=self.private_key,
        ))
        for col_name in self.active_experiment.collaborators:
            queue = self.__col_exp_queues[col_name]
            await queue.put(self.active_experiment.name)
        await run_aggregator
        self._finish_active()

    def __getitem__(self, key: str) -> Experiment:
        """Get experiment by name."""
        return self.__dict[key]

    def get(self, key: str, default=None) -> Experiment:
        """Get experiment by name."""
        return self.__dict.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if experiment exists."""
        return key in self.__dict

    def _finish_active(self) -> None:
        self.__dict[self.__active_experiment].__aggregator_grpc_server = None
        self.__archived_experiments.insert(0, self.__active_experiment)
        self.__active_experiment = None
