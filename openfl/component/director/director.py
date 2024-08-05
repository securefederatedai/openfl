# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Director module."""

import asyncio
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Union

from openfl.component.director.experiment import Experiment, ExperimentsRegistry, Status
from openfl.transport.grpc.exceptions import ShardNotFoundError

logger = logging.getLogger(__name__)


class Director:
    """Director class. The Director is the central node of the federation
    (Director-Based Workflow).

    Attributes:
        tls (bool): A flag indicating if TLS should be used for connections.
        root_certificate (Union[Path, str]): The path to the root certificate
            for TLS.
        private_key (Union[Path, str]): The path to the private key for TLS.
        certificate (Union[Path, str]): The path to the certificate for TLS.
        sample_shape (list): The shape of the sample data.
        target_shape (list): The shape of the target data.
        review_plan_callback (Union[None, Callable]): A callback function for
            reviewing the plan.
        envoy_health_check_period (int): The period for health check of envoys
            in seconds.
        install_requirements (bool): A flag indicating if the requirements
            should be installed.
        _shard_registry (dict): A dictionary to store the shard registry.
        experiments_registry (ExperimentsRegistry): An object of
            ExperimentsRegistry to store the experiments.
        col_exp_queues (defaultdict): A defaultdict to store the experiment
            queues for collaborators.
        col_exp (dict): A dictionary to store the experiments for
            collaborators.
        logger (Logger): A logger for logging activities.
    """

    def __init__(
        self,
        *,
        tls: bool = True,
        root_certificate: Union[Path, str] = None,
        private_key: Union[Path, str] = None,
        certificate: Union[Path, str] = None,
        sample_shape: list = None,
        target_shape: list = None,
        review_plan_callback: Union[None, Callable] = None,
        envoy_health_check_period: int = 60,
        install_requirements: bool = False,
    ) -> None:
        """Initialize the Director object.

        Args:
            tls (bool, optional): A flag indicating if TLS should be used for
                connections. Defaults to True.
            root_certificate (Union[Path, str], optional): The path to the
                root certificate for TLS. Defaults to None.
            private_key (Union[Path, str], optional): The path to the private
                key for TLS. Defaults to None.
            certificate (Union[Path, str], optional): The path to the
                certificate for TLS. Defaults to None.
            sample_shape (list, optional): The shape of the sample data.
                Defaults to None.
            target_shape (list, optional): The shape of the target data.
                Defaults to None.
            review_plan_callback (Union[None, Callable], optional): A callback
                function for reviewing the plan. Defaults to None.
            envoy_health_check_period (int, optional): The period for health
                check of envoys in seconds. Defaults to 60.
            install_requirements (bool, optional): A flag indicating if the
                requirements should be installed. Defaults to False.
        """
        self.sample_shape, self.target_shape = sample_shape, target_shape
        self._shard_registry = {}
        self.tls = tls
        self.root_certificate = root_certificate
        self.private_key = private_key
        self.certificate = certificate
        self.experiments_registry = ExperimentsRegistry()
        self.col_exp_queues = defaultdict(asyncio.Queue)
        self.col_exp = {}
        self.review_plan_callback = review_plan_callback
        self.envoy_health_check_period = envoy_health_check_period
        self.install_requirements = install_requirements

    def acknowledge_shard(self, shard_info: dict) -> bool:
        """Save shard info to shard registry if it's acceptable.

        Args:
            shard_info (dict): The shard info dictionary should be able to
                store registries.

        Returns:
            is_accepted (bool): Bool value to accept o deny the addition of
                the shard info.
        """
        is_accepted = False
        if (
            self.sample_shape != shard_info["sample_shape"]
            or self.target_shape != shard_info["target_shape"]
        ):
            logger.info(
                "Director did not accept shard for %s",
                shard_info["node_info"]["name"],
            )
            return is_accepted
        logger.info("Director accepted shard for %s", shard_info["node_info"]["name"])
        self._shard_registry[shard_info["node_info"]["name"]] = {
            "shard_info": shard_info,
            "is_online": True,
            "is_experiment_running": False,
            "valid_duration": 2 * self.envoy_health_check_period,
            "last_updated": time.time(),
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
        experiment_archive_path: Path,
    ) -> bool:
        """Set new experiment.

        Args:
            experiment_name (str): String id for experiment.
            sender_name (str): The name of the sender.
            tensor_dict (dict): Dictionary of tensors.
            collaborator_names (Iterable[str]): Names of collaborators.
            experiment_archive_path (Path): Path of the experiment.

        Returns:
            bool : Boolean returned if the experiment register was successful.
        """
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

    async def get_experiment_status(self, experiment_name: str, caller: str):
        """Get experiment status.

        Args:
            experiment_name (str): String id for experiment.
            caller (str): String id for experiment owner.

        Returns:
            str: The status of the experiment can be one of the following:
                - PENDING = 'pending'
                - FINISHED = 'finished'
                - IN_PROGRESS = 'in_progress'
                - FAILED = 'failed'
                - REJECTED = 'rejected'
        """
        if (
            experiment_name not in self.experiments_registry
            or caller not in self.experiments_registry[experiment_name].users
        ):
            logger.error("No experiment data in the stash")
            return None
        return self.experiments_registry[experiment_name].status

    def get_trained_model(self, experiment_name: str, caller: str, model_type: str):
        """Get trained model.

        Args:
            experiment_name (str): String id for experiment.
            caller (str): String id for experiment owner.
            model_type (str): The type of the model.

        Returns:
            None: One of the following: [No experiment data in the stash] or
                [Aggregator have no aggregated model to return] or [Unknown
                model type required].
            dict: Dictionary of tensors from the aggregator when the model
                type is 'best' or 'last'.
        """
        if (
            experiment_name not in self.experiments_registry
            or caller not in self.experiments_registry[experiment_name].users
        ):
            logger.error("No experiment data in the stash")
            return None

        aggregator = self.experiments_registry[experiment_name].aggregator

        if aggregator.last_tensor_dict is None:
            logger.error("Aggregator has no aggregated model to return")
            return None

        if model_type == "best":
            return aggregator.best_tensor_dict
        elif model_type == "last":
            return aggregator.last_tensor_dict
        else:
            logger.error("Unknown model type required.")
            return None

    def get_experiment_data(self, experiment_name: str) -> Path:
        """Get experiment data.

        Args:
            experiment_name (str): String id for experiment.

        Returns:
            str: Path of archive.
        """
        return self.experiments_registry[experiment_name].archive_path

    async def wait_experiment(self, envoy_name: str) -> str:
        """Wait an experiment.

        Args:
            envoy_name (str): The name of the envoy.

        Returns:
            str: The name of the experiment on the queue.
        """
        experiment_name = self.col_exp.get(envoy_name)
        if experiment_name and experiment_name in self.experiments_registry:
            # Experiment already set, but the envoy hasn't received experiment
            # name (e.g. was disconnected)
            experiment = self.experiments_registry[experiment_name]
            if experiment.aggregator.round_number < experiment.aggregator.rounds_to_train:
                return experiment_name

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
        return [shard_status["shard_info"] for shard_status in self._shard_registry.values()]

    async def stream_metrics(self, experiment_name: str, caller: str):
        """Stream metrics from the aggregator.

        This method takes next metric dictionary from the aggregator's queue
        and returns it to the caller.

        Args:
            experiment_name (str): String id for experiment.
            caller (str): String id for experiment owner.

        Returns:
            metric_dict: {'metric_origin','task_name','metric_name','metric_value','round'}
                if the queue is not empty.
            None: queue is empty but the experiment is still running.

        Raises:
            StopIteration: if the experiment is finished and there is no more metrics to report.
        """
        if (
            experiment_name not in self.experiments_registry
            or caller not in self.experiments_registry[experiment_name].users
        ):
            raise Exception(
                f'No experiment name "{experiment_name}" in experiments list, or caller "{caller}"'
                f" does not have access to this experiment"
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
        """Remove experiment data from stash.

        Args:
            experiment_name (str): String id for experiment.
            caller (str): String id for experiment owner.
        """
        if (
            experiment_name in self.experiments_registry
            and caller in self.experiments_registry[experiment_name].users
        ):
            self.experiments_registry.remove(experiment_name)

    def set_experiment_failed(self, *, experiment_name: str, collaborator_name: str):
        """Envoys Set experiment failed RPC.

        Args:
            experiment_name (str): String id for experiment.
            collaborator_name (str): String id for collaborator.

        Return:
            None
        """
        """This method shoud call `experiment.abort()` and all the code should
        be pushed down to that method.

        It would be also good to be able to interrupt aggregator async task
        with the following code:
        ```
        run_aggregator_atask = self.experiments_registry[experiment_name].run_aggregator_atask
        if asyncio.isfuture(run_aggregator_atask) and not run_aggregator_atask.done():
            run_aggregator_atask.cancel()
        ```
        unfortunately, cancelling does not work on already awaited future.
        """

        if experiment_name not in self.experiments_registry:
            return
        aggregator = self.experiments_registry[experiment_name].aggregator
        aggregator.stop(failed_collaborator=collaborator_name)
        self.experiments_registry[experiment_name].status = Status.FAILED

    def update_envoy_status(
        self,
        *,
        envoy_name: str,
        is_experiment_running: bool,
        cuda_devices_status: list = None,
    ) -> int:
        """Accept health check from envoy.

        Args:
            envoy_name (str): String id for envoy.
            is_experiment_running (bool): Boolean value for the status of the
                experiment.
            cuda_devices_status (list, optional): List of cuda devices and
                status. Defaults to None.

        Raises:
            ShardNotFoundError: When Unknown shard {envoy_name}.

        Returns:
            int: Value of the envoy_health_check_period.
        """
        shard_info = self._shard_registry.get(envoy_name)
        if not shard_info:
            raise ShardNotFoundError(f"Unknown shard {envoy_name}")

        shard_info["is_online"]: True
        shard_info["is_experiment_running"] = is_experiment_running
        shard_info["valid_duration"] = 2 * self.envoy_health_check_period
        shard_info["last_updated"] = time.time()

        if cuda_devices_status is not None:
            for i in range(len(cuda_devices_status)):
                shard_info["shard_info"]["node_info"]["cuda_devices"][i] = cuda_devices_status[i]

        return self.envoy_health_check_period

    def get_envoys(self) -> list:
        """Get a status information about envoys.

        Returns:
            list: List with the status information about envoys.
        """
        logger.debug("Shard registry: %s", self._shard_registry)
        for envoy_info in self._shard_registry.values():
            envoy_info["is_online"] = time.time() < envoy_info.get(
                "last_updated", 0
            ) + envoy_info.get("valid_duration", 0)
            envoy_name = envoy_info["shard_info"]["node_info"]["name"]
            envoy_info["experiment_name"] = self.col_exp[envoy_name]

        return self._shard_registry.values()

    def get_experiments_list(self, caller: str) -> list:
        """Get experiments list for specific user.

        Args:
            caller (str): String id for experiment owner.
        Returns:
            list: List with the info of the experiment for specific user.
        """
        experiments = self.experiments_registry.get_user_experiments(caller)
        result = []
        for exp in experiments:
            exp_data = {
                "name": exp.name,
                "status": exp.status,
                "collaborators_amount": len(exp.collaborators),
            }
            progress = _get_experiment_progress(exp)
            if progress is not None:
                exp_data["progress"] = progress
            if exp.aggregator:
                tasks_amount = len(
                    {task["function"] for task in exp.aggregator.assigner.tasks.values()}
                )
                exp_data["tasks_amount"] = tasks_amount
            result.append(exp_data)

        return result

    def get_experiment_description(self, caller: str, name: str) -> dict:
        """Get a experiment information by name for specific user.

        Args:
            caller (str): String id for experiment owner.
            name (str): String id for experiment name.

        Returns:
            dict: Dictionary with the info from the experiment.
        """
        exp = self.experiments_registry.get(name)
        if not exp or caller not in exp.users:
            return {}
        progress = _get_experiment_progress(exp)
        model_statuses = _get_model_download_statuses(exp)
        tasks = _get_experiment_tasks(exp)
        collaborators = _get_experiment_collaborators(exp)
        result = {
            "name": name,
            "status": exp.status,
            "current_round": exp.aggregator.round_number,
            "total_rounds": exp.aggregator.rounds_to_train,
            "download_statuses": {
                "models": model_statuses,
                "logs": [{"name": "aggregator", "status": "ready"}],
            },
            "collaborators": collaborators,
            "tasks": tasks,
            "progress": progress,
        }
        return result

    async def start_experiment_execution_loop(self):
        """Run task to monitor and run experiments."""
        loop = asyncio.get_event_loop()
        while True:
            async with self.experiments_registry.get_next_experiment() as experiment:

                # Review experiment block starts.
                if self.review_plan_callback:
                    if not await experiment.review_experiment(self.review_plan_callback):
                        logger.info(
                            f'"{experiment.name}" Plan was rejected by the Director manager.'
                        )
                        continue
                # Review experiment block ends.

                run_aggregator_future = loop.create_task(
                    experiment.start(
                        root_certificate=self.root_certificate,
                        certificate=self.certificate,
                        private_key=self.private_key,
                        tls=self.tls,
                        install_requirements=self.install_requirements,
                    )
                )
                # Adding the experiment to collaborators queues
                for col_name in experiment.collaborators:
                    queue = self.col_exp_queues[col_name]
                    await queue.put(experiment.name)
                await run_aggregator_future


def _get_model_download_statuses(experiment) -> List[dict]:
    best_model_status = "ready" if experiment.aggregator.best_tensor_dict else "pending"
    last_model_status = "ready" if experiment.aggregator.last_tensor_dict else "pending"
    model_statuses = [
        {
            "name": "best",
            "status": best_model_status,
        },
        {
            "name": "last",
            "status": last_model_status,
        },
        {"name": "init", "status": "ready"},
    ]
    return model_statuses


def _get_experiment_progress(experiment) -> Union[float, None]:
    if experiment.status == Status.IN_PROGRESS:
        return experiment.aggregator.round_number / experiment.aggregator.rounds_to_train


def _get_experiment_tasks(experiment) -> List[dict]:
    return [
        {
            "name": task["function"],
            "description": "Task description Mock",
        }
        for task in experiment.aggregator.assigner.tasks.values()
    ]


def _get_experiment_collaborators(experiment) -> List[dict]:
    return [
        {
            "name": name,
            "status": "pending_mock",
            "progress": 0.0,
            "round": 0,
            "current_task": "Current Task Mock",
            "next_task": "Next Task Mock",
        }
        for name in experiment.aggregator.authorized_cols
    ]
