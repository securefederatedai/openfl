# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Collaborator module."""

import importlib
import logging
from enum import Enum
from time import sleep
from typing import List, Optional

import openfl.callbacks as callbacks_module
from openfl.databases import TensorDB
from openfl.pipelines import NoCompressionPipeline, TensorCodec
from openfl.protocols import utils
from openfl.transport.grpc.aggregator_client import AggregatorGRPCClient
from openfl.utilities import TensorKey

logger = logging.getLogger(__name__)


class OptTreatment(Enum):
    """Optimizer Methods.

    Attributes:
        RESET (int): Resets the optimizer state at the beginning of each round.
        CONTINUE_LOCAL (int): Continues with the local optimizer state from
            the previous round.
        CONTINUE_GLOBAL (int): Continues with the federally averaged optimizer
            state from the previous round.
    """

    RESET = 1
    CONTINUE_LOCAL = 2
    CONTINUE_GLOBAL = 3


class Collaborator:
    r"""The Collaborator object class.

    Attributes:
        collaborator_name (str): The common name for the collaborator.
        aggregator_uuid (str): The unique id for the client.
        federation_uuid (str): The unique id for the federation.
        client (object): The client object.
        task_runner (object): The task runner object.
        task_config (dict): The task configuration.
        opt_treatment (str)*: The optimizer state treatment.
        device_assignment_policy (str): [Deprecated] The device assignment policy.
        use_delta_updates (bool)*: If True, only model delta gets sent. If False,
            whole model gets sent to collaborator.
        compression_pipeline (object): The compression pipeline.
        db_store_rounds (int): The number of rounds to store in the database.
        single_col_cert_common_name (str): The common name for the single
            column certificate.

    .. note::
        \* - Plan setting.
    """

    def __init__(
        self,
        collaborator_name,
        aggregator_uuid,
        federation_uuid,
        client: AggregatorGRPCClient,
        task_runner,
        task_config,
        opt_treatment="RESET",
        device_assignment_policy="CPU_ONLY",
        use_delta_updates=False,
        compression_pipeline=None,
        db_store_rounds=1,
        log_memory_usage=False,
        write_logs=False,
        callbacks: Optional[List] = [],
        secure_aggregation=False,
        interop_mode=False,
    ):
        """Initialize the Collaborator object.

        Args:
            collaborator_name (str): The common name for the collaborator.
            aggregator_uuid (str): The unique id for the client.
            federation_uuid (str): The unique id for the federation.
            client (object): The client object.
            task_runner (object): The task runner object.
            task_config (dict): The task configuration.
            opt_treatment (str, optional): The optimizer state treatment.
                Defaults to 'RESET'.
            device_assignment_policy (str, optional): The device assignment
                policy. Defaults to 'CPU_ONLY'.
            use_delta_updates (bool, optional): If True, only model delta gets
                sent. If False, whole model gets sent to collaborator.
                Defaults to False.
            compression_pipeline (object, optional): The compression pipeline.
                Defaults to None.
            db_store_rounds (int, optional): The number of rounds to store in
                the database. Defaults to 1.
            callbacks (list, optional): List of callbacks. Defaults to None.
        """
        # for protobuf compatibility we would really want this as an object
        self.single_col_cert_common_name = ""

        self.collaborator_name = collaborator_name
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid

        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()
        self.tensor_codec = TensorCodec(self.compression_pipeline)
        self.tensor_db = TensorDB()
        self.db_store_rounds = db_store_rounds

        self.task_runner = task_runner
        self.use_delta_updates = use_delta_updates

        self.client = client

        self.task_config = task_config

        # RESET/CONTINUE_LOCAL/CONTINUE_GLOBAL
        if hasattr(OptTreatment, opt_treatment):
            self.opt_treatment = OptTreatment[opt_treatment]
        else:
            logger.error("Unknown opt_treatment: %s.", opt_treatment.name)
            raise NotImplementedError(f"Unknown opt_treatment: {opt_treatment}.")
        self.task_runner.set_optimizer_treatment(self.opt_treatment.name)

        logger.warning(
            "Argument `device_assignment_policy` is deprecated and will be removed in the future."
        )
        del device_assignment_policy

        # Secure aggregation
        self._secure_aggregation_enabled = secure_aggregation
        if self._secure_aggregation_enabled:
            self._private_mask = None
            self._shared_mask = None
            secure_aggregation_callback = callbacks_module.SecAggBootstrapping()
            if isinstance(callbacks, callbacks_module.Callback):
                callbacks = [callbacks, secure_aggregation_callback]
            elif isinstance(callbacks, list):
                callbacks.append(secure_aggregation_callback)
            else:
                callbacks = [secure_aggregation_callback]

        # Interoperability mode
        self._interop_mode_enabled = interop_mode
        if self._interop_mode_enabled:
            callbacks.append(
                callbacks_module.LambdaCallback(
                    on_experiment_begin=lambda logs=None: self.prepare_interop_server()
                )
            )

        # Callbacks
        self.callbacks = callbacks_module.CallbackList(
            callbacks,
            add_memory_profiler=log_memory_usage,
            add_metric_writer=write_logs,
            tensor_db=self.tensor_db,
            origin=self.collaborator_name,
            client=self.client,
        )

    def ping(self):
        """Ping the Aggregator."""
        self.client.ping()

    def run(self):
        """Run the collaborator."""
        # Experiment begin
        self.callbacks.on_experiment_begin()

        while True:
            tasks, round_num, sleep_time, time_to_quit = self.client.get_tasks()

            if time_to_quit:
                break

            if not tasks:
                sleep(sleep_time)
                continue

            # Round begin
            logger.info("Round: %d Received Tasks: %s", round_num, tasks)
            self.callbacks.on_round_begin(round_num)

            # Run tasks
            logs = {}
            for task in tasks:
                logger.info("Task: `%s`", task.name)
                metrics = self.do_task(task, round_num)
                logs.update(metrics)

            # Round end
            self.tensor_db.clean_up(self.db_store_rounds)
            self.callbacks.on_round_end(round_num, logs)

        # Experiment end
        self.callbacks.on_experiment_end()
        logger.info("Received shutdown signal. Exiting...")

    def do_task(self, task, round_number) -> dict:
        """Perform the specified task.

        Args:
            task: Task proto.
            round_number (int): Round number.

        Returns:
            A dictionary of reportable metrics of the current collaborator for the task.
        """
        func_name = self.task_config[task.name]["function"]
        kwargs = self.task_config[task.name]["kwargs"]

        # this would return a list of what tensors we require as TensorKeys
        # models actually return "relative" tensorkeys of (name, LOCAL|GLOBAL,
        # round_offset) so we need to update these keys to their "absolute values"
        tensor_keys = self.task_runner.get_required_tensorkeys_for_function(func_name, **kwargs)
        global_keys, local_keys = [], []
        for tensor_key in tensor_keys:
            if tensor_key.origin == "GLOBAL":
                tensor_key = tensor_key._replace(
                    origin=self.aggregator_uuid, round_number=round_number
                )
                global_keys.append(tensor_key)

            elif tensor_key.origin == "LOCAL":
                tensor_key = tensor_key._replace(
                    origin=self.collaborator_name, round_number=round_number
                )
                local_keys.append(tensor_key)

        # Prepare input tensor dict for this task
        self.fetch_tensors_from_aggregator(global_keys)
        input_tensor_dict = {}
        for tk in local_keys:
            value = self.tensor_db.get_tensor_from_cache(tk)
            if value is None:
                raise ValueError(f"Value corresponding to local tensor `{tk}` not found.")
            input_tensor_dict[tk.tensor_name] = value

        for tk in global_keys:
            value = self.tensor_db.get_tensor_from_cache(tk)
            if value is None:
                raise ValueError(f"Value corresponding to global tensor `{tk}` not found.")
            input_tensor_dict[tk.tensor_name] = value

        self.callbacks.on_task_begin(task.name, round_number)

        # now we have whatever the model needs to do the task
        # Tasks are defined as methods of TaskRunner
        func = getattr(self.task_runner, func_name)
        global_output_tensor_dict, local_output_tensor_dict = func(
            col_name=self.collaborator_name,
            round_num=round_number,
            input_tensor_dict=input_tensor_dict,
            **kwargs,
        )

        self.callbacks.on_task_end(task.name, round_number)

        # If secure aggregation is enabled, add masks to the dict to be shared
        # with the aggregator.
        if self._secure_aggregation_enabled:
            self._apply_masks(global_output_tensor_dict)

        # Save global and local output_tensor_dicts to TensorDB
        self.tensor_db.cache_tensor(global_output_tensor_dict)
        self.tensor_db.cache_tensor(local_output_tensor_dict)

        # send the results for this tasks; delta and compression will occur in
        # this function
        metrics = self.send_task_results(global_output_tensor_dict, round_number, task.name)

        return metrics

    def fetch_tensors_from_aggregator(self, tensor_keys: List[TensorKey]):
        """Fetches tensors from the aggregator and stores them locally.

        This function checks if the tensors are already cached in the local database
        and fetches them from the aggregator if not. The fetched tensors are then
        cached in the local database.

        Args:
            tensor_keys (list): List of TensorKeys to fetch.
        """
        tensor_dict = {}
        tensor_keys = list(
            filter(lambda k: self.tensor_db.get_tensor_from_cache(k) is None, tensor_keys)
        )
        if len(tensor_keys) > 0:
            logger.info("Fetching %d tensors from the aggregator", len(tensor_keys))
            named_tensors = self.client.get_aggregated_tensors(tensor_keys, require_lossless=True)

            # Deserialize tensors and mark them as coming from the aggregator.
            for tensor_key, named_tensor in zip(tensor_keys, named_tensors):
                tensor_key, nparray = utils.deserialize_tensor(named_tensor, self.tensor_codec)
                tensor_key = tensor_key._replace(origin=self.aggregator_uuid)
                tensor_dict[tensor_key] = nparray

        self.tensor_db.cache_tensor(tensor_dict)

    def send_task_results(self, tensor_dict, round_number, task_name) -> dict:
        """Send task results to the aggregator.

        Args:
            tensor_dict (dict): Tensor dictionary.
            round_number (int):  Actual round number.
            task_name (string): Task name.

        Returns:
            A dictionary of reportable metrics of the current collaborator for the task.
        """
        # for general tasks, there may be no notion of data size to send.
        # But that raises the question how to properly aggregate results.

        data_size = -1

        if "train" in task_name:
            data_size = self.task_runner.get_train_data_size()

        if "valid" in task_name:
            data_size = self.task_runner.get_valid_data_size()

        logger.debug("%s data size = %s", task_name, data_size)

        metrics = {}
        for tensor in tensor_dict:
            tensor_name, origin, fl_round, report, tags = tensor

            if report:
                # Reportable metric must be a scalar
                value = float(tensor_dict[tensor])
                metrics.update({f"{self.collaborator_name}/{task_name}/{tensor_name}": value})

        # Serialize tensors to be sent to the aggregator
        named_tensors = [
            utils.serialize_tensor(k, v, self.tensor_codec, lossless=True)
            for k, v in tensor_dict.items()
        ]

        self.client.send_local_task_results(
            round_number,
            task_name,
            data_size,
            named_tensors,
        )

        return metrics

    def _apply_masks(
        self,
        tensor_dict,
    ):
        """
        Calculate masked input vectors for secure aggregation.

        This function fetches private and shared masks from the tensor database if
        they are not provided, and applies these masks to the input tensors.

        Args:
            tensor_dict (dict): A dictionary of tensors to be masked.
        """
        import numpy as np

        # Fetch private mask from tensor db if not already fetched.
        if not self._private_mask:
            self._private_mask = self.tensor_db.get_tensor_from_cache(
                TensorKey("private_mask", self.collaborator_name, -1, False, ("secagg",))
            )[0]
        # Fetch shared mask from tensor db if not already fetched.
        if not self._shared_mask:
            self._shared_mask = self.tensor_db.get_tensor_from_cache(
                TensorKey("shared_mask", self.collaborator_name, -1, False, ("secagg",))
            )[0]

        for tensor_key in tensor_dict:
            _, _, _, _, tags = tensor_key
            if "metric" in tags:
                continue
            masked_metric = np.add(self._private_mask, tensor_dict[tensor_key])
            tensor_dict[tensor_key] = np.add(masked_metric, self._shared_mask)

    def prepare_interop_server(self):
        """
        Prepare the interoperability server.

        This function initializes the interoperability server and sets up
        the callback for receiving messages from the interop server.
        It also sets the interop server in the task configuration to be used
        by the Task Runner.
        """

        # Initialize the interop server
        framework = self.task_config["settings"]["interop_server"]
        module = importlib.import_module(framework)

        def receive_message_from_interop(message):
            """Receive message from interop server."""
            # Process the request and return a response
            response = self.client.send_message_to_server(message, self.collaborator_name)
            return response

        interop_server = module.FlowerInteropServer(receive_message_from_interop)
        self.task_config["prepare_for_interop"]["kwargs"]["interop_server"] = interop_server
