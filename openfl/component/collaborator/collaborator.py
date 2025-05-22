# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Collaborator module."""

import importlib
import logging
from enum import Enum
from os.path import splitext
from time import sleep
from typing import List, Optional

import openfl.callbacks as callbacks_module
from openfl.databases import TensorDB
from openfl.pipelines import NoCompressionPipeline, TensorCodec
from openfl.protocols import utils
from openfl.transport.grpc.aggregator_client import AggregatorClientInterface
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
        client: AggregatorClientInterface,
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
            task (list_of_str): List of tasks.
            round_number (int): Actual round number.

        Returns:
            A dictionary of reportable metrics of the current collaborator for the task.
        """
        # map this task to an actual function name and kwargs
        if isinstance(task, str):
            task_name = task
        else:
            task_name = task.name
        func_name = self.task_config[task_name]["function"]
        kwargs = self.task_config[task_name]["kwargs"]

        # this would return a list of what tensors we require as TensorKeys
        required_tensorkeys_relative = self.task_runner.get_required_tensorkeys_for_function(
            func_name, **kwargs
        )

        # models actually return "relative" tensorkeys of (name, LOCAL|GLOBAL,
        # round_offset)
        # so we need to update these keys to their "absolute values"
        required_tensorkeys = []
        for (
            tname,
            origin,
            rnd_num,
            report,
            tags,
        ) in required_tensorkeys_relative:
            if origin == "GLOBAL":
                origin = self.aggregator_uuid
            else:
                origin = self.collaborator_name

            # rnd_num is the relative round. So if rnd_num is -1, get the
            # tensor from the previous round
            required_tensorkeys.append(
                TensorKey(tname, origin, rnd_num + round_number, report, tags)
            )

        # print('Required tensorkeys = {}'.format(
        # [tk[0] for tk in required_tensorkeys]))
        input_tensor_dict = {
            k.tensor_name: self.get_data_for_tensorkey(k) for k in required_tensorkeys
        }
        self.callbacks.on_task_begin(task_name, round_number)
        # now we have whatever the model needs to do the task
        # Tasks are defined as methods of TaskRunner
        func = getattr(self.task_runner, func_name)
        logger.debug("Using TaskRunner subclassing API")

        global_output_tensor_dict, local_output_tensor_dict = func(
            col_name=self.collaborator_name,
            round_num=round_number,
            input_tensor_dict=input_tensor_dict,
            **kwargs,
        )

        self.callbacks.on_task_end(task_name, round_number)

        # If secure aggregation is enabled, add masks to the dict to be shared
        # with the aggregator.
        if self._secure_aggregation_enabled:
            self._apply_masks(global_output_tensor_dict)

        # Save global and local output_tensor_dicts to TensorDB
        self.tensor_db.cache_tensor(global_output_tensor_dict)
        self.tensor_db.cache_tensor(local_output_tensor_dict)

        # send the results for this tasks; delta and compression will occur in
        # this function
        metrics = self.send_task_results(global_output_tensor_dict, round_number, task_name)

        return metrics

    def get_data_for_tensorkey(self, tensor_key):
        """Resolve the tensor corresponding to the requested tensorkey.

        Args:
            tensor_key (namedtuple): Tensorkey that will be resolved locally or
            remotely. May be the product of other tensors.

        Returns:
            nparray: The decompressed tensor associated with the requested
                tensor key.
        """
        # try to get from the store
        tensor_name, origin, round_number, report, tags = tensor_key
        logger.debug("Attempting to retrieve tensor %s from local store", tensor_key)
        nparray = self.tensor_db.get_tensor_from_cache(tensor_key)

        # if None and origin is our client, request it from the client
        if nparray is None:
            if origin == self.collaborator_name:
                logger.info(
                    f"Attempting to find locally stored {tensor_name} tensor from prior round..."
                )
                prior_round = round_number - 1
                while prior_round >= 0:
                    nparray = self.tensor_db.get_tensor_from_cache(
                        TensorKey(tensor_name, origin, prior_round, report, tags)
                    )
                    if nparray is not None:
                        logger.debug(
                            f"Found tensor {tensor_name} in local TensorDB for round {prior_round}"
                        )
                        return nparray
                    prior_round -= 1
                logger.info(f"Cannot find any prior version of tensor {tensor_name} locally...")
            # Determine whether there are additional compression related
            # dependencies.
            # Typically, dependencies are only relevant to model layers
            tensor_dependencies = self.tensor_codec.find_dependencies(
                tensor_key, self.use_delta_updates
            )
            logger.debug(
                "Unable to get tensor from local store..."
                "attempting to retrieve from client len tensor_dependencies"
                f" tensor_key {tensor_key}"
            )
            if len(tensor_dependencies) > 0:
                # Resolve dependencies
                # tensor_dependencies[0] corresponds to the prior version
                # of the model.
                # If it exists locally, should pull the remote delta because
                # this is the least costly path
                prior_model_layer = self.tensor_db.get_tensor_from_cache(tensor_dependencies[0])
                if prior_model_layer is not None:
                    uncompressed_delta = self.get_aggregated_tensor_from_aggregator(
                        tensor_dependencies[1]
                    )
                    new_model_tk, nparray = self.tensor_codec.apply_delta(
                        tensor_dependencies[1],
                        uncompressed_delta,
                        prior_model_layer,
                        creates_model=True,
                    )
                    self.tensor_db.cache_tensor({new_model_tk: nparray})
                else:
                    logger.info(
                        "Could not find previous model layer.Fetching latest layer from aggregator"
                    )
                    # The original model tensor should be fetched from aggregator
                    nparray = self.get_aggregated_tensor_from_aggregator(
                        tensor_key, require_lossless=True
                    )
            elif "model" in tags:
                # Pulling the model for the first time
                nparray = self.get_aggregated_tensor_from_aggregator(
                    tensor_key, require_lossless=True
                )
            else:
                # we should try fetching the tensor from aggregator
                tensor_name, origin, round_number, report, tags = tensor_key
                tags = (self.collaborator_name,) + tags
                tensor_key = (tensor_name, origin, round_number, report, tags)
                logger.info(
                    "Could not find previous model layer."
                    f"Fetching latest layer from aggregator {tensor_key}"
                )
                nparray = self.get_aggregated_tensor_from_aggregator(
                    tensor_key, require_lossless=True
                )
        else:
            logger.debug("Found tensor %s in local TensorDB", tensor_key)

        return nparray

    def get_aggregated_tensor_from_aggregator(self, tensor_key, require_lossless=False):
        """
        Return the decompressed tensor associated with the requested tensor key.

        If the key requests a compressed tensor (in the tag), the tensor will
        be decompressed before returning.
        If the key specifies an uncompressed tensor (or just omits a compressed
        tag), the decompression operation will be skipped.

        Args:
            tensor_key (namedtuple): The requested tensor.
            require_lossless (bool): Should compression of the tensor be
                allowed in flight? For the initial model, it may affect
                convergence to apply lossy compression. And metrics shouldn't
                be compressed either.

        Returns:
            nparray : The decompressed tensor associated with the requested
                tensor key.
        """
        tensor_name, origin, round_number, report, tags = tensor_key

        logger.debug("Requesting aggregated tensor %s", tensor_key)
        tensor = self.client.get_aggregated_tensor(
            tensor_name,
            round_number,
            report,
            tags,
            require_lossless,
        )

        # this translates to a numpy array and includes decompression, as
        # necessary
        nparray = self.named_tensor_to_nparray(tensor)

        # cache this tensor
        self.tensor_db.cache_tensor({tensor_key: nparray})

        return nparray

    def send_task_results(self, tensor_dict, round_number, task_name) -> dict:
        """Send task results to the aggregator.

        Args:
            tensor_dict (dict): Tensor dictionary.
            round_number (int):  Actual round number.
            task_name (string): Task name.

        Returns:
            A dictionary of reportable metrics of the current collaborator for the task.
        """
        named_tensors = [self.nparray_to_named_tensor(k, v) for k, v in tensor_dict.items()]

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

        self.client.send_local_task_results(
            round_number,
            task_name,
            data_size,
            named_tensors,
        )

        return metrics

    def nparray_to_named_tensor(self, tensor_key, nparray):
        """Construct the NamedTensor Protobuf.

        Includes logic to create delta, compress tensors with the TensorCodec,
        etc.

        Args:
            tensor_key (namedtuple): Tensorkey that will be resolved locally or
                remotely. May be the product of other tensors.
            nparray: The decompressed tensor associated with the requested
                tensor key.

        Returns:
            named_tensor (protobuf) : The tensor constructed from the nparray.
        """
        # if we have an aggregated tensor, we can make a delta
        tensor_name, origin, round_number, report, tags = tensor_key
        if "trained" in tags and self.use_delta_updates:
            # Should get the pretrained model to create the delta. If training
            # has happened,
            # Model should already be stored in the TensorDB
            model_nparray = self.tensor_db.get_tensor_from_cache(
                TensorKey(tensor_name, origin, round_number, report, ("model",))
            )

            # The original model will not be present for the optimizer on the
            # first round.
            if model_nparray is not None:
                delta_tensor_key, delta_nparray = self.tensor_codec.generate_delta(
                    tensor_key, nparray, model_nparray
                )
                delta_comp_tensor_key, delta_comp_nparray, metadata = self.tensor_codec.compress(
                    delta_tensor_key, delta_nparray
                )

                named_tensor = utils.construct_named_tensor(
                    delta_comp_tensor_key,
                    delta_comp_nparray,
                    metadata,
                    lossless=False,
                )
                return named_tensor

        # Assume every other tensor requires lossless compression
        compressed_tensor_key, compressed_nparray, metadata = self.tensor_codec.compress(
            tensor_key, nparray, require_lossless=True
        )
        named_tensor = utils.construct_named_tensor(
            compressed_tensor_key, compressed_nparray, metadata, lossless=True
        )

        return named_tensor

    def named_tensor_to_nparray(self, named_tensor):
        """Convert named tensor to a numpy array.

        Args:
            named_tensor (protobuf): The tensor to convert to nparray.

        Returns:
            decompressed_nparray (nparray): The nparray converted.
        """
        # do the stuff we do now for decompression and frombuffer and stuff
        # This should probably be moved back to protoutils
        raw_bytes = named_tensor.data_bytes
        metadata = [
            {
                "int_to_float": proto.int_to_float,
                "int_list": proto.int_list,
                "bool_list": proto.bool_list,
            }
            for proto in named_tensor.transformer_metadata
        ]
        # The tensor has already been transferred to collaborator, so
        # the newly constructed tensor should have the collaborator origin
        tensor_key = TensorKey(
            named_tensor.name,
            self.collaborator_name,
            named_tensor.round_number,
            named_tensor.report,
            tuple(named_tensor.tags),
        )
        *_, tags = tensor_key
        if "compressed" in tags:
            decompressed_tensor_key, decompressed_nparray = self.tensor_codec.decompress(
                tensor_key,
                data=raw_bytes,
                transformer_metadata=metadata,
                require_lossless=True,
            )
        elif "lossy_compressed" in tags:
            decompressed_tensor_key, decompressed_nparray = self.tensor_codec.decompress(
                tensor_key, data=raw_bytes, transformer_metadata=metadata
            )
        else:
            # There could be a case where the compression pipeline is bypassed
            # entirely
            logger.warning("Bypassing tensor codec...")
            decompressed_tensor_key = tensor_key
            decompressed_nparray = raw_bytes

        self.tensor_db.cache_tensor({decompressed_tensor_key: decompressed_nparray})

        return decompressed_nparray

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
        interop_server_template = self.task_config["settings"]["interop_server"]
        interop_server_class = splitext(interop_server_template)[1].strip(".")
        interop_server_module_path = splitext(interop_server_template)[0]
        interop_server_module = importlib.import_module(interop_server_module_path)

        def receive_message_from_interop(message):
            """Receive message from interop server."""
            # Process the request and return a response
            response = self.client.send_message_to_server(message, self.collaborator_name)
            return response

        interop_server = getattr(interop_server_module, interop_server_class)(
            receive_message_from_interop
        )
        # Pass all keys in self.task_config['settings'] through to prepare_for_interop kwargs
        self.task_config["prepare_for_interop"]["kwargs"].update(
            self.task_config.get("settings", {})
        )
        self.task_config["prepare_for_interop"]["kwargs"]["interop_server"] = interop_server
