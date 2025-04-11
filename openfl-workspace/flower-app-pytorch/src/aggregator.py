# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Aggregator module."""

import logging
import queue
from threading import Lock
from typing import List, Optional

import openfl.callbacks as callbacks_module
from openfl.component.aggregator.straggler_handling import CutoffTimePolicy, StragglerPolicy
from openfl.databases import PersistentTensorDB, TensorDB
from openfl.pipelines import NoCompressionPipeline, TensorCodec
from openfl.protocols import base_pb2, utils
from openfl.utilities import TaskResultKey

from openfl.component import Aggregator

logger = logging.getLogger(__name__)

class AggregatorFlower(Aggregator):
    def __init__(
        self,
        aggregator_uuid,
        federation_uuid,
        authorized_cols,
        init_state_path,
        best_state_path,
        last_state_path,
        assigner,
        connector,
        use_delta_updates=True,
        straggler_handling_policy: StragglerPolicy = CutoffTimePolicy,
        rounds_to_train=256,
        single_col_cert_common_name=None,
        compression_pipeline=None,
        db_store_rounds=1,
        initial_tensor_dict=None,
        log_memory_usage=False,
        write_logs=False,
        callbacks: Optional[List] = [],
        persist_checkpoint=True,
        persistent_db_path=None,
        secure_aggregation=False,
    ):
        self.round_number = 0
        self.next_model_round_number = 0

        if single_col_cert_common_name:
            logger.warning(
                "You are running in single collaborator certificate mode. "
                "This mode is intended for development settings only and does not "
                "provide proper Public Key Infrastructure (PKI) security. "
                "Please use this mode with caution."
            )
        # FIXME: "" instead of None is for protobuf compatibility.
        self.single_col_cert_common_name = single_col_cert_common_name or ""

        self.straggler_handling_policy = straggler_handling_policy()

        self.rounds_to_train = rounds_to_train
        self.assigner = assigner
        if self.assigner.is_task_group_evaluation():
            self.rounds_to_train = 1
            logger.info(f"For evaluation tasks setting rounds_to_train = {self.rounds_to_train}")

        self._end_of_round_check_done = [False] * rounds_to_train
        self.stragglers = []

        # if the collaborator requests a delta, this value is set to true
        self.authorized_cols = authorized_cols
        self.uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.connector = connector

        self.quit_job_sent_to = []

        self.tensor_db = TensorDB()
        if persist_checkpoint:
            persistent_db_path = persistent_db_path or "tensor.db"
            logger.info(
                "Persistent checkpoint is enabled, setting persistent db at path %s",
                persistent_db_path,
            )
            self.persistent_db = PersistentTensorDB(persistent_db_path)
        else:
            logger.info("Persistent checkpoint is disabled")
            self.persistent_db = None
        # FIXME: I think next line generates an error on the second round
        # if it is set to 1 for the aggregator.
        self.db_store_rounds = db_store_rounds

        self.best_model_score = None
        self.metric_queue = queue.Queue()

        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()
        self.tensor_codec = TensorCodec(self.compression_pipeline)

        self.init_state_path = init_state_path
        self.best_state_path = best_state_path
        self.last_state_path = last_state_path

        # TODO: Remove. Used in deprecated interactive and native APIs
        self.best_tensor_dict: dict = {}
        self.last_tensor_dict: dict = {}
        # these enable getting all tensors for a task
        self.collaborator_tasks_results = {}  # {TaskResultKey: list of TensorKeys}
        self.collaborator_task_weight = {}  # {TaskResultKey: data_size}

        # maintain a list of collaborators that have completed task and
        # reported results in a given round
        self.collaborators_done = []
        # Initialize a lock for thread safety
        self.lock = Lock()
        self.use_delta_updates = use_delta_updates

        self.model = None  # Initialize the model attribute to None

        # Callbacks
        self.callbacks = callbacks_module.CallbackList(
            callbacks,
            add_memory_profiler=log_memory_usage,
            add_metric_writer=write_logs,
            origin="aggregator",
        )

        self.collaborator_tensor_results = {}  # {TensorKey: nparray}}

        if initial_tensor_dict:
            self._load_initial_tensors_from_dict(initial_tensor_dict)
            self.model = utils.construct_model_proto(
                tensor_dict=initial_tensor_dict,
                round_number=0,
                tensor_pipe=self.compression_pipeline,
            )
        else:
            if self.connector:
                # The model definition will be handled by the respective framework
                self.model = {}
            else:
                self.model: base_pb2.ModelProto = utils.load_proto(self.init_state_path)
                self._load_initial_tensors()  # keys are TensorKeys

        self._secure_aggregation_enabled = secure_aggregation
        if self._secure_aggregation_enabled:
            from openfl.utilities.secagg.bootstrap import SecAggSetup

            self.secagg = SecAggSetup(self.uuid, self.authorized_cols, self.tensor_db)

        if self.persistent_db and self._recover():
            logger.info("Recovered state of aggregator")

        # TODO: Aggregator has no concrete notion of round_begin.
        # https://github.com/securefederatedai/openfl/pull/1195#discussion_r1879479537
        self.callbacks.on_experiment_begin()
        self.callbacks.on_round_begin(self.round_number)

    def process_task_results(
        self,
        collaborator_name,
        round_number,
        task_name,
        data_size,
        named_tensors,
    ):
        if self._time_to_quit() or collaborator_name in self.stragglers:
            logger.warning(
                f"STRAGGLER: Collaborator {collaborator_name} is reporting results "
                f"after task {task_name} has finished."
            )
            return

        if self.round_number != round_number:
            logger.warning(
                f"Collaborator {collaborator_name} is reporting results"
                f" for the wrong round: {round_number}. Ignoring..."
            )
            return

        if self.is_connector_available():
            # Skip to end of round check
            with self.lock:
                self._is_collaborator_done(collaborator_name, round_number)
                self._end_of_round_with_stragglers_check()

        task_key = TaskResultKey(task_name, collaborator_name, round_number)

        # we mustn't have results already
        if self._collaborator_task_completed(collaborator_name, task_name, round_number):
            logger.warning(
                f"Aggregator already has task results from collaborator {collaborator_name}"
                f" for task {task_key}"
            )
            return

        # By giving task_key it's own weight, we can support different
        # training/validation weights
        # As well as eventually supporting weights that change by round
        # (if more data is added)
        self.collaborator_task_weight[task_key] = data_size

        # initialize the list of tensors that go with this task
        # Setting these incrementally is leading to missing values
        task_results = []

        for named_tensor in named_tensors:
            # quite a bit happens in here, including decompression, delta
            # handling, etc...
            tensor_key, value = self._process_named_tensor(named_tensor, collaborator_name)

            if "metric" in tensor_key.tags:
                # Caution: This schema must be followed. It is also used in
                # gRPC message streams for director/envoy.
                metrics = {
                    "round": round_number,
                    "metric_origin": collaborator_name,
                    "task_name": task_name,
                    "metric_name": tensor_key.tensor_name,
                    "metric_value": float(value),
                }
                self.metric_queue.put(metrics)

            task_results.append(tensor_key)

        self.collaborator_tasks_results[task_key] = task_results

        with self.lock:
            self._is_collaborator_done(collaborator_name, round_number)

            self._end_of_round_with_stragglers_check()

    def is_connector_available(self):
        """
        Check if the OpenFL Connector is available.

        Returns:
            bool: True if connector is available, False otherwise.
        """
        return self.connector is not None

    def start_connector(self):
        """
        Start the OpenFL Connector.

        Raises:
            RuntimeError: If OpenFL Connector has not been enabled.
        """
        if not self.is_connector_available():
            raise RuntimeError("OpenFL Connector has not been enabled.")
        return self.connector.start()

    def stop_connector(self):
        """
        Stop the OpenFL Connector.

        Raises:
            RuntimeError: If OpenFL Connector has not been enabled.
        """
        if not self.is_connector_available():
            raise RuntimeError("OpenFL Connector has not been enabled.")
        return self.connector.stop()

    def get_interop_client(self):
        """
        Get the local gRPC client for the OpenFL Connector.

        Raises:
            RuntimeError: If OpenFL Connector has not been enabled.
        """
        if not self.is_connector_available():
            raise RuntimeError("OpenFL Connector has not been enabled.")
        return self.connector.get_interop_client()

    def _end_of_round_check(self):
        """Check if the round complete.

        If so, perform many end of round operations,
        such as model aggregation, metric reporting, delta generation (+
        associated tensorkey labeling), and save the model.

        Args:
            None

        Returns:
            None
        """
        if self._end_of_round_check_done[self.round_number]:
            return

        if not self.is_connector_available():
        # Compute all validation related metrics
            logs = {}
            for task_name in self.assigner.get_all_tasks_for_round(self.round_number):
                logs.update(self._compute_validation_related_task_metrics(task_name))

            # End of round callbacks.
            self.callbacks.on_round_end(self.round_number, logs)

        # Once all of the task results have been processed
        self._end_of_round_check_done[self.round_number] = True

        # Save the latest model
        if not self.is_connector_available():
            logger.info("Saving round %s model...", self.round_number)
            self._save_model(self.round_number, self.last_state_path)

        self.round_number += 1
        # resetting stragglers for task for a new round
        self.stragglers = []
        # resetting collaborators_done for next round
        self.collaborators_done = []

        # TODO This needs to be fixed!
        if self._time_to_quit():
            logger.info("Experiment Completed. Cleaning up...")
        else:
            logger.info("Starting round %s...", self.round_number)
            # https://github.com/securefederatedai/openfl/pull/1195#discussion_r1879479537
            self.callbacks.on_round_begin(self.round_number)

        # Cleaning tensor db
        self.tensor_db.clean_up(self.db_store_rounds)
        # Reset straggler handling policy for the next round.
        self.straggler_handling_policy.reset_policy_for_round()
