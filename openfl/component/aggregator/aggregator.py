# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Aggregator module."""
import queue
import time
from logging import getLogger
from threading import Lock

from openfl.component.straggler_handling_functions import CutoffTimeBasedStragglerHandling
from openfl.databases import TensorDB
from openfl.interface.aggregation_functions import WeightedAverage
from openfl.pipelines import NoCompressionPipeline, TensorCodec
from openfl.protocols import base_pb2, utils
from openfl.utilities import TaskResultKey, TensorKey, change_tags
from openfl.utilities.logs import write_metric


class Aggregator:
    """An Aggregator is the central node in federated learning.

    Attributes:
        round_number (int): Current round number.
        single_col_cert_common_name (str): Common name for single
            collaborator certificate.
        straggler_handling_policy: Policy for handling stragglers.
        _end_of_round_check_done (list of bool): Indicates if end of round
            check is done for each round.
        stragglers (list): List of stragglers.
        rounds_to_train (int): Number of rounds to train.
        authorized_cols (list of str): IDs of enrolled collaborators.
        uuid (int): Aggregator UUID.
        federation_uuid (str): Federation UUID.
        assigner: Object assigning tasks to collaborators.
        quit_job_sent_to (list): Collaborators sent a quit job.
        tensor_db (TensorDB): Object for tensor database.
        db_store_rounds* (int): Rounds to store in TensorDB.
        logger: Object for logging.
        write_logs (bool): Flag to enable log writing.
        log_metric_callback: Callback for logging metrics.
        best_model_score (optional): Score of the best model. Defaults to
            None.
        metric_queue (queue.Queue): Queue for metrics.
        compression_pipeline: Pipeline for compressing data.
        tensor_codec (TensorCodec): Codec for tensor compression.
        init_state_path* (str): Initial weight file location.
        best_state_path* (str): Where to store the best model weight.
        last_state_path* (str): Where to store the latest model weight.
        best_tensor_dict (dict): Dict of the best tensors.
        last_tensor_dict (dict): Dict of the last tensors.
        collaborator_tensor_results (dict): Dict of collaborator tensor
            results.
        collaborator_tasks_results (dict): Dict of collaborator tasks
            results.
        collaborator_task_weight (dict): Dict of col task weight.
        lock: A threading Lock object used to ensure thread-safe operations.

    .. note::
        - plan setting
    """

    def __init__(
        self,
        aggregator_uuid,
        federation_uuid,
        authorized_cols,
        init_state_path,
        best_state_path,
        last_state_path,
        assigner,
        straggler_handling_policy=None,
        rounds_to_train=256,
        single_col_cert_common_name=None,
        compression_pipeline=None,
        db_store_rounds=1,
        write_logs=False,
        log_metric_callback=None,
        **kwargs,
    ):
        """Initializes the Aggregator.

        Args:
            aggregator_uuid (int): Aggregation ID.
            federation_uuid (str): Federation ID.
            authorized_cols (list of str): The list of IDs of enrolled
                collaborators.
            init_state_path (str): The location of the initial weight file.
            best_state_path (str): The file location to store the weight of
                the best model.
            last_state_path (str): The file location to store the latest
                weight.
            assigner: Assigner object.
            straggler_handling_policy (optional): Straggler handling policy.
                Defaults to CutoffTimeBasedStragglerHandling.
            rounds_to_train (int, optional): Number of rounds to train.
                Defaults to 256.
            single_col_cert_common_name (str, optional): Common name for single
                collaborator certificate. Defaults to None.
            compression_pipeline (optional): Compression pipeline. Defaults to
                NoCompressionPipeline.
            db_store_rounds (int, optional): Rounds to store in TensorDB.
                Defaults to 1.
            write_logs (bool, optional): Whether to write logs. Defaults to
                False.
            log_metric_callback (optional): Callback for log metric. Defaults
                to None.
            **kwargs: Additional keyword arguments.
        """
        self.round_number = 0
        self.single_col_cert_common_name = single_col_cert_common_name

        if self.single_col_cert_common_name is not None:
            self._log_big_warning()
        else:
            # FIXME: '' instead of None is just for protobuf compatibility.
            # Cleaner solution?
            self.single_col_cert_common_name = ""

        self.straggler_handling_policy = (
            straggler_handling_policy or CutoffTimeBasedStragglerHandling()
        )
        self._end_of_round_check_done = [False] * rounds_to_train
        self.stragglers = []

        self.rounds_to_train = rounds_to_train

        # if the collaborator requests a delta, this value is set to true
        self.authorized_cols = authorized_cols
        self.uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.assigner = assigner
        self.quit_job_sent_to = []

        self.tensor_db = TensorDB()
        # FIXME: I think next line generates an error on the second round
        # if it is set to 1 for the aggregator.
        self.db_store_rounds = db_store_rounds

        # Gathered together logging-related objects
        self.logger = getLogger(__name__)
        self.write_logs = write_logs
        self.log_metric_callback = log_metric_callback

        if self.write_logs:
            self.log_metric = write_metric
            if self.log_metric_callback:
                self.log_metric = log_metric_callback
                self.logger.info("Using custom log metric: %s", self.log_metric)

        self.best_model_score = None
        self.metric_queue = queue.Queue()

        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()
        self.tensor_codec = TensorCodec(self.compression_pipeline)

        self.init_state_path = init_state_path
        self.best_state_path = best_state_path
        self.last_state_path = last_state_path

        self.best_tensor_dict: dict = {}
        self.last_tensor_dict: dict = {}

        if kwargs.get("initial_tensor_dict", None) is not None:
            self._load_initial_tensors_from_dict(kwargs["initial_tensor_dict"])
            self.model = utils.construct_model_proto(
                tensor_dict=kwargs["initial_tensor_dict"],
                round_number=0,
                tensor_pipe=self.compression_pipeline,
            )
        else:
            self.model: base_pb2.ModelProto = utils.load_proto(self.init_state_path)
            self._load_initial_tensors()  # keys are TensorKeys

        self.collaborator_tensor_results = {}  # {TensorKey: nparray}}

        # these enable getting all tensors for a task
        self.collaborator_tasks_results = {}  # {TaskResultKey: list of TensorKeys}

        self.collaborator_task_weight = {}  # {TaskResultKey: data_size}

        # maintain a list of collaborators that have completed task and
        # reported results in a given round
        self.collaborators_done = []

        # Initialize a lock for thread safety
        self.lock = Lock()

    def _load_initial_tensors(self):
        """Load all of the tensors required to begin federated learning.

        Required tensors are: \
            1. Initial model.

        Returns:
            None
        """
        tensor_dict, round_number = utils.deconstruct_model_proto(
            self.model, compression_pipeline=self.compression_pipeline
        )

        if round_number > self.round_number:
            self.logger.info(
                f"Starting training from round {round_number} of previously saved model"
            )
            self.round_number = round_number
        tensor_key_dict = {
            TensorKey(k, self.uuid, self.round_number, False, ("model",)): v
            for k, v in tensor_dict.items()
        }
        # all initial model tensors are loaded here
        self.tensor_db.cache_tensor(tensor_key_dict)
        self.logger.debug("This is the initial tensor_db: %s", self.tensor_db)

    def _load_initial_tensors_from_dict(self, tensor_dict):
        """Load all of the tensors required to begin federated learning.

        Required tensors are: \
            1. Initial model.

        Returns:
            None
        """
        tensor_key_dict = {
            TensorKey(k, self.uuid, self.round_number, False, ("model",)): v
            for k, v in tensor_dict.items()
        }
        # all initial model tensors are loaded here
        self.tensor_db.cache_tensor(tensor_key_dict)
        self.logger.debug("This is the initial tensor_db: %s", self.tensor_db)

    def _save_model(self, round_number, file_path):
        """Save the best or latest model.

        Args:
            round_number (int): Model round to be saved.
            file_path (str): Either the best model or latest model file path.

        Returns:
            None
        """
        # Extract the model from TensorDB and set it to the new model
        og_tensor_dict, _ = utils.deconstruct_model_proto(
            self.model, compression_pipeline=self.compression_pipeline
        )
        tensor_keys = [
            TensorKey(k, self.uuid, round_number, False, ("model",))
            for k, v in og_tensor_dict.items()
        ]
        tensor_dict = {}
        for tk in tensor_keys:
            tk_name, _, _, _, _ = tk
            tensor_dict[tk_name] = self.tensor_db.get_tensor_from_cache(tk)
            if tensor_dict[tk_name] is None:
                self.logger.info(
                    "Cannot save model for round %s. Continuing...",
                    round_number,
                )
                return
        if file_path == self.best_state_path:
            self.best_tensor_dict = tensor_dict
        if file_path == self.last_state_path:
            self.last_tensor_dict = tensor_dict
        self.model = utils.construct_model_proto(
            tensor_dict, round_number, self.compression_pipeline
        )
        utils.dump_proto(self.model, file_path)

    def valid_collaborator_cn_and_id(self, cert_common_name, collaborator_common_name):
        """
        Determine if the collaborator certificate and ID are valid for this federation.

        Args:
            cert_common_name (str): Common name for security certificate.
            collaborator_common_name (str): Common name for collaborator.

        Returns:
            bool: True means the collaborator common name matches the name in
                the security certificate.
        """
        # if self.test_mode_whitelist is None, then the common_name must
        # match collaborator_common_name and be in authorized_cols
        # FIXME: '' instead of None is just for protobuf compatibility.
        #  Cleaner solution?
        if self.single_col_cert_common_name == "":
            return (
                cert_common_name == collaborator_common_name
                and collaborator_common_name in self.authorized_cols
            )
        # otherwise, common_name must be in whitelist and
        # collaborator_common_name must be in authorized_cols
        else:
            return (
                cert_common_name == self.single_col_cert_common_name
                and collaborator_common_name in self.authorized_cols
            )

    def all_quit_jobs_sent(self):
        """Assert all quit jobs are sent to collaborators.

        Returns:
            bool: True if all quit jobs are sent, False otherwise.
        """
        return set(self.quit_job_sent_to) == set(self.authorized_cols)

    @staticmethod
    def _get_sleep_time():
        """Sleep 10 seconds.

        Returns:
            int: Sleep time.
        """
        # Decrease sleep period for finer discretezation
        return 10

    def _time_to_quit(self):
        """If all rounds are complete, it's time to quit.

        Returns:
            bool: True if it's time to quit, False otherwise.
        """
        if self.round_number >= self.rounds_to_train:
            return True
        return False

    def get_tasks(self, collaborator_name):
        """RPC called by a collaborator to determine which tasks to perform.

        Args:
            collaborator_name (str): Requested collaborator name.

        Returns:
            tasks (list[str]): List of tasks to be performed by the requesting
                collaborator for the current round.
            round_number (int): Actual round number.
            sleep_time (int): Sleep time.
            time_to_quit (bool): Whether it's time to quit.
        """
        self.logger.debug(
            f"Aggregator GetTasks function reached from collaborator {collaborator_name}..."
        )

        # first, if it is time to quit, inform the collaborator
        if self._time_to_quit():
            self.logger.info(
                "Sending signal to collaborator %s to shutdown...",
                collaborator_name,
            )
            self.quit_job_sent_to.append(collaborator_name)

            tasks = None
            sleep_time = 0
            time_to_quit = True

            return tasks, self.round_number, sleep_time, time_to_quit

        time_to_quit = False
        # otherwise, get the tasks from our task assigner
        tasks = self.assigner.get_tasks_for_collaborator(collaborator_name, self.round_number)

        # if no tasks, tell the collaborator to sleep
        if len(tasks) == 0:
            tasks = None
            sleep_time = self._get_sleep_time()

            return tasks, self.round_number, sleep_time, time_to_quit

        # if we do have tasks, remove any that we already have results for
        if isinstance(tasks[0], str):
            # backward compatibility
            tasks = [
                t
                for t in tasks
                if not self._collaborator_task_completed(collaborator_name, t, self.round_number)
            ]
            if collaborator_name in self.stragglers:
                tasks = []

        else:
            tasks = [
                t
                for t in tasks
                if not self._collaborator_task_completed(
                    collaborator_name, t.name, self.round_number
                )
            ]
            if collaborator_name in self.stragglers:
                tasks = []

        # Do the check again because it's possible that all tasks have
        # been completed
        if len(tasks) == 0:
            tasks = None
            sleep_time = self._get_sleep_time()

            return tasks, self.round_number, sleep_time, time_to_quit

        self.logger.info(
            f"Sending tasks to collaborator {collaborator_name} for round {self.round_number}"
        )
        sleep_time = 0

        # Start straggler handling policy for timer based callback is required
        # for %age based policy callback is not required
        self.straggler_handling_policy.start_policy(callback=self._straggler_cutoff_time_elapsed)

        return tasks, self.round_number, sleep_time, time_to_quit

    def _straggler_cutoff_time_elapsed(self) -> None:
        """
        This method is called by the straggler handling policy when cutoff timer is elapsed.
        It applies straggler handling policy and ends the round early.

        Returns:
            None
        """
        self.logger.warning(
            f"Round number: {self.round_number} cutoff timer elapsed after "
            f"{self.straggler_handling_policy.straggler_cutoff_time}s. "
            f"Applying {self.straggler_handling_policy.__class__.__name__} policy."
        )

        with self.lock:
            # Check if minimum collaborators reported results
            self._end_of_round_with_stragglers_check()

    def get_aggregated_tensor(
        self,
        collaborator_name,
        tensor_name,
        round_number,
        report,
        tags,
        require_lossless,
    ):
        """
        RPC called by collaborator.

        Performs local lookup to determine if there is an aggregated tensor available
        that matches the request.

        Args:
            collaborator_name (str): Requested tensor key collaborator name.
            tensor_name (str): Name of the tensor.
            round_number (int): Actual round number.
            report (bool): Whether to report.
            tags (tuple[str, ...]): Tags.
            require_lossless (bool): Whether to require lossless.

        Returns:
            named_tensor (protobuf) :  NamedTensor, the tensor requested by the collaborator.

        Raises:
            ValueError: if Aggregator does not have an aggregated tensor for {tensor_key}.
        """
        self.logger.debug(
            f"Retrieving aggregated tensor {tensor_name},{round_number},{tags} "
            f"for collaborator {collaborator_name}"
        )

        if "compressed" in tags or require_lossless:
            compress_lossless = True
        else:
            compress_lossless = False

        # TODO the TensorDB doesn't support compressed data yet.
        #  The returned tensor will
        # be recompressed anyway.
        if "compressed" in tags:
            tags = change_tags(tags, remove_field="compressed")
        if "lossy_compressed" in tags:
            tags = change_tags(tags, remove_field="lossy_compressed")

        tensor_key = TensorKey(tensor_name, self.uuid, round_number, report, tags)
        tensor_name, origin, round_number, report, tags = tensor_key

        if "aggregated" in tags and "delta" in tags and round_number != 0:
            agg_tensor_key = TensorKey(tensor_name, origin, round_number, report, ("aggregated",))
        else:
            agg_tensor_key = tensor_key

        nparray = self.tensor_db.get_tensor_from_cache(agg_tensor_key)

        start_retrieving_time = time.time()
        while nparray is None:
            self.logger.debug("Waiting for tensor_key %s", agg_tensor_key)
            time.sleep(5)
            nparray = self.tensor_db.get_tensor_from_cache(agg_tensor_key)
            if (time.time() - start_retrieving_time) > 60:
                break

        if nparray is None:
            raise ValueError(f"Aggregator does not have an aggregated tensor for {tensor_key}")

        # quite a bit happens in here, including compression, delta handling,
        # etc...
        # we might want to cache these as well
        named_tensor = self._nparray_to_named_tensor(
            agg_tensor_key, nparray, send_model_deltas=True, compress_lossless=compress_lossless
        )

        return named_tensor

    def _nparray_to_named_tensor(self, tensor_key, nparray, send_model_deltas, compress_lossless):
        """Construct the NamedTensor Protobuf.

        Also includes logic to create delta, compress tensors with the
            TensorCodec, etc.

        Args:
            tensor_key (TensorKey): Tensor key.
            nparray (np.array): Numpy array.
            send_model_deltas (bool): Whether to send model deltas.
            compress_lossless (bool): Whether to compress lossless.

        Returns:
            tensor_key (TensorKey): Tensor key.
            nparray (np.array): Numpy array.

        """
        tensor_name, origin, round_number, report, tags = tensor_key
        # if we have an aggregated tensor, we can make a delta
        if "aggregated" in tags and send_model_deltas:
            # Should get the pretrained model to create the delta. If training
            # has happened, Model should already be stored in the TensorDB
            model_tk = TensorKey(tensor_name, origin, round_number - 1, report, ("model",))

            model_nparray = self.tensor_db.get_tensor_from_cache(model_tk)

            assert model_nparray is not None, (
                "The original model layer should be present if the latest "
                "aggregated model is present"
            )
            delta_tensor_key, delta_nparray = self.tensor_codec.generate_delta(
                tensor_key, nparray, model_nparray
            )
            delta_comp_tensor_key, delta_comp_nparray, metadata = self.tensor_codec.compress(
                delta_tensor_key, delta_nparray, lossless=compress_lossless
            )
            named_tensor = utils.construct_named_tensor(
                delta_comp_tensor_key,
                delta_comp_nparray,
                metadata,
                lossless=compress_lossless,
            )

        else:
            # Assume every other tensor requires lossless compression
            compressed_tensor_key, compressed_nparray, metadata = self.tensor_codec.compress(
                tensor_key, nparray, require_lossless=True
            )
            named_tensor = utils.construct_named_tensor(
                compressed_tensor_key,
                compressed_nparray,
                metadata,
                lossless=compress_lossless,
            )

        return named_tensor

    def _collaborator_task_completed(self, collaborator, task_name, round_num):
        """Check if the collaborator has completed the task for the round.

         The aggregator doesn't actually know which tensors should be sent from
         the collaborator so it must to rely specifically on the presence of
         previous results.

        Args:
         collaborator (str): Collaborator to check if their task has been
             completed.
         task_name (str): The name of the task (TaskRunner function).
         round_num (int): Round number.

         Returns:
             bool: Whether or not the collaborator has completed the task for
                 this round.
        """
        task_key = TaskResultKey(task_name, collaborator, round_num)
        return task_key in self.collaborator_tasks_results

    def send_local_task_results(
        self,
        collaborator_name,
        round_number,
        task_name,
        data_size,
        named_tensors,
    ):
        """
        RPC called by collaborator.

        Transmits collaborator's task results to the aggregator.

        Args:
            collaborator_name (str): Collaborator name.
            round_number (int): Round number.
            task_name (str): Task name.
            data_size (int): Data size.
            named_tensors (protobuf NamedTensor): Named tensors.

        Returns:
            None
        """
        if self._time_to_quit() or collaborator_name in self.stragglers:
            self.logger.warning(
                f"STRAGGLER: Collaborator {collaborator_name} is reporting results "
                f"after task {task_name} has finished."
            )
            return

        if self.round_number != round_number:
            self.logger.warning(
                f"Collaborator {collaborator_name} is reporting results"
                f" for the wrong round: {round_number}. Ignoring..."
            )
            return

        self.logger.info(
            f"Collaborator {collaborator_name} is sending task results "
            f"for {task_name}, round {round_number}"
        )

        task_key = TaskResultKey(task_name, collaborator_name, round_number)

        # we mustn't have results already
        if self._collaborator_task_completed(collaborator_name, task_name, round_number):
            self.logger.warning(
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
                self.logger.metric("%s", str(metrics))

            task_results.append(tensor_key)

        self.collaborator_tasks_results[task_key] = task_results

        with self.lock:
            self._is_collaborator_done(collaborator_name, round_number)

            self._end_of_round_with_stragglers_check()

    def _end_of_round_with_stragglers_check(self):
        """
        Checks if the minimum required collaborators have reported their results,
        identifies any stragglers, and initiates an early round end if necessary.

        Returns:
            None
        """
        if self.straggler_handling_policy.straggler_cutoff_check(
            len(self.collaborators_done), len(self.authorized_cols)
        ):
            self.stragglers = [
                collab_name
                for collab_name in self.authorized_cols
                if collab_name not in self.collaborators_done
            ]
            if len(self.stragglers) != 0:
                self.logger.warning(f"Identified stragglers: {self.stragglers}")
            self._end_of_round_check()

    def _process_named_tensor(self, named_tensor, collaborator_name):
        """Extract the named tensor fields.

        Performs decompression, delta computation, and inserts results into
        TensorDB.

        Args:
            named_tensor (protobuf NamedTensor): Named tensor.
                protobuf that will be extracted from and processed
            collaborator_name (str): Collaborator name.
                Collaborator name is needed for proper tagging of resulting
                tensorkeys.

        Returns:
            tensor_key (TensorKey): Tensor key.
                The tensorkey extracted from the protobuf.
            nparray (np.array): Numpy array.
                The numpy array associated with the returned tensorkey.
        """
        raw_bytes = named_tensor.data_bytes
        metadata = [
            {
                "int_to_float": proto.int_to_float,
                "int_list": proto.int_list,
                "bool_list": proto.bool_list,
            }
            for proto in named_tensor.transformer_metadata
        ]
        # The tensor has already been transfered to aggregator,
        # so the newly constructed tensor should have the aggregator origin
        tensor_key = TensorKey(
            named_tensor.name,
            self.uuid,
            named_tensor.round_number,
            named_tensor.report,
            tuple(named_tensor.tags),
        )
        tensor_name, origin, round_number, report, tags = tensor_key
        assert (
            "compressed" in tags or "lossy_compressed" in tags
        ), f"Named tensor {tensor_key} is not compressed"
        if "compressed" in tags:
            dec_tk, decompressed_nparray = self.tensor_codec.decompress(
                tensor_key,
                data=raw_bytes,
                transformer_metadata=metadata,
                require_lossless=True,
            )
            dec_name, dec_origin, dec_round_num, dec_report, dec_tags = dec_tk
            # Need to add the collaborator tag to the resulting tensor
            new_tags = change_tags(dec_tags, add_field=collaborator_name)

            # layer.agg.n.trained.delta.col_i
            decompressed_tensor_key = TensorKey(
                dec_name, dec_origin, dec_round_num, dec_report, new_tags
            )
        if "lossy_compressed" in tags:
            dec_tk, decompressed_nparray = self.tensor_codec.decompress(
                tensor_key,
                data=raw_bytes,
                transformer_metadata=metadata,
                require_lossless=False,
            )
            dec_name, dec_origin, dec_round_num, dec_report, dec_tags = dec_tk
            new_tags = change_tags(dec_tags, add_field=collaborator_name)
            # layer.agg.n.trained.delta.lossy_decompressed.col_i
            decompressed_tensor_key = TensorKey(
                dec_name, dec_origin, dec_round_num, dec_report, new_tags
            )

        if "delta" in tags:
            base_model_tensor_key = TensorKey(tensor_name, origin, round_number, report, ("model",))
            base_model_nparray = self.tensor_db.get_tensor_from_cache(base_model_tensor_key)
            if base_model_nparray is None:
                raise ValueError(f"Base model {base_model_tensor_key} not present in TensorDB")
            final_tensor_key, final_nparray = self.tensor_codec.apply_delta(
                decompressed_tensor_key,
                decompressed_nparray,
                base_model_nparray,
            )
        else:
            final_tensor_key = decompressed_tensor_key
            final_nparray = decompressed_nparray

        assert final_nparray is not None, f"Could not create tensorkey {final_tensor_key}"
        self.tensor_db.cache_tensor({final_tensor_key: final_nparray})
        self.logger.debug("Created TensorKey: %s", final_tensor_key)

        return final_tensor_key, final_nparray

    def _prepare_trained(self, tensor_name, origin, round_number, report, agg_results):
        """Prepare aggregated tensorkey tags.

        Args:
            tensor_name (str): Tensor name.
            origin: Origin.
            round_number (int): Round number.
            report (bool): Whether to report.
            agg_results (np.array): Aggregated results.
        """
        # The aggregated tensorkey tags should have the form of
        # 'trained' or 'trained.lossy_decompressed'
        # They need to be relabeled to 'aggregated' and
        # reinserted. Then delta performed, compressed, etc.
        # then reinserted to TensorDB with 'model' tag

        # First insert the aggregated model layer with the
        # correct tensorkey
        agg_tag_tk = TensorKey(tensor_name, origin, round_number + 1, report, ("aggregated",))
        self.tensor_db.cache_tensor({agg_tag_tk: agg_results})

        # Create delta and save it in TensorDB
        base_model_tk = TensorKey(tensor_name, origin, round_number, report, ("model",))
        base_model_nparray = self.tensor_db.get_tensor_from_cache(base_model_tk)
        if base_model_nparray is not None:
            delta_tk, delta_nparray = self.tensor_codec.generate_delta(
                agg_tag_tk, agg_results, base_model_nparray
            )
        else:
            # This condition is possible for base model
            # optimizer states (i.e. Adam/iter:0, SGD, etc.)
            # These values couldn't be present for the base
            # model because no training occurs on the aggregator
            delta_tk, delta_nparray = agg_tag_tk, agg_results

        # Compress lossless/lossy
        compressed_delta_tk, compressed_delta_nparray, metadata = self.tensor_codec.compress(
            delta_tk, delta_nparray
        )

        # TODO extend the TensorDB so that compressed data is
        #  supported. Once that is in place
        # the compressed delta can just be stored here instead
        # of recreating it for every request

        # Decompress lossless/lossy
        decompressed_delta_tk, decompressed_delta_nparray = self.tensor_codec.decompress(
            compressed_delta_tk, compressed_delta_nparray, metadata
        )

        self.tensor_db.cache_tensor({decompressed_delta_tk: decompressed_delta_nparray})

        # Apply delta (unless delta couldn't be created)
        if base_model_nparray is not None:
            self.logger.debug("Applying delta for layer %s", decompressed_delta_tk[0])
            new_model_tk, new_model_nparray = self.tensor_codec.apply_delta(
                decompressed_delta_tk,
                decompressed_delta_nparray,
                base_model_nparray,
            )
        else:
            new_model_tk, new_model_nparray = (
                decompressed_delta_tk,
                decompressed_delta_nparray,
            )

        # Now that the model has been compressed/decompressed
        # with delta operations,
        # Relabel the tags to 'model'
        (
            new_model_tensor_name,
            new_model_origin,
            new_model_round_number,
            new_model_report,
            new_model_tags,
        ) = new_model_tk
        final_model_tk = TensorKey(
            new_model_tensor_name,
            new_model_origin,
            new_model_round_number,
            new_model_report,
            ("model",),
        )

        # Finally, cache the updated model tensor
        self.tensor_db.cache_tensor({final_model_tk: new_model_nparray})

    def _compute_validation_related_task_metrics(self, task_name):
        """Compute all validation related metrics.

        Args:
            task_name (str): Task name.
        """
        # By default, print out all of the metrics that the validation
        # task sent
        # This handles getting the subset of collaborators that may be
        # part of the validation task
        all_collaborators_for_task = self.assigner.get_collaborators_for_task(
            task_name, self.round_number
        )
        # Leave out straggler for the round even if they've paritally
        # completed given tasks
        collaborators_for_task = []
        collaborators_for_task = [
            c for c in all_collaborators_for_task if c in self.collaborators_done
        ]

        # The collaborator data sizes for that task
        collaborator_weights_unnormalized = {
            c: self.collaborator_task_weight[TaskResultKey(task_name, c, self.round_number)]
            for c in collaborators_for_task
        }
        weight_total = sum(collaborator_weights_unnormalized.values())
        collaborator_weight_dict = {
            k: v / weight_total for k, v in collaborator_weights_unnormalized.items()
        }

        # The validation task should have just a couple tensors (i.e.
        # metrics) associated with it. Because each collaborator should
        # have sent the same tensor list, we can use the first
        # collaborator in our subset, and apply the correct
        # transformations to the tensorkey to resolve the aggregated
        # tensor for that round
        task_agg_function = self.assigner.get_aggregation_type_for_task(task_name)
        task_key = TaskResultKey(task_name, collaborators_for_task[0], self.round_number)

        for tensor_key in self.collaborator_tasks_results[task_key]:
            tensor_name, origin, round_number, report, tags = tensor_key
            assert (
                collaborators_for_task[0] in tags
            ), f"Tensor {tensor_key} in task {task_name} has not been processed correctly"
            # Strip the collaborator label, and lookup aggregated tensor
            new_tags = change_tags(tags, remove_field=collaborators_for_task[0])
            agg_tensor_key = TensorKey(tensor_name, origin, round_number, report, new_tags)
            agg_function = WeightedAverage() if "metric" in tags else task_agg_function
            agg_results = self.tensor_db.get_aggregated_tensor(
                agg_tensor_key,
                collaborator_weight_dict,
                aggregation_function=agg_function,
            )

            if report:
                # Caution: This schema must be followed. It is also used in
                # gRPC message streams for director/envoy.
                metrics = {
                    "metric_origin": "aggregator",
                    "task_name": task_name,
                    "metric_name": tensor_key.tensor_name,
                    "metric_value": float(agg_results),
                    "round": round_number,
                }

                self.metric_queue.put(metrics)
                self.logger.metric("%s", metrics)

                # FIXME: Configurable logic for min/max criteria in saving best.
                if "validate_agg" in tags:
                    # Compare the accuracy of the model, potentially save it
                    if self.best_model_score is None or self.best_model_score < agg_results:
                        self.logger.metric(
                            f"Round {round_number}: saved the best "
                            f"model with score {agg_results:f}"
                        )
                        self.best_model_score = agg_results
                        self._save_model(round_number, self.best_state_path)
            if "trained" in tags:
                self._prepare_trained(tensor_name, origin, round_number, report, agg_results)

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

        # Compute all validation related metrics
        all_tasks = self.assigner.get_all_tasks_for_round(self.round_number)
        for task_name in all_tasks:
            self._compute_validation_related_task_metrics(task_name)

        # Once all of the task results have been processed
        self._end_of_round_check_done[self.round_number] = True
        self.round_number += 1
        # resetting stragglers for task for a new round
        self.stragglers = []
        # resetting collaborators_done for next round
        self.collaborators_done = []

        # Save the latest model
        self.logger.info("Saving round %s model...", self.round_number)
        self._save_model(self.round_number, self.last_state_path)

        # TODO This needs to be fixed!
        if self._time_to_quit():
            self.logger.info("Experiment Completed. Cleaning up...")
        else:
            self.logger.info("Starting round %s...", self.round_number)

        # Cleaning tensor db
        self.tensor_db.clean_up(self.db_store_rounds)
        # Reset straggler handling policy for the next round.
        self.straggler_handling_policy.reset_policy_for_round()

    def _is_collaborator_done(self, collaborator_name: str, round_number: int) -> None:
        """
        Check if all tasks given to the collaborator are completed then,
        completed or not.

        Args:
            collaborator_name (str): Collaborator name.
            round_number (int): Round number.

        Returns:
            None
        """
        if self.round_number != round_number:
            self.logger.warning(
                f"Collaborator {collaborator_name} is reporting results"
                f" for the wrong round: {round_number}. Ignoring..."
            )
            return

        # Get all tasks given to the collaborator for current round
        all_tasks = self.assigner.get_tasks_for_collaborator(collaborator_name, self.round_number)
        # Check if all given tasks are completed by the collaborator
        all_tasks_completed = True
        for task in all_tasks:
            if hasattr(task, "name"):
                task = task.name
            all_tasks_completed = all_tasks_completed and self._collaborator_task_completed(
                collaborator=collaborator_name, task_name=task, round_num=self.round_number
            )
        # If the collaborator has completed ALL tasks for current round,
        # update collaborators_done
        if all_tasks_completed:
            self.collaborators_done.append(collaborator_name)
            self.logger.info(
                f"Round: {self.round_number}, Collaborators that have completed all tasks: "
                f"{self.collaborators_done}"
            )

    def _log_big_warning(self):
        """Warn user about single collaborator cert mode."""
        self.logger.warning(
            f"\n{the_dragon}\nYOU ARE RUNNING IN SINGLE COLLABORATOR CERT MODE! THIS IS"
            f" NOT PROPER PKI AND "
            f"SHOULD ONLY BE USED IN DEVELOPMENT SETTINGS!!!! YE HAVE BEEN"
            f" WARNED!!!"
        )

    def stop(self, failed_collaborator: str = None) -> None:
        """Stop aggregator execution.

        Args:
            failed_collaborator (str, optional): Failed collaborator. Defaults to None.

        Returns:
            None
        """
        self.logger.info("Force stopping the aggregator execution.")
        # We imitate quit_job_sent_to the failed collaborator
        # So the experiment set to a finished state
        if failed_collaborator:
            self.quit_job_sent_to.append(failed_collaborator)

        # This code does not actually send `quit` tasks to collaborators,
        # it just mimics it by filling arrays.
        for collaborator_name in filter(lambda c: c != failed_collaborator, self.authorized_cols):
            self.logger.info(
                "Sending signal to collaborator %s to shutdown...",
                collaborator_name,
            )
            self.quit_job_sent_to.append(collaborator_name)


the_dragon = """

 ,@@.@@+@@##@,@@@@.`@@#@+  *@@@@ #@##@  `@@#@# @@@@@   @@    @@@@` #@@@ :@@ `@#`@@@#.@
  @@ #@ ,@ +. @@.@* #@ :`   @+*@ .@`+.   @@ *@::@`@@   @@#  @@  #`;@`.@@ @@@`@`#@* +:@`
  @@@@@ ,@@@  @@@@  +@@+    @@@@ .@@@    @@ .@+:@@@:  .;+@` @@ ,;,#@` @@ @@@@@ ,@@@* @
  @@ #@ ,@`*. @@.@@ #@ ,;  `@+,@#.@.*`   @@ ,@::@`@@` @@@@# @@`:@;*@+ @@ @`:@@`@ *@@ `
 .@@`@@,+@+;@.@@ @@`@@;*@  ;@@#@:*@+;@  `@@;@@ #@**@+;@ `@@:`@@@@  @@@@.`@+ .@ +@+@*,@
  `` ``     ` ``  .     `     `      `     `    `  .` `  ``   ``    ``   `       .   `



                                            .**
                                      ;`  `****:
                                     @**`*******
                         ***        +***********;
                        ,@***;` .*:,;************
                        ;***********@@***********
                        ;************************,
                        `*************************
                         *************************
                         ,************************
                          **#*********************
                          *@****`     :**********;
                          +**;          .********.
                          ;*;            `*******#:                       `,:
                                          ****@@@++::                ,,;***.
                                          *@@@**;#;:         +:      **++*,
                                          @***#@@@:          +*;     ,****
                                          @*@+****           ***`     ****,
                                         ,@#******.  ,       ****     **;,**.
                                         * ******** :,       ;*:*+    **  :,**
                                        #  ********::      *,.*:**`   *      ,*;
                                        .  *********:      .+,*:;*:   :      `:**
                                       ;   :********:       ***::**   `       ` **
                                       +   :****::***  ,    *;;::**`             :*
                                      ``   .****::;**:::    *;::::*;              ;*
                                      *     *****::***:.    **::::**               ;:
                                      #     *****;:****     ;*::;***               ,*`
                                      ;     ************`  ,**:****;               ::*
                                      :     *************;:;*;*++:                   *.
                                      :     *****************;*                      `*
                                     `.    `*****************;  :                     *.
                                     .`    .*+************+****;:                     :*
                                     `.    :;+***********+******;`    :              .,*
                                      ;    ::*+*******************. `::              .`:.
                                      +    :::**********************;;:`                *
                                      +    ,::;*************;:::*******.                *
                                      #    `:::+*************:::;********  :,           *
                                      @     :::***************;:;*********;:,           *
                                      @     ::::******:*********************:         ,:*
                                      @     .:::******:;*********************,         :*
                                      #      :::******::******###@*******;;****        *,
                                      #      .::;*****::*****#****@*****;:::***;  ``  **
                                      *       ::;***********+*****+#******::*****,,,,**
                                      :        :;***********#******#******************
                                      .`       `;***********#******+****+************
                                      `,        ***#**@**+***+*****+**************;`
                                       ;         *++**#******#+****+`      `.,..
                                       +         `@***#*******#****#
                                       +          +***@********+**+:
                                       *         .+**+;**;;;**;#**#
                                      ,`         ****@         +*+:
                                      #          +**+         :+**
                                      @         ;**+,       ,***+
                                      #      #@+****      *#****+
                                     `;     @+***+@      `#**+#++
                                     #      #*#@##,      .++:.,#
                                    `*      @#            +.
                                  @@@
                                 # `@
                                  ,                                                        """
