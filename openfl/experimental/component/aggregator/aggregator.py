# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Experimental Aggregator module."""
import inspect
import pickle
import queue
import time
from logging import getLogger
from threading import Event
from typing import Any, Callable, Dict, List, Tuple

from openfl.experimental.interface import FLSpec
from openfl.experimental.runtime import FederatedRuntime
from openfl.experimental.utilities import aggregator_to_collaborator, checkpoint
from openfl.experimental.utilities.metaflow_utils import MetaflowInterface


class Aggregator:
    r"""An Aggregator is the central node in federated learning.

    Args:
        aggregator_uuid (str): Aggregation ID.
        federation_uuid (str): Federation ID.
        authorized_cols (list of str): The list of IDs of enrolled
            collaborators.
        flow (Any): Flow class.
        rounds_to_train (int): External loop rounds.
        checkpoint (bool): Whether to save checkpoint or noe (default=False).
        private_attrs_callable (Callable): Function for Aggregator private
            attriubtes
        (default=None).
        private_attrs_kwargs (Dict): Arguments to call private_attrs_callable
            (default={}).

    Returns:
        None
    """

    def __init__(
        self,
        aggregator_uuid: str,
        federation_uuid: str,
        authorized_cols: List,
        flow: Any,
        rounds_to_train: int = 1,
        checkpoint: bool = False,
        private_attributes_callable: Callable = None,
        private_attributes_kwargs: Dict = {},
        private_attributes: Dict = {},
        single_col_cert_common_name: str = None,
        log_metric_callback: Callable = None,
        **kwargs,
    ) -> None:

        self.logger = getLogger(__name__)

        self.single_col_cert_common_name = single_col_cert_common_name
        if self.single_col_cert_common_name is not None:
            self._log_big_warning()
        else:
            # FIXME: "" instead of None is just for protobuf compatibility.
            # Cleaner solution?
            self.single_col_cert_common_name = ""

        self.log_metric_callback = log_metric_callback
        if log_metric_callback is not None:
            self.log_metric = log_metric_callback
            self.logger.info(f"Using custom log metric: {self.log_metric}")

        self.uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.authorized_cols = authorized_cols

        self.rounds_to_train = rounds_to_train
        self.current_round = 1
        self.collaborators_counter = 0
        self.quit_job_sent_to = []
        self.time_to_quit = False

        # Event to inform aggregator that collaborators have sent the results
        self.collaborator_task_results = Event()
        # A queue for each task
        self.__collaborator_tasks_queue = {collab: queue.Queue() for collab in self.authorized_cols}

        self.flow = flow
        self.checkpoint = checkpoint
        self.flow._foreach_methods = []
        self.logger.info("MetaflowInterface creation.")
        self.flow._metaflow_interface = MetaflowInterface(self.flow.__class__, "single_process")
        self.flow._run_id = self.flow._metaflow_interface.create_run()
        self.flow.runtime = FederatedRuntime()
        self.flow.runtime.aggregator = "aggregator"
        self.flow.runtime.collaborators = self.authorized_cols

        self.__private_attrs_callable = private_attributes_callable
        self.__private_attrs = private_attributes
        self.connected_collaborators = []
        self.tasks_sent_to_collaborators = 0
        self.collaborator_results_received = []

        if self.__private_attrs_callable is not None:
            self.logger.info("Initializing aggregator private attributes...")
            self.__initialize_private_attributes(private_attributes_kwargs)

    def __initialize_private_attributes(self, kwargs: Dict) -> None:
        """Call private_attrs_callable function set attributes to
        self.__private_attrs."""
        self.__private_attrs = self.__private_attrs_callable(**kwargs)

    def __set_attributes_to_clone(self, clone: Any) -> None:
        """Set private_attrs to clone as attributes."""
        if len(self.__private_attrs) > 0:
            for name, attr in self.__private_attrs.items():
                setattr(clone, name, attr)

    def __delete_agg_attrs_from_clone(self, clone: Any, replace_str: str = None) -> None:
        """
        Remove aggregator private attributes from FLSpec clone before
        transition from Aggregator step to collaborator steps.
        """
        # Update aggregator private attributes by taking latest
        # parameters from clone, then delete attributes from clone.
        if len(self.__private_attrs) > 0:
            for attr_name in self.__private_attrs:
                if hasattr(clone, attr_name):
                    self.__private_attrs.update({attr_name: getattr(clone, attr_name)})
                    if replace_str:
                        setattr(clone, attr_name, replace_str)
                    else:
                        delattr(clone, attr_name)

    def _log_big_warning(self) -> None:
        """Warn user about single collaborator cert mode."""
        self.logger.warning(
            f"\n{the_dragon}\nYOU ARE RUNNING IN SINGLE COLLABORATOR CERT MODE! THIS IS"
            f" NOT PROPER PKI AND "
            f"SHOULD ONLY BE USED IN DEVELOPMENT SETTINGS!!!! YE HAVE BEEN"
            f" WARNED!!!"
        )

    @staticmethod
    def _get_sleep_time() -> int:
        """Sleep 10 seconds.

        Returns:
            sleep_time: int
        """
        return 10

    def run_flow(self) -> None:
        """Start the execution and run flow until transition."""
        # Start function will be the first step if any flow
        f_name = "start"

        self.logger.info(f"Starting round {self.current_round}...")
        while True:
            next_step = self.do_task(f_name)

            if self.time_to_quit:
                self.logger.info("Experiment Completed.")
                self.quit_job_sent_to = self.authorized_cols
                break

            # Prepare queue for collaborator task, with clones
            for k, v in self.__collaborator_tasks_queue.items():
                if k in self.selected_collaborators:
                    v.put((next_step, self.clones_dict[k]))
                else:
                    self.logger.info(f"Tasks will not be sent to {k}")

            while not self.collaborator_task_results.is_set():
                len_sel_collabs = len(self.selected_collaborators)
                len_connected_collabs = len(self.connected_collaborators)
                if len_connected_collabs < len_sel_collabs:
                    # Waiting for collaborators to connect.
                    self.logger.info(
                        "Waiting for "
                        + f"{len_connected_collabs}/{len_sel_collabs}"
                        + " collaborators to connect..."
                    )
                elif self.tasks_sent_to_collaborators != len_sel_collabs:
                    self.logger.info(
                        "Waiting for "
                        + f"{self.tasks_sent_to_collaborators}/{len_sel_collabs}"
                        + " to make requests for tasks..."
                    )
                else:
                    # Waiting for selected collaborators to send the results.
                    self.logger.info(
                        "Waiting for "
                        + f"{self.collaborators_counter}/{len_sel_collabs}"
                        + " collaborators to send results..."
                    )
                time.sleep(Aggregator._get_sleep_time())

            self.collaborator_task_results.clear()
            f_name = self.next_step
            if hasattr(self, "instance_snapshot"):
                self.flow.restore_instance_snapshot(self.flow, list(self.instance_snapshot))
                delattr(self, "instance_snapshot")

    def call_checkpoint(self, ctx: Any, f: Callable, stream_buffer: bytes = None) -> None:
        """
        Perform checkpoint task.

        Args:
            ctx (FLSpec / bytes): Collaborator FLSpec object for which
                checkpoint is to be performed.
            f (Callable / bytes): Collaborator Step (Function) which is to be
                checkpointed.
            stream_buffer (bytes): Captured object for output and error
                (default=None).
            reserved_attributes (List[str]): List of attribute names which is
                to be excluded from checkpoint (default=[]).

        Returns:
            None
        """
        if self.checkpoint:

            # Check if arguments are pickled, if yes then unpickle
            if not isinstance(ctx, FLSpec):
                ctx = pickle.loads(ctx)
                # Updating metaflow interface object
                ctx._metaflow_interface = self.flow._metaflow_interface
            if not isinstance(f, Callable):
                f = pickle.loads(f)
            if isinstance(stream_buffer, bytes):
                # Set stream buffer as function parameter
                setattr(f.__func__, "_stream_buffer", pickle.loads(stream_buffer))

            checkpoint(ctx, f)

    def get_tasks(self, collaborator_name: str) -> Tuple:
        """RPC called by a collaborator to determine which tasks to perform.
        Tasks will only be sent to selected collaborators.

        Args:
            collaborator_name (str): Collaborator name which requested tasks.

        Returns:
            next_step (str): Next function to be executed by collaborator
            clone_bytes (bytes): Function execution context for collaborator
        """
        # If requesting collaborator is not registered as connected
        # collaborator, then register it
        if collaborator_name not in self.connected_collaborators:
            self.logger.info(f"Collaborator {collaborator_name} is connected.")
            self.connected_collaborators.append(collaborator_name)

        self.logger.debug(
            f"Aggregator GetTasks function reached from collaborator {collaborator_name}..."
        )

        # If queue of requesting collaborator is empty
        while self.__collaborator_tasks_queue[collaborator_name].qsize() == 0:
            # If it is time to then inform the collaborator
            if self.time_to_quit:
                self.logger.info(
                    f"Sending signal to collaborator {collaborator_name} to shutdown..."
                )
                # FIXME: 0, and "" instead of None is just for protobuf compatibility.
                #  Cleaner solution?
                return (
                    0,
                    "",
                    None,
                    Aggregator._get_sleep_time(),
                    self.time_to_quit,
                )

            # If not time to quit then sleep for 10 seconds
            time.sleep(Aggregator._get_sleep_time())

        # Get collaborator step, and clone for requesting collaborator
        next_step, clone = self.__collaborator_tasks_queue[collaborator_name].get()

        self.tasks_sent_to_collaborators += 1
        self.logger.info(
            "Sending tasks to collaborator"
            + f" {collaborator_name} for round {self.current_round}..."
        )
        return (
            self.current_round,
            next_step,
            pickle.dumps(clone),
            0,
            self.time_to_quit,
        )

    def do_task(self, f_name: str) -> Any:
        """Execute aggregator steps until transition.

        Args:
            f_name (str): Aggregator step

        Returns:
            string / None: Next collaborator function or None end of the flow.
        """
        # Set aggregator private attributes to flow object
        self.__set_attributes_to_clone(self.flow)

        not_at_transition_point = True
        # Run a loop to execute flow steps until not_at_transition_point
        # is False
        while not_at_transition_point:
            f = getattr(self.flow, f_name)
            # Get the list of parameters of function f
            args = inspect.signature(f)._parameters

            if f.__name__ == "end":
                f()
                # Take the checkpoint of "end" step
                self.__delete_agg_attrs_from_clone(self.flow, "Private attributes: Not Available.")
                self.call_checkpoint(self.flow, f)
                self.__set_attributes_to_clone(self.flow)
                # Check if all rounds of external loop is executed
                if self.current_round is self.rounds_to_train:
                    # All rounds execute, it is time to quit
                    self.time_to_quit = True
                    # It is time to quit - Break the loop
                    not_at_transition_point = False
                # Start next round of execution
                else:
                    self.current_round += 1
                    self.logger.info(f"Starting round {self.current_round}...")
                    f_name = "start"
                continue

            selected_clones = ()
            # If function requires arguments then it is join step of the flow
            if len(args) > 0:
                # Check if total number of collaborators and number of
                # selected collaborators are the same
                if len(self.selected_collaborators) != len(self.clones_dict):
                    # Create list of selected collaborator clones
                    selected_clones = ([],)
                    for name, clone in self.clones_dict.items():
                        # Check if collaboraotr is in the list of selected
                        # collaborators
                        if name in self.selected_collaborators:
                            selected_clones[0].append(clone)
                else:
                    # Number of selected collaborators, and number of total
                    # collaborators are same
                    selected_clones = (list(self.clones_dict.values()),)
            # Call the join function with selected collaborators
            # clones are arguments
            f(*selected_clones)

            self.__delete_agg_attrs_from_clone(self.flow, "Private attributes: Not Available.")
            # Take the checkpoint of executed step
            self.call_checkpoint(self.flow, f)
            self.__set_attributes_to_clone(self.flow)

            # Next function in the flow
            _, f, parent_func = self.flow.execute_task_args[:3]
            f_name = f.__name__

            self.flow._display_transition_logs(f, parent_func)

            # Transition check
            if aggregator_to_collaborator(f, parent_func):
                # Extract clones, instance snapshot and kwargs when reached
                # foreach loop first time
                if len(self.flow.execute_task_args) > 4:
                    temp = self.flow.execute_task_args[3:]
                    self.clones_dict, self.instance_snapshot, self.kwargs = temp

                    self.selected_collaborators = getattr(self.flow, self.kwargs["foreach"])
                else:
                    self.kwargs = self.flow.execute_task_args[3]

                # Transition encountered, break the loop
                not_at_transition_point = False

        # Delete private attributes from flow object
        self.__delete_agg_attrs_from_clone(self.flow)

        return f_name if f_name != "end" else None

    def send_task_results(
        self,
        collab_name: str,
        round_number: int,
        next_step: str,
        clone_bytes: bytes,
    ) -> None:
        """After collaborator execution, collaborator will call this function
        via gRPc to send next function.

        Args:
            collab_name (str): Collaborator name which is sending results
            round_number (int): Round number for which collaborator is sending
                results
            next_step (str): Next aggregator step in the flow
            clone_bytes (bytes): Collaborator FLSpec object

        Returns:
            None
        """
        # Log a warning if collaborator is sending results for old round
        if round_number is not self.current_round:
            self.logger.warning(
                f"Collaborator {collab_name} is reporting results"
                f" for the wrong round: {round_number}. Ignoring..."
            )
        else:
            self.logger.info(
                f"Collaborator {collab_name} sent task results" f" for round {round_number}."
            )
        # Unpickle the clone (FLSpec object)
        clone = pickle.loads(clone_bytes)
        # Update the clone in clones_dict dictionary
        self.clones_dict[clone.input] = clone
        self.next_step = next_step[0]

        self.collaborators_counter += 1
        # If selected collaborator have sent the results
        if self.collaborators_counter is len(self.selected_collaborators):
            self.collaborators_counter = 0
            # Set the event to inform aggregator to resume the flow execution
            self.collaborator_task_results.set()
            # Empty tasks_sent_to_collaborators list for next time.
            if self.tasks_sent_to_collaborators == len(self.selected_collaborators):
                self.tasks_sent_to_collaborators = 0

    def valid_collaborator_cn_and_id(
        self, cert_common_name: str, collaborator_common_name: str
    ) -> bool:
        """Determine if the collaborator certificate and ID are valid for this
        federation.

        Args:
            cert_common_name: Common name for security certificate
            collaborator_common_name: Common name for collaborator

        Returns:
            bool: True means the collaborator common name matches the name in
                  the security certificate.
        """
        # if self.test_mode_whitelist is None, then the common_name must
        # match collaborator_common_name and be in authorized_cols
        # FIXME: "" instead of None is just for protobuf compatibility.
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

    def all_quit_jobs_sent(self) -> bool:
        """Assert all quit jobs are sent to collaborators."""
        return set(self.quit_job_sent_to) == set(self.authorized_cols)


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
