# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experimental Aggregator module."""
import sys
import time
import queue
import pickle
import inspect
from threading import Event
from copy import deepcopy
from logging import getLogger
from typing import Any, Dict, Tuple
from typing import List, Callable

from openfl.experimental.utilities import aggregator_to_collaborator
from openfl.experimental.runtime import FederatedRuntime
from openfl.experimental.utilities import checkpoint
from openfl.experimental.utilities.metaflow_utils import (
    MetaflowInterface
)


class Aggregator:
    r"""An Aggregator is the central node in federated learning.

    Args:
        aggregator_uuid (str): Aggregation ID.
        federation_uuid (str): Federation ID.
        authorized_cols (list of str): The list of IDs of enrolled collaborators.

        flow (Any): Flow class.
        runtime (FederatedRuntime): FedeatedRuntime object.

        private_attrs_callable (Callable): Function for Aggregator private attriubtes.
        private_attrs_kwargs (Dict): Arguments to call private_attrs_callable.        

    Note:
        \* - plan setting.
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

            single_col_cert_common_name: Any = None,

            log_metric_callback: Callable = None,
            **kwargs) -> None:

        self.uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.authorized_cols = authorized_cols

        self.round_number = rounds_to_train
        self.collaborators_counter = 0
        self.quit_job_sent_to = []
        self.time_to_quit = False

        self.logger = getLogger(__name__)

        self.single_col_cert_common_name = single_col_cert_common_name
        if self.single_col_cert_common_name is not None:
            self._log_big_warning()
        else:
            # FIXME: '' instead of None is just for protobuf compatibility.
            # Cleaner solution?
            self.single_col_cert_common_name = ''

        self.log_metric_callback = log_metric_callback
        self.collaborator_task_results = Event()

        if self.log_metric_callback is not None:
            self.log_metric = log_metric_callback
            self.logger.info(f'Using custom log metric: {self.log_metric}')

        self.checkpoint = checkpoint
        self.flow = flow
        self.logger.info(f"MetaflowInterface creation.")
        self.flow._metaflow_interface = MetaflowInterface(
            self.flow.__class__, "single_process"
        )
        self.flow._run_id = self.flow._metaflow_interface.create_run()
        self.flow.runtime = FederatedRuntime()
        self.flow.runtime.aggregator = "aggregator"
        self.flow.runtime.collaborators = self.authorized_cols

        self.collaborator_tasks_queue = {collab: queue.Queue() for collab
                                         in self.authorized_cols}

        self.__private_attrs = {}
        self.connected_collaborators = []
        self.collaborator_results_received = []
        self.private_attrs_callable = private_attributes_callable

        if self.private_attrs_callable is not None:
            self.logger.info("Initialiaing aggregator.")
            self.initialize_private_attributes(private_attributes_kwargs)

    def initialize_private_attributes(self, kwargs: Dict) -> None:
        """
        Call private_attrs_callable function set 
            attributes to self.__private_attrs.
        """
        self.__private_attrs = self.private_attrs_callable(
            **kwargs
        )

    # TODO: rename the method.
    def run_flow_until_transition(self) -> None:
        """
        Start the execution and run flow until transition.
        """
        f_name = self.flow.run()

        while True:
            next_step = self.do_task(f_name)

            if self.time_to_quit:
                self.logger.info("Flow execution completed.")
                sys.exit(0)

            # Prepare queue for collaborator task, with clones
            for k, v in self.collaborator_tasks_queue.items():
                if k in self.selected_collaborators:
                    v.put((next_step, self.clones_dict[k]))
                else:
                    self.logger.info(f"Tasks will not be sent to {k}")

            while not self.collaborator_task_results.is_set():
                # Waiting for selected collaborators to send the results.
                self.logger.info(f"Waiting for "
                                 + f"{self.collaborators_counter}/{len(self.selected_collaborators)}"
                                 + " collaborators to send results.")
                time.sleep(Aggregator._get_sleep_time())

            self.collaborator_task_results.clear()
            f_name = self.next_step
            if hasattr(self, "instance_snapshot"):
                self.flow.restore_instance_snapshot(self.flow, list(self.instance_snapshot))
                delattr(self, "instance_snapshot")

    def call_checkpoint(self, ctx: Any, f: Callable, stream_buffer: bytes = None,
                        reserved_attributes: List[str] = []) -> None:
        """Perform checkpoint task."""
        if self.checkpoint:
            # with SystemMutex("critical_section_1"):
            from openfl.experimental.interface import (
                FLSpec,
            )

            if not isinstance(ctx, FLSpec):
                ctx = pickle.loads(ctx)
                # Updating metaflow interface object
                ctx._metaflow_interface = self.flow._metaflow_interface
            if not isinstance(f, Callable):
                f = pickle.loads(f)
            if isinstance(stream_buffer, bytes):
                setattr(f.__func__, "_stream_buffer", pickle.loads(stream_buffer))

            for attr in reserved_attributes:
                if hasattr(ctx, attr):
                    setattr(ctx, attr, "Private attributes: Not Available.")

            checkpoint(ctx, f)

    def __set_attributes_to_clone(self, clone: Any) -> None:
        """
        Set private_attrs to clone as attributes.
        """
        # if hasattr(self, "private_attrs"):
        if len(self.__private_attrs) > 0:
            for name, attr in self.__private_attrs.items():
                setattr(clone, name, attr)

    def __delete_agg_attrs_from_clone(self, clone: Any) -> None:
        """
        Remove aggregator private attributes from FLSpec clone before
        transition from Aggregator step to collaborator steps.
        """
        # Update aggregator private attributes by taking latest
        # parameters from clone, then delete attributes from clone.
        # if hasattr(self, "private_attrs"):
        if len(self.__private_attrs) > 0:
            for attr_name in self.__private_attrs:
                if hasattr(clone, attr_name):
                    self.__private_attrs.update({attr_name: getattr(clone, attr_name)})
                    delattr(clone, attr_name)

    def _log_big_warning(self) -> None:
        """Warn user about single collaborator cert mode."""
        self.logger.warning(
            f'\n{the_dragon}\nYOU ARE RUNNING IN SINGLE COLLABORATOR CERT MODE! THIS IS'
            f' NOT PROPER PKI AND '
            f'SHOULD ONLY BE USED IN DEVELOPMENT SETTINGS!!!! YE HAVE BEEN'
            f' WARNED!!!'
        )

    @staticmethod
    def _get_sleep_time() -> int:
        """
        Sleep 10 seconds.

        Returns:
            sleep_time: int
        """
        # Decrease sleep period for finer discretezation
        return 10

    def get_tasks(self, collaborator_name: str) -> Tuple:
        """
        RPC called by a collaborator to determine which tasks to perform.

        Args:
            collaborator_name: str
                Requested collaborator name

        Returns:
            next_step: str
                next function to be executed by collaborator
            clone_bytes: bytes
                Function execution context for collaborator                    
        """
        if collaborator_name not in self.connected_collaborators:
            self.logger.info(f"Collaborator {collaborator_name} is connected.")
            self.connected_collaborators.append(collaborator_name)

        while self.collaborator_tasks_queue[collaborator_name].qsize() == 0:
            # FIXME: 0, and '' instead of None is just for protobuf compatibility.
            #  Cleaner solution?
            if not self.time_to_quit:
                time.sleep(Aggregator._get_sleep_time())
            else:
                return 0, '', None, Aggregator._get_sleep_time(), self.time_to_quit

        next_step, clone = self.collaborator_tasks_queue[
            collaborator_name].get()

        return 0, next_step, pickle.dumps(clone), 0, self.time_to_quit

    def do_task(self, f_name: str) -> Any:
        """Execute aggregator steps until transition."""
        self.__set_attributes_to_clone(self.flow)

        not_at_transition_point = True
        while not_at_transition_point:
            f = getattr(self.flow, f_name)
            args = inspect.signature(f)._parameters

            if f.__name__ == "end":
                f()
                self.call_checkpoint(deepcopy(self.flow), f,
                                     reserved_attributes=list(self.__private_attrs.keys()))
                self.time_to_quit = True
                not_at_transition_point = False
                continue

            if len(args) > 0:
                # Get clone for each selected collaborator only
                if len(self.selected_collaborators) != len(self.clones_dict):
                    selected_clones = {}
                    for name, clone in self.clones_dict.items():
                        if name in self.selected_collaborators:
                            selected_clones[name] = clone
                else:
                    selected_clones = self.clones_dict

                # Call the join function with selected collaborators
                # clones are arguments
                f(list(selected_clones.values()))
            else: f()

            self.call_checkpoint(deepcopy(self.flow), f,
                                 reserved_attributes=list(self.__private_attrs.keys()))

            _, f, parent_func = self.flow.execute_task_args[:3]
            f_name = f.__name__

            if aggregator_to_collaborator(f, parent_func):
                not_at_transition_point = False
                if len(self.flow.execute_task_args) > 4:
                    self.clones_dict, self.instance_snapshot, self.kwargs = \
                        self.flow.execute_task_args[3:]

                    self.selected_collaborators = getattr(self.flow, self.kwargs["foreach"])
                else:
                    self.kwargs = self.flow.execute_task_args[3]

        self.__delete_agg_attrs_from_clone(self.flow)

        return f_name if f_name != "end" else False

    def send_task_results(self, collab_name: str, round_number: int, next_step: str,
                          clone_bytes: bytes) -> None:
        """
        After collaborator execution, collaborator will call this function via gRPc
            to send next function.
        """
        self.logger.info(f"Aggregator step received from {collab_name} for "
                         + f"round number: {round_number}.")

        # TODO: Think about taking values from collaborators.
        # Do not take rn.
        self.round_number = round_number
        clone = pickle.loads(clone_bytes)
        self.clones_dict[clone.input] = clone
        self.next_step = next_step[0]

        self.collaborators_counter += 1
        # No need to wait for all collaborators
        if self.collaborators_counter == len(self.selected_collaborators):
            self.collaborators_counter = 0
            self.collaborator_task_results.set()

    def valid_collaborator_cn_and_id(self, cert_common_name: str,
                                     collaborator_common_name: str) -> bool:
        """
        Determine if the collaborator certificate and ID are valid for this federation.

        Args:
            cert_common_name: Common name for security certificate
            collaborator_common_name: Common name for collaborator

        Returns:
            bool: True means the collaborator common name matches the name in
                  the security certificate.

        """
        # if self.test_mode_whitelist is None, then the common_name must
        # match collaborator_common_name and be in authorized_cols
        # FIXME: '' instead of None is just for protobuf compatibility.
        #  Cleaner solution?
        if self.single_col_cert_common_name == '':
            return (cert_common_name == collaborator_common_name
                    and collaborator_common_name in self.authorized_cols)
        # otherwise, common_name must be in whitelist and
        # collaborator_common_name must be in authorized_cols
        else:
            return (cert_common_name == self.single_col_cert_common_name
                    and collaborator_common_name in self.authorized_cols)

    def all_quit_jobs_sent(self) -> bool:
        """Assert all quit jobs are sent to collaborators."""
        return set(self.quit_job_sent_to) == set(self.authorized_cols)


the_dragon = '''

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
                                  ,                                                        '''
