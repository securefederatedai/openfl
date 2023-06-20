# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experimental Aggregator module."""
import queue
import pickle
from copy import deepcopy
from logging import getLogger
from typing import Any, Dict
from typing import List, Callable

from openfl.utilities.logs import write_metric

from openfl.experimental.utilities import aggregator_to_collaborator
from openfl.experimental.runtime import FederatedRuntime
from openfl.experimental.utilities import checkpoint


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
            name: str,
            aggregator_uuid: str,
            federation_uuid: str,
            authorized_cols: List,

            flow: Any,
            runtime: FederatedRuntime,
            private_attrs_callable: Callable = None,
            private_attrs_kwargs: Dict = {},

            single_col_cert_common_name: Any = None,
            compression_pipeline: Any = None, # TBD: Could be used later

            write_logs: bool = False,
            log_metric_callback: Callable = None,
            **kwargs) -> None:

        self.name = name
        self.uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.authorized_cols = authorized_cols

        self.single_col_cert_common_name = single_col_cert_common_name

        if self.single_col_cert_common_name is not None:
            self._log_big_warning()
        else:
            # FIXME: '' instead of None is just for protobuf compatibility.
            # Cleaner solution?
            self.single_col_cert_common_name = ''

        self.compression_pipeline = compression_pipeline

        self.quit_job_sent_to = []
        self.time_to_quit = False

        self.logger = getLogger(__name__)
        self.write_logs = write_logs
        self.log_metric_callback = log_metric_callback

        if self.write_logs:
            self.log_metric = write_metric
            if self.log_metric_callback:
                self.log_metric = log_metric_callback
                self.logger.info(f'Using custom log metric: {self.log_metric}')

        self.flow = flow
        self.flow.runtime = runtime
        self.flow.runtime.aggregator = self.name
        self.flow.runtime.collaborators = self.authorized_cols

        self.collaborator_tasks = queue.Queue()
        self.connected_collaborators = []
        self.private_attrs_callable = private_attrs_callable
        # FIXME: Save private_attrs_kwargs to self, or directly pass
        # private_attrs_kwargs to initialize_private_attributes ?
        self.private_attrs_kwargs = private_attrs_kwargs

        self.initialize_private_attributes()


    def run_flow_until_transition(self):
        """
        Start the execution and run flow until transition.
        """
        start_step = self.flow.run()
        # self.clones_dict = clones_dict
        next_step = self.do_task(start_step)

        if isinstance(next_step, bool):
            self.logger.info("Flow execution completed.")
        else:
            self.collaborator_tasks.put((next_step, self.clones))


    def initialize_private_attributes(self):
        """
        Call private_attrs_callable function set 
            attributes to self.private_attrs.
        """
        self.private_attrs = self.private_attrs_callable(
            **self.private_attrs_kwargs
        )


    def __set_attributes_to_clone(self, clone):
        """
        Set private_attrs to clone as attributes.
        """
        for name, attr in self.private_attrs.items():
            setattr(clone, name, attr)


    def __delete_agg_attrs_from_clone(self, clone: Any) -> None:
        """
        Remove aggregator private attributes from FLSpec clone before
        transition from Aggregator step to collaborator steps.
        """
        # Update aggregator private attributes by taking latest
        # parameters from clone, then delete attributes from clone.
        for attr_name in self.private_attrs:
            if hasattr(clone, attr_name):
                self.private_attrs.update({attr_name: getattr(clone, attr_name)})
                delattr(clone, attr_name)


    def _log_big_warning(self):
        """Warn user about single collaborator cert mode."""
        self.logger.warning(
            f'\n{the_dragon}\nYOU ARE RUNNING IN SINGLE COLLABORATOR CERT MODE! THIS IS'
            f' NOT PROPER PKI AND '
            f'SHOULD ONLY BE USED IN DEVELOPMENT SETTINGS!!!! YE HAVE BEEN'
            f' WARNED!!!'
        )


    @staticmethod
    def _get_sleep_time():
        """
        Sleep 10 seconds.

        Returns:
            sleep_time: int
        """
        # Decrease sleep period for finer discretezation
        return 10


    # Get this logic checked by Sachin.
    def get_tasks(self, collaborator_name):
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
            self.connected_collaborators.append(collaborator_name)

        if self.collaborator_tasks.qsize() == 0:
            # FIXME: 0, and '' instead of None is just for protobuf compatibility.
            #  Cleaner solution?
            return 0, '', None, Aggregator._get_sleep_time(), self.time_to_quit

        next_step, clones_dict = self.collaborator_tasks.get()

        clone = deepcopy(clones_dict[collaborator_name])

        # Cannot delete clone from dictionary there may be new collaborator
        # steps which require clones
        del clones_dict[collaborator_name]

        if len(clones_dict) > 0:
            self.collaborator_tasks.put((next_step, clones_dict))

        return 0, next_step, pickle.dumps(clone), 0, self.time_to_quit


    def do_task(self, f_name, args=None):
        """Execute aggregator steps until transition
        """
        self.__set_attributes_to_clone(self.flow)

        not_at_transition_point = True
        while not_at_transition_point:
            f = getattr(self.flow, f_name)
            f(args) if args else f()
            args = None

            _, f, parent_func = self.flow.execute_task_args[:3]
            f_name = f.__name__

            if aggregator_to_collaborator(f, parent_func):
                not_at_transition_point = False
                if len(self.flow.execute_task_args) > 4:
                    self.clones, self.instance_snapshot, self.kwargs = self.flow.execute_task_args[3:]
                else:
                    self.kwargs = self.flow.execute_task_args[3]

            if f.__name__ == "end":
                f = getattr(self.flow, f_name)
                f()
                checkpoint(self.flow, f)
                self.time_to_quit = True
                not_at_transition_point = False

        self.__delete_agg_attrs_from_clone(self.flow)

        return f_name if f_name != "end" else False


    def send_task_results(self, collab_name, round_number, next_step, clone_bytes):
        """
        After collaborator execution, collaborator will call this function via gRPc
            to send next function.
        """
        self.logger.info(f"Aggregator step received from {collab_name}")

        self.round_number = round_number
        clone = pickle.loads(clone_bytes)
        self.clones[clone.input] = clone

        self.flow.restore_instance_snapshot(self.flow, self.instance_snapshot)
        # FIXME: do_task should not be called from here.
        # Cleaner solution?
        self.do_task(next_step[0], list(self.clones.values()))

        return True


    def valid_collaborator_cn_and_id(self, cert_common_name,
                                     collaborator_common_name):
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


    def all_quit_jobs_sent(self):
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
