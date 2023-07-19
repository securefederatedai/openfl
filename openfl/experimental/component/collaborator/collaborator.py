# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experimental Collaborator module."""
import time
import pickle

from typing import Any, Callable, Dict, Tuple
from logging import getLogger


class Collaborator:
    r"""The Collaborator object class.

    Args:
        collaborator_name (str): The common name for the collaborator.
        aggregator_uuid (str): The unique id for the client.
        federation_uuid (str): The unique id for the federation.

        client (AggregatorGRPCClient): GRPC Client to connect to 
        Aggregator Server.

        private_attrs_callable (Callable): Function for Collaborator 
        private attriubtes.
        private_attrs_kwargs (Dict): Arguments to call private_attrs_callable.        
        compression_pipeline: The compression pipeline (Defaults to None).

    Note:
        \* - Plan setting.
    """

    def __init__(self,
                 collaborator_name: str,
                 aggregator_uuid: str,
                 federation_uuid: str,
                 client: Any,
                 private_attributes_callable: Any = None,
                 private_attributes_kwargs: Dict = {},
                 **kwargs) -> None:

        self.name = collaborator_name
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid

        self.client = client

        self.logger = getLogger(__name__)

        self.private_attrs_callable = private_attributes_callable

        self.__private_attrs = {}
        if self.private_attrs_callable is not None:
            self.logger.info("Initialiaing collaborator.")
            self.initialize_private_attributes(private_attributes_kwargs)

    def initialize_private_attributes(self, kwrags: Dict) -> None:
        """
        Call private_attrs_callable function set 
            attributes to self.__private_attrs
        """
        self.__private_attrs = self.private_attrs_callable(
            **kwrags
        )

    def __set_attributes_to_clone(self, clone: Any) -> None:
        """
        Set private_attrs to clone as attributes.
        """
        if len(self.__private_attrs) > 0:
            for name, attr in self.__private_attrs.items():
                setattr(clone, name, attr)

    def __delete_agg_attrs_from_clone(self, clone: Any) -> None:
        """
        Remove aggregator private attributes from FLSpec clone before
        transition from Aggregator step to collaborator steps
        """
        # Update aggregator private attributes by taking latest
        # parameters from clone, then delete attributes from clone.
        if len(self.__private_attrs) > 0:
            for attr_name in self.__private_attrs:
                if hasattr(clone, attr_name):
                    self.__private_attrs.update({attr_name: getattr(clone, attr_name)})
                    delattr(clone, attr_name)

    def call_checkpoint(self, ctx: Any, f: Callable, stream_buffer: Any) -> None:
        """Call checkpoint gRPC."""
        self.client.call_checkpoint(
            self.name,
            pickle.dumps(ctx), pickle.dumps(f), pickle.dumps(stream_buffer),
            list(self.__private_attrs.keys())
        )

    def run(self) -> None:
        """Run the collaborator."""
        while True:
            next_step, clone, sleep_time, time_to_quit = self.get_tasks()
            if time_to_quit:
                break
            elif sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.logger.info(f"Received {next_step} step from Aggregator.")
                f_name, ctx = self.do_task(next_step, clone)
                self.send_task_results(f_name, ctx)

    def send_task_results(self, next_step: str, clone: Any) -> None:
        """
        After collaborator is executed, send next aggregator
            step to Aggregator for continue execution.
        """
        self.logger.info(f"Sending results to aggregator...")
        self.client.send_task_results(
            self.name, self.round_number,
            next_step, pickle.dumps(clone)
        )

    def get_tasks(self) -> Tuple:
        """Get tasks from the aggregator."""
        self.logger.info('Waiting for tasks...')

        self.round_number, next_step, clone_bytes, sleep_time, \
            time_to_quit = self.client.get_tasks(self.name)

        return next_step, pickle.loads(clone_bytes), sleep_time, time_to_quit

    def do_task(self, f_name: str, ctx: Any) -> Tuple:
        """Run collaborator steps until transition."""
        self.__set_attributes_to_clone(ctx)

        not_at_transition_point = True
        while not_at_transition_point:
            f = getattr(ctx, f_name)
            f()
            self.call_checkpoint(ctx, f, f._stream_buffer)

            _, f, parent_func = ctx.execute_task_args[:3]
            if ctx._is_at_transition_point(f, parent_func):
                not_at_transition_point = False

            f_name = f.__name__

        self.__delete_agg_attrs_from_clone(ctx)

        return f_name, ctx


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
