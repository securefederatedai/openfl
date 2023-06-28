# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experimental Collaborator module."""
import sys
sys.path.append("src")

import time
import pickle

from typing import Any, Dict
from logging import getLogger

from openfl.pipelines import NoCompressionPipeline


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
                 private_attrs_callable: Any = None,
                 private_attrs_kwargs: Dict = {},
                 compression_pipeline: Any = None, # No sure if keep it or not
                 **kwargs):

        self.collaborator_name = collaborator_name
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.client = client
        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()

        self.logger = getLogger(__name__)

        self.private_attrs_callable = private_attrs_callable
        self.private_attrs_kwargs = private_attrs_kwargs

        self.round_number = 0

        self.logger.info("Initialiaing collaborator.")
        self.initialize_private_attributes()


    def initialize_private_attributes(self):
        """
        Call private_attrs_callable function set 
            attributes to self.private_attrs
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


    def __delete_agg_attrs_from_clone(self, clone) -> None:
        """
        Remove aggregator private attributes from FLSpec clone before
        transition from Aggregator step to collaborator steps
        """
        # Update aggregator private attributes by taking latest
        # parameters from clone, then delete attributes from clone.
        for attr_name in self.private_attrs:
            if hasattr(clone, attr_name):
                self.private_attrs.update({attr_name: getattr(clone, attr_name)})
                delattr(clone, attr_name)


    def call_checkpoint(self, ctx, f, sb):
        """Call checkpoint gRPC."""
        self.client.call_checkpoint(
            self.collaborator_name,
            pickle.dumps(ctx), pickle.dumps(f) #, pickle.dumps(sb)
        )


    def run(self):
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


    def send_task_results(self, next_step, clone):
        """
        After collaborator is executed, send next aggregator
            step to Aggregator for continue execution.
        """
        self.logger.info(f"Sending results to aggregator...")
        self.client.send_task_results(
            self.collaborator_name, self.round_number,
            next_step, pickle.dumps(clone)
        )


    def get_tasks(self):
        """Get tasks from the aggregator."""
        self.logger.info('Waiting for tasks...')

        self.round_number, next_step, clone_bytes, sleep_time, \
            time_to_quit = self.client.get_tasks(self.collaborator_name)

        return next_step, pickle.loads(clone_bytes), sleep_time, time_to_quit


    def do_task(self, f_name, ctx):
        """Run collaborator steps until transition."""
        self.__set_attributes_to_clone(ctx)

        not_at_transition_point = True
        while not_at_transition_point:
            f = getattr(ctx, f_name)
            f()
            # self.call_checkpoint(ctx, f, f._stream_buffer)

            _, f, parent_func = ctx.execute_task_args[:3]
            if ctx._is_at_transition_point(f, parent_func):
                not_at_transition_point = False

            f_name = f.__name__

        self.__delete_agg_attrs_from_clone(ctx)

        self.round_number += 1
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
