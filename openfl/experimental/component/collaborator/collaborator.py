# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Experimental Collaborator module."""
import pickle
import time
from logging import getLogger
from typing import Any, Callable, Dict, Tuple


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

    Note:
        \* - Plan setting.
    """

    def __init__(
        self,
        collaborator_name: str,
        aggregator_uuid: str,
        federation_uuid: str,
        client: Any,
        private_attributes_callable: Any = None,
        private_attributes_kwargs: Dict = {},
        private_attributes: Dict = {},
        **kwargs,
    ) -> None:

        self.name = collaborator_name
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid

        self.client = client

        self.logger = getLogger(__name__)

        self.__private_attrs_callable = private_attributes_callable

        self.__private_attrs = private_attributes
        if self.__private_attrs_callable is not None:
            self.logger.info("Initializing collaborator.")
            self.__initialize_private_attributes(private_attributes_kwargs)

    def __initialize_private_attributes(self, kwargs: Dict) -> None:
        """Call private_attrs_callable function set attributes to
        self.__private_attrs.

        Args:
            kwargs (Dict): Private attributes callable function arguments

        Returns:
            None
        """
        self.__private_attrs = self.__private_attrs_callable(**kwargs)

    def __set_attributes_to_clone(self, clone: Any) -> None:
        """Set private_attrs to clone as attributes.

        Args:
            clone (FLSpec): Clone to which private attributes are to be
            set

        Returns:
            None
        """
        if len(self.__private_attrs) > 0:
            for name, attr in self.__private_attrs.items():
                setattr(clone, name, attr)

    def __delete_agg_attrs_from_clone(self, clone: Any, replace_str: str = None) -> None:
        """
        Remove aggregator private attributes from FLSpec clone before
        transition from Aggregator step to collaborator steps

        Args:
            clone (FLSpec): Clone from which private attributes are to be
            removed

        Returns:
            None
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

    def call_checkpoint(self, ctx: Any, f: Callable, stream_buffer: Any) -> None:
        """
        Call checkpoint gRPC.

        Args:
            ctx (FLSpec): FLSPec object.
            f (Callable): Flow step which is be checkpointed.
            stream_buffer (Any): Captured object for output and error.

        Returns:
            None
        """
        self.client.call_checkpoint(
            self.name,
            pickle.dumps(ctx),
            pickle.dumps(f),
            pickle.dumps(stream_buffer),
        )

    def run(self) -> None:
        """Run the collaborator.

        Args:
            None

        Returns:
            None
        """
        while True:
            next_step, clone, sleep_time, time_to_quit = self.get_tasks()
            if time_to_quit:
                break
            elif sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.logger.info(f"Received the following tasks: {next_step}.")
                f_name, ctx = self.do_task(next_step, clone)
                self.send_task_results(f_name, ctx)

        self.logger.info("End of Federation reached. Exiting...")

    def send_task_results(self, next_step: str, clone: Any) -> None:
        """After collaborator is executed, send next aggregator step to
        Aggregator for continue execution.

        Args:
            next_step (str): Send next function to aggregator
            clone (FLSpec): Updated clone object (Private attributes atr not
                included)

        Returns:
            None
        """
        self.logger.info(
            f"Round {self.round_number}," f" collaborator {self.name} is sending results..."
        )
        self.client.send_task_results(self.name, self.round_number, next_step, pickle.dumps(clone))

    def get_tasks(self) -> Tuple:
        """Get tasks from the aggregator.

        Args:
            None

        Returns:
            next_step (str): Next collaborator function to start execution from
            ctx (FLSpec): Function context
            sleep_time (int): Sleep for given seconds if not ready yet
            time_to_quit (bool): True if end of reached
        """
        self.logger.info("Waiting for tasks...")
        temp = self.client.get_tasks(self.name)
        self.round_number, next_step, clone_bytes, sleep_time, time_to_quit = temp

        return next_step, pickle.loads(clone_bytes), sleep_time, time_to_quit

    def do_task(self, f_name: str, ctx: Any) -> Tuple:
        """Run collaborator steps until transition.

        Args:
            f_name (str): Function name which is to be executed.
            ctx (FLSpec): Function context.

        Returns:
            Tuple(str, FLSpec): Next aggregator function, and updated context.
        """
        # Set private attributes to context
        self.__set_attributes_to_clone(ctx)

        # Loop control variable
        not_at_transition_point = True
        while not_at_transition_point:
            f = getattr(ctx, f_name)
            f()
            # Checkpoint the function
            self.__delete_agg_attrs_from_clone(ctx, "Private attributes: Not Available.")
            self.call_checkpoint(ctx, f, f._stream_buffer)
            self.__set_attributes_to_clone(ctx)

            _, f, parent_func = ctx.execute_task_args[:3]
            # Display transition logs if transition
            ctx._display_transition_logs(f, parent_func)

            # If transition break the loop
            if ctx._is_at_transition_point(f, parent_func):
                not_at_transition_point = False

            # Update the function name
            f_name = f.__name__

        # Reomve private attributes from context
        self.__delete_agg_attrs_from_clone(ctx)

        return f_name, ctx
