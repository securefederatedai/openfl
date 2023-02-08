# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.interface.flspec module."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Type, List, Callable
from openfl.experimental.utilities import (
    MetaflowInterface,
    SerializationError,
    aggregator_to_collaborator,
    collaborator_to_aggregator,
    should_transfer,
    filter_attributes,
    checkpoint,
)
from openfl.experimental.runtime import Runtime

final_attributes = []


class FLSpec:

    _clones = []
    _initial_state = None

    def __init__(self, checkpoint: bool = False):
        self._foreach_methods = []
        self._checkpoint = checkpoint

    @classmethod
    def _create_clones(cls, instance: Type[FLSpec], names: List[str]) -> None:
        """Creates clones for instance for each collaborator in names"""
        cls._clones = {name: deepcopy(instance) for name in names}

    @classmethod
    def _reset_clones(cls):
        """Reset clones"""
        cls._clones = []

    @classmethod
    def save_initial_state(cls, instance: Type[FLSpec]) -> None:
        """Save initial state of instance before executing the flow"""
        cls._initial_state = deepcopy(instance)

    def run(self) -> None:
        """Starts the execution of the flow"""
        # Submit flow to Runtime
        self._metaflow_interface = MetaflowInterface(
            self.__class__, self.runtime.backend
        )
        self._run_id = self._metaflow_interface.create_run()
        if str(self._runtime) == "LocalRuntime":
            # Setup any necessary ShardDescriptors through the LocalEnvoys
            # Assume that first task always runs on the aggregator
            self._setup_aggregator()
            self._foreach_methods = []
            FLSpec._reset_clones()
            FLSpec._create_clones(self, self.runtime.collaborators)
            # the start function can just be invoked locally
            if self._checkpoint:
                print(f"Created flow {self.__class__.__name__}")
            try:
                self.start()
            except Exception as e:
                if "cannot pickle" in str(e) or "Failed to unpickle" in str(e):
                    msg = (
                        "\nA serialization error was encountered that could not"
                        "\nbe handled by the ray backend."
                        "\nTry rerunning the flow without ray as follows:\n"
                        "\nLocalRuntime(...,backend='single_process')\n"
                        "\n or for more information about the original error,"
                        "\nPlease see the official Ray documentation"
                        "\nhttps://docs.ray.io/en/latest/ray-core/objects/serialization.html"
                    )
                    raise SerializationError(str(e) + msg)
                else:
                    raise e
            for name, attr in final_attributes:
                setattr(self, name, attr)
        elif str(self._runtime) == "FederatedRuntime":
            raise Exception("Submission to remote runtime not available yet")
        else:
            raise Exception("Runtime not implemented")

    def _setup_aggregator(self):
        """Sets aggregator private attributes as self attributes"""
        for name, attr in self.runtime._aggregator.private_attributes.items():
            setattr(self, name, attr)

    @property
    def runtime(self) -> Type[Runtime]:
        """Returns flow runtime"""
        return self._runtime

    @runtime.setter
    def runtime(self, runtime: Type[Runtime]) -> None:
        """Sets flow runtime"""
        if isinstance(runtime, Runtime):
            self._runtime = runtime
        else:
            raise TypeError(f"{runtime} is not a valid OpenFL Runtime")

    def _capture_instance_snapshot(self, kwargs):
        """
        Takes backup of self before exclude or include filtering

        Args:
            kwargs: Key word arguments originally passed to the next function.
                    If include or exclude are in the kwargs, the state of the
                    aggregator needs to be retained
        """
        return_objs = []
        if "exclude" in kwargs or "include" in kwargs:
            backup = deepcopy(self)
            return_objs.append(backup)
        return return_objs

    def _is_at_transition_point(self, f: Callable, parent_func: Callable) -> bool:
        """
        Has the collaborator finished its current sequence?

        Args:
            f:           The next function to be executed
            parent_func: The previous function executed
        """
        if parent_func.__name__ in self._foreach_methods:
            self._foreach_methods.append(f.__name__)
            if should_transfer(f, parent_func):
                print(
                    f"Should transfer from {parent_func.__name__} to {f.__name__}"
                )
                self.execute_next = f.__name__
                return True
        return False

    def _display_transition_logs(self, f: Callable, parent_func: Callable) -> None:
        """
        Prints aggregator to collaborators or
        collaborators to aggregator state transition logs
        """
        if aggregator_to_collaborator(f, parent_func):
            print("Sending state from aggregator to collaborators")

        elif collaborator_to_aggregator(f, parent_func):
            print("Sending state from collaborator to aggregator")

    def next(self, f: Callable, **kwargs) -> None:
        """
        Next task in the flow to execute

        Args:
            f: The next task that will be executed in the flow
        """

        # Get the name and reference to the calling function
        parent = inspect.stack()[1][3]
        parent_func = getattr(self, parent)

        # Checkpoint current attributes (if checkpoint==True)
        checkpoint(self, parent_func)

        # Take back-up of current state of self
        agg_to_collab_ss = []
        if aggregator_to_collaborator(f, parent_func):
            agg_to_collab_ss = self._capture_instance_snapshot(kwargs=kwargs)

        # Remove included / excluded attributes from next task
        filter_attributes(self, f, **kwargs)

        if self._is_at_transition_point(f, parent_func):
            # Collaborator is done executing for now
            return

        self._display_transition_logs(f, parent_func)

        self._runtime.execute_task(
            self,
            f,
            parent_func,
            instance_snapshot=agg_to_collab_ss,
            **kwargs,
        )
