# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.interface.flspec module."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Callable, List, Type

from openfl.experimental.utilities import (
    MetaflowInterface,
    SerializationError,
    aggregator_to_collaborator,
    checkpoint,
    collaborator_to_aggregator,
    filter_attributes,
    generate_artifacts,
    should_transfer,
)


class FLSpec:
    _clones = []
    _initial_state = None

    def __init__(self, checkpoint: bool = False):
        """Initializes the FLSpec object.

        Args:
            checkpoint (bool, optional): Determines whether to checkpoint or
                not. Defaults to False.
        """
        self._foreach_methods = []
        self._checkpoint = checkpoint

    @classmethod
    def _create_clones(cls, instance: Type[FLSpec], names: List[str]) -> None:
        """Creates clones for instance for each collaborator in names.

        Args:
            instance (Type[FLSpec]): The instance to be cloned.
            names (List[str]): The list of names for the clones.
        """
        cls._clones = {name: deepcopy(instance) for name in names}

    @classmethod
    def _reset_clones(cls):
        """Resets the clones of the class."""

        cls._clones = []

    @classmethod
    def save_initial_state(cls, instance: Type[FLSpec]) -> None:
        """Saves the initial state of an instance before executing the flow.

        Args:
            instance (Type[FLSpec]): The instance whose initial state is to be
                saved.
        """
        cls._initial_state = deepcopy(instance)

    def run(self) -> None:
        """Starts the execution of the flow."""

        # Submit flow to Runtime
        if str(self._runtime) == "LocalRuntime":
            self._metaflow_interface = MetaflowInterface(self.__class__, self.runtime.backend)
            self._run_id = self._metaflow_interface.create_run()
            # Initialize aggregator private attributes
            self.runtime.initialize_aggregator()
            self._foreach_methods = []
            FLSpec._reset_clones()
            FLSpec._create_clones(self, self.runtime.collaborators)
            # Initialize collaborator private attributes
            self.runtime.initialize_collaborators()
            if self._checkpoint:
                print(f"Created flow {self.__class__.__name__}")
            try:
                # Execute all Participant (Aggregator & Collaborator) tasks and
                # retrieve the final attributes
                # start step is the first task & invoked on aggregator through
                # runtime.execute_task
                final_attributes = self.runtime.execute_task(
                    self,
                    self.start,
                )
            except Exception as e:
                if "cannot pickle" in str(e) or "Failed to unpickle" in str(e):
                    msg = (
                        "\nA serialization error was encountered that could not"
                        "\nbe handled by the ray backend."
                        "\nTry rerunning the flow without ray as follows:\n"
                        "\nLocalRuntime(...,backend='single_process')\n"
                        "\n or for more information about the original error,"
                        "\nPlease see the official Ray documentation"
                        "\nhttps://docs.ray.io/en/releases-2.2.0/ray-core/\
                        objects/serialization.html"
                    )
                    raise SerializationError(str(e) + msg)
                else:
                    raise e
            for name, attr in final_attributes:
                setattr(self, name, attr)
        elif str(self._runtime) == "FederatedRuntime":
            pass
        else:
            raise Exception("Runtime not implemented")

    @property
    def runtime(self):
        """Returns flow runtime.

        Returns:
            Type[Runtime]: The runtime of the flow.
        """
        return self._runtime

    @runtime.setter
    def runtime(self, runtime) -> None:
        """Sets flow runtime.

        Args:
            runtime (Type[Runtime]): The runtime to be set.

        Raises:
            TypeError: If the provided runtime is not a valid OpenFL Runtime.
        """
        if str(runtime) not in ["LocalRuntime", "FederatedRuntime"]:
            raise TypeError(f"{runtime} is not a valid OpenFL Runtime")
        self._runtime = runtime

    def _capture_instance_snapshot(self, kwargs):
        """Takes backup of self before exclude or include filtering.

        Args:
            kwargs: Key word arguments originally passed to the next function.
                    If include or exclude are in the kwargs, the state of the
                    aggregator needs to be retained.

        Returns:
            return_objs (list): A list of return objects.
        """
        return_objs = []
        if "exclude" in kwargs or "include" in kwargs:
            backup = deepcopy(self)
            return_objs.append(backup)
        return return_objs

    def _is_at_transition_point(self, f: Callable, parent_func: Callable) -> bool:
        """
        Determines if the collaborator has finished its current sequence.

        Args:
            f (Callable): The next function to be executed.
            parent_func (Callable): The previous function executed.

        Returns:
            bool: True if the collaborator has finished its current sequence,
                False otherwise.
        """
        if parent_func.__name__ in self._foreach_methods:
            self._foreach_methods.append(f.__name__)
            if should_transfer(f, parent_func):
                print(f"Should transfer from {parent_func.__name__} to {f.__name__}")
                self.execute_next = f.__name__
                return True
        return False

    def _display_transition_logs(self, f: Callable, parent_func: Callable) -> None:
        """
        Prints aggregator to collaborators or collaborators to aggregator
        state transition logs.

        Args:
            f (Callable): The next function to be executed.
            parent_func (Callable): The previous function executed.
        """
        if aggregator_to_collaborator(f, parent_func):
            print("Sending state from aggregator to collaborators")

        elif collaborator_to_aggregator(f, parent_func):
            print("Sending state from collaborator to aggregator")

    def filter_exclude_include(self, f, **kwargs):
        """Filters exclude/include attributes for a given task within the flow.

        Args:
            f (Callable): The task to be executed within the flow.
            **kwargs (dict): Additional keyword arguments. These should
                include:
                - "foreach" (str): The attribute name that contains the list
                of selected collaborators.
                - "exclude" (list, optional): List of attribute names to
                exclude. If an attribute name is present in this list and the
                clone has this attribute, it will be filtered out.
                - "include" (list, optional): List of attribute names to
                include. If an attribute name is present in this list and the
                clone has this attribute, it will be included.
        """
        selected_collaborators = getattr(self, kwargs["foreach"])

        for col in selected_collaborators:
            clone = FLSpec._clones[col]
            clone.input = col
            if ("exclude" in kwargs and hasattr(clone, kwargs["exclude"][0])) or (
                "include" in kwargs and hasattr(clone, kwargs["include"][0])
            ):
                filter_attributes(clone, f, **kwargs)
            artifacts_iter, _ = generate_artifacts(ctx=self)
            for name, attr in artifacts_iter():
                setattr(clone, name, deepcopy(attr))
            clone._foreach_methods = self._foreach_methods

    def restore_instance_snapshot(self, ctx: FLSpec, instance_snapshot: List[FLSpec]):
        """Restores attributes from backup (in instance snapshot) to ctx.

        Args:
            ctx (FLSpec): The context to restore the attributes to.
            instance_snapshot (List[FLSpec]): The list of FLSpec instances
                that serve as the backup.
        """
        for backup in instance_snapshot:
            artifacts_iter, _ = generate_artifacts(ctx=backup)
            for name, attr in artifacts_iter():
                if not hasattr(ctx, name):
                    setattr(ctx, name, attr)

    def get_clones(self, kwargs):
        """Create, and prepare clones."""
        FLSpec._reset_clones()
        FLSpec._create_clones(self, self.runtime.collaborators)
        selected_collaborators = self.__getattribute__(kwargs["foreach"])

        for col in selected_collaborators:
            clone = FLSpec._clones[col]
            clone.input = col
            artifacts_iter, _ = generate_artifacts(ctx=clone)
            attributes = artifacts_iter()
            for name, attr in attributes:
                setattr(clone, name, deepcopy(attr))
            clone._foreach_methods = self._foreach_methods
            clone._metaflow_interface = self._metaflow_interface

    def next(self, f, **kwargs):
        """Specifies the next task in the flow to execute.

        Args:
            f (Callable): The next task that will be executed in the flow.
            **kwargs: Additional keyword arguments.
        """
        # Get the name and reference to the calling function
        parent = inspect.stack()[1][3]
        parent_func = getattr(self, parent)

        if str(self._runtime) == "LocalRuntime":
            # Checkpoint current attributes (if checkpoint==True)
            checkpoint(self, parent_func)

        # Take back-up of current state of self
        agg_to_collab_ss = None
        if aggregator_to_collaborator(f, parent_func):
            agg_to_collab_ss = self._capture_instance_snapshot(kwargs=kwargs)

            if str(self._runtime) == "FederatedRuntime":
                if len(FLSpec._clones) == 0:
                    self.get_clones(kwargs)

        # Remove included / excluded attributes from next task
        filter_attributes(self, f, **kwargs)

        if str(self._runtime) == "FederatedRuntime":
            if f.collaborator_step and not f.aggregator_step:
                self._foreach_methods.append(f.__name__)

            if "foreach" in kwargs:
                self.filter_exclude_include(f, **kwargs)
                # if "foreach" in kwargs:
                self.execute_task_args = (
                    self,
                    f,
                    parent_func,
                    FLSpec._clones,
                    agg_to_collab_ss,
                    kwargs,
                )
            else:
                self.execute_task_args = (self, f, parent_func, kwargs)

        elif str(self._runtime) == "LocalRuntime":
            # update parameters required to execute execute_task function
            self.execute_task_args = [f, parent_func, agg_to_collab_ss, kwargs]
