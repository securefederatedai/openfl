# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.workflow.interface.flspec module."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Callable, List, Type

from openfl.experimental.workflow.utilities import (
    MetaflowInterface,
    aggregator_to_collaborator,
    collaborator_to_aggregator,
    filter_attributes,
    generate_artifacts,
    should_transfer,
)


class FLSpec:
    """FLSpec Class

    A class representing a Federated Learning Specification. It manages clones,
    maintains the initial state, and supports checkpointing.

    Attributes:
        _clones (list): A list of clones created for the FLSpec instance.
        _initial_state (FLSpec or None): The saved initial state of the FLSpec instance.
        _foreach_methods (list): A list of methods to be applied iteratively.
        _checkpoint (bool): A flag indicating whether checkpointing is enabled.
        _collaborators (list): A list of collaborators associated with the runtime.
        _metaflow_interface (MetaflowInterface): The interface to the Metaflow runtime.
        _run_id (str): The ID of the current run.
    """

    _clones = []
    _initial_state = None

    def __init__(self, checkpoint: bool = False) -> None:
        """Initializes the FLSpec object.

        Args:
            checkpoint (bool, optional): Determines whether to checkpoint or
                not. Defaults to False.
        """
        self._foreach_methods = []
        self._checkpoint = checkpoint

    @classmethod
    def _reset_and_create_clones(cls, instance: Type[FLSpec], names: List[str]) -> None:
        """Resets and creates clones for instance for each collaborator in names.

        Args:
            instance (Type[FLSpec]): The instance to be cloned.
            names (List[str]): The list of names for the clones.
        """
        cls._clones = []
        cls._clones = {name: deepcopy(instance) for name in names}

    @classmethod
    def get_clones(cls) -> dict:
        """Returns the _clones dictionary.

        Returns:
            dict: A dictionary mapping collaborator names to their respective
        cloned FLSpec instances.
        """
        return cls._clones

    @classmethod
    def save_initial_state(cls, instance: Type[FLSpec]) -> None:
        """Saves the initial state of an instance before executing the flow.

        Args:
            instance (Type[FLSpec]): The instance whose initial state is to be
                saved.
        """
        cls._initial_state = deepcopy(instance)

    @property
    def checkpoint(self) -> bool:
        """Getter for the checkpoint attribute.

        Returns:
            bool: The current value of the checkpoint.
        """
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, value: bool) -> None:
        """Setter for the checkpoint attribute.

        Args:
            value (bool): The new value for the checkpoint.

        Raises:
            ValueError: If the provided value is not a boolean.
        """
        if not isinstance(value, bool):
            raise ValueError("checkpoint must be a boolean value.")
        self._checkpoint = value

    @property
    def collaborators(self) -> List:
        """Get the list of collaborators.

        Returns:
            _collaborators: A list of collaborators
        """
        return self._collaborators

    @collaborators.setter
    def collaborators(self, collaborators: List) -> None:
        """Set the list of collaborators.

        Args:
            collaborators (List): A list of collaborators to be assigned.
        """
        self._collaborators = collaborators

    def _update_from_flspec_obj(self, flspec_obj: FLSpec) -> None:
        """Update self with attributes from the updated flspec instance.

        Args:
            flspec_obj (FLSpec): Updated Flspec instance
        """
        artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
        for name, attr in artifacts_iter():
            setattr(self, name, deepcopy(attr))

        self._foreach_methods = flspec_obj._foreach_methods

    def _capture_instance_snapshot(self, kwargs) -> List:
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

    def initialize_flow_state(self, collaborators: List, backend: str = "single_process") -> None:
        """
        Sets up the flow's initial state

        Args:
            collaborators (list): A list of collaborators
            backend (str): The runtime backend
        """
        self.collaborators = collaborators
        print("MetaflowInterface creation.")
        self._metaflow_interface = MetaflowInterface(self.__class__, backend)
        self._run_id = self._metaflow_interface.create_run()
        self._foreach_methods = []
        FLSpec._reset_and_create_clones(self, self.collaborators)
        if self._checkpoint:
            print(f"Created flow {self.__class__.__name__}")

    def filter_exclude_include(self, f, **kwargs) -> None:
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

    def restore_instance_snapshot(self, ctx: FLSpec, instance_snapshot: List[FLSpec]) -> None:
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

    def next(self, f, **kwargs) -> None:
        """Specifies the next task in the flow to execute.

        Args:
            f (Callable): The next task that will be executed in the flow.
            **kwargs: Additional keyword arguments.
        """
        # Get the name and reference to the calling function
        parent = inspect.stack()[1][3]
        parent_func = getattr(self, parent)

        # Take back-up of current state of self
        agg_to_collab_ss = None
        if aggregator_to_collaborator(f, parent_func):
            agg_to_collab_ss = self._capture_instance_snapshot(kwargs=kwargs)

        # Remove included / excluded attributes from next task
        filter_attributes(self, f, **kwargs)

        if f.collaborator_step and not f.aggregator_step:
            self._foreach_methods.append(f.__name__)

        # update parameters required to execute next steps
        self.execute_task_args = (
            f,
            parent_func,
            agg_to_collab_ss,
            kwargs,
        )
