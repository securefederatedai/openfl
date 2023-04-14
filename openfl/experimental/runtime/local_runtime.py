# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime package LocalRuntime class."""

from __future__ import annotations
from copy import deepcopy
import ray
import gc
from openfl.experimental.runtime import Runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openfl.experimental.interface import Aggregator, Collaborator, FLSpec
from openfl.experimental.placement import RayExecutor
from openfl.experimental.utilities import (
    aggregator_to_collaborator,
    generate_artifacts,
    filter_attributes,
    checkpoint,
)
from typing import List
from typing import Type
from typing import Callable
import importlib


class LocalRuntime(Runtime):
    def __init__(
        self,
        aggregator: Type[Aggregator] = None,
        collaborators: List[Type[Collaborator]] = None,
        backend: str = "single_process",
        **kwargs,
    ) -> None:
        """
        Use single node to run the flow

        Args:
            aggregator:    The aggregator instance that holds private attributes
            collaborators: A list of collaborators; each with their own private attributes
            backend:       The backend that will execute the tasks. Available options are:

                           'single_process': (default) Executes every task within the same process

                           'ray':            Executes tasks using the Ray library. Each participant
                                             runs tasks in their own isolated process. Also
                                             supports GPU isolation using Ray's 'num_gpus'
                                             argument, which can be passed in through the
                                             collaborator placement decorator.

                                             Example:
                                             @collaborator(num_gpus=1)
                                             def some_collaborator_task(self):
                                                 ...


                                             By selecting num_gpus=1, the task is guaranteed
                                             exclusive GPU access. If the system has one GPU,
                                             collaborator tasks will run sequentially.
        """
        super().__init__()
        if backend not in ["ray", "single_process"]:
            raise ValueError(
                f"Invalid 'backend' value '{backend}', accepted values are "
                + "'ray', or 'single_process'"
            )
        if backend == "ray":
            if not ray.is_initialized():
                dh = kwargs.get("dashboard_host", "127.0.0.1")
                dp = kwargs.get("dashboard_port", 5252)
                ray.init(dashboard_host=dh, dashboard_port=dp)
        self.backend = backend
        if aggregator is not None:
            self.aggregator = aggregator
        # List of envoys should be iterable, so that a subset can be selected at runtime
        # The envoys is the superset of envoys that can be selected during the experiment
        if collaborators is not None:
            self.collaborators = collaborators

    @property
    def aggregator(self) -> str:
        """Returns name of _aggregator"""
        return self._aggregator.name

    @aggregator.setter
    def aggregator(self, aggregator: Type[Aggregator]):
        """Set LocalRuntime _aggregator"""
        self._aggregator = aggregator

    @property
    def collaborators(self) -> List[str]:
        """
        Return names of collaborators. Don't give direct access to private attributes
        """
        return list(self.__collaborators.keys())

    @collaborators.setter
    def collaborators(self, collaborators: List[Type[Collaborator]]):
        """Set LocalRuntime collaborators"""
        self.__collaborators = {
            collaborator.name: collaborator for collaborator in collaborators
        }

    def restore_instance_snapshot(
        self, ctx: Type[FLSpec], instance_snapshot: List[Type[FLSpec]]
    ):
        """Restores attributes from backup (in instance snapshot) to ctx"""
        for backup in instance_snapshot:
            artifacts_iter, _ = generate_artifacts(ctx=backup)
            for name, attr in artifacts_iter():
                if not hasattr(ctx, name):
                    setattr(ctx, name, attr)

    def execute_task(
        self,
        flspec_obj: Type[FLSpec],
        f: Callable,
        parent_func: Callable,
        instance_snapshot: List[Type[FLSpec]] = [],
        **kwargs,
    ):
        """
        Defines which function to be executed based on name and kwargs
        Updates the arguments and executes until end is not reached

        Args:
            flspec_obj:        Reference to the FLSpec (flow) object. Contains information
                               about task sequence, flow attributes.
            f:                 The next task to be executed within the flow
            parent_func:       The prior task executed in the flow
            instance_snapshot: A prior FLSpec state that needs to be restored from
                               (i.e. restoring aggregator state after collaborator
                               execution)
        """

        while f.__name__ != "end":
            if "foreach" in kwargs:
                f, parent_func, instance_snapshot, kwargs = self.execute_foreach_task(
                    flspec_obj, f, parent_func, instance_snapshot, **kwargs
                )
            else:
                f, parent_func, instance_snapshot, kwargs = self.execute_agg_task(
                    flspec_obj, f
                )
        else:
            self.execute_end_task(flspec_obj, f)

    def execute_agg_task(self, flspec_obj, f):
        """
        Performs execution of aggregator task
        Args:
            flspec_obj : Reference to the FLSpec (flow) object
            f          :  The task to be executed within the flow

        Returns:
            list: updated arguments to be executed
        """

        to_exec = getattr(flspec_obj, f.__name__)
        to_exec()
        return flspec_obj.execute_task_args

    def execute_end_task(self, flspec_obj, f):
        """
        Performs execution of end task
        Args:
            flspec_obj : Reference to the FLSpec (flow) object
            f          :  The task to be executed within the flow

        Returns:
            list: updated arguments to be executed
        """

        global final_attributes
        final_attr_module = importlib.import_module("openfl.experimental.interface")
        final_attributes = getattr(final_attr_module, "final_attributes")

        to_exec = getattr(flspec_obj, f.__name__)
        to_exec()
        checkpoint(flspec_obj, f)
        artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
        final_attributes = artifacts_iter()
        return

    def execute_foreach_task(
        self, flspec_obj, f, parent_func, instance_snapshot, **kwargs
    ):
        """
        Performs
            1. Filter include/exclude
            2. Remove aggregator private attributes
            3. Set runtime, collab private attributes , metaflow_interface
            4. Execution of all collaborator for each task
            5. Remove collaborator private attributes
            6. Execute the next function after transition

        Args:
            flspec_obj  :  Reference to the FLSpec (flow) object
            f           :  The task to be executed within the flow
            parent_func : The prior task executed in the flow
            instance_snapshot : A prior FLSpec state that needs to be restored

        Returns:
            list: updated arguments to be executed
        """

        flspec_module = importlib.import_module("openfl.experimental.interface")
        flspec_class = getattr(flspec_module, "FLSpec")
        flspec_obj._foreach_methods.append(f.__name__)
        selected_collaborators = getattr(flspec_obj, kwargs["foreach"])

        # filter exclude/include attributes for clone
        self.filter_exclude_include(flspec_obj, f, selected_collaborators, **kwargs)

        # Remove aggregator private attributes
        for col in selected_collaborators:
            clone = flspec_class._clones[col]
            if aggregator_to_collaborator(f, parent_func):
                for attr in self._aggregator.private_attributes:
                    self._aggregator.private_attributes[attr] = getattr(
                        flspec_obj, attr
                    )
                    if hasattr(clone, attr):
                        delattr(clone, attr)

        if self.backend == "ray":
            ray_executor = RayExecutor()

        # set runtime,collab private attributes and metaflowinterface
        for col in selected_collaborators:
            clone = flspec_class._clones[col]
            # Set new LocalRuntime for clone as it is required
            # new runtime object will not contain private attributes of
            # aggregator or other collaborators
            clone.runtime = LocalRuntime(backend="single_process")

            # set collab private attributes
            for name, attr in self.__collaborators[
                clone.input
            ].private_attributes.items():
                setattr(clone, name, attr)

            # write the clone to the object store
            # ensure clone is getting latest _metaflow_interface
            clone._metaflow_interface = flspec_obj._metaflow_interface

        # For initial step assume there is no trasition from collab_to_agg
        not_at_transition_point = True

        # loop until there is no transition
        while not_at_transition_point:
            # execute to_exec for for each collab
            for collab in selected_collaborators:
                clone = flspec_class._clones[collab]
                # get the function to be executed
                to_exec = getattr(clone, f.__name__)

                if self.backend == "ray":
                    ray_executor.ray_call_put(clone, to_exec)
                else:
                    to_exec()

            if self.backend == "ray":
                # Execute the collab steps
                clones = ray_executor.get_remote_clones()
                flspec_class._clones.update(zip(selected_collaborators, clones))

            # update the next arguments
            f, parent_func, _, kwargs = flspec_class._clones[collab].execute_task_args

            # check for transition
            if flspec_class._clones[collab]._is_at_transition_point(f, parent_func):
                not_at_transition_point = False

        # remove clones after transition
        if self.backend == "ray":
            del ray_executor
            del clones
            gc.collect()

        # Removes collaborator private attributes after transition
        for col in selected_collaborators:
            clone = flspec_class._clones[col]
            for attr in self.__collaborators[clone.input].private_attributes:
                if hasattr(clone, attr):
                    self.__collaborators[clone.input].private_attributes[
                        attr
                    ] = getattr(clone, attr)
                    delattr(clone, attr)

        # Restore the flspec_obj state if back-up is taken
        self.restore_instance_snapshot(flspec_obj, instance_snapshot)
        del instance_snapshot

        g = getattr(flspec_obj, f.__name__)
        gc.collect()
        g([flspec_class._clones[col] for col in selected_collaborators])
        return flspec_obj.execute_task_args

    def filter_exclude_include(self, flspec_obj, f, selected_collaborators, **kwargs):
        """
        This function filters exclude/include attributes
        Args:
            flspec_obj  :  Reference to the FLSpec (flow) object
            f           :  The task to be executed within the flow
            selected_collaborators : all collaborators
        """

        flspec_module = importlib.import_module("openfl.experimental.interface")
        flspec_class = getattr(flspec_module, "FLSpec")

        for col in selected_collaborators:
            clone = flspec_class._clones[col]
            clone.input = col
            if ("exclude" in kwargs and hasattr(clone, kwargs["exclude"][0])) or (
                "include" in kwargs and hasattr(clone, kwargs["include"][0])
            ):
                filter_attributes(clone, f, **kwargs)
            artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
            for name, attr in artifacts_iter():
                setattr(clone, name, deepcopy(attr))
            clone._foreach_methods = flspec_obj._foreach_methods

    def __repr__(self):
        return "LocalRuntime"
