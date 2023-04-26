# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime package LocalRuntime class."""

from __future__ import annotations
from copy import deepcopy
import importlib
import ray
import os
import gc
from openfl.experimental.runtime import Runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from openfl.experimental.interface import Aggregator, Collaborator, FLSpec

from openfl.experimental.utilities import (
    ResourcesNotAvailableError,
    aggregator_to_collaborator,
    generate_artifacts,
    filter_attributes,
    checkpoint,
    get_number_of_gpus,
)
from typing import List, Any
from typing import Dict, Type, Callable


class RayExecutor:
    def __init__(self):
        """Create RayExecutor object"""
        self.__remote_contexts = []

    def ray_call_put(self, collaborator: Collaborator, ctx: Any,
                     f_name: str, callback: Callable) -> None:
        """
        Execute f_name from inside collaborator class with the context 
        of clone (ctx)
        """
        self.__remote_contexts.append(
            collaborator.execute_func.remote(ctx, f_name, callback)
        )

    def get_remote_clones(self) -> List[Any]:
        """
        Get remote clones and delete ray references of clone (ctx) and,
        reclaim memory
        """
        clones = ray.get(self.__remote_contexts)
        del self.__remote_contexts
        self.__remote_contexts = []

        return clones


class LocalRuntime(Runtime):
    def __init__(
        self,
        aggregator: Dict = None,
        collaborators: Dict = None,
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

        if collaborators is not None:
            self.collaborators == self.__get_collaborator_object(collaborators)

    def __get_collaborator_object(self, collaborators: List) -> Any:
        """Get collaborator object based on localruntime backend"""

        if self.backend == "single_process":
            return collaborators

        total_available_cpus = os.cpu_count()
        total_available_gpus = get_number_of_gpus()

        total_required_gpus = sum([collaborator.num_gpus for collaborator in collaborators])
        total_required_cpus = sum([collaborator.num_cpus for collaborator in collaborators])

        if total_available_gpus < total_required_gpus:
            raise ResourcesNotAvailableError(
                    f"cannot assign more than available GPUs ({total_required_gpus} < {total_available_gpus})."
                )
        if total_available_cpus < total_required_cpus:
            raise ResourcesNotAvailableError(
                    f"cannot assign more than available CPUs ({total_required_cpus} < {total_available_cpus})."
            )
        interface_module = importlib.import_module("openfl.experimental.interface")
        collaborator_class = getattr(interface_module, "Collaborator")

        collaborator_ray_refs = []
        for collaborator in collaborators:
            num_cpus = collaborator.num_cpus
            num_gpus = collaborator.num_gpus

            collaborator_actor = ray.remote(collaborator_class).options(
                num_cpus=num_cpus, num_gpus=num_gpus)

            collaborator_ray_refs.append(collaborator_actor.remote(
                name=collaborator.get_name(),
                private_attributes_callable=collaborator.private_attributes_callable,
                **collaborator.kwargs
            ))
        return collaborator_ray_refs

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
        if self.backend == "single_process":
            get_collab_name = lambda collab: collab.get_name()
        else:
            get_collab_name = lambda collab: ray.get(collab.get_name.remote())

        self.__collaborators = {
            get_collab_name(collaborator): collaborator
            for collaborator in collaborators
        }

    def initialize_aggregator(self):
        """initialize aggregator private attributes"""
        self._aggregator.initialize_private_attributes()

    def initialize_collaborators(self):
        """initialize collaborator private attributes"""
        if self.backend == "single_process":
            init_private_attrs = lambda collab: collab.initialize_private_attributes()
        else:
            init_private_attrs = lambda collab: collab.initialize_private_attributes.remote()

        for collaborator in self.__collaborators.values():
            init_private_attrs(collaborator)

    def restore_instance_snapshot(
            self,
            ctx: Type[FLSpec],
            instance_snapshot: List[Type[FLSpec]]
    ):
        """Restores attributes from backup (in instance snapshot) to ctx"""
        for backup in instance_snapshot:
            artifacts_iter, _ = generate_artifacts(ctx=backup)
            for name, attr in artifacts_iter():
                if not hasattr(ctx, name):
                    setattr(ctx, name, attr)

    def execute_collaborator_steps(self, ctx: Any, f_name: str):
        """
        Execute collaborator steps for each 
        collaborator until at transition point
        """
        not_at_transition_point = True
        while not_at_transition_point:
            f = getattr(ctx, f_name)
            f()

            f, parent_func = ctx.execute_task_args[:2]
            if ctx._is_at_transition_point(f, parent_func):
                not_at_transition_point = False
            f_name = f.__name__

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

        interface_module = importlib.import_module("openfl.experimental.interface")
        final_attributes = getattr(interface_module, "final_attributes")

        to_exec = getattr(flspec_obj, f.__name__)
        to_exec()
        checkpoint(flspec_obj, f)
        artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
        final_attributes = artifacts_iter()

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

        from openfl.experimental.interface import (
            FLSpec,
        )

        flspec_obj._foreach_methods.append(f.__name__)
        selected_collaborators = getattr(flspec_obj, kwargs["foreach"])

        # filter exclude/include attributes for clone
        self.filter_exclude_include(flspec_obj, f, selected_collaborators, **kwargs)

        # Remove aggregator private attributes from FLSpec._clones
        for col in selected_collaborators:
            clone = FLSpec._clones[col]
            if aggregator_to_collaborator(f, parent_func):
                for attr in self._aggregator.private_attributes:
                    self._aggregator.private_attributes[attr] = getattr(
                        clone, attr
                    )
                    if hasattr(clone, attr):
                        delattr(clone, attr)

        if self.backend == "ray":
            ray_executor = RayExecutor()
        # set runtime,collab private attributes and metaflowinterface
        for col in selected_collaborators:
            clone = FLSpec._clones[col]
            # Set new LocalRuntime for clone as it is required
            # new runtime object will not contain private attributes of
            # aggregator or other collaborators
            clone.runtime = LocalRuntime(backend="single_process")

            # write the clone to the object store
            # ensure clone is getting latest _metaflow_interface
            clone._metaflow_interface = flspec_obj._metaflow_interface

        for collab_name in selected_collaborators:
            clone = FLSpec._clones[collab_name]
            collaborator = self.__collaborators[collab_name]

            if self.backend == "ray":
                ray_executor.ray_call_put(collaborator, clone, f.__name__, self.execute_collaborator_steps)
            else:
                collaborator.execute_func(clone, f.__name__, self.execute_collaborator_steps)

        if self.backend == "ray":
            clones = ray_executor.get_remote_clones()
            FLSpec._clones.update(zip(selected_collaborators, clones))
            f = getattr(clones[0], "execute_next")
            del ray_executor
            del clones
            gc.collect()
        else:
            f = getattr(clone, "execute_next")

        # Restore the flspec_obj state if back-up is taken
        self.restore_instance_snapshot(flspec_obj, instance_snapshot)
        del instance_snapshot

        g = getattr(flspec_obj, f)
        gc.collect()
        g([FLSpec._clones[col] for col in selected_collaborators])
        return flspec_obj.execute_task_args

    def filter_exclude_include(self, flspec_obj, f, selected_collaborators, **kwargs):
        """
        This function filters exclude/include attributes
        Args:
            flspec_obj  :  Reference to the FLSpec (flow) object
            f           :  The task to be executed within the flow
            selected_collaborators : all collaborators
        """

        from openfl.experimental.interface import (
            FLSpec,
        )

        for col in selected_collaborators:
            clone = FLSpec._clones[col]
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
