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
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from openfl.experimental.interface import Aggregator, Collaborator, FLSpec

from openfl.experimental.utilities import (
    ResourcesNotAvailableError,
    aggregator_to_collaborator,
    generate_artifacts,
    filter_attributes,
    checkpoint,
    get_number_of_gpus,
    check_resource_allocation,
)
from typing import List, Any
from typing import Dict, Type, Callable


class RayExecutor:
    def __init__(self):
        """Create RayExecutor object"""
        self.__remote_contexts = []

    def ray_call_put(
        self, participant: Any, ctx: Any, f_name: str, callback: Callable,
        clones: Optional[Any] = None
    ) -> None:
        """
        Execute f_name from inside participant (Aggregator or Collaborator) class with the context
        of clone (ctx)
        """
        if clones is not None:
            self.__remote_contexts.append(
                participant.execute_func.remote(ctx, f_name, callback, clones)
            )
        else:
            self.__remote_contexts.append(
                participant.execute_func.remote(ctx, f_name, callback)
            )

    def ray_call_get(self) -> List[Any]:
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
            self.aggregator = self.__get_aggregator_object(aggregator)

        if collaborators is not None:
            self.collaborators = self.__get_collaborator_object(collaborators)

    def __get_aggregator_object(self, aggregator: Type[Aggregator]) -> Any:
        """Get aggregator object based on localruntime backend"""

        if self.backend == "single_process":
            return aggregator

        total_available_cpus = os.cpu_count()
        total_available_gpus = get_number_of_gpus()

        agg_cpus = aggregator.num_cpus
        agg_gpus = aggregator.num_gpus

        if agg_gpus > 0:
            check_resource_allocation(
                total_available_gpus,
                {aggregator.get_name(): agg_gpus},
            )

        if total_available_gpus < agg_gpus:
            raise ResourcesNotAvailableError(
                f"cannot assign more than available GPUs \
                    ({agg_gpus} < {total_available_gpus})."
            )
        if total_available_cpus < agg_cpus:
            raise ResourcesNotAvailableError(
                f"cannot assign more than available CPUs \
                    ({agg_cpus} < {total_available_cpus})."
            )

        interface_module = importlib.import_module("openfl.experimental.interface")
        aggregator_class = getattr(interface_module, "Aggregator")

        aggregator_actor = ray.remote(aggregator_class).options(
            num_cpus=agg_cpus, num_gpus=agg_gpus
        )
        aggregator_actor_ref = aggregator_actor.remote(
            name=aggregator.get_name(),
            private_attributes_callable=aggregator.private_attributes_callable,
            **aggregator.kwargs,
        )

        return aggregator_actor_ref

    def __get_collaborator_object(self, collaborators: List) -> Any:
        """Get collaborator object based on localruntime backend"""

        if self.backend == "single_process":
            return collaborators

        total_available_cpus = os.cpu_count()
        total_available_gpus = get_number_of_gpus()

        total_required_gpus = sum(
            [collaborator.num_gpus for collaborator in collaborators]
        )
        total_required_cpus = sum(
            [collaborator.num_cpus for collaborator in collaborators]
        )
        if total_required_gpus > 0:
            check_resource_allocation(
                total_available_gpus,
                {collab.get_name(): collab.num_gpus for collab in collaborators},
            )

        if total_available_gpus < total_required_gpus:
            raise ResourcesNotAvailableError(
                f"cannot assign more than available GPUs \
                    ({total_required_gpus} < {total_available_gpus})."
            )
        if total_available_cpus < total_required_cpus:
            raise ResourcesNotAvailableError(
                f"cannot assign more than available CPUs \
                    ({total_required_cpus} < {total_available_cpus})."
            )
        interface_module = importlib.import_module("openfl.experimental.interface")
        collaborator_class = getattr(interface_module, "Collaborator")

        collaborator_ray_refs = []
        for collaborator in collaborators:
            num_cpus = collaborator.num_cpus
            num_gpus = collaborator.num_gpus

            collaborator_actor = ray.remote(collaborator_class).options(
                num_cpus=num_cpus, num_gpus=num_gpus
            )

            collaborator_ray_refs.append(
                collaborator_actor.remote(
                    name=collaborator.get_name(),
                    private_attributes_callable=collaborator.private_attributes_callable,
                    **collaborator.kwargs,
                )
            )
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
            def get_collab_name(collab):
                return collab.get_name()
        else:
            def get_collab_name(collab):
                return ray.get(collab.get_name.remote())

        self.__collaborators = {
            get_collab_name(collaborator): collaborator
            for collaborator in collaborators
        }

    def initialize_aggregator(self):
        """initialize aggregator private attributes"""
        if self.backend == "single_process":
            self._aggregator.initialize_private_attributes()
        else:
            self._aggregator.initialize_private_attributes.remote()

    def initialize_collaborators(self):
        """initialize collaborator private attributes"""
        if self.backend == "single_process":
            def init_private_attrs(collab):
                return collab.initialize_private_attributes()
        else:
            def init_private_attrs(collab):
                return collab.initialize_private_attributes.remote()

        for collaborator in self.__collaborators.values():
            init_private_attrs(collaborator)

    def restore_instance_snapshot(
        self, ctx: Type[FLSpec], instance_snapshot: List[Type[FLSpec]]
    ):
        """Restores attributes from backup (in instance snapshot) to ctx"""
        for backup in instance_snapshot:
            artifacts_iter, _ = generate_artifacts(ctx=backup)
            for name, attr in artifacts_iter():
                if not hasattr(ctx, name):
                    setattr(ctx, name, attr)

    def execute_agg_steps(self, ctx: Any, f_name: str, clones: Optional[Any] = None):
        """
        Execute aggregator steps until at transition point
        """
        if clones is not None:
            f = getattr(ctx, f_name)
            f(clones)
        else:
            not_at_transition_point = True
            while not_at_transition_point:
                f = getattr(ctx, f_name)
                f()

                f, parent_func = ctx.execute_task_args[:2]
                if aggregator_to_collaborator(f, parent_func) or f.__name__ == "end":
                    not_at_transition_point = False

                f_name = f.__name__

    def execute_collab_steps(self, ctx: Any, f_name: str):
        """
        Execute collaborator steps until at transition point
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
        **kwargs
    ):
        """
        Defines which function to be executed based on name and kwargs
        Updates the arguments and executes until end is not reached

        Args:
            flspec_obj:        Reference to the FLSpec (flow) object. Contains information
                               about task sequence, flow attributes.
            f:                 The next task to be executed within the flow

        Returns:
            artifacts_iter: Iterator with updated sequence of values
        """
        parent_func = None
        instance_snapshot = None
        self.join_step = False

        while f.__name__ != "end":
            if "foreach" in kwargs:
                flspec_obj = self.execute_collab_task(
                    flspec_obj, f, parent_func, instance_snapshot, **kwargs
                )
            else:
                flspec_obj = self.execute_agg_task(flspec_obj, f)
            f, parent_func, instance_snapshot, kwargs = flspec_obj.execute_task_args
        else:
            flspec_obj = self.execute_agg_task(flspec_obj, f)
            f = flspec_obj.execute_task_args[0]

            checkpoint(flspec_obj, f)
            artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
            return artifacts_iter()

    def execute_agg_task(self, flspec_obj, f):
        """
        Performs execution of aggregator task
        Args:
            flspec_obj : Reference to the FLSpec (flow) object
            f          :  The task to be executed within the flow

        Returns:
            flspec_obj: updated FLSpec (flow) object
        """
        from openfl.experimental.interface import FLSpec
        aggregator = self._aggregator
        clones = None

        if self.join_step:
            clones = [FLSpec._clones[col] for col in self.selected_collaborators]
            self.join_step = False

        if self.backend == "ray":
            ray_executor = RayExecutor()
            ray_executor.ray_call_put(
                aggregator, flspec_obj,
                f.__name__, self.execute_agg_steps,
                clones
            )
            flspec_obj = ray_executor.ray_call_get()[0]
            del ray_executor
        else:
            aggregator.execute_func(
                flspec_obj, f.__name__, self.execute_agg_steps,
                clones
            )

        gc.collect()
        return flspec_obj

    def execute_collab_task(
        self, flspec_obj, f, parent_func, instance_snapshot, **kwargs
    ):
        """
        Performs
            1. Filter include/exclude
            2. Set runtime, collab private attributes , metaflow_interface
            3. Execution of all collaborator for each task
            4. Remove collaborator private attributes
            5. Execute the next function after transition

        Args:
            flspec_obj  :  Reference to the FLSpec (flow) object
            f           :  The task to be executed within the flow
            parent_func : The prior task executed in the flow
            instance_snapshot : A prior FLSpec state that needs to be restored

        Returns:
            flspec_obj: updated FLSpec (flow) object
        """

        from openfl.experimental.interface import (
            FLSpec,
        )

        flspec_obj._foreach_methods.append(f.__name__)
        selected_collaborators = getattr(flspec_obj, kwargs["foreach"])
        self.selected_collaborators = selected_collaborators

        # filter exclude/include attributes for clone
        self.filter_exclude_include(flspec_obj, f, selected_collaborators, **kwargs)

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
                ray_executor.ray_call_put(
                    collaborator, clone, f.__name__, self.execute_collab_steps
                )
            else:
                collaborator.execute_func(clone, f.__name__, self.execute_collab_steps)

        if self.backend == "ray":
            clones = ray_executor.ray_call_get()
            FLSpec._clones.update(zip(selected_collaborators, clones))
            clone = clones[0]
            del clones

        flspec_obj.execute_task_args = clone.execute_task_args

        # Restore the flspec_obj state if back-up is taken
        self.restore_instance_snapshot(flspec_obj, instance_snapshot)
        del instance_snapshot

        gc.collect()
        # Setting the join_step to indicate to aggregator to collect clones
        self.join_step = True
        return flspec_obj

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