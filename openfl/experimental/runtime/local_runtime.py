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

    def execute_task(
        self,
        flspec_obj: Type[FLSpec],
        f: Callable,
        parent_func: Callable,
        instance_snapshot: List[Type[FLSpec]] = [],
        **kwargs
    ):
        """
        Performs the execution of a task as defined by the
        implementation and underlying backend (single_process, ray, etc)
        on a single node
        Args:
            flspec_obj:        Reference to the FLSpec (flow) object. Contains information
                               about task sequence, flow attributes, that are needed to
                               execute a future task
            f:                 The next task to be executed within the flow
            parent_func:       The prior task executed in the flow
            instance_snapshot: A prior FLSpec state that needs to be restored from
                               (i.e. restoring aggregator state after collaborator
                               execution)
        """
        while f.__name__ != "end":
            if "foreach" in kwargs:
                # save collab first info
                self._collab_start_func,self._collab_start_parent_func,self._collab_start_kwargs,= f, parent_func, kwargs
                f, parent_func, instance_snapshot, kwargs = self.execute_foreach_task(
                    flspec_obj, f, parent_func, instance_snapshot, **kwargs )
            else:
                f,parent_func,instance_snapshot,kwargs,= self.execute_no_transition_task(flspec_obj)
        else:
            self.execute_end_task(flspec_obj, f)

    def execute_no_transition_task(self, flspec_obj):
        flspec_obj.to_exec()
        # update the params
        return flspec_obj.execute_task_args

    def execute_end_task(self, flspec_obj, f):
        from openfl.experimental.interface import (final_attributes)
        global final_attributes
        flspec_obj.to_exec()
        checkpoint(flspec_obj, f)
        artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
        final_attributes = artifacts_iter()
        return

    def execute_foreach_task(
        self, flspec_obj, f, parent_func, instance_snapshot, **kwargs
    ):
        from openfl.experimental.interface import (
            FLSpec,
        )

        agg_func = None
        flspec_obj._foreach_methods.append(f.__name__)
        selected_collaborators = flspec_obj.__getattribute__(kwargs["foreach"])

        self.filter_exclude_include_private_attr(
            flspec_obj, f, parent_func, selected_collaborators, **kwargs
        )
        
        if self.backend == "ray":
            ray_executor = RayExecutor()

        for col in selected_collaborators:
            clone = FLSpec._clones[col]
            # Set new LocalRuntime for clone as it is required
            # and also new runtime object will not contain private attributes of
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

            # execute all collab steps for each collab
            for each_step in flspec_obj._foreach_methods:
                to_exec = getattr(clone, f.__name__)
                if self.backend == "ray":
                    ray_executor.ray_call_put(clone, to_exec)
                else:
                    to_exec()
                    f, parent_func, _, kwargs = clone.execute_task_args
                    if clone._is_at_transition_point(f, parent_func):
                        # get collab starting point for next collab to execute
                        f, parent_func, kwargs = self._collab_start_func,self._collab_start_parent_func,self._collab_start_kwargs
                        break

        if self.backend == "ray":
            # get the initial collab put methods
            clones = ray_executor.get_remote_clones()

            # iterate until all collab steps re finished and get the next set of collab steps
            while not hasattr( clones[0], 'execute_next'):
                for clone_obj in clones:
                    func_name = clone_obj.execute_task_args[0].name
                    to_exec = getattr(clone_obj,func_name)
                    ray_executor.ray_call_put(clone_obj, to_exec)
                
                # update clone
                clones = ray_executor.get_remote_clones()
            
            clone = clones[0]
            FLSpec._clones.update(zip(selected_collaborators, clones))
            del ray_executor
            del clones
            gc.collect()
            
        self.remove_collab_private_attr(selected_collaborators)

        # Restore the flspec_obj state if back-up is taken
        self.restore_instance_snapshot(flspec_obj, instance_snapshot)
        del instance_snapshot

        # get next aggregator function to be executed 
        agg_func = clone.execute_next
        
        g = getattr(flspec_obj, agg_func)
        # remove private collaborator state
        gc.collect()
        g([FLSpec._clones[col] for col in selected_collaborators])
        return flspec_obj.execute_task_args

    def remove_collab_private_attr(self, selected_collaborators):
        # Removes private attributes of collaborator after transition 
        from openfl.experimental.interface import (
            FLSpec,
        )

        for col in selected_collaborators:
            clone = FLSpec._clones[col]
            for attr in self.__collaborators[clone.input].private_attributes:
                if hasattr(clone, attr):
                    self.__collaborators[clone.input].private_attributes[
                        attr
                    ] = getattr(clone, attr)
                    delattr(clone, attr)

    def filter_exclude_include_private_attr(
        self, flspec_obj, f, parent_func, selected_collaborators, **kwargs
    ):
        # This function filters exclude/include attributes
        # Removes private attributes of aggregator
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

            # remove private aggregator state
            if aggregator_to_collaborator(f, parent_func):
                for attr in self._aggregator.private_attributes:
                    self._aggregator.private_attributes[attr] = getattr(
                        flspec_obj, attr
                    )
                    if hasattr(clone, attr):
                        delattr(clone, attr)

    def __repr__(self):
        return "LocalRuntime"
