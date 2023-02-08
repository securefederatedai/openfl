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
        from openfl.experimental.interface import (
            FLSpec,
            final_attributes,
        )

        global final_attributes

        if "foreach" in kwargs:
            flspec_obj._foreach_methods.append(f.__name__)
            selected_collaborators = flspec_obj.__getattribute__(
                kwargs["foreach"]
            )

            for col in selected_collaborators:
                clone = FLSpec._clones[col]
                if (
                    "exclude" in kwargs and hasattr(clone, kwargs["exclude"][0])
                ) or (
                    "include" in kwargs and hasattr(clone, kwargs["include"][0])
                ):
                    filter_attributes(clone, f, **kwargs)
                artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
                for name, attr in artifacts_iter():
                    setattr(clone, name, deepcopy(attr))
                clone._foreach_methods = flspec_obj._foreach_methods

            for col in selected_collaborators:
                clone = FLSpec._clones[col]
                clone.input = col
                if aggregator_to_collaborator(f, parent_func):
                    # remove private aggregator state
                    for attr in self._aggregator.private_attributes:
                        self._aggregator.private_attributes[attr] = getattr(
                            flspec_obj, attr
                        )
                        if hasattr(clone, attr):
                            delattr(clone, attr)

            func = None
            if self.backend == "ray":
                ray_executor = RayExecutor()
            for col in selected_collaborators:
                clone = FLSpec._clones[col]
                # Set new LocalRuntime for clone as it is required
                # for calling execute_task and also new runtime
                # object will not contain private attributes of
                # aggregator or other collaborators
                clone.runtime = LocalRuntime(backend="single_process")
                for name, attr in self.__collaborators[
                    clone.input
                ].private_attributes.items():
                    setattr(clone, name, attr)
                to_exec = getattr(clone, f.__name__)
                # write the clone to the object store
                # ensure clone is getting latest _metaflow_interface
                clone._metaflow_interface = flspec_obj._metaflow_interface
                if self.backend == "ray":
                    ray_executor.ray_call_put(clone, to_exec)
                else:
                    to_exec()
            if self.backend == "ray":
                clones = ray_executor.get_remote_clones()
                FLSpec._clones.update(
                    {
                        col: obj
                        for col, obj in zip(selected_collaborators, clones)
                    }
                )
                del ray_executor
                del clones
                gc.collect()
            for col in selected_collaborators:
                clone = FLSpec._clones[col]
                func = clone.execute_next
                for attr in self.__collaborators[
                    clone.input
                ].private_attributes:
                    if hasattr(clone, attr):
                        self.__collaborators[clone.input].private_attributes[
                            attr
                        ] = getattr(clone, attr)
                        delattr(clone, attr)
            # Restore the flspec_obj state if back-up is taken
            self.restore_instance_snapshot(flspec_obj, instance_snapshot)
            del instance_snapshot

            g = getattr(flspec_obj, func)
            # remove private collaborator state
            gc.collect()
            g([FLSpec._clones[col] for col in selected_collaborators])
        else:
            to_exec = getattr(flspec_obj, f.__name__)
            to_exec()
            if f.__name__ == "end":
                checkpoint(flspec_obj, f)
                artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
                final_attributes = artifacts_iter()

    def __repr__(self):
        return "LocalRuntime"
