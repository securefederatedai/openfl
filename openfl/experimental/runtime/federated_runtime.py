# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" openfl.experimental.runtime package LocalRuntime class."""

from __future__ import annotations
from copy import deepcopy
import gc
from openfl.experimental.runtime import Runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openfl.experimental.interface import Aggregator, Collaborator, FLSpec

from openfl.experimental.utilities import (
    generate_artifacts,
    filter_attributes,
    checkpoint,
)
from typing import List, Type, Callable


class FederatedRuntime(Runtime):
    def __init__(
        self,
        aggregator: str = None,
        collaborators: List[str] = None,
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
        if aggregator is not None:
            self.aggregator = aggregator

        if collaborators is not None:
            self.collaborators = collaborators

    @property
    def aggregator(self) -> str:
        """Returns name of _aggregator"""
        return self._aggregator

    @aggregator.setter
    def aggregator(self, aggregator_name: Type[Aggregator]):
        """Set LocalRuntime _aggregator"""
        self._aggregator = aggregator_name

    @property
    def collaborators(self) -> List[str]:
        """
        Return names of collaborators. Don't give direct access to private attributes
        """
        return self.__collaborators

    @collaborators.setter
    def collaborators(self, collaborators: List[Type[Collaborator]]):
        """Set LocalRuntime collaborators"""
        self.__collaborators = collaborators

    # def restore_instance_snapshot(
    #     self, ctx: Type[FLSpec], instance_snapshot: List[Type[FLSpec]]
    # ):
    #     """Restores attributes from backup (in instance snapshot) to ctx"""
    #     for backup in instance_snapshot:
    #         artifacts_iter, _ = generate_artifacts(ctx=backup)
    #         for name, attr in artifacts_iter():
    #             if not hasattr(ctx, name):
    #                 setattr(ctx, name, attr)

    # def execute_task(
    #     self,
    #     flspec_obj: Type[FLSpec],
    #     f: Callable,
    #     **kwargs
    # ):
    #     """
    #     Defines which function to be executed based on name and kwargs
    #     Updates the arguments and executes until end is not reached

    #     Args:
    #         flspec_obj:        Reference to the FLSpec (flow) object. Contains information
    #                            about task sequence, flow attributes.
    #         f:                 The next task to be executed within the flow

    #     Returns:
    #         artifacts_iter: Iterator with updated sequence of values
    #     """
    #     parent_func = None
    #     instance_snapshot = None

    #     while f.__name__ != "end":
    #         if "foreach" in kwargs:
    #             flspec_obj = self.execute_collab_task(
    #                 flspec_obj, f, parent_func, instance_snapshot, **kwargs
    #             )
    #         else:
    #             flspec_obj = self.execute_agg_task(flspec_obj, f)
    #         f, parent_func, instance_snapshot, kwargs = flspec_obj.execute_task_args
    #         # return self.get_collaborator_clone(flspec_obj, f, **kwargs)
    #     else:
    #         flspec_obj = self.execute_agg_task(flspec_obj, f)
    #         f = flspec_obj.execute_task_args[0]

    #         checkpoint(flspec_obj, f)
    #         artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
    #         final_attributes = artifacts_iter()

    # def execute_agg_task(self, flspec_obj, f):
    #     """
    #     Performs execution of aggregator task
    #     Args:
    #         flspec_obj : Reference to the FLSpec (flow) object
    #         f          :  The task to be executed within the flow

    #     Returns:
    #         flspec_obj: updated FLSpec (flow) object
    #     """
    #     aggregator = self._aggregator

    #     flspec_obj = aggregator.execute_func(flspec_obj, f.__name__, self.execute_agg_steps)

    #     print(f"execute_agg_task: f.__name__: {f.__name__}")
    #     return flspec_obj


    # def execute_collab_task(
    #     self, flspec_obj, f, parent_func, instance_snapshot, **kwargs
    # ):
    #     """
    #     Performs
    #         1. Filter include/exclude
    #         2. Set runtime, collab private attributes , metaflow_interface
    #         3. Execution of all collaborator for each task
    #         4. Remove collaborator private attributes
    #         5. Execute the next function after transition

    #     Args:
    #         flspec_obj  :  Reference to the FLSpec (flow) object
    #         f           :  The task to be executed within the flow
    #         parent_func : The prior task executed in the flow
    #         instance_snapshot : A prior FLSpec state that needs to be restored

    #     Returns:
    #         flspec_obj: updated FLSpec (flow) object
    #     """
    #     from openfl.experimental.interface import (
    #         FLSpec,
    #     )

    #     flspec_obj._foreach_methods.append(f.__name__)
    #     selected_collaborators = getattr(flspec_obj, kwargs["foreach"])

    #     # filter exclude/include attributes for clone
    #     self.filter_exclude_include(flspec_obj, f, selected_collaborators, **kwargs)

    #     # set runtime,collab private attributes and metaflowinterface
    #     for col in selected_collaborators:
    #         clone = FLSpec._clones[col]
    #         # Set new LocalRuntime for clone as it is required
    #         # new runtime object will not contain private attributes of
    #         # aggregator or other collaborators
    #         clone.runtime = FederatedRuntime()

    #         # write the clone to the object store
    #         # ensure clone is getting latest _metaflow_interface
    #         clone._metaflow_interface = flspec_obj._metaflow_interface

    #     for collab_name in selected_collaborators:
    #         clone = FLSpec._clones[collab_name]
    #         collaborator = self.__collaborators[collab_name]

    #         collaborator.execute_func(clone, f.__name__, self.execute_collab_steps)

    #     f = getattr(clone, "execute_next")

    #     # Restore the flspec_obj state if back-up is taken
    #     self.restore_instance_snapshot(flspec_obj, instance_snapshot)
    #     del instance_snapshot

    #     g = getattr(flspec_obj, f)

    #     self._aggregator.execute_func(
    #         flspec_obj, g.__name__,
    #         self.execute_agg_steps,
    #         [FLSpec._clones[col] for col in selected_collaborators]
    #     )

    #     gc.collect()
    #     return flspec_obj

    # def filter_exclude_include(self, flspec_obj, f, selected_collaborators, **kwargs):
    #     """
    #     This function filters exclude/include attributes

    #     Args:
    #         flspec_obj  :  Reference to the FLSpec (flow) object
    #         f           :  The task to be executed within the flow
    #         selected_collaborators : all collaborators
    #     """
    #     from openfl.experimental.interface import (
    #         FLSpec,
    #     )

    #     for col in selected_collaborators:
    #         clone = FLSpec._clones[col]
    #         clone.input = col
    #         if ("exclude" in kwargs and hasattr(clone, kwargs["exclude"][0])) or (
    #             "include" in kwargs and hasattr(clone, kwargs["include"][0])
    #         ):
    #             filter_attributes(clone, f, **kwargs)
    #         artifacts_iter, _ = generate_artifacts(ctx=flspec_obj)
    #         for name, attr in artifacts_iter():
    #             setattr(clone, name, deepcopy(attr))
    #         clone._foreach_methods = flspec_obj._foreach_methods

    def __repr__(self):
        return "FederatedRuntime"
